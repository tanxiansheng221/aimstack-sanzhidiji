/* aimp M2: 前向引擎实现
 * 权重驻留: Q8_0/F32 原始块驻留, matmul 现场反量化 (省内存: 2.49GB vs 10.6GB fp32)
 * 反量化单乘精确: q8 值 * f32 scale, 无二次舍入
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include "forward.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

static const aim_tensor *find_tensor(const aim_gguf *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (!strcmp(g->tensors[i].name, name)) return &g->tensors[i];
    return NULL;
}

static uint8_t *tensor_ptr(aim_model *m, const aim_tensor *t) {
    return m->blob + t->offset;
}

/* 反量化 Q8_0 块 [n_blocks x 34B] -> fp32 [n_blocks*32] */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { float z; bits = sign; memcpy(&z, &bits, 4); return z; }
        float v = (float)man * 5.9604644775390625e-8f;
        return sign ? -v : v;
    }
    if (exp == 31) bits = sign | 0x7F800000u | (man << 13);
    else           bits = sign | ((exp + 112u) << 23) | (man << 13);
    float r; memcpy(&r, &bits, 4); return r;
}

static void deq_q8_0(const uint8_t *src, float *dst, uint64_t n_blocks) {
    for (uint64_t b = 0; b < n_blocks; b++) {
        uint16_t h16;
        memcpy(&h16, src + b * 34, 2);
        float scale = f16_to_f32(h16);
        const int8_t *q = (const int8_t *)(src + b * 34 + 2);
        float *d = dst + b * 32;
        for (int i = 0; i < 32; i++)
            d[i] = scale * (float)q[i];   /* 单乘精确 */
    }
}

/* ---------- M2.11: 行级线程池 (Win32 自旋, 零依赖) ----------
 * 行间零依赖: 每行内部求和序不变 (契约不破坏); 分块动态领取;
 * rows < min_rows 走串行, 线程数 = 逻辑核 - 1 */
typedef struct {
    const void *w; const float *x; float *y;
    uint32_t in_dim; float scale; const float *lut;
    uint32_t bk;       /* 批量列数 (0=单列): x 列主 [k][in_dim], y 列主 [k][ystride] */
    uint32_t ystride;  /* 批量 y 列跨度 (=rows); 单列时忽略 */
    int32_t  sumx;     /* mt patch: VNNI path, sum of the int8 activations */
} mv_ctx;

#ifdef _WIN32
#define AIM_MAX_WORKERS 15
static HANDLE g_th[AIM_MAX_WORKERS];
static volatile LONG g_nw = 0, g_pool_on = 0;
static volatile LONG g_gen = 0, g_next = 0, g_total = 0, g_remaining = 0;
static volatile LONG g_chunk = 16;
static void (*g_fn)(const mv_ctx *, uint32_t);
static const mv_ctx *g_cx;

static volatile LONG g_dbg = 0;

static DWORD WINAPI aim_worker(LPVOID p) {
    LONG id = (LONG)(LONG_PTR)p;
    LONG my = 0;
    for (;;) {
        LONG v = g_gen;
        if (v != my) {
            if (g_dbg) fprintf(stderr, "[w%ld] start gen=%ld\n", id, v);
            my = v;
            LONG got = 0;
            LARGE_INTEGER t0, t1, fq;
            QueryPerformanceCounter(&t0);
            for (;;) {
                LONG i = InterlockedExchangeAdd(&g_next, g_chunk);
                if (i >= g_total) break;
                LONG e = i + g_chunk;
                if (e > g_total) e = g_total;
                got += e - i;
                for (LONG u = i; u < e; u++) g_fn(g_cx, (uint32_t)u);
            }
            QueryPerformanceCounter(&t1);
            QueryPerformanceFrequency(&fq);
            if (g_dbg) fprintf(stderr, "[w%ld] done gen=%ld got=%ld %.1fms\n", id, v, got,
                               (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)fq.QuadPart);
            InterlockedDecrement(&g_remaining);
            WakeByAddressAll((void *)&g_remaining);
        } else {
            if (g_dbg) fprintf(stderr, "[w%ld] sleep v=%ld\n", id, v);
            /* 阻塞等发车: *g_gen == v 期间休眠; 500ms 超时兜底防丢醒 */
            WaitOnAddress((volatile VOID *)&g_gen, &v, sizeof(LONG), 500);
        }
    }
    return 0;
}

/* 自校准 dummy: 纯算术行负载 (无内存带宽干扰, 量调度/并行效率) */
static void cal_row(const mv_ctx *c, uint32_t o) {
    const float *x = c->x;
    float acc = 0.0f;
    for (uint32_t r = 0; r < 64u; r++)
        for (uint32_t i = 0; i < c->in_dim; i++)
            acc += x[i] * (float)(i * (o + r + 1u));
    c->y[o] = acc;
}

static double now_ms(void) {
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)(t.QuadPart) * 1000.0 / (double)f.QuadPart;
}

static void aim_pool_init(void) {
    /* 默认串行: 弱核/带宽受限机器上多线程负收益 (实测 lm_head 并行慢 8 倍)。
     * AIM_WORKERS=n 显式开启; 自校准护栏会把不快的并行自动降级。 */
    LONG nw = 0;
    char eb[16];
    if (GetEnvironmentVariableA("AIM_WORKERS", eb, sizeof(eb)) > 0)
        nw = atol(eb);
    if (GetEnvironmentVariableA("AIM_PAR_DBG", eb, sizeof(eb)) > 0)
        g_dbg = atol(eb);
    if (nw > AIM_MAX_WORKERS) nw = AIM_MAX_WORKERS;
    /* 杀 EcoQoS: 禁用进程级电源节流 (Win10 1709+) —— STALL 真身候选:
     * worker 被宿主 EcoQoS 降速冻结 (曾实测睡眠后 5s 不被唤醒)。
     * StateMask=0 = 节流关闭; 失败静默 (旧系统无此 API, 语义照旧)。 */
    {
        PROCESS_POWER_THROTTLING_STATE st;
        st.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
        st.StateMask = 0;
        SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                              &st, sizeof(st));
    }
    for (LONG i = 0; i < nw; i++) {
        g_th[i] = CreateThread(NULL, 0, aim_worker, (LPVOID)(LONG_PTR)i, 0, NULL);
        if (!g_th[i]) break;
        SetThreadPriority(g_th[i], THREAD_PRIORITY_ABOVE_NORMAL);
        g_nw = i + 1;
    }
    g_pool_on = 1;
    if (g_nw > 0 && nw > 0) {
        /* 自校准: 同一负载并行 vs 串行实测; 并行不显著更快 -> 降级串行
         * (防沙箱/Job CPU 限额环境线程轮饿, 真机不受影响) */
        enum { CAL_ROWS = 2048, CAL_DIM = 256 };
        static float cal_x[CAL_DIM], cal_y1[CAL_ROWS], cal_y2[CAL_ROWS];
        for (uint32_t i = 0; i < CAL_DIM; i++) cal_x[i] = (float)i * 0.001f;
        mv_ctx cc;
        cc.w = NULL; cc.x = cal_x; cc.y = cal_y1; cc.in_dim = CAL_DIM;
        cc.scale = 1.0f; cc.lut = NULL; cc.bk = 0; cc.ystride = 0;
        LONG nw_save = g_nw;
        double t0 = now_ms();
        for (uint32_t o = 0; o < CAL_ROWS; o++) cal_row(&cc, o);
        double ts = now_ms() - t0;
        g_nw = nw_save;
        cc.y = cal_y2;
        t0 = now_ms();
        {
            LONG save_total = g_total, save_next = g_next, save_chunk = g_chunk;
            void (*save_fn)(const mv_ctx *, uint32_t) = g_fn;
            const mv_ctx *save_cx = g_cx;
            LONG save_rem = g_remaining;
            g_fn = cal_row; g_cx = &cc; g_total = CAL_ROWS; g_next = 0;
            g_chunk = CAL_ROWS / ((g_nw + 1) * 8);
            if (g_chunk < 1) g_chunk = 1;
            g_remaining = g_nw;
            InterlockedIncrement(&g_gen);
            WakeByAddressAll((void *)&g_gen);
            for (;;) {
                LONG i2 = InterlockedExchangeAdd(&g_next, g_chunk);
                if (i2 >= g_total) break;
                LONG e2 = i2 + g_chunk;
                if (e2 > g_total) e2 = g_total;
                for (LONG u = i2; u < e2; u++) cal_row(&cc, (uint32_t)u);
            }
            while (g_remaining > 0) {
                LONG r = g_remaining;
                WaitOnAddress((volatile VOID *)&g_remaining, &r, sizeof(LONG), 5000);
            }
            g_total = save_total; g_next = save_next; g_chunk = save_chunk;
            g_fn = save_fn; g_cx = save_cx; g_remaining = save_rem;
        }
        double tp = now_ms() - t0;
        int mismatch = memcmp(cal_y1, cal_y2, CAL_ROWS * sizeof(float)) != 0;
        if (g_dbg)
            fprintf(stderr, "[par] calib serial=%.1fms parallel=%.1fms nw=%ld mismatch=%d\n",
                    ts, tp, g_nw, mismatch);
        if (mismatch || tp > ts * 0.75) {
            if (g_dbg) fprintf(stderr, "[par] fallback -> serial\n");
            g_nw = 0;                      /* 降级: 全串行 */
        }
    }
}

static void parallel_for(const mv_ctx *c, uint32_t rows, uint32_t min_rows,
                         void (*fn)(const mv_ctx *, uint32_t)) {
    if (!g_pool_on) aim_pool_init();
    if (g_nw <= 0 || rows < min_rows) {
        for (uint32_t o = 0; o < rows; o++) fn(c, o);
        return;
    }
    g_fn = fn;
    g_cx = c;
    g_total = (LONG)rows;
    g_next = 0;
    LONG chunk = (LONG)rows / ((g_nw + 1) * 8);
    if (chunk < 1) chunk = 1;
    g_chunk = chunk;
    g_remaining = g_nw;
    InterlockedIncrement(&g_gen);          /* 发车 (locked = 全屏障) */
    WakeByAddressAll((void *)&g_gen);      /* 唤醒休眠 worker */
    for (;;) {                             /* 主线程同领块干活 */
        LONG i = InterlockedExchangeAdd(&g_next, g_chunk);
        if (i >= g_total) break;
        LONG e = i + chunk;
        if (e > g_total) e = g_total;
        for (LONG u = i; u < e; u++) fn(c, (uint32_t)u);
    }
    while (g_remaining > 0) {
        LONG r = g_remaining;
        WaitOnAddress((volatile VOID *)&g_remaining, &r, sizeof(LONG), 5000);
        if (g_remaining > 0) {
            /* 5s 未齐: 诊断现场 (AIM_PAR_DBG=1 时打印) */
            static int diag_shown = 0;
            if (g_remaining > 0 && !diag_shown) {
                diag_shown = 1;
                fprintf(stderr, "[par] STALL rows=%u nw=%ld chunk=%ld remaining=%ld next=%ld total=%ld\n",
                        rows, g_nw, chunk, g_remaining, g_next, g_total);
            }
        }
    }
}
#else
/* ---------- POSIX row pool (mt patch): pthread + C11 atomics + futex, same design as the Win32 pool above.
 * Rows are independent and each row's summation order is unchanged, so threaded output is bit-identical to serial.
 * Env: AIM_WORKERS=n extra worker threads (main thread works too; default 0 = serial), AIM_SPIN=n spin iterations
 * before a worker sleeps on the futex (default 30000), AIM_MIN_ROWS=n override the min-rows threshold. */
#include <pthread.h>
#include <stdatomic.h>
#include <limits.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <sched.h>
#define AIM_MAX_WORKERS 63
#define CPU_RELAX() __builtin_ia32_pause()
static int g_cpus[1024], g_ncpus = 0;
/* one hardware thread per physical core first (SMT siblings last), so N threads land on N cores */
static void pick_cpus(void) {
    cpu_set_t all; CPU_ZERO(&all);
    if (sched_getaffinity(0, sizeof all, &all) != 0) return;
    for (int pass = 0; pass < 2; pass++)
        for (int c = 0; c < 1024; c++) {
            if (!CPU_ISSET(c, &all)) continue;
            char p[128]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
            int first = c; FILE *f = fopen(p, "r");
            if (f) { if (fscanf(f, "%d", &first) != 1) first = c; fclose(f); }
            int primary = (first == c);
            if ((pass == 0) != primary) continue;
            g_cpus[g_ncpus++] = c;
        }
}
static pthread_t g_th[AIM_MAX_WORKERS];
static int g_nw = 0, g_pool_on = 0;
static uint32_t g_min_rows_override = 0;
static long g_spin = 30000;
static _Alignas(64) _Atomic uint32_t g_gen = 0;
static _Alignas(64) _Atomic uint32_t g_next = 0;
static _Alignas(64) _Atomic uint32_t g_remaining = 0;
static _Alignas(64) _Atomic uint32_t g_sleepers = 0;
static _Alignas(64) _Atomic uint32_t g_main_sleep = 0;
static uint32_t g_total = 0, g_chunk = 16;
static void (*g_fn)(const mv_ctx *, uint32_t);
static const mv_ctx *g_cx;
static long futex_op(_Atomic uint32_t *addr, int op, uint32_t val) {
    return syscall(SYS_futex, (uint32_t *)addr, op | FUTEX_PRIVATE_FLAG, val, NULL, NULL, 0);
}
static inline void claim_loop(void) {
    for (;;) {
        uint32_t i = atomic_fetch_add(&g_next, g_chunk);
        if (i >= g_total) break;
        uint32_t e = i + g_chunk; if (e > g_total) e = g_total;
        for (uint32_t u = i; u < e; u++) g_fn(g_cx, u);
    }
}
static void *aim_worker(void *p) {
    (void)p;
    uint32_t my = 0;
    for (;;) {
        uint32_t v = atomic_load(&g_gen);
        if (v == my) {
            for (long s = 0; s < g_spin; s++) { CPU_RELAX(); if ((v = atomic_load(&g_gen)) != my) break; }
            if (v == my) {
                atomic_fetch_add(&g_sleepers, 1);
                if (atomic_load(&g_gen) == my) futex_op(&g_gen, FUTEX_WAIT, my);
                atomic_fetch_sub(&g_sleepers, 1);
                continue;
            }
        }
        my = v;
        claim_loop();
        if (atomic_fetch_sub(&g_remaining, 1) == 1 && atomic_load(&g_main_sleep)) futex_op(&g_remaining, FUTEX_WAKE, 1);
    }
    return NULL;
}
static void aim_pool_init(void) {
    const char *e;
    int nw = 0;
    if ((e = getenv("AIM_WORKERS"))) nw = atoi(e);
    if ((e = getenv("AIM_SPIN"))) g_spin = atol(e);
    if ((e = getenv("AIM_MIN_ROWS"))) g_min_rows_override = (uint32_t)atol(e);
    if (nw > AIM_MAX_WORKERS) nw = AIM_MAX_WORKERS;
    int pin = 1; if ((e = getenv("AIM_PIN"))) pin = atoi(e);
    if (nw > 0 && pin) pick_cpus();
    for (int i = 0; i < nw; i++) {
        if (pthread_create(&g_th[i], NULL, aim_worker, NULL)) break;
        g_nw = i + 1;
        if (pin && g_ncpus > 0) {
            cpu_set_t one; CPU_ZERO(&one); CPU_SET(g_cpus[(i + 1) % g_ncpus], &one);
            pthread_setaffinity_np(g_th[i], sizeof one, &one);
        }
    }
    if (nw > 0 && pin && g_ncpus > 0) { cpu_set_t one; CPU_ZERO(&one); CPU_SET(g_cpus[0], &one); sched_setaffinity(0, sizeof one, &one); }
    g_pool_on = 1;
}
static void parallel_for(const mv_ctx *c, uint32_t rows, uint32_t min_rows,
                         void (*fn)(const mv_ctx *, uint32_t)) {
    if (!g_pool_on) aim_pool_init();
    if (g_min_rows_override && min_rows > 1) min_rows = g_min_rows_override;
    if (g_nw <= 0 || rows < min_rows) {
        for (uint32_t o = 0; o < rows; o++) fn(c, o);
        return;
    }
    g_fn = fn; g_cx = c; g_total = rows;
    uint32_t chunk = rows / ((uint32_t)(g_nw + 1) * 8u);
    if (chunk < 1) chunk = 1;
    g_chunk = chunk;
    atomic_store(&g_next, 0);
    atomic_store(&g_remaining, (uint32_t)g_nw);
    atomic_fetch_add(&g_gen, 1);
    if (atomic_load(&g_sleepers)) futex_op(&g_gen, FUTEX_WAKE, INT_MAX);
    claim_loop();                              /* main thread works too */
    long s = 0;
    while (atomic_load(&g_remaining) > 0) {
        if (++s < g_spin * 4) { CPU_RELAX(); continue; }
        atomic_store(&g_main_sleep, 1);
        uint32_t r = atomic_load(&g_remaining);
        if (r > 0) futex_op(&g_remaining, FUTEX_WAIT, r);
        atomic_store(&g_main_sleep, 0);
    }
}
#endif

/* 行主序 [rows, in_dim] 权重 × 向量 h[in_dim] -> y[rows]
 * Q8_0 现场反量化; 求和序契约:
 *   AVX2 (in_dim%32==0 且 nb%4==0): 链4×8序 —— 链 ch(ch=0..3) 车道 j(j=0..7)
 *   累计 i=32b+8k+j (b≡ch mod 4, k=0..3) 升序; 车道内合并 s=((a0+a1)+a2)+a3,
 *   横向合并 ((((((s0+s1)+s2)+s3)+s4)+s5)+s6)+s7; 每码 (q*scale 先舍入)×h 后舍入
 *   —— 与 numpy 镜像 matvec 非 ternp 分支逐位一致.
 *   v1 标量 = 4cyc/元素串行链 → ~0.27B/cyc (Q8 349MB/token 占一半时间的主因);
 *   v2 每块 4 加法进独立链, 块间 4 链并行 → 上界=uop 流量, 带宽受限.
 * 否则: 标量 i 升序逐值累加 (契约 v1) */
static void mv_row_q8(const mv_ctx *c, uint32_t o) {
    const uint8_t *w = (const uint8_t *)c->w;
    const float *h = c->x;
    uint32_t in_dim = c->in_dim;
    uint32_t nb = in_dim / 32;
    const uint8_t *row = w + (uint64_t)o * in_dim / 32 * 34;
#if defined(__AVX2__)
#define Q8BLK4(av, bb) do { \
    uint16_t h16_; memcpy(&h16_, row + (size_t)(bb) * 34, 2); \
    const __m256 sv_ = _mm256_set1_ps(f16_to_f32(h16_)); \
    const uint8_t *qb_ = row + (size_t)(bb) * 34 + 2; \
    const float *hb_ = h + (size_t)(bb) * 32; \
    __m128i A_ = _mm_loadu_si128((const __m128i *)(const void *)qb_); \
    __m128i B_ = _mm_loadu_si128((const __m128i *)(const void *)(qb_ + 16)); \
    __m256 p_; \
    p_ = _mm256_mul_ps(_mm256_loadu_ps(hb_), \
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(A_)), sv_)); \
    av = _mm256_add_ps(av, p_); \
    p_ = _mm256_mul_ps(_mm256_loadu_ps(hb_ + 8), \
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(A_, 8))), sv_)); \
    av = _mm256_add_ps(av, p_); \
    p_ = _mm256_mul_ps(_mm256_loadu_ps(hb_ + 16), \
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(B_)), sv_)); \
    av = _mm256_add_ps(av, p_); \
    p_ = _mm256_mul_ps(_mm256_loadu_ps(hb_ + 24), \
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(B_, 8))), sv_)); \
    av = _mm256_add_ps(av, p_); \
} while (0)
    if (in_dim % 32 == 0 && nb % 4 == 0) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        _mm_prefetch((const char *)(const void *)(row + (uint64_t)nb * 34), _MM_HINT_T0);
        _mm_prefetch((const char *)(const void *)(row + (uint64_t)nb * 34 + 64), _MM_HINT_T0);
        for (uint32_t b = 0; b < nb; b += 4) {
            Q8BLK4(a0, b);
            Q8BLK4(a1, b + 1);
            Q8BLK4(a2, b + 2);
            Q8BLK4(a3, b + 3);
        }
        __m256 sv2 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), a2), a3);
        float f[8];
        _mm256_storeu_ps(f, sv2);
        float accS = f[0] + f[1];
        accS += f[2]; accS += f[3]; accS += f[4];
        accS += f[5]; accS += f[6]; accS += f[7];
        c->y[o] = accS;
        return;
    }
#undef Q8BLK4
#endif
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        uint16_t h16;
        memcpy(&h16, row + b * 34, 2);
        float scale = f16_to_f32(h16);
        const int8_t *q = (const int8_t *)(row + b * 34 + 2);
        const float *hv = h + b * 32;
        float part = 0.0f;
        for (int i = 0; i < 32; i++)
            part += hv[i] * (float)q[i];
        acc += scale * part;
    }
    c->y[o] = acc;
}

/* 批量包装器: k 列外循环复用同一权重行 (首列拉入 DRAM, 后续列 L1 命中),
 * 逐列调用单列函数 —— 求和序与单列完全一致, batch logits == 串行 logits 构造性成立 */
static void mv_row_q8_b(const mv_ctx *c, uint32_t o) {
    uint32_t in_dim = c->in_dim;
    for (uint32_t t = 0; t < c->bk; t++) {
        mv_ctx c1 = *c;
        c1.x = c->x + (size_t)t * in_dim;
        c1.y = c->y + (size_t)t * c->ystride;
        c1.bk = 0;
        mv_row_q8(&c1, o);
    }
}

static int32_t g_sign_tab[256][4];       /* 前置声明 (定义在 mv_row_ternp 前) */

/* 熔核 k=2 (fused): 解码跨列共享 —— 每 8 码解码/缩放 1 次, 逐列仅 load+mul+add。
 * FP 运算每列 4→3, 总 uop ~16→11/8码; 寄存器账 4链×2列=8 累加向量正好 (k=4 爆)。
 * 求和序不变: 链 c 车道 j 升序原样, 逐列与单列逐位一致 (零镜像改动) */
static void mv_row_ternp_f2(const mv_ctx *c, uint32_t o) {
    const uint8_t *pk = (const uint8_t *)c->w;
    uint32_t in_dim = c->in_dim;
    uint64_t base = (uint64_t)o * in_dim;
#if defined(__AVX2__)
    const __m256 vs = _mm256_set1_ps(c->scale);
    const float *x0 = c->x, *x1 = c->x + in_dim;
    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
    _mm_prefetch((const char *)(const void *)(pk + ((base + in_dim) >> 2)), _MM_HINT_T0);
    _mm_prefetch((const char *)(const void *)(pk + ((base + in_dim) >> 2) + 64), _MM_HINT_T0);
    for (uint32_t i = 0; i < in_dim; i += 32) {
        const uint8_t *pp = pk + ((base + i) >> 2);
        __m256i s0 = _mm256_set_m128i(
            _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[1]]),
            _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[0]]));
        __m256i s1 = _mm256_set_m128i(
            _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[3]]),
            _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[2]]));
        __m256i s2 = _mm256_set_m128i(
            _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[5]]),
            _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[4]]));
        __m256i s3 = _mm256_set_m128i(
            _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[7]]),
            _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[6]]));
        __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(s0), vs);
        __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(s1), vs);
        __m256 w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(s2), vs);
        __m256 w3 = _mm256_mul_ps(_mm256_cvtepi32_ps(s3), vs);
        a00 = _mm256_add_ps(a00, _mm256_mul_ps(_mm256_loadu_ps(x0 + i), w0));
        a01 = _mm256_add_ps(a01, _mm256_mul_ps(_mm256_loadu_ps(x1 + i), w0));
        a10 = _mm256_add_ps(a10, _mm256_mul_ps(_mm256_loadu_ps(x0 + i + 8), w1));
        a11 = _mm256_add_ps(a11, _mm256_mul_ps(_mm256_loadu_ps(x1 + i + 8), w1));
        a20 = _mm256_add_ps(a20, _mm256_mul_ps(_mm256_loadu_ps(x0 + i + 16), w2));
        a21 = _mm256_add_ps(a21, _mm256_mul_ps(_mm256_loadu_ps(x1 + i + 16), w2));
        a30 = _mm256_add_ps(a30, _mm256_mul_ps(_mm256_loadu_ps(x0 + i + 24), w3));
        a31 = _mm256_add_ps(a31, _mm256_mul_ps(_mm256_loadu_ps(x1 + i + 24), w3));
    }
    __m256 v0 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a00, a10), a20), a30);
    __m256 v1 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a01, a11), a21), a31);
    float f[8];
    _mm256_storeu_ps(f, v0);
    float acc0 = f[0] + f[1];
    acc0 += f[2]; acc0 += f[3]; acc0 += f[4];
    acc0 += f[5]; acc0 += f[6]; acc0 += f[7];
    c->y[o] = acc0;
    _mm256_storeu_ps(f, v1);
    float acc1 = f[0] + f[1];
    acc1 += f[2]; acc1 += f[3]; acc1 += f[4];
    acc1 += f[5]; acc1 += f[6]; acc1 += f[7];
    c->y[c->ystride + o] = acc1;
    return;
#endif
}

static void mv_row_f32(const mv_ctx *c, uint32_t o) {
    const float *w = (const float *)c->w;
    const float *h = c->x;
    uint32_t in_dim = c->in_dim;
    const float *row = w + (uint64_t)o * in_dim;
    float acc = 0.0f;
    for (uint32_t i = 0; i < in_dim; i++)
        acc += h[i] * row[i];
    c->y[o] = acc;
}

/* ---------- Q4_K (GGML qtype 12): 体积切入点 ----------
 * super-block 256 权重 / 144B: 2B f16 d + 2B f16 dmin
 * + 12B scales(8 子块 x 6bit) + 128B nibbles(4 组 x 32B, 低 nibble 前 32 权重/高 nibble 后 32)
 * 反量化: w = d*sc*n - dmin*m (非对称区间), 每子块独立 scale = 分组量化压动态范围
 * 求和序: super-block 升序 -> 子块升序 -> 块内 l 升序 = 全局 i 升序 (契约) */
static void q4k_scale_min(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = (uint8_t)(q[j] & 63);
        *m = (uint8_t)(q[j + 4] & 63);
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static void mv_row_q4k(const mv_ctx *c, uint32_t o) {
    const uint8_t *w = (const uint8_t *)c->w;
    const float *x = c->x;
    uint32_t in_dim = c->in_dim;
    uint32_t sb = in_dim / 256;
    const uint8_t *blk = w + (uint64_t)o * (in_dim / 256) * 144;
    float acc = 0.0f;
    for (uint32_t s = 0; s < sb; s++) {
        const uint8_t *b = blk + (size_t)s * 144;
        uint16_t h16;
        memcpy(&h16, b, 2);
        float d = f16_to_f32(h16);
        memcpy(&h16, b + 2, 2);
        float dmin = f16_to_f32(h16);
        const uint8_t *sc = b + 4;
        const uint8_t *qs = b + 16;
        int is = 0;
        for (int j = 0; j < 8; j += 2) {
            uint8_t s0d, s0m, s1d, s1m;
            q4k_scale_min(is, sc, &s0d, &s0m);
            q4k_scale_min(is + 1, sc, &s1d, &s1m);
            is += 2;
            float d1 = d * s0d, m1 = dmin * s0m;
            float d2 = d * s1d, m2 = dmin * s1m;
            const uint8_t *qg = qs + (size_t)(j / 2) * 32;
            for (int l = 0; l < 32; l++)
                acc += x[s * 256 + (uint32_t)j * 32 + l]
                     * (d1 * (qg[l] & 0xF) - m1);
            for (int l = 0; l < 32; l++)
                acc += x[s * 256 + (uint32_t)(j + 1) * 32 + l]
                     * (d2 * (qg[l] >> 4) - m2);
        }
    }
    c->y[o] = acc;
}

static void deq_q4k_row(const uint8_t *w, uint32_t id, float *x, uint32_t D) {
    const uint8_t *blk = w + (uint64_t)id * (D / 256) * 144;
    for (uint32_t s = 0; s < D / 256; s++) {
        const uint8_t *b = blk + (size_t)s * 144;
        uint16_t h16;
        memcpy(&h16, b, 2);
        float d = f16_to_f32(h16);
        memcpy(&h16, b + 2, 2);
        float dmin = f16_to_f32(h16);
        const uint8_t *sc = b + 4;
        const uint8_t *qs = b + 16;
        int is = 0;
        for (int j = 0; j < 8; j += 2) {
            uint8_t s0d, s0m, s1d, s1m;
            q4k_scale_min(is, sc, &s0d, &s0m);
            q4k_scale_min(is + 1, sc, &s1d, &s1m);
            is += 2;
            float d1 = d * s0d, m1 = dmin * s0m;
            float d2 = d * s1d, m2 = dmin * s1m;
            const uint8_t *qg = qs + (size_t)(j / 2) * 32;
            for (int l = 0; l < 32; l++)
                x[s * 256 + (uint32_t)j * 32 + l] = d1 * (qg[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++)
                x[s * 256 + (uint32_t)(j + 1) * 32 + l] = d2 * (qg[l] >> 4) - m2;
        }
    }
}

/* ---------- AIM-W v2 (qtype 33): Q4_K 容器 + 学习型点阵 ----------
 * 与 mv_row_q4k 唯一差异: nibble 值经 lut 查表 (点阵索引非层级值)
 * 求和序契约同 q4_K: 超块升序 -> 子块升序 -> 块内 l 升序 */
static void mv_row_q4kv2(const mv_ctx *c, uint32_t o) {
    const float *lut = c->lut;
    const uint8_t *w = (const uint8_t *)c->w;
    const float *x = c->x;
    uint32_t in_dim = c->in_dim;
    uint32_t sb = in_dim / 256;
    const uint8_t *blk = w + (uint64_t)o * (in_dim / 256) * 144;
    float acc = 0.0f;
    for (uint32_t s = 0; s < sb; s++) {
        const uint8_t *b = blk + (size_t)s * 144;
        uint16_t h16;
        memcpy(&h16, b, 2);
        float d = f16_to_f32(h16);
        memcpy(&h16, b + 2, 2);
        float dmin = f16_to_f32(h16);
        const uint8_t *sc = b + 4;
        const uint8_t *qs = b + 16;
        int is = 0;
        for (int j = 0; j < 8; j += 2) {
            uint8_t s0d, s0m, s1d, s1m;
            q4k_scale_min(is, sc, &s0d, &s0m);
            q4k_scale_min(is + 1, sc, &s1d, &s1m);
            is += 2;
            float d1 = d * s0d, m1 = dmin * s0m;
            float d2 = d * s1d, m2 = dmin * s1m;
            const uint8_t *qg = qs + (size_t)(j / 2) * 32;
            for (int l = 0; l < 32; l++)
                acc += x[s * 256 + (uint32_t)j * 32 + l]
                     * (d1 * lut[qg[l] & 0xF] - m1);
            for (int l = 0; l < 32; l++)
                acc += x[s * 256 + (uint32_t)(j + 1) * 32 + l]
                     * (d2 * lut[qg[l] >> 4] - m2);
        }
    }
    c->y[o] = acc;
}

static void deq_q4kv2_row(const aim_tref *t, uint32_t id, float *x, uint32_t D) {
    const float *lut = t->lut;
    const uint8_t *w = t->data;
    const uint8_t *blk = w + (uint64_t)id * (D / 256) * 144;
    for (uint32_t s = 0; s < D / 256; s++) {
        const uint8_t *b = blk + (size_t)s * 144;
        uint16_t h16;
        memcpy(&h16, b, 2);
        float d = f16_to_f32(h16);
        memcpy(&h16, b + 2, 2);
        float dmin = f16_to_f32(h16);
        const uint8_t *sc = b + 4;
        const uint8_t *qs = b + 16;
        int is = 0;
        for (int j = 0; j < 8; j += 2) {
            uint8_t s0d, s0m, s1d, s1m;
            q4k_scale_min(is, sc, &s0d, &s0m);
            q4k_scale_min(is + 1, sc, &s1d, &s1m);
            is += 2;
            float d1 = d * s0d, m1 = dmin * s0m;
            float d2 = d * s1d, m2 = dmin * s1m;
            const uint8_t *qg = qs + (size_t)(j / 2) * 32;
            for (int l = 0; l < 32; l++)
                x[s * 256 + (uint32_t)j * 32 + l] = d1 * lut[qg[l] & 0xF] - m1;
            for (int l = 0; l < 32; l++)
                x[s * 256 + (uint32_t)(j + 1) * 32 + l] = d2 * lut[qg[l] >> 4] - m2;
        }
    }
}

/* ---------- M2.7: 统一张量表 + 三值通道 ---------- */

static const aim_tref *mt(const aim_model *m, const char *name) {
    for (uint32_t i = 0; i < m->n_trefs; i++)
        if (!strcmp(m->trefs[i].name, name)) return &m->trefs[i];
    return NULL;
}

/* 三值 int8 逐码: 求和序契约 acc += x[i]*(code*scale), i 升序 (与 numpy 基线逐位一致) */
static void mv_row_tern(const mv_ctx *c, uint32_t o) {
    const int8_t *cc = (const int8_t *)c->w;
    const float *x = c->x;
    uint32_t in_dim = c->in_dim;
    const int8_t *row = cc + (uint64_t)o * in_dim;
    float acc = 0.0f;
    for (uint32_t i = 0; i < in_dim; i++)
        acc += x[i] * ((float)row[i] * c->scale);
    c->y[o] = acc;
}

/* 符号表解码 (aim_fast 同形态): byte(4 码, 低位在前) -> 4×int32 ∈ {-1,0,+1}
 * 码 0/1/2 = 0/+1/-1; cvtepi32_ps(±1) 精确 = ±1.0f
 * —— 与 srlv 拆码逐位等价 (同值同序), numpy 镜像零改动。
 * 4KB 常驻 L1; 初始化只在主线程 (matvec 分派 / aim_pool_init), worker 只读 */
static int32_t g_sign_tab[256][4];
static int g_sign_tab_ready = 0;
static void ensure_sign_tab(void) {
    if (g_sign_tab_ready) return;
    for (uint32_t b = 0; b < 256; b++)
        for (uint32_t k = 0; k < 4; k++) {
            uint32_t cd = (b >> (k * 2)) & 3u;
            g_sign_tab[b][k] = (cd == 1u) ? 1 : (cd == 2u) ? -1 : 0;
        }
    g_sign_tab_ready = 1;
}

/* 2bit 打包: 元素 g = o*in_dim+i 落在字节 g>>2 的位 (g&3)*2, 低位在前 (spec)
 * 码 3 已在装载期拒绝, 运行时零检查
 * AVX2 快路 v2 (链4×8序, 拆单累加器延迟墙):
 *   in_dim%32==0: 4 条独立累加向量, 链 c(c=0..3) 的车道 j(j=0..7) 累计元素
 *   i=8t+j (t≡c mod 4) 升序; 车道内合并 s=((a0+a1)+a2)+a3,
 *   横向合并 ((((((s0+s1)+s2)+s3)+s4)+s5)+s6)+s7
 *   —— 与 numpy 基线 matvec(ternp=1) 的链4×8序逐位一致 (mul/add 独立舍入, 无 FMA 收缩);
 *   每码舍入序不变: (cvt(u)*vs 先舍入) 再乘 x 后舍入.
 *   v1 单 acc = 4cyc/8码 串行延迟墙 → 0.5B/cyc ≈ 1GB/s (0.9GB/s 实测身份);
 *   v2 4 链摊薄加法延迟, 上界=uop 流量 (~2.4cyc/8码 → ~1.5GB/s 预期).
 * in_dim%8==0 (非 32 倍): 保 v1 单 acc 8 车道序; 否则标量 W4 查表. */
static void mv_row_ternp(const mv_ctx *c, uint32_t o) {
    const uint8_t *pk = (const uint8_t *)c->w;
    const float *x = c->x;
    uint32_t in_dim = c->in_dim;
    uint64_t base = (uint64_t)o * in_dim;
#if defined(__AVX2__)
    if (in_dim % 32 == 0) {
        const __m256 vs = _mm256_set1_ps(c->scale);
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        /* 预取下一行首 2 条 cache line (行内流式交给硬件预取器) */
        _mm_prefetch((const char *)(const void *)(pk + ((base + in_dim) >> 2)), _MM_HINT_T0);
        _mm_prefetch((const char *)(const void *)(pk + ((base + in_dim) >> 2) + 64), _MM_HINT_T0);
        for (uint32_t i = 0; i < in_dim; i += 32) {
            const uint8_t *pp = pk + ((base + i) >> 2);
            /* 符号表解码: 每链 2 byte = 8 码 = 2×4 lane 查表拼 __m256i */
            __m256i s0 = _mm256_set_m128i(
                _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[1]]),
                _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[0]]));
            __m256i s1 = _mm256_set_m128i(
                _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[3]]),
                _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[2]]));
            __m256i s2 = _mm256_set_m128i(
                _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[5]]),
                _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[4]]));
            __m256i s3 = _mm256_set_m128i(
                _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[7]]),
                _mm_loadu_si128((const __m128i *)(const void *)g_sign_tab[pp[6]]));
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(x + i),
                _mm256_mul_ps(_mm256_cvtepi32_ps(s0), vs)));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(x + i + 8),
                _mm256_mul_ps(_mm256_cvtepi32_ps(s1), vs)));
            a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(x + i + 16),
                _mm256_mul_ps(_mm256_cvtepi32_ps(s2), vs)));
            a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(x + i + 24),
                _mm256_mul_ps(_mm256_cvtepi32_ps(s3), vs)));
        }
        __m256 sv = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), a2), a3);
        float f[8];
        _mm256_storeu_ps(f, sv);
        float accS = f[0] + f[1];
        accS += f[2]; accS += f[3]; accS += f[4];
        accS += f[5]; accS += f[6]; accS += f[7];
        c->y[o] = accS;
        return;
    }
    if (in_dim % 8 == 0) {
        /* v1 单 acc 8 车道序 (非 32 倍形状回退, numpy 镜像 8 车道分支) */
        const __m256 vs = _mm256_set1_ps(c->scale);
        const __m256i sh = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
        const __m256i m1 = _mm256_set1_epi32(1);
        __m256 acc = _mm256_setzero_ps();
        for (uint32_t i = 0; i < in_dim; i += 8) {
            const uint8_t *pp = pk + ((base + i) >> 2);
            uint32_t w32 = (uint32_t)pp[0] | ((uint32_t)pp[1] << 8);
            __m256i cd = _mm256_srlv_epi32(_mm256_set1_epi32((int)w32), sh);
            __m256i u = _mm256_sub_epi32(_mm256_and_si256(cd, m1),
                                         _mm256_and_si256(_mm256_srli_epi32(cd, 1), m1));
            __m256 wv = _mm256_mul_ps(_mm256_cvtepi32_ps(u), vs);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(x + i), wv));
        }
        float f[8];
        _mm256_storeu_ps(f, acc);
        float accS = f[0] + f[1];
        accS += f[2]; accS += f[3]; accS += f[4];
        accS += f[5]; accS += f[6]; accS += f[7];
        c->y[o] = accS;
        return;
    }
#endif
    {
        float W4[4] = {0.0f, c->scale, -c->scale, 0.0f};
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        uint32_t i = 0;
        for (; i + 8 <= in_dim; i += 8) {
            uint8_t b = pk[(base + i) >> 2];
            a0 += x[i] * W4[b & 3u];
            a1 += x[i + 1] * W4[(b >> 2) & 3u];
            a2 += x[i + 2] * W4[(b >> 4) & 3u];
            a3 += x[i + 3] * W4[(b >> 6) & 3u];
            b = pk[(base + i + 4) >> 2];
            a4 += x[i + 4] * W4[b & 3u];
            a5 += x[i + 5] * W4[(b >> 2) & 3u];
            a6 += x[i + 6] * W4[(b >> 4) & 3u];
            a7 += x[i + 7] * W4[(b >> 6) & 3u];
        }
        float acc = ((((((a0 + a1) + a2) + a3) + a4) + a5) + a6) + a7;
        for (; i < in_dim; i++) {
            uint64_t g = base + i;
            uint8_t code = (uint8_t)((pk[g >> 2] >> ((g & 3u) * 2u)) & 3u);
            acc += x[i] * W4[code];
        }
        c->y[o] = acc;
    }
}

/* TERNP 批量包装器 (同 Q8: 列外循环, 权重行 L1 复用, 逐列逐位同构) */
/* Q8 熔核 k=2: 反量化+缩放跨列共享 (每块 1 次), 逐列仅 load+mul+add。
 * 求和序: 链 ch 车道 j 累计 i=32b+8k+j (b≡ch mod 4) 升序原样, 逐列逐位一致 */
static void mv_row_q8_f2(const mv_ctx *c, uint32_t o) {
    const uint8_t *w = (const uint8_t *)c->w;
    uint32_t in_dim = c->in_dim;
    uint32_t nb = in_dim / 32;
    const uint8_t *row = w + (uint64_t)o * in_dim / 32 * 34;
#if defined(__AVX2__)
    if (in_dim % 32 == 0 && nb % 4 == 0) {
        const float *h0 = c->x, *h1 = c->x + in_dim;
        __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
        __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
        __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
        __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
        _mm_prefetch((const char *)(const void *)(row + (uint64_t)nb * 34), _MM_HINT_T0);
        _mm_prefetch((const char *)(const void *)(row + (uint64_t)nb * 34 + 64), _MM_HINT_T0);
#define Q8BLK4F2(av0, av1, bb) do { \
    uint16_t h16_; memcpy(&h16_, row + (size_t)(bb) * 34, 2); \
    const __m256 sv_ = _mm256_set1_ps(f16_to_f32(h16_)); \
    const uint8_t *qb_ = row + (size_t)(bb) * 34 + 2; \
    const float *b0_ = h0 + (size_t)(bb) * 32, *b1_ = h1 + (size_t)(bb) * 32; \
    __m128i A_ = _mm_loadu_si128((const __m128i *)(const void *)qb_); \
    __m128i B_ = _mm_loadu_si128((const __m128i *)(const void *)(qb_ + 16)); \
    __m256 w0_ = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(A_)), sv_); \
    __m256 w1_ = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(A_, 8))), sv_); \
    __m256 w2_ = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(B_)), sv_); \
    __m256 w3_ = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_bsrli_si128(B_, 8))), sv_); \
    av0 = _mm256_add_ps(av0, _mm256_mul_ps(_mm256_loadu_ps(b0_), w0_)); \
    av1 = _mm256_add_ps(av1, _mm256_mul_ps(_mm256_loadu_ps(b1_), w0_)); \
    av0 = _mm256_add_ps(av0, _mm256_mul_ps(_mm256_loadu_ps(b0_ + 8), w1_)); \
    av1 = _mm256_add_ps(av1, _mm256_mul_ps(_mm256_loadu_ps(b1_ + 8), w1_)); \
    av0 = _mm256_add_ps(av0, _mm256_mul_ps(_mm256_loadu_ps(b0_ + 16), w2_)); \
    av1 = _mm256_add_ps(av1, _mm256_mul_ps(_mm256_loadu_ps(b1_ + 16), w2_)); \
    av0 = _mm256_add_ps(av0, _mm256_mul_ps(_mm256_loadu_ps(b0_ + 24), w3_)); \
    av1 = _mm256_add_ps(av1, _mm256_mul_ps(_mm256_loadu_ps(b1_ + 24), w3_)); \
} while (0)
        for (uint32_t b = 0; b < nb; b += 4) {
            Q8BLK4F2(a00, a01, b);
            Q8BLK4F2(a10, a11, b + 1);
            Q8BLK4F2(a20, a21, b + 2);
            Q8BLK4F2(a30, a31, b + 3);
        }
#undef Q8BLK4F2
        __m256 v0 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a00, a10), a20), a30);
        __m256 v1 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a01, a11), a21), a31);
        float f[8];
        _mm256_storeu_ps(f, v0);
        float acc0 = f[0] + f[1];
        acc0 += f[2]; acc0 += f[3]; acc0 += f[4];
        acc0 += f[5]; acc0 += f[6]; acc0 += f[7];
        c->y[o] = acc0;
        _mm256_storeu_ps(f, v1);
        float acc1 = f[0] + f[1];
        acc1 += f[2]; acc1 += f[3]; acc1 += f[4];
        acc1 += f[5]; acc1 += f[6]; acc1 += f[7];
        c->y[c->ystride + o] = acc1;
        return;
    }
#endif
}

static void mv_row_ternp_b(const mv_ctx *c, uint32_t o) {
    uint32_t in_dim = c->in_dim;
    for (uint32_t t = 0; t < c->bk; t++) {
        mv_ctx c1 = *c;
        c1.x = c->x + (size_t)t * in_dim;
        c1.y = c->y + (size_t)t * c->ystride;
        c1.bk = 0;
        mv_row_ternp(&c1, o);
    }
}

#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
/* ---------- mt patch: int8-activation ternary kernel (AVX-512 VNNI), opt-in via AIM_VNNI=1 ----------
 * Activations are quantized once per matvec to int8 (per-vector absmax/127, the BitNet W1.58A8 recipe) and de-interleaved into
 * 4 phase arrays ph[k][i] = xq[4i+k]. Weight bytes are re-encoded at load so each 2-bit field = w+1 in {0,1,2} (unsigned).
 * y[o] = scale * xs * ( sum_k dpbusd(field_k, ph[k]) - sum(xq) ). Integer accumulation: exact and order-independent. */
#define AIM_VNNI 1
static int g_vnni = 0;
static int8_t g_ph[4][8192] __attribute__((aligned(64)));
static int8_t g_phb[32][8448] __attribute__((aligned(64)));   /* col stride 8448 B / phase stride 2112 B: avoids L1 set aliasing */   /* batch: [col][4 phases x 2048] */
static float g_bxs[32];
static int32_t g_bsumx[32];
static void vnni_quant_to(const float *x, uint32_t n, int8_t *ph, uint32_t phs, float *xs, int32_t *sumx) {
    __m512 vmax = _mm512_setzero_ps();
    const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    for (uint32_t i = 0; i < n; i += 16) vmax = _mm512_max_ps(vmax, _mm512_and_ps(_mm512_loadu_ps(x + i), absmask));
    float amax = _mm512_reduce_max_ps(vmax);
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    *xs = amax > 0.0f ? amax / 127.0f : 1.0f;
    int8_t tmp[8192] __attribute__((aligned(64)));
    __m512 vinv = _mm512_set1_ps(inv);
    __m512i vs = _mm512_setzero_si512();
    for (uint32_t i = 0; i < n; i += 16) {
        __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + i), vinv));
        vs = _mm512_add_epi32(vs, q);
        _mm_storeu_si128((__m128i *)(void *)(tmp + i), _mm512_cvtsepi32_epi8(q));
    }
    *sumx = _mm512_reduce_add_epi32(vs);
    for (uint32_t i = 0; i < n; i += 64) {
        __m512i w = _mm512_loadu_si512((const void *)(tmp + i));
        uint32_t o = i >> 2;
        _mm_storeu_si128((__m128i *)(void *)(ph + 0u * phs + o), _mm512_cvtepi32_epi8(w));
        _mm_storeu_si128((__m128i *)(void *)(ph + 1u * phs + o), _mm512_cvtepi32_epi8(_mm512_srli_epi32(w, 8)));
        _mm_storeu_si128((__m128i *)(void *)(ph + 2u * phs + o), _mm512_cvtepi32_epi8(_mm512_srli_epi32(w, 16)));
        _mm_storeu_si128((__m128i *)(void *)(ph + 3u * phs + o), _mm512_cvtepi32_epi8(_mm512_srli_epi32(w, 24)));
    }
}
static inline void vnni_quant_x(const float *x, uint32_t n, float *xs, int32_t *sumx) { vnni_quant_to(x, n, &g_ph[0][0], 8192u, xs, sumx); }
static void mv_row_tern_vnni(const mv_ctx *c, uint32_t o) {
    const uint8_t *pk = (const uint8_t *)c->w + (((uint64_t)o * c->in_dim) >> 2);
    const uint32_t nb = c->in_dim >> 2;
    const __m512i m3 = _mm512_set1_epi8(3);
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    for (uint32_t b = 0; b < nb; b += 64) {
        __m512i v = _mm512_loadu_si512((const void *)(pk + b));
        a0 = _mm512_dpbusd_epi32(a0, _mm512_and_si512(v, m3), _mm512_load_si512((const void *)(g_ph[0] + b)));
        a1 = _mm512_dpbusd_epi32(a1, _mm512_and_si512(_mm512_srli_epi16(v, 2), m3), _mm512_load_si512((const void *)(g_ph[1] + b)));
        a2 = _mm512_dpbusd_epi32(a2, _mm512_and_si512(_mm512_srli_epi16(v, 4), m3), _mm512_load_si512((const void *)(g_ph[2] + b)));
        a3 = _mm512_dpbusd_epi32(a3, _mm512_and_si512(_mm512_srli_epi16(v, 6), m3), _mm512_load_si512((const void *)(g_ph[3] + b)));
    }
    int32_t dot = _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3)));
    c->y[o] = (float)(dot - c->sumx) * c->scale;
}
/* batched: unpack each 64-byte weight chunk once, feed up to 8 token columns (dpbusd) per tile */
static inline __attribute__((always_inline)) void vnni_tile(const uint8_t *pk, uint32_t nb, uint32_t t0, const int NT,
                                                            const mv_ctx *c, uint32_t o) {
    const __m512i m3 = _mm512_set1_epi8(3);
    __m512i acc[16];
    for (int j = 0; j < NT; j++) acc[j] = _mm512_setzero_si512();
    for (uint32_t b = 0; b < nb; b += 64) {
        __m512i v = _mm512_loadu_si512((const void *)(pk + b));
        __m512i f0 = _mm512_and_si512(v, m3), f1 = _mm512_and_si512(_mm512_srli_epi16(v, 2), m3);
        __m512i f2 = _mm512_and_si512(_mm512_srli_epi16(v, 4), m3), f3 = _mm512_and_si512(_mm512_srli_epi16(v, 6), m3);
        for (int j = 0; j < NT; j++) {
            const int8_t *p = &g_phb[t0 + (uint32_t)j][0] + b;
            __m512i a = acc[j];
            a = _mm512_dpbusd_epi32(a, f0, _mm512_load_si512((const void *)p));
            a = _mm512_dpbusd_epi32(a, f1, _mm512_load_si512((const void *)(p + 2112)));
            a = _mm512_dpbusd_epi32(a, f2, _mm512_load_si512((const void *)(p + 4224)));
            a = _mm512_dpbusd_epi32(a, f3, _mm512_load_si512((const void *)(p + 6336)));
            acc[j] = a;
        }
    }
    for (int j = 0; j < NT; j++) {
        int32_t dot = _mm512_reduce_add_epi32(acc[j]);
        c->y[(size_t)(t0 + (uint32_t)j) * c->ystride + o] = (float)(dot - g_bsumx[t0 + (uint32_t)j]) * (c->scale * g_bxs[t0 + (uint32_t)j]);
    }
}
static int g_tile = 8, g_tile_big = 8;
static void mv_row_tern_vnni_b(const mv_ctx *c, uint32_t o) {
    const uint8_t *pk = (const uint8_t *)c->w + (((uint64_t)o * c->in_dim) >> 2);
    const uint32_t nb = c->in_dim >> 2, K = c->bk;
    const uint32_t TW = (uint32_t)(c->in_dim > 4096 ? g_tile_big : g_tile);
    for (uint32_t t0 = 0; t0 < K; t0 += TW) {
        uint32_t n = K - t0 < TW ? K - t0 : TW;
        switch (n) {
#define TC(N) case N: vnni_tile(pk, nb, t0, N, c, o); break;
        TC(16) TC(15) TC(14) TC(13) TC(12) TC(11) TC(10) TC(9) TC(8) TC(7) TC(6) TC(5) TC(4) TC(3) TC(2)
#undef TC
        default: vnni_tile(pk, nb, t0, 1, c, o); break;
        }
    }
}
/* generic "for each token column" task on the pool */
typedef struct { void (*fn)(uint32_t, const void *); const void *a; } tk_task;
static tk_task g_tk;
static void tk_unit(const mv_ctx *c, uint32_t t) { (void)c; g_tk.fn(t, g_tk.a); }
static void par_tokens(uint32_t k, void (*fn)(uint32_t, const void *), const void *a) {
    g_tk.fn = fn; g_tk.a = a;
    mv_ctx d; memset(&d, 0, sizeof d);
    parallel_for(&d, k, 1, tk_unit);
}
static const float *g_qx; static uint32_t g_qn;
static void quant_col_unit(uint32_t col, const void *a) {
    (void)a;
    vnni_quant_to(g_qx + (size_t)col * g_qn, g_qn, &g_phb[col][0], 2112u, &g_bxs[col], &g_bsumx[col]);
}
static void vnni_batch_block(const aim_tref *t, const float *x, float *y, uint32_t rows, uint32_t in_dim, uint32_t k) {
    mv_ctx c;
    c.w = t->data; c.x = x; c.y = y; c.in_dim = in_dim; c.scale = t->scale; c.lut = NULL; c.bk = k; c.ystride = rows; c.sumx = 0;
    g_qx = x; g_qn = in_dim;
    par_tokens(k, quant_col_unit, NULL);
    parallel_for(&c, rows, 2048, mv_row_tern_vnni_b);
}
/* ---------- mt patch: Q8_0 x int8-activation kernel for the output layer (AVX-512 VNNI + VL), opt-in AIM_Q8V=1 ----------
 * block = f16 d + 32 x int8 (34 B). dot via vpsignb trick: |w| (unsigned) x sign-adjusted xq, exact, no correction term. */
static int g_q8v = 0;
static int8_t g_xq8[8192] __attribute__((aligned(64)));
static int8_t g_xq8b[32][8256] __attribute__((aligned(64)));
static float g_q8xs[32];
static void q8v_quant(const float *x, uint32_t n, int8_t *dst, float *xs) {
    __m512 vmax = _mm512_setzero_ps();
    const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    for (uint32_t i = 0; i < n; i += 16) vmax = _mm512_max_ps(vmax, _mm512_and_ps(_mm512_loadu_ps(x + i), absmask));
    float amax = _mm512_reduce_max_ps(vmax);
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    *xs = amax > 0.0f ? amax / 127.0f : 1.0f;
    __m512 vinv = _mm512_set1_ps(inv);
    for (uint32_t i = 0; i < n; i += 16)
        _mm_storeu_si128((__m128i *)(void *)(dst + i), _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + i), vinv))));
}
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo)); lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
static void mv_row_q8_vnni(const mv_ctx *c, uint32_t o) {
    const uint32_t nbk = c->in_dim >> 5;
    const uint8_t *row = (const uint8_t *)c->w + (size_t)o * nbk * 34;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    uint32_t b = 0;
    for (; b + 4 <= nbk; b += 4) {
#define Q8STEP(A, K) { const uint8_t *blk = row + (size_t)(b + K) * 34; uint16_t h; memcpy(&h, blk, 2); \
            __m256i w = _mm256_loadu_si256((const __m256i *)(const void *)(blk + 2)); \
            __m256i xq = _mm256_load_si256((const __m256i *)(const void *)(g_xq8 + (size_t)(b + K) * 32)); \
            __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(w, w), _mm256_sign_epi8(xq, w)); \
            A = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dp), _mm256_set1_ps(_cvtsh_ss(h)), A); }
        Q8STEP(a0, 0) Q8STEP(a1, 1) Q8STEP(a2, 2) Q8STEP(a3, 3)
    }
    for (; b < nbk; b++) Q8STEP(a0, 0)
#undef Q8STEP
    c->y[o] = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3))) * c->scale;
}
static inline __attribute__((always_inline)) void q8_tile(const uint8_t *row, uint32_t nbk, uint32_t t0, const int NT, const mv_ctx *c, uint32_t o) {
    __m256 acc[8];
    for (int j = 0; j < NT; j++) acc[j] = _mm256_setzero_ps();
    for (uint32_t b = 0; b < nbk; b++) {
        const uint8_t *blk = row + (size_t)b * 34; uint16_t h; memcpy(&h, blk, 2);
        __m256i w = _mm256_loadu_si256((const __m256i *)(const void *)(blk + 2));
        __m256i aw = _mm256_sign_epi8(w, w);
        __m256 d = _mm256_set1_ps(_cvtsh_ss(h));
        for (int j = 0; j < NT; j++) {
            __m256i xq = _mm256_load_si256((const __m256i *)(const void *)(&g_xq8b[t0 + (uint32_t)j][0] + (size_t)b * 32));
            __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), aw, _mm256_sign_epi8(xq, w));
            acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dp), d, acc[j]);
        }
    }
    for (int j = 0; j < NT; j++) c->y[(size_t)(t0 + (uint32_t)j) * c->ystride + o] = hsum256(acc[j]) * g_q8xs[t0 + (uint32_t)j];
}
static void mv_row_q8_vnni_b(const mv_ctx *c, uint32_t o) {
    const uint32_t nbk = c->in_dim >> 5, K = c->bk;
    const uint8_t *row = (const uint8_t *)c->w + (size_t)o * nbk * 34;
    for (uint32_t t0 = 0; t0 < K; t0 += 8) {
        switch (K - t0 < 8 ? K - t0 : 8) {
#define QC(N) case N: q8_tile(row, nbk, t0, N, c, o); break;
        QC(8) QC(7) QC(6) QC(5) QC(4) QC(3) QC(2)
#undef QC
        default: q8_tile(row, nbk, t0, 1, c, o); break;
        }
    }
}
static const float *g_q8qx; static uint32_t g_q8qn;
static void q8_quant_col_unit(uint32_t col, const void *a) { (void)a; q8v_quant(g_q8qx + (size_t)col * g_q8qn, g_q8qn, &g_xq8b[col][0], &g_q8xs[col]); }
static void q8v_batch_block(const aim_tref *t, const float *x, float *y, uint32_t rows, uint32_t in_dim, uint32_t k) {
    mv_ctx c; memset(&c, 0, sizeof c);
    c.w = t->data; c.x = x; c.y = y; c.in_dim = in_dim; c.bk = k; c.ystride = rows;
    g_q8qx = x; g_q8qn = in_dim;
    par_tokens(k, q8_quant_col_unit, NULL);
    parallel_for(&c, rows, 2048, mv_row_q8_vnni_b);
}
/* ---------- mt patch (LOSSY, opt-in AIM_LM4=1): 4-bit copy of the output layer for single-token logits ----------
 * 18 B per 32 weights: f16 d + 16 bytes (low nibble = w[j], high = w[j+16]), q = round(w/d)+8 with d = absmax/7. */
static int g_lm4 = 0;
static int32_t g_bsum[256];
static void q4v_quant(const float *x, uint32_t n, float *xs) {
    q8v_quant(x, n, g_xq8, xs);
    for (uint32_t b = 0; b < (n >> 5); b++) {
        __m256i v = _mm256_load_si256((const __m256i *)(const void *)(g_xq8 + (size_t)b * 32));
        __m256i s16 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), v), _mm256_set1_epi16(1));
        __m128i lo = _mm256_castsi256_si128(s16), hi = _mm256_extracti128_si256(s16, 1);
        __m128i t = _mm_add_epi32(lo, hi); t = _mm_add_epi32(t, _mm_srli_si128(t, 8)); t = _mm_add_epi32(t, _mm_srli_si128(t, 4));
        g_bsum[b] = _mm_cvtsi128_si32(t);
    }
}
static void mv_row_q4_vnni(const mv_ctx *c, uint32_t o) {
    const uint32_t nbk = c->in_dim >> 5;
    const uint8_t *row = (const uint8_t *)c->w + (size_t)o * nbk * 18;
    const __m128i m0F = _mm_set1_epi8(0x0F);
    __m256 a0 = _mm256_setzero_ps(), a1 = a0;
    float corr = 0.0f;
    for (uint32_t b = 0; b < nbk; b++) {
        const uint8_t *blk = row + (size_t)b * 18; uint16_t h; memcpy(&h, blk, 2);
        __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(blk + 2));
        __m256i w = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), m0F), _mm_and_si128(v, m0F));
        __m256i xq = _mm256_load_si256((const __m256i *)(const void *)(g_xq8 + (size_t)b * 32));
        __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, xq);
        float d = _cvtsh_ss(h);
        if (b & 1) a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dp), _mm256_set1_ps(d), a1);
        else       a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dp), _mm256_set1_ps(d), a0);
        corr += d * 8.0f * (float)g_bsum[b];
    }
    c->y[o] = (hsum256(_mm256_add_ps(a0, a1)) - corr) * c->scale;
}
static void lm4_prepare(aim_tref *t) {
    const uint64_t n = t->n_elems, nblk = n >> 5;
    uint8_t *out = (uint8_t *)malloc(nblk * 18);
    if (!out) return;
    for (uint64_t b = 0; b < nblk; b++) {
        const uint8_t *blk = t->data + b * 34; uint16_t h; memcpy(&h, blk, 2);
        float d8 = _cvtsh_ss(h), w[32], amax = 0.0f;
        for (int j = 0; j < 32; j++) { w[j] = d8 * (float)(int8_t)blk[2 + j]; float a = w[j] < 0 ? -w[j] : w[j]; if (a > amax) amax = a; }
        float d = amax / 7.0f, inv = d > 0.0f ? 1.0f / d : 0.0f;
        uint16_t hd = _cvtss_sh(d, 0); uint8_t *o = out + b * 18; memcpy(o, &hd, 2);
        for (int j = 0; j < 16; j++) {
            int q0 = (int)lrintf(w[j] * inv), q1 = (int)lrintf(w[j + 16] * inv);
            if (q0 > 7) q0 = 7; if (q0 < -7) q0 = -7; if (q1 > 7) q1 = 7; if (q1 < -7) q1 = -7;
            o[2 + j] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
        }
    }
    t->alt = out;
}
/* re-encode every TERNP tensor: field 0->1 (w=-1... see map), 1->2, 2->0 so field = w+1 given AIM-W code 0/1/2 = 0/+1/-1 */
static void vnni_prepare(aim_tref *trefs, uint32_t n) {
    uint8_t lut[256];
    static const uint8_t mp[4] = {1, 2, 0, 3};
    for (uint32_t b = 0; b < 256; b++)
        lut[b] = (uint8_t)(mp[b & 3] | (mp[(b >> 2) & 3] << 2) | (mp[(b >> 4) & 3] << 4) | (mp[(b >> 6) & 3] << 6));
    for (uint32_t i = 0; i < n; i++) {
        if (trefs[i].qtype != AIM_QT_TERNP) continue;
        uint8_t *d = (uint8_t *)(uintptr_t)trefs[i].data;
        uint64_t nbytes = trefs[i].n_elems >> 2;
        for (uint64_t k = 0; k < nbytes; k++) d[k] = lut[d[k]];
    }
}
#endif

/* 前向唯一 matvec 入口: 按 qtype 分派 (前向不感知来源格式) + 行级并行 */
static void matvec_any(const aim_tref *t, const float *x, float *y,
                       uint32_t rows, uint32_t in_dim) {
    mv_ctx c;
    c.w = t->data;
    c.x = x;
    c.y = y;
    c.in_dim = in_dim;
    c.scale = t->scale;
    c.lut = t->lut;
    c.bk = 0;
    c.ystride = 0;
    void (*fn)(const mv_ctx *, uint32_t) =
        (t->qtype == 8)                ? mv_row_q8
        : (t->qtype == AIM_QT_TERN)    ? mv_row_tern
        : (t->qtype == AIM_QT_TERNP)   ? mv_row_ternp
        : (t->qtype == 12)             ? mv_row_q4k
        : (t->qtype == AIM_QT_Q4KV2)   ? mv_row_q4kv2
                                       : mv_row_f32;
    ensure_sign_tab();                     /* 主线程一次性, worker 只读 */
#ifdef AIM_VNNI
    if (g_lm4 && t->qtype == 8 && t->alt && (in_dim & 31u) == 0 && in_dim <= 8192 && rows > 100000) {
        float xs; q4v_quant(x, in_dim, &xs);
        c.w = t->alt; c.scale = xs;
        parallel_for(&c, rows, 2048, mv_row_q4_vnni);
        return;
    }
    if (g_q8v && t->qtype == 8 && (in_dim & 31u) == 0 && in_dim <= 8192) {
        float xs; q8v_quant(x, in_dim, g_xq8, &xs);
        c.scale = xs;
        parallel_for(&c, rows, 2048, mv_row_q8_vnni);
        return;
    }
    if (g_vnni && t->qtype == AIM_QT_TERNP && (in_dim & 255u) == 0 && in_dim <= 8192) {
        float xs; int32_t sx;
        vnni_quant_x(x, in_dim, &xs, &sx);
        c.scale = t->scale * xs; c.sumx = sx;
        parallel_for(&c, rows, 2048, mv_row_tern_vnni);
        return;
    }
#endif
    parallel_for(&c, rows, 2048, fn);
}

/* 批量 matvec 入口 (投机验证): k 列共用一次权重读。
 * 热路径 (TERNP/Q8) 走列外循环包装器; 其余 qtype 逐列退化为单列调用。
 * x 列主 [k][in_dim], y 列主 [k][rows] */
static void matvec_any_batch(const aim_tref *t, const float *x, float *y,
                             uint32_t rows, uint32_t in_dim, uint32_t k) {
#ifdef AIM_VNNI
    if (g_q8v && t->qtype == 8 && k > 1 && (in_dim & 31u) == 0 && in_dim <= 8192) {
        for (uint32_t tc = 0; tc < k; tc += 32) {
            uint32_t kk2 = k - tc < 32 ? k - tc : 32;
            q8v_batch_block(t, x + (size_t)tc * in_dim, y + (size_t)tc * rows, rows, in_dim, kk2);
        }
        return;
    }
    if (g_vnni && t->qtype == AIM_QT_TERNP) {
        if (k <= 1 || (in_dim & 255u) != 0 || in_dim > 8192) {
            for (uint32_t tc = 0; tc < k; tc++)
                matvec_any(t, x + (size_t)tc * in_dim, y + (size_t)tc * rows, rows, in_dim);
            return;
        }
        for (uint32_t tc = 0; tc < k; tc += 32) {
            uint32_t kk = k - tc < 32 ? k - tc : 32;
            vnni_batch_block(t, x + (size_t)tc * in_dim, y + (size_t)tc * rows, rows, in_dim, kk);
        }
        return;
    }
#endif
    if (k <= 1 || (t->qtype != AIM_QT_TERNP && t->qtype != 8)) {
        for (uint32_t tc = 0; tc < k; tc++)
            matvec_any(t, x + (size_t)tc * in_dim,
                       y + (size_t)tc * rows, rows, in_dim);
        return;
    }
    mv_ctx c;
    c.w = t->data;
    c.x = x;
    c.y = y;
    c.in_dim = in_dim;
    c.scale = t->scale;
    c.lut = t->lut;
    c.bk = k;
    c.ystride = rows;
    /* k==2 走熔核 (解码跨列共享); k>2 走列外循环包装器 (寄存器账限制) */
    void (*fn)(const mv_ctx *, uint32_t) =
        (t->qtype == AIM_QT_TERNP)
            ? (k == 2 ? mv_row_ternp_f2 : mv_row_ternp_b)
            : (k == 2 ? mv_row_q8_f2 : mv_row_q8_b);
    ensure_sign_tab();
    parallel_for(&c, rows, 2048, fn);
}

/* embedding 行反量化: 按 qtype 分派 */
static void deq_embd(const aim_tref *t, uint32_t id, float *x, uint32_t D) {
    if (t->qtype == 8) {
        const uint8_t *row = t->data + (uint64_t)id * D / 32 * 34;
        deq_q8_0(row, x, D / 32);
    } else if (t->qtype == AIM_QT_TERN) {
        const int8_t *row = (const int8_t *)t->data + (uint64_t)id * D;
        for (uint32_t i = 0; i < D; i++)
            x[i] = (float)row[i] * t->scale;
    } else if (t->qtype == AIM_QT_TERNP) {
        static const float MAP[4] = {0.0f, 1.0f, -1.0f, 0.0f};
        const uint8_t *pk = t->data;
        uint64_t base = (uint64_t)id * D;
        for (uint32_t i = 0; i < D; i++) {
            uint64_t g = base + i;
            x[i] = MAP[(pk[g >> 2] >> ((g & 3u) * 2u)) & 3u] * t->scale;
        }
    } else if (t->qtype == 12) {
        deq_q4k_row(t->data, id, x, D);
    } else if (t->qtype == AIM_QT_Q4KV2) {
        deq_q4kv2_row(t, id, x, D);
    } else {
        memcpy(x, (const float *)t->data + (uint64_t)id * D, (size_t)D * 4);
    }
}

/* RMSNorm */
static void rmsnorm(const float *x, const float *weight, float *y,
                    uint32_t n, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++)
        ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0; i < n; i++)
        y[i] = x[i] * inv * weight[i];
}

/* RoPE: half-split 配对 (i 与 i+half), theta_j = base^(-2j/head_dim) */
static void rope(float *v, uint32_t pos, uint32_t head_dim,
                 float base, uint32_t n_seqs_total) {
    uint32_t half = head_dim / 2;
    for (uint32_t j = 0; j < half; j++) {
        float theta = powf(base, -(2.0f * (float)j) / (float)head_dim);
        float ang = (float)pos * theta;
        float c = cosf(ang), s = sinf(ang);
        for (uint32_t seq = 0; seq < n_seqs_total; seq++) {
            float x0 = v[seq * head_dim + j];
            float x1 = v[seq * head_dim + j + half];
            v[seq * head_dim + j] = x0 * c - x1 * s;
            v[seq * head_dim + j + half] = x0 * s + x1 * c;
        }
    }
}

static uint32_t u32_meta(const aim_gguf *g, const char *fmt, const char *base) {
    char full[256];
    snprintf(full, sizeof(full), fmt, base);
    const aim_kv *kv = aim_gguf_find(g, full);
    return kv ? (uint32_t)kv->v.u : 0;
}

/* ---------- 模型装载 (M2.7 双通道) ---------- */

/* KV cache 常驻: [n_layers][kv_cap][kv_dim] 层主序
 * 真模型 ctx_len=131072 满配 = 11.3GB, 8GB 机器必炸 -> 封顶 8192 (~688MB) */
static int alloc_kv(aim_model *m) {
    m->kv_cap = m->ctx_len ? m->ctx_len : 4096;
    if (m->kv_cap > 8192) m->kv_cap = 8192;
    uint64_t slots = (uint64_t)m->n_layers * m->kv_cap
                   * (m->n_kv_heads * m->head_dim);
    m->kv_k = (float *)calloc((size_t)slots, 4);
    m->kv_v = (float *)calloc((size_t)slots, 4);
    if (!m->kv_k || !m->kv_v) {
        free(m->kv_k); free(m->kv_v); m->kv_k = m->kv_v = NULL;
        return 3;
    }
    return 0;
}

static int add_trefs_gguf(aim_model *m) {
    aim_gguf *g = &m->meta;
    m->n_trefs = (uint32_t)g->n_tensors;
    m->trefs = (aim_tref *)calloc(m->n_trefs ? m->n_trefs : 1, sizeof(aim_tref));
    if (!m->trefs) return 3;
    for (uint32_t i = 0; i < g->n_tensors; i++) {
        const aim_tensor *t = &g->tensors[i];
        aim_tref *r = &m->trefs[i];
        snprintf(r->name, sizeof(r->name), "%s", t->name);
        r->qtype   = t->qtype;
        r->data    = m->blob + t->offset;
        r->n_elems = t->n_elems;
        r->scale   = 0.0f;
    }
    return 0;
}

static int open_gguf(const char *path, aim_model *m) {
    int rc = aim_gguf_open(path, &m->meta);
    if (rc) return rc;
    aim_gguf *g = &m->meta;

    const aim_kv *arch = aim_gguf_find(g, "general.architecture");
    const char *base = (arch && arch->vt == AIM_T_STR) ? arch->v.str : "llama";
    m->n_layers   = u32_meta(g, "%s.block_count", base);
    m->n_heads    = u32_meta(g, "%s.attention.head_count", base);
    m->n_kv_heads = u32_meta(g, "%s.attention.head_count_kv", base);
    m->head_dim   = u32_meta(g, "%s.attention.key_length", base);
    m->n_embd     = u32_meta(g, "%s.embedding_length", base);
    m->n_ff       = u32_meta(g, "%s.feed_forward_length", base);
    m->n_vocab    = u32_meta(g, "%s.vocab_size", base);
    m->ctx_len    = u32_meta(g, "%s.context_length", base);
    m->eps  = 1e-5f;
    m->rope_base = 1000000.0f;
    { char full[256];
      snprintf(full, sizeof(full), "%s.attention.layer_norm_rms_epsilon", base);
      const aim_kv *kv = aim_gguf_find(g, full);
      if (kv && kv->vt == AIM_T_F32) m->eps = (float)kv->v.f;
      snprintf(full, sizeof(full), "%s.rope.freq_base", base);
      kv = aim_gguf_find(g, full);
      if (kv && kv->vt == AIM_T_F32) m->rope_base = (float)kv->v.f; }
    if (!m->n_layers || !m->n_heads || !m->n_embd || !m->n_vocab) return 2;
    if (!m->head_dim) m->head_dim = m->n_embd / m->n_heads;
    if (!m->n_kv_heads) m->n_kv_heads = m->n_heads;   /* MHA 无 GQA 键 */

    uint64_t total = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        uint64_t end = g->tensors[i].offset + g->tensors[i].nbytes;
        if (end > total) total = end;
    }
    m->blob_size = total;
    m->blob = (uint8_t *)malloc((size_t)total);
    if (!m->blob) return 3;
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    if (fseek(f, (long)g->data_start, SEEK_SET) ||
        fread(m->blob, 1, (size_t)total, f) != (size_t)total) {
        fclose(f); return 1;
    }
    fclose(f);

    if (!find_tensor(g, "token_embd.weight")) return 2;

    rc = add_trefs_gguf(m);
    if (rc) return rc;
    if (!mt(m, "output.weight") && !mt(m, "output_norm.weight") &&
        !mt(m, "output_norm")) {
        /* 无 output_norm 的极简合成模型不在此列; 交由前向查表时失败 */
    }
    return alloc_kv(m);
}

/* AIM-W 通道: 架构参数从 meta 标量张量读, 缺项推断兜底 */
static uint32_t meta_u32(const aim_model *m, const char *name, uint32_t dflt) {
    const aim_aimw_tensor *t = aim_aimw_find(&m->aw, name);
    if (t && t->dtype == AIM_AIMW_DT_FP32 && t->n_elems == 1) {
        float v;
        memcpy(&v, t->data, 4);
        return (uint32_t)(v > 0.0f ? v + 0.5f : v);
    }
    return dflt;
}

static float meta_f32(const aim_model *m, const char *name, float dflt) {
    const aim_aimw_tensor *t = aim_aimw_find(&m->aw, name);
    if (t && t->dtype == AIM_AIMW_DT_FP32 && t->n_elems == 1) {
        float v;
        memcpy(&v, t->data, 4);
        return v;
    }
    return dflt;
}

static int open_aimw(const char *path, aim_model *m) {
    int rc = aim_aimw_open(path, &m->aw);
    if (rc) return rc;
    m->src_is_aimw = 1;

    const aim_aimw_tensor *te = aim_aimw_find(&m->aw, "token_embd.weight");
    if (!te || te->ndims != 2) return 2;
    m->n_vocab = te->dims[0];          /* 行主序 [VOC, D] */
    m->n_embd  = te->dims[1];
    m->n_layers = meta_u32(m, "meta.nlayers", 0);
    if (!m->n_layers) {                /* 兜底: 扫 blk.N. 前缀计数 */
        for (uint32_t l = 0; ; l++) {
            char nm[64];
            snprintf(nm, sizeof(nm), "blk.%u.attn_q.weight", l);
            if (!aim_aimw_find(&m->aw, nm)) { m->n_layers = l; break; }
        }
    }
    m->n_heads    = meta_u32(m, "meta.n_heads", 0);
    m->n_kv_heads = meta_u32(m, "meta.n_kv_heads", m->n_heads);
    m->head_dim   = meta_u32(m, "meta.head_dim", m->n_heads ? m->n_embd / m->n_heads : 0);
    if (!m->n_heads) m->n_heads = m->head_dim ? m->n_embd / m->head_dim : 1;
    { const aim_aimw_tensor *tg = aim_aimw_find(&m->aw, "blk.0.ffn_gate.weight");
      m->n_ff = meta_u32(m, "meta.n_ff", tg ? tg->dims[0] : 0); }
    m->ctx_len   = meta_u32(m, "meta.ctx_len", 0);
    m->eps       = meta_f32(m, "meta.eps", 1e-5f);
    m->rope_base = meta_f32(m, "meta.rope_base", 1000000.0f);
    if (!m->n_layers || !m->n_heads || !m->n_embd || !m->n_vocab || !m->n_ff)
        return 2;
    /* BitNet 架构探测: attn_sub_norm 在场即为 BitNet (SubLN x2 + relu2) */
    m->bitnet = aim_aimw_find(&m->aw, "blk.0.attn_sub_norm.weight") ? 1u : 0u;
    if (!m->n_kv_heads) m->n_kv_heads = m->n_heads;
    if (!m->head_dim) m->head_dim = m->n_embd / m->n_heads;

    m->n_trefs = m->aw.ntensors;
    m->trefs = (aim_tref *)calloc(m->n_trefs ? m->n_trefs : 1, sizeof(aim_tref));
    if (!m->trefs) return 3;
    for (uint32_t i = 0; i < m->aw.ntensors; i++) {
        const aim_aimw_tensor *t = &m->aw.tensors[i];
        aim_tref *r = &m->trefs[i];
        snprintf(r->name, sizeof(r->name), "%s", t->name);
        r->qtype   = (t->dtype == AIM_AIMW_DT_TERN)  ? AIM_QT_TERN
                   : (t->dtype == AIM_AIMW_DT_TERNP) ? AIM_QT_TERNP
                   : (t->dtype == AIM_AIMW_DT_Q4KV2) ? AIM_QT_Q4KV2
                   : (t->dtype == AIM_AIMW_DT_Q8)    ? 8u : 0u;
        r->data    = t->data;
        r->n_elems = t->n_elems;
        r->scale   = t->scale;
        if (t->dtype == AIM_AIMW_DT_Q4KV2) {
            /* 头 32B = 16 x f16 点阵; data 指向超块区 */
            for (int k = 0; k < 16; k++) {
                uint16_t h16;
                memcpy(&h16, t->data + (size_t)k * 2, 2);
                r->lut[k] = f16_to_f32(h16);
            }
            r->data = t->data + 32;
        }
    }
#ifdef AIM_VNNI
    { const char *l4 = getenv("AIM_LM4");
      if (l4 && l4[0] == '1' && __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vl")) {
          for (uint32_t i = 0; i < m->n_trefs; i++)
              if (m->trefs[i].qtype == 8 && !strcmp(m->trefs[i].name, "token_embd.weight")) { lm4_prepare(&m->trefs[i]); if (m->trefs[i].alt) { g_lm4 = 1; g_q8v = 1; fprintf(stderr, "[mt] LOSSY 4-bit output layer on\n"); } }
      } }
    { const char *q8e = getenv("AIM_Q8V");
      if (q8e && q8e[0] == '1' && __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vl")) { g_q8v = 1; fprintf(stderr, "[mt] Q8 int8 VNNI output-layer kernel on\n"); } }
    { const char *ev = getenv("AIM_VNNI"); { const char *tv = getenv("AIM_TILE"); if (tv) g_tile = g_tile_big = atoi(tv); tv = getenv("AIM_TILE_BIG"); if (tv) g_tile_big = atoi(tv); if (g_tile < 1 || g_tile > 16) g_tile = 8; if (g_tile_big < 1 || g_tile_big > 16) g_tile_big = 8; }
      if (ev && ev[0] == '1' && __builtin_cpu_supports("avx512vnni") && m->bitnet && (m->n_embd & 255u) == 0 && (m->n_ff & 255u) == 0 && ((m->n_heads * m->head_dim) & 255u) == 0 && m->n_embd <= 8192 && m->n_ff <= 8192) { vnni_prepare(m->trefs, m->n_trefs); g_vnni = 1; fprintf(stderr, "[mt] VNNI int8 ternary kernel on\n"); } }
#endif
    return alloc_kv(m);
}

int aim_model_open(const char *path, aim_model *m) {
    memset(m, 0, sizeof(*m));
    /* magic 分流: "AIMW" -> 自研通道, 其余按 GGUF */
    FILE *f0 = fopen(path, "rb");
    if (!f0) return 1;
    char magic[4] = {0, 0, 0, 0};
    size_t got = fread(magic, 1, 4, f0);
    fclose(f0);
    if (got == 4 && !memcmp(magic, "AIMW", 4)) return open_aimw(path, m);
    return open_gguf(path, m);
}

void aim_model_close(aim_model *m) {
    if (!m) return;
    free(m->blob);
    free(m->kv_k);
    free(m->kv_v);
    free(m->trefs);
    aim_aimw_free(&m->aw);
    aim_gguf_free(&m->meta);
    memset(m, 0, sizeof(*m));
}

/* ---------- 单层前向 (增量 KV, M2.6) ----------
 * pos = 本 token 在 cache 中的槽位; 读 cache[0..pos], 写 cache[pos]
 * 数学与 M2 批量版逐位一致: 同一求和序, 同一 softmax 归一 */
#ifdef AIM_VNNI
/* ---------- mt patch: AVX-512 attention head (HD=128), run in parallel over (token, head) ---------- */
typedef struct { const float *q; float *att; const float *kk, *vv; float *ao; uint32_t HD, n_heads, grp, kv_dim, kv_cap, pos0, D; float scale; } attn_g;
static attn_g g_at;
static void attn_unit(const mv_ctx *c, uint32_t idx) {
    (void)c;
    const uint32_t nh = g_at.n_heads, h = idx % nh, t = idx / nh, pos = g_at.pos0 + t, kvh = h / g_at.grp;
    const uint32_t kvd = g_at.kv_dim;
    const float *qh = g_at.q + (size_t)t * g_at.D + (size_t)h * 128;
    float *ah = g_at.att + ((size_t)t * nh + h) * g_at.kv_cap;
    float *oh = g_at.ao + (size_t)t * g_at.D + (size_t)h * 128;
    const float *kb = g_at.kk + (size_t)kvh * 128, *vb = g_at.vv + (size_t)kvh * 128;
    __m512 q0 = _mm512_loadu_ps(qh), q1 = _mm512_loadu_ps(qh + 16), q2 = _mm512_loadu_ps(qh + 32), q3 = _mm512_loadu_ps(qh + 48);
    __m512 q4 = _mm512_loadu_ps(qh + 64), q5 = _mm512_loadu_ps(qh + 80), q6 = _mm512_loadu_ps(qh + 96), q7 = _mm512_loadu_ps(qh + 112);
    float mx = -1e30f;
    for (uint32_t u = 0; u <= pos; u++) {
        const float *kh = kb + (size_t)u * kvd;
        __m512 a0 = _mm512_mul_ps(q0, _mm512_loadu_ps(kh)), a1 = _mm512_mul_ps(q1, _mm512_loadu_ps(kh + 16));
        __m512 a2 = _mm512_mul_ps(q2, _mm512_loadu_ps(kh + 32)), a3 = _mm512_mul_ps(q3, _mm512_loadu_ps(kh + 48));
        a0 = _mm512_fmadd_ps(q4, _mm512_loadu_ps(kh + 64), a0); a1 = _mm512_fmadd_ps(q5, _mm512_loadu_ps(kh + 80), a1);
        a2 = _mm512_fmadd_ps(q6, _mm512_loadu_ps(kh + 96), a2); a3 = _mm512_fmadd_ps(q7, _mm512_loadu_ps(kh + 112), a3);
        float sc = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3))) * g_at.scale;
        ah[u] = sc;
        if (sc > mx) mx = sc;
    }
    float sum = 0.0f;
    for (uint32_t u = 0; u <= pos; u++) { ah[u] = expf(ah[u] - mx); sum += ah[u]; }
    const float inv = 1.0f / sum;
    __m512 o0 = _mm512_setzero_ps(), o1 = o0, o2 = o0, o3 = o0, o4 = o0, o5 = o0, o6 = o0, o7 = o0;
    for (uint32_t u = 0; u <= pos; u++) {
        const float *vh = vb + (size_t)u * kvd;
        __m512 w = _mm512_set1_ps(ah[u] * inv);
        o0 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vh), o0); o1 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vh + 16), o1);
        o2 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vh + 32), o2); o3 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vh + 48), o3);
        o4 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vh + 64), o4); o5 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vh + 80), o5);
        o6 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vh + 96), o6); o7 = _mm512_fmadd_ps(w, _mm512_loadu_ps(vh + 112), o7);
    }
    _mm512_storeu_ps(oh, o0); _mm512_storeu_ps(oh + 16, o1); _mm512_storeu_ps(oh + 32, o2); _mm512_storeu_ps(oh + 48, o3);
    _mm512_storeu_ps(oh + 64, o4); _mm512_storeu_ps(oh + 80, o5); _mm512_storeu_ps(oh + 96, o6); _mm512_storeu_ps(oh + 112, o7);
}
static void attn_run(const float *q, float *att, const float *kk, const float *vv, float *ao, uint32_t n_heads, uint32_t n_kv,
                     uint32_t kv_dim, uint32_t kv_cap, uint32_t pos0, uint32_t D, uint32_t ntok, float scale) {
    g_at.q = q; g_at.att = att; g_at.kk = kk; g_at.vv = vv; g_at.ao = ao; g_at.HD = 128; g_at.n_heads = n_heads; g_at.grp = n_heads / n_kv;
    g_at.kv_dim = kv_dim; g_at.kv_cap = kv_cap; g_at.pos0 = pos0; g_at.D = D; g_at.scale = scale;
    mv_ctx dummy; memset(&dummy, 0, sizeof dummy);
    parallel_for(&dummy, ntok * n_heads, 1, attn_unit);
}
#endif

static void layer_push(aim_model *m, uint32_t layer, float *x, uint32_t pos,
                       float *xn, float *q, float *k, float *v,
                       float *ao, float *proj, float *att,
                       float *gate, float *up) {
    char name[128];
    uint32_t D = m->n_embd, HD = m->head_dim;
    uint32_t kv_dim = m->n_kv_heads * HD;
    float *kk = m->kv_k + (size_t)layer * m->kv_cap * kv_dim;
    float *vv = m->kv_v + (size_t)layer * m->kv_cap * kv_dim;

    snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", layer);
    const aim_tref *tn = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_q.weight", layer);
    const aim_tref *tq = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_k.weight", layer);
    const aim_tref *tk = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_v.weight", layer);
    const aim_tref *tv = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_output.weight", layer);
    const aim_tref *to = mt(m, name);

    rmsnorm(x, (const float *)tn->data, xn, D, m->eps);
    matvec_any(tq, xn, q, D, D);
    matvec_any(tk, xn, k, kv_dim, D);
    matvec_any(tv, xn, v, kv_dim, D);
    for (uint32_t h = 0; h < m->n_heads; h++)
        rope(q + h * HD, pos, HD, m->rope_base, 1);
    for (uint32_t h = 0; h < m->n_kv_heads; h++)
        rope(k + h * HD, pos, HD, m->rope_base, 1);
    memcpy(kk + (size_t)pos * kv_dim, k, (size_t)kv_dim * 4);
    memcpy(vv + (size_t)pos * kv_dim, v, (size_t)kv_dim * 4);

    float scale = 1.0f / sqrtf((float)HD);
#ifdef AIM_VNNI
    if (g_vnni && HD == 128) attn_run(q, att, kk, vv, ao, m->n_heads, m->n_kv_heads, kv_dim, m->kv_cap, pos, D, 1, scale);
    else
#endif
    for (uint32_t h = 0; h < m->n_heads; h++) {
        uint32_t kvh = h / (m->n_heads / m->n_kv_heads);
        const float *qh = q + h * HD;
        float *ah = att + (size_t)h * m->kv_cap;
        float mx = -1e30f;
        for (uint32_t t = 0; t <= pos; t++) {
            const float *kh = kk + (size_t)t * kv_dim + kvh * HD;
            float s = 0.0f;
            for (uint32_t i = 0; i < HD; i++)
                s += qh[i] * kh[i];
            s *= scale;
            ah[t] = s;
            if (s > mx) mx = s;
        }
        float sum = 0.0f;
        for (uint32_t t = 0; t <= pos; t++) {
            ah[t] = expf(ah[t] - mx);
            sum += ah[t];
        }
        float inv = 1.0f / sum;
        float *oh = ao + h * HD;
        memset(oh, 0, HD * 4);
        for (uint32_t t = 0; t <= pos; t++) {
            const float *vh = vv + (size_t)t * kv_dim + kvh * HD;
            float w = ah[t] * inv;
            for (uint32_t i = 0; i < HD; i++)
                oh[i] += w * vh[i];
        }
    }
    if (m->bitnet) {
        /* SubLN: attention 输出先过 sub_norm 再进 o_proj (BitNet 特有) */
        snprintf(name, sizeof(name), "blk.%u.attn_sub_norm.weight", layer);
        tn = mt(m, name);
        rmsnorm(ao, (const float *)tn->data, ao, D, m->eps);
    }
    matvec_any(to, ao, proj, D, D);
    for (uint32_t i = 0; i < D; i++)
        x[i] += proj[i];

    /* FFN (SwiGLU; BitNet: relu2 + ffn_sub_norm) */
    snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", layer);
    tn = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", layer);
    const aim_tref *tg = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", layer);
    const aim_tref *tu = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", layer);
    const aim_tref *td = mt(m, name);

    rmsnorm(x, (const float *)tn->data, xn, D, m->eps);
    matvec_any(tg, xn, gate, m->n_ff, D);
    matvec_any(tu, xn, up, m->n_ff, D);
    if (m->bitnet) {
        /* relu2(gate) * up -> ffn_sub_norm -> down (SubLN 在激活后) */
        for (uint32_t i = 0; i < m->n_ff; i++) {
            float g = gate[i];
            gate[i] = (g > 0.0f ? g * g : 0.0f) * up[i];
        }
        snprintf(name, sizeof(name), "blk.%u.ffn_sub_norm.weight", layer);
        tn = mt(m, name);
        rmsnorm(gate, (const float *)tn->data, gate, m->n_ff, m->eps);
    } else {
        for (uint32_t i = 0; i < m->n_ff; i++) {
            float gv = gate[i];
            gate[i] = gv / (1.0f + expf(-gv)) * up[i];
        }
    }
    matvec_any(td, gate, proj, D, m->n_ff);
    for (uint32_t i = 0; i < D; i++)
        x[i] += proj[i];
}

/* ---------- 前向主入口 ---------- */

void aim_kv_reset(aim_model *m) {
    m->kv_len = 0;
}

/* 单 token 入 cache: 位置 = kv_len, attention 读 cache[0..pos] */
int aim_forward_push(aim_model *m, uint32_t id, float *logits) {
    uint32_t D = m->n_embd;
    uint32_t kv_dim = m->n_kv_heads * m->head_dim;
    if (!m->kv_cap) return 1;              /* cache 未分配 */
    if (m->kv_len >= m->kv_cap) return 2;  /* cache 满 */
    uint32_t pos = m->kv_len;

    float *x   = (float *)malloc((size_t)D * 4);
    float *xn  = (float *)malloc((size_t)D * 4);
    float *q   = (float *)malloc((size_t)D * 4);
    float *k   = (float *)malloc((size_t)kv_dim * 4);
    float *v   = (float *)malloc((size_t)kv_dim * 4);
    float *ao  = (float *)calloc((size_t)D, 4);
    float *proj = (float *)malloc((size_t)D * 4);
    float *att = (float *)malloc((size_t)m->n_heads * m->kv_cap * 4);
    float *gate = (float *)malloc((size_t)m->n_ff * 4);
    float *up  = (float *)malloc((size_t)m->n_ff * 4);
    if (!x || !xn || !q || !k || !v || !ao || !proj || !att || !gate || !up) {
        free(x); free(xn); free(q); free(k); free(v);
        free(ao); free(proj); free(att); free(gate); free(up);
        return 3;
    }

    /* embedding row (按 qtype 现场反量化) */
    deq_embd(mt(m, "token_embd.weight"), id, x, D);

    for (uint32_t l = 0; l < m->n_layers; l++)
        layer_push(m, l, x, pos, xn, q, k, v, ao, proj, att, gate, up);

    rmsnorm(x, (const float *)mt(m, "output_norm.weight")->data, xn, D, m->eps);
    const aim_tref *ot = mt(m, "output.weight");
    if (!ot) ot = mt(m, "token_embd.weight");   /* llama tied embedding 回落 */
    matvec_any(ot, xn, logits, m->n_vocab, D);

    m->kv_len = pos + 1;
    free(x); free(xn); free(q); free(k); free(v);
    free(ao); free(proj); free(att); free(gate); free(up);
    return 0;
}

/* ---------- 批量前向 (投机验证, M2.12) ----------
 * k 列共用一次权重读: 权重行首列拉入 DRAM 后 L1 复用 —— 带宽摊薄 k 倍。
 * 逐列计算与 aim_forward_push 串行完全同构 (同求和序/同 softmax 归一),
 * batch logits == 串行 logits 为构造性结论, 无需新 numpy 镜像。
 * 所有缓冲列主 [k][dim]; ATT 列主 [k][n_heads*kv_cap] */
#ifdef AIM_VNNI
typedef struct { float *X, *Xn, *Q, *K, *V, *AO, *PROJ, *GATE, *UP, *kk, *vv; const float *w; uint32_t D, kv_dim, HD, nh, nkv, nff, pos0; float eps, rope_base; } bl_ctx;
static void u_rms_x_xn(uint32_t t, const void *a) { const bl_ctx *b = (const bl_ctx *)a; rmsnorm(b->X + (size_t)t * b->D, b->w, b->Xn + (size_t)t * b->D, b->D, b->eps); }
static void u_rms_ao(uint32_t t, const void *a) { const bl_ctx *b = (const bl_ctx *)a; rmsnorm(b->AO + (size_t)t * b->D, b->w, b->AO + (size_t)t * b->D, b->D, b->eps); }
static void u_rope(uint32_t t, const void *a) {
    const bl_ctx *b = (const bl_ctx *)a;
    for (uint32_t h = 0; h < b->nh; h++) rope(b->Q + (size_t)t * b->D + h * b->HD, b->pos0 + t, b->HD, b->rope_base, 1);
    for (uint32_t h = 0; h < b->nkv; h++) rope(b->K + (size_t)t * b->kv_dim + h * b->HD, b->pos0 + t, b->HD, b->rope_base, 1);
    memcpy(b->kk + (size_t)(b->pos0 + t) * b->kv_dim, b->K + (size_t)t * b->kv_dim, (size_t)b->kv_dim * 4);
    memcpy(b->vv + (size_t)(b->pos0 + t) * b->kv_dim, b->V + (size_t)t * b->kv_dim, (size_t)b->kv_dim * 4);
}
static void u_resid(uint32_t t, const void *a) { const bl_ctx *b = (const bl_ctx *)a; for (uint32_t i = 0; i < b->D; i++) b->X[(size_t)t * b->D + i] += b->PROJ[(size_t)t * b->D + i]; }
static void u_ffn_act(uint32_t t, const void *a) {
    const bl_ctx *b = (const bl_ctx *)a;
    float *g = b->GATE + (size_t)t * b->nff; const float *u = b->UP + (size_t)t * b->nff;
    for (uint32_t i = 0; i < b->nff; i++) { float x = g[i]; g[i] = (x > 0.0f ? x * x : 0.0f) * u[i]; }
    rmsnorm(g, b->w, g, b->nff, b->eps);
}
#endif
#include <time.h>
static double g_prof[8]; static int g_prof_on = -1;
static inline double prof_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6; }
#define PROF_BEGIN double _pt = 0; if (g_prof_on) _pt = prof_now();
#define PROF(i) do { if (g_prof_on) { double _n = prof_now(); g_prof[i] += _n - _pt; _pt = _n; } } while (0)
static void layer_push_batch(aim_model *m, uint32_t layer, float *X, uint32_t pos0,
                             uint32_t k, float *Xn, float *Q, float *K, float *V,
                             float *AO, float *PROJ, float *ATT,
                             float *GATE, float *UP) {
    if (g_prof_on < 0) g_prof_on = getenv("AIM_PROF") ? 1 : 0;
    PROF_BEGIN
    char name[128];
    uint32_t D = m->n_embd, HD = m->head_dim;
    uint32_t kv_dim = m->n_kv_heads * HD;
    float *kk = m->kv_k + (size_t)layer * m->kv_cap * kv_dim;
    float *vv = m->kv_v + (size_t)layer * m->kv_cap * kv_dim;

    snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", layer);
    const aim_tref *tn = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_q.weight", layer);
    const aim_tref *tq = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_k.weight", layer);
    const aim_tref *tk = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_v.weight", layer);
    const aim_tref *tv = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.attn_output.weight", layer);
    const aim_tref *to = mt(m, name);

#ifdef AIM_VNNI
    bl_ctx bl; bl.X = X; bl.Xn = Xn; bl.Q = Q; bl.K = K; bl.V = V; bl.AO = AO; bl.PROJ = PROJ; bl.GATE = GATE; bl.UP = UP; bl.kk = kk; bl.vv = vv; bl.w = NULL;
    bl.D = D; bl.kv_dim = kv_dim; bl.HD = HD; bl.nh = m->n_heads; bl.nkv = m->n_kv_heads; bl.nff = m->n_ff; bl.pos0 = pos0; bl.eps = m->eps; bl.rope_base = m->rope_base;
#endif
#ifdef AIM_VNNI
    if (g_vnni) { bl.w = (const float *)tn->data; par_tokens(k, u_rms_x_xn, &bl); } else
#endif
    for (uint32_t t = 0; t < k; t++)
        rmsnorm(X + (size_t)t * D, (const float *)tn->data, Xn + (size_t)t * D, D, m->eps);
    PROF(0);
    matvec_any_batch(tq, Xn, Q, D, D, k);
    matvec_any_batch(tk, Xn, K, kv_dim, D, k);
    matvec_any_batch(tv, Xn, V, kv_dim, D, k);
    PROF(1);
#ifdef AIM_VNNI
    if (g_vnni) par_tokens(k, u_rope, &bl); else
#endif
    for (uint32_t t = 0; t < k; t++) {
        for (uint32_t h = 0; h < m->n_heads; h++)
            rope(Q + (size_t)t * D + h * HD, pos0 + t, HD, m->rope_base, 1);
        for (uint32_t h = 0; h < m->n_kv_heads; h++)
            rope(K + (size_t)t * kv_dim + h * HD, pos0 + t, HD, m->rope_base, 1);
        memcpy(kk + (size_t)(pos0 + t) * kv_dim, K + (size_t)t * kv_dim, (size_t)kv_dim * 4);
        memcpy(vv + (size_t)(pos0 + t) * kv_dim, V + (size_t)t * kv_dim, (size_t)kv_dim * 4);
    }

    PROF(0);
    float scale = 1.0f / sqrtf((float)HD);
#ifdef AIM_VNNI
    if (g_vnni && HD == 128) attn_run(Q, ATT, kk, vv, AO, m->n_heads, m->n_kv_heads, kv_dim, m->kv_cap, pos0, D, k, scale);
    else
#endif
    for (uint32_t t = 0; t < k; t++) {
        uint32_t pos = pos0 + t;
        for (uint32_t h = 0; h < m->n_heads; h++) {
            uint32_t kvh = h / (m->n_heads / m->n_kv_heads);
            const float *qh = Q + (size_t)t * D + h * HD;
            float *ah = ATT + (size_t)t * m->n_heads * m->kv_cap + (size_t)h * m->kv_cap;
            float mx = -1e30f;
            for (uint32_t u = 0; u <= pos; u++) {
                const float *kh = kk + (size_t)u * kv_dim + kvh * HD;
                float s = 0.0f;
                for (uint32_t i = 0; i < HD; i++)
                    s += qh[i] * kh[i];
                s *= scale;
                ah[u] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.0f;
            for (uint32_t u = 0; u <= pos; u++) {
                ah[u] = expf(ah[u] - mx);
                sum += ah[u];
            }
            float inv = 1.0f / sum;
            float *oh = AO + (size_t)t * D + h * HD;
            memset(oh, 0, HD * 4);
            for (uint32_t u = 0; u <= pos; u++) {
                const float *vh = vv + (size_t)u * kv_dim + kvh * HD;
                float w = ah[u] * inv;
                for (uint32_t i = 0; i < HD; i++)
                    oh[i] += w * vh[i];
            }
        }
    }
    PROF(2);
    if (m->bitnet) {
        snprintf(name, sizeof(name), "blk.%u.attn_sub_norm.weight", layer);
        tn = mt(m, name);
#ifdef AIM_VNNI
        if (g_vnni) { bl.w = (const float *)tn->data; par_tokens(k, u_rms_ao, &bl); } else
#endif
        for (uint32_t t = 0; t < k; t++)
            rmsnorm(AO + (size_t)t * D, (const float *)tn->data,
                    AO + (size_t)t * D, D, m->eps);
    }
    PROF(0);
    matvec_any_batch(to, AO, PROJ, D, D, k);
    PROF(1);
#ifdef AIM_VNNI
    if (g_vnni) par_tokens(k, u_resid, &bl); else
#endif
    for (uint32_t t = 0; t < k; t++)
        for (uint32_t i = 0; i < D; i++)
            X[(size_t)t * D + i] += PROJ[(size_t)t * D + i];

    snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", layer);
    tn = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", layer);
    const aim_tref *tg = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", layer);
    const aim_tref *tu = mt(m, name);
    snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", layer);
    const aim_tref *td = mt(m, name);

#ifdef AIM_VNNI
    if (g_vnni) { bl.w = (const float *)tn->data; par_tokens(k, u_rms_x_xn, &bl); } else
#endif
    for (uint32_t t = 0; t < k; t++)
        rmsnorm(X + (size_t)t * D, (const float *)tn->data, Xn + (size_t)t * D, D, m->eps);
    PROF(0);
    matvec_any_batch(tg, Xn, GATE, m->n_ff, D, k);
    matvec_any_batch(tu, Xn, UP, m->n_ff, D, k);
    PROF(1);
    if (m->bitnet) {
#ifdef AIM_VNNI
      if (g_vnni) {
        snprintf(name, sizeof(name), "blk.%u.ffn_sub_norm.weight", layer);
        tn = mt(m, name);
        bl.w = (const float *)tn->data;
        par_tokens(k, u_ffn_act, &bl);
      } else {
#endif
        for (uint32_t t = 0; t < k; t++)
            for (uint32_t i = 0; i < m->n_ff; i++) {
                float g = GATE[(size_t)t * m->n_ff + i];
                GATE[(size_t)t * m->n_ff + i] =
                    (g > 0.0f ? g * g : 0.0f) * UP[(size_t)t * m->n_ff + i];
            }
        snprintf(name, sizeof(name), "blk.%u.ffn_sub_norm.weight", layer);
        tn = mt(m, name);
        for (uint32_t t = 0; t < k; t++)
            rmsnorm(GATE + (size_t)t * m->n_ff, (const float *)tn->data,
                    GATE + (size_t)t * m->n_ff, m->n_ff, m->eps);
#ifdef AIM_VNNI
      }
#endif
    } else {
        for (uint32_t t = 0; t < k; t++)
            for (uint32_t i = 0; i < m->n_ff; i++) {
                float gv = GATE[(size_t)t * m->n_ff + i];
                GATE[(size_t)t * m->n_ff + i] =
                    gv / (1.0f + expf(-gv)) * UP[(size_t)t * m->n_ff + i];
            }
    }
    PROF(0);
    matvec_any_batch(td, GATE, PROJ, D, m->n_ff, k);
    PROF(1);
#ifdef AIM_VNNI
    if (g_vnni) par_tokens(k, u_resid, &bl); else
#endif
    for (uint32_t t = 0; t < k; t++)
        for (uint32_t i = 0; i < D; i++)
            X[(size_t)t * D + i] += PROJ[(size_t)t * D + i];
    PROF(0);
}

/* k token 批量入 cache (位置 pos0..pos0+k-1), logits [k][n_vocab] 列主 */
static int forward_push_batch_ex(aim_model *m, const uint32_t *ids, uint32_t k,
                           float *logits, int head_mode) {
    uint32_t D = m->n_embd;
    uint32_t kv_dim = m->n_kv_heads * m->head_dim;
    if (!m->kv_cap) return 1;
    if (m->kv_len + k > m->kv_cap) return 2;
    uint32_t pos0 = m->kv_len;
    size_t nhc = (size_t)m->n_heads * m->kv_cap;

    float *X    = (float *)malloc((size_t)k * D * 4);
    float *xn   = (float *)malloc((size_t)k * D * 4);
    float *q    = (float *)malloc((size_t)k * D * 4);
    float *kb   = (float *)malloc((size_t)k * kv_dim * 4);
    float *v    = (float *)malloc((size_t)k * kv_dim * 4);
    float *ao   = (float *)calloc((size_t)k * D, 4);
    float *proj = (float *)malloc((size_t)k * D * 4);
    float *att  = (float *)malloc((size_t)k * nhc * 4);
    float *gate = (float *)malloc((size_t)k * m->n_ff * 4);
    float *up   = (float *)malloc((size_t)k * m->n_ff * 4);
    if (!X || !xn || !q || !kb || !v || !ao || !proj || !att || !gate || !up) {
        free(X); free(xn); free(q); free(kb); free(v);
        free(ao); free(proj); free(att); free(gate); free(up);
        return 3;
    }

    for (uint32_t t = 0; t < k; t++)
        deq_embd(mt(m, "token_embd.weight"), ids[t], X + (size_t)t * D, D);

    for (uint32_t l = 0; l < m->n_layers; l++)
        layer_push_batch(m, l, X, pos0, k, xn, q, kb, v, ao, proj, att, gate, up);

    for (uint32_t t = 0; t < k; t++)
        rmsnorm(X + (size_t)t * D, (const float *)mt(m, "output_norm.weight")->data,
                xn + (size_t)t * D, D, m->eps);
    const aim_tref *ot = mt(m, "output.weight");
    if (!ot) ot = mt(m, "token_embd.weight");
    if (head_mode == 0) matvec_any_batch(ot, xn, logits, m->n_vocab, D, k);
    else if (head_mode == 1) matvec_any(ot, xn + (size_t)(k - 1) * D, logits, m->n_vocab, D);   /* prefill: only the last column's logits */

    m->kv_len = pos0 + k;
    free(X); free(xn); free(q); free(kb); free(v);
    free(ao); free(proj); free(att); free(gate); free(up);
    return 0;
}

int aim_forward_push_batch(aim_model *m, const uint32_t *ids, uint32_t k, float *logits) {
    return forward_push_batch_ex(m, ids, k, logits, 0);
}

/* KV 回滚 (投机拒绝段废弃; 被回滚位置下轮覆盖) */
void aim_kv_rollback(aim_model *m, uint32_t new_len) {
    if (new_len <= m->kv_len) m->kv_len = new_len;
}

/* 全量重算语义 (对账口径): 清 cache 后逐 push
 * 与 M2 批量版逐位一致 —— 单 token matvec/softmax 求和序不变 */
int aim_forward_last(aim_model *m, const uint32_t *ids, uint32_t n,
                     float *logits) {
    aim_kv_reset(m);
    for (uint32_t p = 0; p < n; p++) {
        int rc = aim_forward_push(m, ids[p], logits);
        if (rc) return rc;
    }
    return 0;
}

/* mt patch: batched prefill helper (AIM_PREFILL_BATCH=k), falls back to one push per token */
static int prefill_ids(aim_model *m, const uint32_t *ids, uint32_t n, float *logits) {
    const char *pb = getenv("AIM_PREFILL_BATCH");
    uint32_t pkb = pb ? (uint32_t)atoi(pb) : 0;
    if (pkb > 1 && n > 1) {
        for (uint32_t p = 0; p < n;) {
            uint32_t ch = n - p < pkb ? n - p : pkb;
            int rc = forward_push_batch_ex(m, ids + p, ch, logits, (p + ch == n) ? 1 : 2);
            if (rc) return rc;
            p += ch;
        }
        return 0;
    }
    for (uint32_t p = 0; p < n; p++) {
        int rc = aim_forward_push(m, ids[p], logits);
        if (rc) return rc;
    }
    return 0;
}

int aim_generate(aim_model *m, const uint32_t *ids, uint32_t n,
                 uint32_t n_next, uint32_t *out_ids) {
    { const char *sp = getenv("AIM_SPEC"); if (sp && atoi(sp) > 1) return aim_generate_spec(m, ids, n, n_next, (uint32_t)atoi(sp), out_ids); }
    /* prefill 一次 + 每步增量 push: O(n^2)->O(1)/token */
    float *logits = (float *)malloc((size_t)m->n_vocab * 4);
    if (!logits) return 3;
    aim_kv_reset(m);
    { int rcp = prefill_ids(m, ids, n, logits); if (rcp) { free(logits); return rcp; } }
    for (uint32_t s = 0; s < n_next; s++) {
        uint32_t best = 0;
        for (uint32_t v = 1; v < m->n_vocab; v++)
            if (logits[v] > logits[best]) best = v;
        out_ids[s] = best;
        if (s + 1 < n_next) {
            int rc = aim_forward_push(m, best, logits);
            if (rc) { free(logits); return rc; }
        }
    }
    free(logits);
    return 0;
}

/* ---------- 投机解码 (prompt-lookup 起草 + 批量验证, M2.12) ----------
 * 数学: 批量验证 k 草稿 = 权重读一遍 (带宽摊薄 k 倍); 接受最长 greedy 前缀,
 * 输出与串行 aim_generate 构造性逐 id 一致 (接受判据=与贪心预测全等)。
 * 起草: d[0]=贪心悬挂预测 (必接受), d[1..]=context 末 token 的最近后继复制。
 * 接受走查: a = 连续正确数 (>=1); a==k 时悬挂预测 = L[k-1] argmax,
 * a<k 时回滚 kv_len 到 pos0+a 且悬挂预测 = L[a-1] argmax —— 零额外前向。
 * 坏情况 (零命中) k=1 等价串行, 不亏。 */
static uint32_t argmax32(const float *v, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

/* mt patch: n-gram draft. S[0..n-1] with S[n-1] = pending token (= d[0]). Find the longest earlier suffix match (up to 8 tokens,
 * most recent on ties) and draft the tokens that followed it. Returns k >= 1 (d[0] set by the caller). */
static uint32_t draft_ngram(const uint32_t *S, uint32_t n, uint32_t *d, uint32_t kmax) {
    uint32_t bestL = 0, besti = 0;
    for (uint32_t i = n - 1; i-- > 0;) {
        uint32_t L = 0;
        while (L < 8 && L <= i && S[i - L] == S[n - 1 - L]) L++;
        if (L > bestL) { bestL = L; besti = i; if (L >= 8) break; }
    }
    uint32_t k = 1;
    uint32_t cap = 1 + 3 * bestL; if (cap > kmax) cap = kmax;
    if (bestL >= 1) while (k < cap && besti + k < n) { d[k] = S[besti + k]; k++; }
    return k;
}
static uint32_t g_spec_steps = 0, g_spec_tokens = 0;
int aim_generate_spec(aim_model *m, const uint32_t *ids, uint32_t n,
                      uint32_t n_next, uint32_t kmax, uint32_t *out_ids) {
    if (kmax <= 1) return aim_generate(m, ids, n, n_next, out_ids);
    if (kmax > 16) kmax = 16;
    uint32_t V = m->n_vocab;
    float *logits = (float *)malloc((size_t)kmax * V * 4);
    uint32_t *ctx = (uint32_t *)malloc(((size_t)n + n_next + kmax + 16) * 4);
    uint32_t *d = (uint32_t *)malloc((size_t)kmax * 4);
    if (!logits || !ctx || !d) { free(logits); free(ctx); free(d); return 3; }
    aim_kv_reset(m);
    for (uint32_t p = 0; p < n; p++) ctx[p] = ids[p];
    int rc0 = prefill_ids(m, ids, n, logits);
    if (rc0) { free(logits); free(ctx); free(d); return rc0; }
    uint32_t ctx_n = n;
    uint32_t pend = argmax32(logits, V);
    uint32_t out_n = 0;
    g_spec_steps = 0; g_spec_tokens = 0;
    while (out_n < n_next) {
        d[0] = pend;
        ctx[ctx_n] = pend;
        uint32_t k = draft_ngram(ctx, ctx_n + 1, d, kmax);
        uint32_t pos0 = m->kv_len;
        int rc = (k == 1) ? aim_forward_push(m, d[0], logits) : aim_forward_push_batch(m, d, k, logits);
        if (rc) { free(logits); free(ctx); free(d); return rc; }
        uint32_t a = 1;
        while (a < k && d[a] == argmax32(logits + (size_t)(a - 1) * V, V)) a++;
        pend = argmax32(logits + (size_t)(a - 1) * V, V);
        aim_kv_rollback(m, pos0 + a);
        g_spec_steps++; g_spec_tokens += a;
        for (uint32_t t = 0; t < a; t++) {
            ctx[ctx_n++] = d[t];
            if (out_n < n_next) out_ids[out_n++] = d[t];
        }
    }
    if (getenv("AIM_PROF")) fprintf(stderr, "[spec] kmax=%u steps=%u tokens=%u avg accepted/step=%.2f\n", kmax, g_spec_steps, g_spec_tokens, g_spec_steps ? (double)g_spec_tokens / g_spec_steps : 0.0);
    free(logits); free(ctx); free(d);
    return 0;
}
