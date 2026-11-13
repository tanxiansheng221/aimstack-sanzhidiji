/* aimp M2: 前向引擎实现
 * 权重驻留: Q8_0/F32 原始块驻留, matmul 现场反量化 (省内存: 2.49GB vs 10.6GB fp32)
 * 反量化单乘精确: q8 值 * f32 scale, 无二次舍入
 */
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
static void parallel_for(const mv_ctx *c, uint32_t rows, uint32_t min_rows,
                         void (*fn)(const mv_ctx *, uint32_t)) {
    (void)min_rows;
    for (uint32_t o = 0; o < rows; o++) fn(c, o);
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
    parallel_for(&c, rows, 2048, fn);
}

/* 批量 matvec 入口 (投机验证): k 列共用一次权重读。
 * 热路径 (TERNP/Q8) 走列外循环包装器; 其余 qtype 逐列退化为单列调用。
 * x 列主 [k][in_dim], y 列主 [k][rows] */
static void matvec_any_batch(const aim_tref *t, const float *x, float *y,
                             uint32_t rows, uint32_t in_dim, uint32_t k) {
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
static void layer_push_batch(aim_model *m, uint32_t layer, float *X, uint32_t pos0,
                             uint32_t k, float *Xn, float *Q, float *K, float *V,
                             float *AO, float *PROJ, float *ATT,
                             float *GATE, float *UP) {
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

    for (uint32_t t = 0; t < k; t++)
        rmsnorm(X + (size_t)t * D, (const float *)tn->data, Xn + (size_t)t * D, D, m->eps);
    matvec_any_batch(tq, Xn, Q, D, D, k);
    matvec_any_batch(tk, Xn, K, kv_dim, D, k);
    matvec_any_batch(tv, Xn, V, kv_dim, D, k);
    for (uint32_t t = 0; t < k; t++) {
        for (uint32_t h = 0; h < m->n_heads; h++)
            rope(Q + (size_t)t * D + h * HD, pos0 + t, HD, m->rope_base, 1);
        for (uint32_t h = 0; h < m->n_kv_heads; h++)
            rope(K + (size_t)t * kv_dim + h * HD, pos0 + t, HD, m->rope_base, 1);
        memcpy(kk + (size_t)(pos0 + t) * kv_dim, K + (size_t)t * kv_dim, (size_t)kv_dim * 4);
        memcpy(vv + (size_t)(pos0 + t) * kv_dim, V + (size_t)t * kv_dim, (size_t)kv_dim * 4);
    }

    float scale = 1.0f / sqrtf((float)HD);
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
    if (m->bitnet) {
        snprintf(name, sizeof(name), "blk.%u.attn_sub_norm.weight", layer);
        tn = mt(m, name);
        for (uint32_t t = 0; t < k; t++)
            rmsnorm(AO + (size_t)t * D, (const float *)tn->data,
                    AO + (size_t)t * D, D, m->eps);
    }
    matvec_any_batch(to, AO, PROJ, D, D, k);
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

    for (uint32_t t = 0; t < k; t++)
        rmsnorm(X + (size_t)t * D, (const float *)tn->data, Xn + (size_t)t * D, D, m->eps);
    matvec_any_batch(tg, Xn, GATE, m->n_ff, D, k);
    matvec_any_batch(tu, Xn, UP, m->n_ff, D, k);
    if (m->bitnet) {
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
    } else {
        for (uint32_t t = 0; t < k; t++)
            for (uint32_t i = 0; i < m->n_ff; i++) {
                float gv = GATE[(size_t)t * m->n_ff + i];
                GATE[(size_t)t * m->n_ff + i] =
                    gv / (1.0f + expf(-gv)) * UP[(size_t)t * m->n_ff + i];
            }
    }
    matvec_any_batch(td, GATE, PROJ, D, m->n_ff, k);
    for (uint32_t t = 0; t < k; t++)
        for (uint32_t i = 0; i < D; i++)
            X[(size_t)t * D + i] += PROJ[(size_t)t * D + i];
}

/* k token 批量入 cache (位置 pos0..pos0+k-1), logits [k][n_vocab] 列主 */
int aim_forward_push_batch(aim_model *m, const uint32_t *ids, uint32_t k,
                           float *logits) {
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
    matvec_any_batch(ot, xn, logits, m->n_vocab, D, k);

    m->kv_len = pos0 + k;
    free(X); free(xn); free(q); free(kb); free(v);
    free(ao); free(proj); free(att); free(gate); free(up);
    return 0;
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

int aim_generate(aim_model *m, const uint32_t *ids, uint32_t n,
                 uint32_t n_next, uint32_t *out_ids) {
    /* prefill 一次 + 每步增量 push: O(n^2)->O(1)/token */
    float *logits = (float *)malloc((size_t)m->n_vocab * 4);
    if (!logits) return 3;
    aim_kv_reset(m);
    for (uint32_t p = 0; p < n; p++) {
        int rc = aim_forward_push(m, ids[p], logits);
        if (rc) { free(logits); return rc; }
    }
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

int aim_generate_spec(aim_model *m, const uint32_t *ids, uint32_t n,
                      uint32_t n_next, uint32_t kmax, uint32_t *out_ids) {
    if (kmax <= 1) return aim_generate(m, ids, n, n_next, out_ids);
    uint32_t V = m->n_vocab;
    float *logits = (float *)malloc((size_t)kmax * V * 4);
    uint32_t *ctx = (uint32_t *)malloc(((size_t)n + n_next + kmax + 16) * 4);
    uint32_t *d = (uint32_t *)malloc((size_t)kmax * 4);
    if (!logits || !ctx || !d) {
        free(logits); free(ctx); free(d);
        return 3;
    }
    aim_kv_reset(m);
    for (uint32_t p = 0; p < n; p++) {
        int rc = aim_forward_push(m, ids[p], logits);
        if (rc) { free(logits); free(ctx); free(d); return rc; }
        ctx[p] = ids[p];
    }
    uint32_t ctx_n = n;
    uint32_t pend = argmax32(logits, V);
    uint32_t out_n = 0;
    while (out_n < n_next) {
        /* 起草: d[0]=悬挂贪心; d[1..]=ctx 末 token 最近出现处的后继复制 */
        d[0] = pend;
        uint32_t k = 1;
        if (ctx_n > 0) {
            uint32_t c0 = ctx[ctx_n - 1];
            for (uint32_t i = ctx_n - 1; i-- > 0;) {
                if (ctx[i] == c0) {
                    while (k < kmax && i + k < ctx_n) { d[k] = ctx[i + k]; k++; }
                    break;
                }
            }
        }
        uint32_t pos0 = m->kv_len;
        int rc = aim_forward_push_batch(m, d, k, logits);
        if (rc) { free(logits); free(ctx); free(d); return rc; }
        uint32_t a = 1;                       /* d[0]=pend 恒正确 */
        while (a < k && d[a] == argmax32(logits + (size_t)(a - 1) * V, V)) a++;
        pend = argmax32(logits + (size_t)(a - 1) * V, V);
        aim_kv_rollback(m, pos0 + a);
        for (uint32_t t = 0; t < a; t++) {
            ctx[ctx_n++] = d[t];
            if (out_n < n_next) out_ids[out_n++] = d[t];
        }
    }
    free(logits); free(ctx); free(d);
    return 0;
}
