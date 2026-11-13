/* aimp M1: GGUF v3 解析实现 (纯 C 标准库, 零第三方)
 * 对账: tools/make_test_gguf.py 合成文件 + 真实模型双验证
 */
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { FILE *f; int err; } Rd;

static int rd_bytes(Rd *r, void *buf, size_t n) {
    if (r->err) return 1;
    if (fread(buf, 1, n, r->f) != n) { r->err = 1; return 1; }
    return 0;
}
static uint8_t  rd_u8 (Rd *r) { uint8_t  v = 0; rd_bytes(r, &v, 1); return v; }
static uint16_t rd_u16(Rd *r) { uint16_t v = 0; rd_bytes(r, &v, 2); return v; }
static uint32_t rd_u32(Rd *r) { uint32_t v = 0; rd_bytes(r, &v, 4); return v; }
static uint64_t rd_u64(Rd *r) { uint64_t v = 0; rd_bytes(r, &v, 8); return v; }
static int16_t  rd_i16(Rd *r) { int16_t v; rd_bytes(r, &v, 2); return v; }
static int32_t  rd_i32(Rd *r) { int32_t v; rd_bytes(r, &v, 4); return v; }
static int64_t  rd_i64(Rd *r) { int64_t v; rd_bytes(r, &v, 8); return v; }

static char *rd_str(Rd *r) {
    uint64_t n = rd_u64(r);
    if (r->err || n > ((uint64_t)1 << 30)) { r->err = r->err ? r->err : 2; return NULL; }
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) { r->err = 3; return NULL; }
    if (n && rd_bytes(r, s, (size_t)n)) { free(s); return NULL; }
    s[n] = 0;
    return s;
}

static int scalar_size(uint32_t vt) {
    switch (vt) {
    case AIM_T_U8: case AIM_T_I8: case AIM_T_BOOL: return 1;
    case AIM_T_U16: case AIM_T_I16: return 2;
    case AIM_T_U32: case AIM_T_I32: case AIM_T_F32: return 4;
    case AIM_T_U64: case AIM_T_I64: case AIM_T_F64: return 8;
    default: return -1;
    }
}

/* 量化布局: (每块字节, 每块值数); 元素级 (元素字节, 1) */
typedef struct { uint32_t bytes, vals; } QtLayout;
static const QtLayout QL[] = {
    {4,1},{2,1},{18,32},{20,32},{0,0},{0,0},{22,32},{24,32}, /* F32,F16,Q4_0,Q4_1,-,-,Q5_0,Q5_1 */
    {34,32},{36,32},                                         /* Q8_0,Q8_1 */
    {84,256},{110,256},{144,256},{176,256},{210,256},        /* Q2_K..Q6_K */
    {0,0},                                                   /* Q8_K 特例 */
    {0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},   /* IQ 占位 */
    {0,0},{0,0},{0,0},{0,0},{0,0},
    {2,1},                                                   /* BF16 */
};
static const char *QT_N[] = {
    "F32","F16","Q4_0","Q4_1","Q4_2","Q4_3","Q5_0","Q5_1",
    "Q8_0","Q8_1","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K","Q8_K",
    "IQ2_XXS","IQ2_XS","IQ3_XXS","IQ1_S","IQ4_NL","IQ3_S","IQ2_S","IQ4_XS",
    "I8","I16","I32","I64","F64","T29","BF16"
};
const char *aim_gguf_qt_name(uint32_t qt) {
    return qt < sizeof(QT_N)/sizeof(QT_N[0]) ? QT_N[qt] : "UNK";
}

static uint64_t tensor_nbytes(uint32_t qtype, uint64_t n_elems) {
    if (qtype < sizeof(QL)/sizeof(QL[0])) {
        QtLayout l = QL[qtype];
        if (l.vals == 1) return n_elems * l.bytes;
        if (l.vals) return (n_elems + l.vals - 1) / l.vals * l.bytes;
        if (qtype == 15) return n_elems / 256 * 292; /* Q8_K */
    }
    return 0;
}

static int read_value(Rd *r, uint32_t vt, aim_kv *kv) {
    memset(&kv->v, 0, sizeof(kv->v));
    uint8_t b[8];
    switch (vt) {
    case AIM_T_U8:   rd_bytes(r, b, 1); kv->v.u = b[0]; return r->err;
    case AIM_T_I8:   rd_bytes(r, b, 1); kv->v.u = (uint64_t)(int64_t)(int8_t)b[0]; return r->err;
    case AIM_T_BOOL: rd_bytes(r, b, 1); kv->v.u = b[0]; return r->err;
    case AIM_T_U16:  rd_bytes(r, b, 2); kv->v.u = *(uint16_t *)b; return r->err;
    case AIM_T_I16:  rd_bytes(r, b, 2); kv->v.u = (uint64_t)(int64_t)*(int16_t *)b; return r->err;
    case AIM_T_U32:  rd_bytes(r, b, 4); kv->v.u = *(uint32_t *)b; return r->err;
    case AIM_T_I32:  rd_bytes(r, b, 4); kv->v.u = (uint64_t)(int64_t)*(int32_t *)b; return r->err;
    case AIM_T_U64:  rd_bytes(r, b, 8); kv->v.u = *(uint64_t *)b; return r->err;
    case AIM_T_I64:  rd_bytes(r, b, 8); kv->v.u = *(uint64_t *)b; return r->err;
    case AIM_T_F32:  rd_bytes(r, b, 4); { float f; memcpy(&f, b, 4); kv->v.f = f; } return r->err;
    case AIM_T_F64:  rd_bytes(r, b, 8); { double d; memcpy(&d, b, 8); kv->v.f = d; } return r->err;
    case AIM_T_STR:
        kv->v.str = rd_str(r);
        return kv->v.str ? 0 : (r->err ? r->err : 3);
    case AIM_T_ARR: {
        kv->elem_vt = rd_u32(r);
        uint64_t cnt = rd_u64(r);
        kv->elem_count = cnt;
        if (kv->elem_vt == AIM_T_STR) {
            kv->v.strs = (char **)malloc((size_t)cnt * sizeof(char *));
            if (!kv->v.strs) return 3;
            for (uint64_t i = 0; i < cnt; i++) {
                kv->v.strs[i] = rd_str(r);
                if (!kv->v.strs[i]) return r->err ? r->err : 3;
            }
        } else {
            int sz = scalar_size(kv->elem_vt);
            if (sz < 0) return 2;
            kv->v.arr = malloc((size_t)(cnt * (uint64_t)sz));
            if (!kv->v.arr) return 3;
            if (cnt && rd_bytes(r, kv->v.arr, (size_t)(cnt * (uint64_t)sz))) return 1;
        }
        return 0;
    }
    default: return 2;
    }
}

static void free_kv(aim_kv *kv) {
    free(kv->key);
    if (kv->vt == AIM_T_STR) { free(kv->v.str); }
    else if (kv->vt == AIM_T_ARR) {
        if (kv->elem_vt == AIM_T_STR) {
            for (uint64_t i = 0; i < kv->elem_count; i++) free(kv->v.strs[i]);
            free(kv->v.strs);
        } else free(kv->v.arr);
    }
}

int aim_gguf_open(const char *path, aim_gguf *out) {
    memset(out, 0, sizeof(*out));
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    Rd rds = { f, 0 };
    Rd *r = &rds;

    uint32_t magic = rd_u32(r);
    if (r->err || magic != AIM_GGUF_MAGIC) { fclose(f); return 2; }
    uint32_t version = rd_u32(r);
    if (r->err || version < 1 || version > 3) { fclose(f); return 2; }
    out->version = version;

    uint64_t n_tensors = rd_u64(r);
    uint64_t n_kv = rd_u64(r);
    out->n_tensors = n_tensors;
    out->n_kv = n_kv;
    if (n_kv > 100000u || n_tensors > 1000000u) { fclose(f); return 2; }

    out->kvs = (aim_kv *)calloc((size_t)n_kv, sizeof(aim_kv));
    if (!out->kvs) { fclose(f); return 3; }

    int rc = 0;
    for (uint64_t i = 0; i < n_kv && !rc; i++) {
        aim_kv *kv = &out->kvs[i];
        kv->key = rd_str(r);
        if (!kv->key) { rc = r->err ? r->err : 2; break; }
        kv->vt = rd_u32(r);
        rc = read_value(r, kv->vt, kv);
    }

    out->align = 32;
    for (uint64_t i = 0; i < n_kv && !rc; i++) {
        if (!strcmp(out->kvs[i].key, "general.alignment") &&
            out->kvs[i].vt == AIM_T_U32)
            out->align = (uint32_t)out->kvs[i].v.u;
    }
    if (out->align == 0 || out->align > 4096) out->align = 32;

    if (!rc) {
        out->tensors = (aim_tensor *)calloc((size_t)n_tensors, sizeof(aim_tensor));
        if (!out->tensors) rc = 3;
    }
    for (uint64_t i = 0; i < n_tensors && !rc; i++) {
        aim_tensor *t = &out->tensors[i];
        t->name = rd_str(r);
        if (!t->name) { rc = r->err ? r->err : 2; break; }
        t->n_dims = rd_u32(r);
        if (t->n_dims < 1 || t->n_dims > 4) { rc = 2; break; }
        t->dims = (uint64_t *)malloc((size_t)t->n_dims * sizeof(uint64_t));
        if (!t->dims) { rc = 3; break; }
        t->n_elems = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = rd_u64(r);
            t->n_elems *= t->dims[d];
        }
        t->qtype = rd_u32(r);
        t->offset = rd_u64(r);
        t->nbytes = tensor_nbytes(t->qtype, t->n_elems);
        if (t->nbytes == 0) { rc = 2; break; }
    }

    if (r->err && !rc) rc = 1;
    if (rc) { fclose(f); aim_gguf_free(out); return rc; }

    /* data 段起始 = header 尾向上对齐 align */
    long end = ftell(f);
    uint64_t hend = (uint64_t)end;
    out->header_end = hend;
    out->data_start = (hend + out->align - 1) / out->align * out->align;
    fclose(f);
    return 0;
}

void aim_gguf_free(aim_gguf *g) {
    if (!g) return;
    for (uint64_t i = 0; i < g->n_kv; i++) free_kv(&g->kvs[i]);
    free(g->kvs);
    if (g->tensors) {
        for (uint64_t i = 0; i < g->n_tensors; i++) {
            free(g->tensors[i].name);
            free(g->tensors[i].dims);
        }
        free(g->tensors);
    }
    memset(g, 0, sizeof(*g));
}

const aim_kv *aim_gguf_find(const aim_gguf *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (!strcmp(g->kvs[i].key, key)) return &g->kvs[i];
    return NULL;
}
