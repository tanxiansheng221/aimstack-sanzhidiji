/* aimp M2/M2.6/M2.7: 前向引擎 (纯 C, 零第三方)
 * 对账纪律 (80.6): 求和序契约 + 反量化单乘精确 + /fp:precise
 * M2.7 双通道: GGUF 与 AIM-W 平级接入, 前向只认统一张量表 (来源不感知)
 */
#ifndef AIM_FORWARD_H
#define AIM_FORWARD_H

#include "gguf.h"
#include "aimw.h"

/* 自研量化码 (GGML qtype 枚举之外的私有段) */
#define AIM_QT_TERN   31u   /* int8 逐码, per-tensor scale */
#define AIM_QT_TERNP  32u   /* 2bit 打包, per-tensor scale */
#define AIM_QT_Q4KV2  33u   /* AIM-W v2: Q4_K 容器 + 学习型 16 点点阵 (lut) */

/* 统一张量引用: GGUF/AIM-W 装载后同形态 */
typedef struct {
    char     name[128];
    uint32_t qtype;        /* 0=F32 8=Q8_0 31=TERN 32=TERNP 33=Q4KV2 */
    const uint8_t *data;
    uint64_t n_elems;
    float    scale;        /* ternary per-tensor; 其余为 0 */
    float    lut[16];      /* Q4KV2: 点阵 f16 解码缓存; 其余为 0 */
} aim_tref;

typedef struct {
    aim_gguf  meta;
    uint8_t  *blob;        /* GGUF 权重数据段全量驻留 */
    uint64_t  blob_size;
    aim_aimw  aw;          /* AIM-W 源 (src_is_aimw=1 时有效) */
    uint32_t  src_is_aimw;
    aim_tref *trefs;       /* 统一张量表 */
    uint32_t  n_trefs;
    /* 架构参数 (从 header/meta 读出, 一码两跑) */
    uint32_t  n_layers, n_heads, n_kv_heads, head_dim;
    uint32_t  n_embd, n_ff, n_vocab;
    uint32_t  bitnet;      /* BitNet 架构: attn/ffn 各带 sub_norm, FFN 用 relu2 */
    uint32_t  ctx_len;
    float     eps, rope_base;
    float    *kv_k, *kv_v;      /* [n_layers][ctx_cap][kv_dim] */
    uint32_t  kv_len, kv_cap;
} aim_model;

int  aim_model_open(const char *path, aim_model *m);   /* 按 magic 分流 GGUF/AIMW */
void aim_model_close(aim_model *m);
int  aim_forward_last(aim_model *m, const uint32_t *ids, uint32_t n,
                      float *logits);
int  aim_forward_push(aim_model *m, uint32_t id, float *logits);
int  aim_forward_push_batch(aim_model *m, const uint32_t *ids, uint32_t k,
                            float *logits);   /* logits [k][n_vocab] 列主 */
void aim_kv_reset(aim_model *m);
void aim_kv_rollback(aim_model *m, uint32_t new_len);
int  aim_generate(aim_model *m, const uint32_t *ids, uint32_t n,
                  uint32_t n_next, uint32_t *out_ids);
int  aim_generate_spec(aim_model *m, const uint32_t *ids, uint32_t n,
                       uint32_t n_next, uint32_t kmax, uint32_t *out_ids);

#endif
