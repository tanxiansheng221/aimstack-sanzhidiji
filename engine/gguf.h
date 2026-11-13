/* aimp - 纯自研模型加载器 M1: GGUF v3 适配层
 * 断供免疫宪章合规: 仅依赖 C 标准库, 零第三方
 * 架构约定: 本层只做「外部格式 -> 统一张量表」;
 *          日后自研 AIM-W 格式平级接入, 前向引擎不感知来源。
 */
#ifndef AIM_GGUF_H
#define AIM_GGUF_H

#include <stdint.h>
#include <stddef.h>

#define AIM_GGUF_MAGIC 0x46554747u /* "GGUF" little-endian */

/* GGUF metadata value types */
enum {
    AIM_T_U8 = 0, AIM_T_I8, AIM_T_U16, AIM_T_I16, AIM_T_U32, AIM_T_I32,
    AIM_T_F32, AIM_T_BOOL, AIM_T_STR, AIM_T_ARR, AIM_T_U64, AIM_T_I64, AIM_T_F64
};

typedef struct {
    char    *key;
    uint32_t vt;          /* AIM_T_* */
    uint32_t elem_vt;     /* 数组元素类型, 非数组为 0 */
    uint64_t elem_count;  /* 数组长度, 非数组为 0 */
    /* 标量统一放 union; 字符串放 str; 数组放 arr (连续标量) 或 strs */
    union {
        uint64_t u;
        double   f;
        char    *str;
        void    *arr;
        char   **strs;
    } v;
} aim_kv;

typedef struct {
    char    *name;
    uint32_t n_dims;
    uint64_t *dims;
    uint32_t qtype;       /* GGML quant type */
    uint64_t offset;      /* 相对 data 段起始 */
    uint64_t n_elems;
    uint64_t nbytes;      /* 量化布局换算 */
} aim_tensor;

typedef struct {
    uint32_t   version;
    uint32_t   align;
    uint64_t   n_kv;
    uint64_t   n_tensors;
    aim_kv    *kvs;
    aim_tensor *tensors;
    uint64_t   header_end;   /* header 逻辑结束偏移 */
    uint64_t   data_start;   /* 对齐后的 data 段起始 */
} aim_gguf;

/* 返回 0 成功; 非 0 失败 (1=IO, 2=格式, 3=内存) */
int  aim_gguf_open(const char *path, aim_gguf *out);
void aim_gguf_free(aim_gguf *g);
const aim_kv *aim_gguf_find(const aim_gguf *g, const char *key);
const char *aim_gguf_qt_name(uint32_t qt);

#endif
