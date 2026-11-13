/* aimp M2.7: AIM-W 权重容器解析 (纯 C, 零第三方)
 * 契约源: aim-kernel/spec/aimw-format.md (v0/v1, 格式规范即主权)
 * 码表: 0->0, 1->+1, 2->-1, 3=保留(必须拒绝)
 * 反量化: w = float(code) * scale, 单次 fp32 乘法精确
 * 错误码: 0 成功; 1=IO; 2=格式(含码3/未知version/未知dtype); 3=内存
 */
#ifndef AIM_AIMW_H
#define AIM_AIMW_H

#include <stdint.h>

#define AIM_AIMW_DT_FP32    0u
#define AIM_AIMW_DT_TERN    1u   /* int8 逐码 1B + 尾随 f32 scale */
#define AIM_AIMW_DT_TERNP   2u   /* 2bit 打包 0.25B/码 + 尾随 f32 scale */
#define AIM_AIMW_DT_Q4KV2   3u   /* Q4_K 容器 + 16 点点点阵 (32B 码表 + 144B/256 权重) */
#define AIM_AIMW_DT_Q8      4u   /* Q8_0 块驻留 (34B/32 权重, f16 scale x int8), 无尾随 */
/* v2 (M2.8): Q4_K 容器 + 全张量学习型非均匀点阵 (AIM-W v2 护城河)
 * payload = 32B 码表(16 x f16 升序) + ceil(n/256) x 144B Q4_K 超块
 * 超块布局同 GGML q4_K: 2B f16 d + 2B f16 dmin + 12B scales(6bit x8x2) + 128B nibbles
 * 反量化: w = d*sc*LUT[n] - dmin*m, n∈[0,15] 为点阵索引
 * 均匀点阵 LUT={0..15} 即退化为 GGML q4_K 语义 (向后兼容基线) */

typedef struct {
    char     name[128];
    uint32_t dtype;
    uint32_t ndims;
    uint32_t dims[8];
    uint64_t n_elems;
    const uint8_t *data;   /* 原始码驻留(不预展开 fp32 -- 三值内存优势所在) */
    float    scale;        /* ternary per-tensor; fp32 时为 0 */
} aim_aimw_tensor;

typedef struct {
    uint32_t version;
    uint32_t ntensors;
    uint8_t *blob;             /* 全文件驻留, tensor.data 指入其中 */
    aim_aimw_tensor *tensors;
} aim_aimw;

int  aim_aimw_open(const char *path, aim_aimw *out);
void aim_aimw_free(aim_aimw *w);
const aim_aimw_tensor *aim_aimw_find(const aim_aimw *w, const char *name);

#endif
