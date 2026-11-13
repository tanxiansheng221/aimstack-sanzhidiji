/* aimp M2.7: AIM-W 解析器实现 (与 aim-kernel/kernel/aim_kernel.c 加载器逐语义对齐)
 * 纪律: 解析器遇到未知 version/dtype/码3 必须拒绝, 禁止猜测 (spec 硬性)
 */
#include "aimw.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 64 位文件寻址: Windows long = 32 位, >2GB 的 .aimw 必须用 _fseeki64/_ftelli64 */
#if defined(_WIN32)
#define AIM_FSEEK64(f, o, w) _fseeki64((f), (o), (w))
#define AIM_FTELL64(f)       _ftelli64(f)
#else
#define AIM_FSEEK64(f, o, w) fseeko((f), (o), (w))
#define AIM_FTELL64(f)       ftello(f)
#endif

int aim_aimw_open(const char *path, aim_aimw *w) {
    memset(w, 0, sizeof(*w));
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    if (AIM_FSEEK64(f, 0, SEEK_END)) { fclose(f); return 1; }
    int64_t fsz = AIM_FTELL64(f);
    if (fsz < 12) { fclose(f); return 2; }
    AIM_FSEEK64(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)fsz);
    if (!buf) { fclose(f); return 3; }
    if (fread(buf, 1, (size_t)fsz, f) != (size_t)fsz) {
        fclose(f); free(buf); return 1;
    }
    fclose(f);

    uint64_t pos = 0;
    if (memcmp(buf, "AIMW", 4) != 0) { free(buf); return 2; }
    pos = 4;
    uint32_t ver, nt;
    memcpy(&ver, buf + pos, 4); pos += 4;
    memcpy(&nt,  buf + pos, 4); pos += 4;
    if (ver != 0u && ver != 1u) { free(buf); return 2; }

    w->version  = ver;
    w->ntensors = nt;
    w->blob     = buf;
    w->tensors  = (aim_aimw_tensor *)calloc(nt ? nt : 1, sizeof(aim_aimw_tensor));
    if (!w->tensors) { free(buf); w->blob = NULL; return 3; }

    for (uint32_t i = 0; i < nt; i++) {
        aim_aimw_tensor *t = &w->tensors[i];
        uint32_t nl;
        if (pos + 4 > (uint64_t)fsz) return 2;
        memcpy(&nl, buf + pos, 4); pos += 4;
        if (nl == 0 || nl >= sizeof(t->name) || pos + nl > (uint64_t)fsz) return 2;
        memcpy(t->name, buf + pos, nl); t->name[nl] = 0; pos += nl;
        if (pos + 8 > (uint64_t)fsz) return 2;
        memcpy(&t->dtype,  buf + pos, 4); pos += 4;
        memcpy(&t->ndims,  buf + pos, 4); pos += 4;
        if (t->ndims == 0 || t->ndims > 8) return 2;
        if (pos + 4u * t->ndims > (uint64_t)fsz) return 2;
        uint64_t n = 1;
        for (uint32_t d = 0; d < t->ndims; d++) {
            memcpy(&t->dims[d], buf + pos, 4); pos += 4;
            n *= t->dims[d];
        }
        t->n_elems = n;
        uint32_t nb;
        if (pos + 4 > (uint64_t)fsz) return 2;
        memcpy(&nb, buf + pos, 4); pos += 4;

        /* 期望 data 尺寸 (spec) */
        uint64_t want;
        uint32_t tail = 0;
        if (t->dtype == AIM_AIMW_DT_FP32) {
            want = (uint64_t)n * 4u;
        } else if (t->dtype == AIM_AIMW_DT_TERN) {
            want = n; tail = 4;
        } else if (t->dtype == AIM_AIMW_DT_TERNP) {
            want = (n + 3u) / 4u; tail = 4;
        } else if (t->dtype == AIM_AIMW_DT_Q4KV2) {
            /* 32B 码表 + 144B/256权重超块; 维度非 256 倍数 = 拒绝 */
            if (n % 256u != 0u) return 2;
            want = 32u + (n / 256u) * 144u; tail = 0;
        } else if (t->dtype == AIM_AIMW_DT_Q8) {
            if (n % 32u != 0u) return 2;
            want = (n / 32u) * 34u; tail = 0;
        } else {
            return 2;   /* 未知 dtype: 拒绝 */
        }
        if (nb != want + tail || pos + nb > (uint64_t)fsz) return 2;

        t->data = buf + pos;
        if (t->dtype == AIM_AIMW_DT_TERN || t->dtype == AIM_AIMW_DT_TERNP) {
            memcpy(&t->scale, buf + pos + want, 4);
            /* 码3 扫描: 装载期一次性拒绝, 运行时零检查 */
            if (t->dtype == AIM_AIMW_DT_TERN) {
                for (uint64_t e = 0; e < n; e++) {
                    int8_t c = (int8_t)t->data[e];
                    if (c != -1 && c != 0 && c != 1) return 2;
                }
            } else {
                uint64_t npacked = want;
                for (uint64_t b = 0; b < npacked; b++) {
                    uint8_t byte = t->data[b];
                    for (uint32_t s = 0; s < 4; s++) {
                        uint64_t gidx = b * 4u + s;
                        uint8_t code = (uint8_t)((byte >> (s * 2u)) & 3u);
                        if (gidx < n) { if (code == 3u) return 2; }
                        else if (code != 0u) return 2;   /* 填充位必须码 0 */
                    }
                }
            }
        }
        pos += nb;
    }
    return 0;
}

void aim_aimw_free(aim_aimw *w) {
    if (!w) return;
    free(w->blob);
    free(w->tensors);
    memset(w, 0, sizeof(*w));
}

const aim_aimw_tensor *aim_aimw_find(const aim_aimw *w, const char *name) {
    for (uint32_t i = 0; i < w->ntensors; i++)
        if (!strcmp(w->tensors[i].name, name)) return &w->tensors[i];
    return NULL;
}
