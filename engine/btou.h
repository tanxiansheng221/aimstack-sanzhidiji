/* GPT-2 bytes_to_unicode 映射 (纯 C, 零依赖) */
#ifndef AIM_BTOU_H
#define AIM_BTOU_H
#include <stdint.h>

uint32_t aim_b2u_cp(uint8_t b);
int aim_cp_utf8(uint32_t cp, char *buf);
int aim_utf8_next(const char *s, uint32_t len, uint32_t pos, uint32_t *cp);

#endif
