/* aimp M2.5: BPE tokenizer (pure C stdlib) */
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t b2u_table[256];
static int b2u_ready = 0;

static void b2u_build(void) {
    if (b2u_ready) return;
    uint32_t bs[256], cs[256];
    uint32_t n = 0;
    for (int b = 33; b < 127; b++) { bs[n] = (uint32_t)b; cs[n] = (uint32_t)b; n++; }
    for (int b = 161; b < 173; b++) { bs[n] = (uint32_t)b; cs[n] = (uint32_t)b; n++; }
    for (int b = 174; b < 256; b++) { bs[n] = (uint32_t)b; cs[n] = (uint32_t)b; n++; }
    uint32_t extra = 256;
    for (int b = 0; b < 256; b++) {
        int direct = 0;
        for (uint32_t i = 0; i < n; i++)
            if (bs[i] == (uint32_t)b) { direct = 1; break; }
        if (!direct) { bs[n] = (uint32_t)b; cs[n] = extra++; n++; }
    }
    for (uint32_t i = 0; i < 256; i++) b2u_table[bs[i]] = cs[i];
    b2u_ready = 1;
}

uint32_t aim_b2u_cp(uint8_t b) {
    b2u_build();
    return b2u_table[b];
}

int aim_cp_utf8(uint32_t cp, char *buf) {
    if (cp < 0x80) { buf[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    buf[0] = (char)(0xE0 | (cp >> 12));
    buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

int aim_utf8_next(const char *s, uint32_t len, uint32_t pos, uint32_t *cp) {
    if (pos >= len) return -1;
    uint8_t c = (uint8_t)s[pos];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && pos + 1 < len) {
        *cp = ((c & 0x1F) << 6) | ((uint8_t)s[pos+1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && pos + 2 < len) {
        *cp = ((c & 0x0F) << 12) | (((uint8_t)s[pos+1] & 0x3F) << 6)
            | ((uint8_t)s[pos+2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && pos + 3 < len) {
        *cp = ((c & 0x07) << 18) | (((uint8_t)s[pos+1] & 0x3F) << 12)
            | (((uint8_t)s[pos+2] & 0x3F) << 6) | ((uint8_t)s[pos+3] & 0x3F);
        return 4;
    }
    return -1;
}

static uint32_t fnv(const char *s, uint32_t len) {
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static int str_find(char **arr, int32_t *head, int32_t *next, uint32_t HT,
                    const char *key, uint32_t klen, uint32_t *out_id) {
    uint32_t h = fnv(key, klen) % HT;
    for (int32_t e = head[h]; e >= 0; e = next[e]) {
        if (strlen(arr[e]) == klen && !memcmp(arr[e], key, klen)) {
            *out_id = (uint32_t)e;
            return 1;
        }
    }
    return 0;
}

/* ---- init / free ---- */
static void tok_hash_build(aim_tok *t) {
    t->HT = 1u << 18;
    t->hash_head = (int32_t *)malloc(t->HT * 4);
    t->hash_next = (int32_t *)malloc((size_t)t->n_tokens * 4);
    for (uint32_t i = 0; i < t->HT; i++) t->hash_head[i] = -1;
    for (uint64_t i = 0; i < t->n_tokens; i++) {
        uint32_t h = fnv(t->tokens[i], (uint32_t)strlen(t->tokens[i]))
                   % t->HT;
        t->hash_next[i] = t->hash_head[h];
        t->hash_head[h] = (int32_t)i;
    }
}

static void mr_hash_build(aim_tok *t) {
    t->MHT = 1u << 18;
    t->mr_head = (int32_t *)malloc(t->MHT * 4);
    t->mr_next = (int32_t *)malloc((size_t)t->n_merges * 4);
    t->mr_key  = (char **)calloc((size_t)t->n_merges, sizeof(char *));
    t->mr_rank = (int32_t *)malloc((size_t)t->n_merges * 4);
    for (uint32_t i = 0; i < t->MHT; i++) t->mr_head[i] = -1;
    for (uint64_t i = 0; i < t->n_merges; i++) {
        const char *m = t->merges[i];
        uint32_t klen = (uint32_t)strlen(m);
        t->mr_key[i] = t->merges[i];
        uint32_t h = fnv(m, klen) % t->MHT;
        t->mr_next[i] = t->mr_head[h];
        t->mr_head[h] = (int32_t)i;
        t->mr_rank[i] = (int32_t)i;
    }
}

int aim_tok_init(const aim_gguf *g, aim_tok *t) {
    memset(t, 0, sizeof(*t));
    t->g = g;
    const aim_kv *kt = aim_gguf_find(g, "tokenizer.ggml.tokens");
    const aim_kv *km = aim_gguf_find(g, "tokenizer.ggml.merges");
    if (!kt || kt->elem_vt != AIM_T_STR) return 2;
    t->n_tokens = kt->elem_count;
    t->tokens   = kt->v.strs;
    fprintf(stderr, "DBG n_tokens=%llu\n", (unsigned long long)t->n_tokens);
    b2u_build();
    fprintf(stderr, "DBG b2u ok\n");
    tok_hash_build(t);
    fprintf(stderr, "DBG hash ok\n");
    for (int b = 0; b < 256; b++) {
        char u[8];
        int ul = aim_cp_utf8(aim_b2u_cp((uint8_t)b), u);
        u[ul] = 0;
        uint32_t id;
        t->byte_id[b] = 0xFFFFFFFFu;
        if (str_find(t->tokens, t->hash_head, t->hash_next, t->HT,
                     u, (uint32_t)ul, &id))
            t->byte_id[b] = id;
    }
    if (km && km->elem_vt == AIM_T_STR) {
        t->n_merges = km->elem_count;
        t->merges   = km->v.strs;
        fprintf(stderr, "DBG merges=%llu\n", (unsigned long long)t->n_merges);
        mr_hash_build(t);
        fprintf(stderr, "DBG mr ok\n");
    }
    return 0;
}

static int mr_find(aim_tok *t, const char *left, uint32_t llen,
                   const char *right, uint32_t rlen, int32_t *rank) {
    char buf[512];
    if (llen + rlen + 2 > sizeof(buf)) return 0;
    memcpy(buf, left, llen);
    buf[llen] = ' ';
    memcpy(buf + llen + 1, right, rlen);
    uint32_t klen = llen + rlen + 1;
    uint32_t h = fnv(buf, klen) % t->MHT;
    for (int32_t e = t->mr_head[h]; e >= 0; e = t->mr_next[e]) {
        if (strlen(t->mr_key[e]) == klen && !memcmp(t->mr_key[e], buf, klen)) {
            *rank = t->mr_rank[e];
            return 1;
        }
    }
    return 0;
}
static int bpe_merge(aim_tok *t, const char *piece, uint32_t plen,
                     uint32_t *ids, uint32_t *n_ids, uint32_t cap) {
    uint32_t local[512];
    uint32_t n = 0, pos = 0;
    while (pos < plen) {
        uint32_t cp;
        int cl = aim_utf8_next(piece, plen, pos, &cp);
        if (cl < 0) { fprintf(stderr, "BM: utf8 fail\n"); return 1; }
        pos += (uint32_t)cl;
        uint8_t byte = 0xFF;
        int found = 0;
        for (int b = 0; b < 256; b++) {
            if (aim_b2u_cp((uint8_t)b) == cp) {
                byte = (uint8_t)b; found = 1; break;
            }
        }
        if (!found || t->byte_id[byte] == 0xFFFFFFFFu) {
            fprintf(stderr, "BM: byte fail cp=%u found=%d\n", cp, found);
            return 1;
        }
        if (n >= 500) return 1;
        local[n++] = t->byte_id[byte];
    }
    while (n > 1) {
        int32_t best_rank = -1;
        uint32_t best_i = 0;
        for (uint32_t i = 0; i + 1 < n; i++) {
            int32_t r;
            if (mr_find(t, t->tokens[local[i]],
                        (uint32_t)strlen(t->tokens[local[i]]),
                        t->tokens[local[i+1]],
                        (uint32_t)strlen(t->tokens[local[i+1]]), &r)) {
                if (best_rank < 0 || r < best_rank) {
                    best_rank = r; best_i = i;
                }
            }
        }
        if (best_rank < 0) break;
        char merged[1024];
        uint32_t l1 = (uint32_t)strlen(t->tokens[local[best_i]]);
        uint32_t l2 = (uint32_t)strlen(t->tokens[local[best_i+1]]);
        if (l1 + l2 + 1 > sizeof(merged)) return 1;
        memcpy(merged, t->tokens[local[best_i]], l1);
        memcpy(merged + l1, t->tokens[local[best_i+1]], l2);
        merged[l1 + l2] = 0;
        uint32_t mid;
        if (!str_find(t->tokens, t->hash_head, t->hash_next, t->HT,
                      merged, l1 + l2, &mid)) {
            fprintf(stderr, "BM: merged not found [%s]\n", merged);
            return 1;
        }
        local[best_i] = mid;
        memmove(local + best_i + 1, local + best_i + 2, (n - best_i - 2) * 4);
        n--;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (*n_ids >= cap) return 1;
        ids[(*n_ids)++] = local[i];
    }
    return 0;
}
/* classify: 0=ws 1=alpha 2=digit 3=symbol */
static int uni_alpha(uint32_t cp) {
    if (cp < 0x80) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    if (cp >= 0xAA && cp <= 0x2AF) return 1;      /* latin-1 ext + ext A/B */
    if (cp >= 0x370 && cp <= 0x3FF) return 1;     /* greek */
    if (cp >= 0x400 && cp <= 0x4FF) return 1;     /* cyrillic */
    if (cp >= 0x531 && cp <= 0x58F) return 1;     /* armenian */
    if (cp >= 0x5D0 && cp <= 0x5EA) return 1;     /* hebrew */
    if (cp >= 0x620 && cp <= 0x64A) return 1;     /* arabic */
    if (cp >= 0x900 && cp <= 0x97F) return 1;     /* devanagari */
    if (cp >= 0xE00 && cp <= 0xE5B) return 1;     /* thai */
    if (cp >= 0x3040 && cp <= 0x30FF) return 1;   /* kana */
    if (cp >= 0x3400 && cp <= 0x9FFF) return 1;   /* CJK */
    if (cp >= 0xAC00 && cp <= 0xD7AF) return 1;   /* hangul */
    if (cp >= 0xF900 && cp <= 0xFAFF) return 1;   /* CJK compat */
    if (cp >= 0xFF21 && cp <= 0xFF3A) return 1;   /* fullwidth A-Z */
    if (cp >= 0xFF41 && cp <= 0xFF5A) return 1;   /* fullwidth a-z */
    if (cp >= 0x20000 && cp <= 0x2FA1F) return 1; /* CJK ext B+ */
    return 0;
}

static int ch_kind(uint32_t cp) {
    if (cp == 0x20 || (cp >= 0x09 && cp <= 0x0D)) return 0;
    if (cp >= 0x30 && cp <= 0x39) return 2;
    if (uni_alpha(cp)) return 1;
    return 3;
}

int aim_tok_encode(aim_tok *t, const char *text,
                   uint32_t **out, uint32_t *n_out) {
    uint32_t tlen = (uint32_t)strlen(text);
    uint32_t cap = tlen * 4 + 16;
    uint32_t *ids = (uint32_t *)malloc(cap * 4);
    uint32_t n = 0;
    uint32_t i = 0;
    while (i < tlen) {
        /* 特殊 token 优先匹配: <|...|> 整段直查词表 (added_tokens 不在
         * BPE merges 里, 走合并路径必被切碎 — BitNet chat template 实测) */
        if (text[i] == '<' && i + 1 < tlen && text[i + 1] == '|') {
            uint32_t j = i + 1;
            while (j + 1 < tlen && !(text[j] == '|' && text[j + 1] == '>')) j++;
            if (j + 1 < tlen) {
                j += 2;
                uint32_t sid;
                if (str_find(t->tokens, t->hash_head, t->hash_next, t->HT,
                             text + i, j - i, &sid)) {
                    if (n + 1 > cap) { free(ids); return 1; }
                    ids[n++] = sid;
                    i = j;
                    continue;
                }
            }
        }
        uint32_t cp, cp2 = 0;
        int cl = aim_utf8_next(text, tlen, i, &cp);
        if (cl < 0) { free(ids); return 1; }
        int k = ch_kind(cp);
        uint32_t j = i;
        uint32_t ncl;
        if (k == 0 && cp == 0x20 && i + 1 < tlen) {
            /* gpt2 \s+(?!\S): in a run of spaces only the LAST one
             * attaches to the following segment */
            uint32_t w = 1;
            while (i + w < tlen && text[i + w] == ' ') w++;
            if (i + w < tlen) {
                /* attaching space = text[i+w-1], following seg starts i+w */
                uint32_t cp2;
                int cl2 = aim_utf8_next(text, tlen, i + w, &cp2);
                int k2 = ch_kind(cp2);
                j = i + w + (uint32_t)cl2;
                if (k2 == 1 || k2 == 2) {
                    while (j < tlen) {
                        uint32_t cpx;
                        int cx = aim_utf8_next(text, tlen, j, &cpx);
                        if (cx < 0 || ch_kind(cpx) != k2) break;
                        j += (uint32_t)cx;
                    }
                }
                /* emit (w-1) standalone spaces first */
                for (uint32_t q = 0; q + 1 < w; q++) {
                    char pc[8];
                    int pl = aim_cp_utf8(aim_b2u_cp(0x20), pc);
                    if (bpe_merge(t, pc, (uint32_t)pl, ids, &n, cap)) {
                        free(ids); return 1;
                    }
                }
                i = i + w - 1;   /* attach segment starts at last space */
            } else {
                j = i + w;       /* trailing spaces: all standalone */
            }
            /* fallthrough: build piece from [i, j) */
            {
                char piece[2048];
                uint32_t plen = 0;
                for (uint32_t p = i; p < j; p++) {
                    char ub[8];
                    int ul = aim_cp_utf8(aim_b2u_cp((uint8_t)text[p]), ub);
                    if (plen + ul + 1 > sizeof(piece)) break;
                    memcpy(piece + plen, ub, ul);
                    plen += (uint32_t)ul;
                }
                if (bpe_merge(t, piece, plen, ids, &n, cap)) {
                    free(ids); return 1;
                }
                i = j;
                continue;
            }
        } else if (k == 3) {
            j += (uint32_t)cl;
        } else {
            j += (uint32_t)cl;
            while (j < tlen) {
                ncl = aim_utf8_next(text, tlen, j, &cp2);
                if (ncl < 0 || ch_kind(cp2) != k) break;
                j += ncl;
                if (k == 0) break;      /* ws: single */
            }
        }
        uint32_t seg_start = i;
        /* piece: bytes -> byte-unicode utf8 string */
        char piece[2048];
        uint32_t plen = 0;
        for (uint32_t p = seg_start; p < j; p++) {
            char ub[8];
            int ul = aim_cp_utf8(aim_b2u_cp((uint8_t)text[p]), ub);
            if (plen + ul + 1 > sizeof(piece)) break;
            memcpy(piece + plen, ub, ul);
            plen += (uint32_t)ul;
        }
        if (bpe_merge(t, piece, plen, ids, &n, cap)) {
            free(ids); return 1;
        }
        i = j;
    }
    *out = ids;
    *n_out = n;
    return 0;
}
int aim_tok_decode(aim_tok *t, const uint32_t *ids, uint32_t n,
                   char **out, uint32_t *len_out) {
    size_t cap = n * 8 + 16;
    char *buf = (char *)malloc(cap);
    size_t bl = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (ids[i] >= t->n_tokens) { free(buf); return 1; }
        const char *tok = t->tokens[ids[i]];
        uint32_t tlen = (uint32_t)strlen(tok);
        uint32_t pos = 0;
        int all_bytes = 1;
        while (pos < tlen) {
            uint32_t cp;
            int cl = aim_utf8_next(tok, tlen, pos, &cp);
            if (cl < 0) { all_bytes = 0; break; }
            int found = 0;
            for (int b = 0; b < 256; b++) {
                if (aim_b2u_cp((uint8_t)b) == cp) { found = 1; break; }
            }
            if (!found) { all_bytes = 0; break; }
            pos += (uint32_t)cl;
        }
        if (all_bytes) {
            pos = 0;
            while (pos < tlen) {
                uint32_t cp;
                int cl = aim_utf8_next(tok, tlen, pos, &cp);
                for (int b = 0; b < 256; b++) {
                    if (aim_b2u_cp((uint8_t)b) == cp) {
                        if (bl + 1 >= cap) { cap *= 2; buf = (char *)realloc(buf, cap); }
                        buf[bl++] = (char)b;
                        break;
                    }
                }
                pos += (uint32_t)cl;
            }
        } else {
            while (bl + tlen + 1 >= cap) { cap *= 2; buf = (char *)realloc(buf, cap); }
            memcpy(buf + bl, tok, tlen);
            bl += tlen;
        }
    }
    buf[bl] = 0;
    *out = buf;
    *len_out = (uint32_t)bl;
    return 0;
}

void aim_tok_free(aim_tok *t) {
    if (!t) return;
    free(t->hash_head);
    free(t->hash_next);
    free(t->mr_head);
    free(t->mr_next);
    free(t->mr_key);
    free(t->mr_rank);
    memset(t, 0, sizeof(*t));
}
