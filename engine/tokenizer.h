/* aimp M2.5: gpt2 式 byte-level BPE 分词器 (纯 C, 零依赖) */
#ifndef AIM_TOKENIZER_H
#define AIM_TOKENIZER_H

#include "gguf.h"

typedef struct {
    const aim_gguf *g;
    uint64_t n_tokens, n_merges;
    char **tokens;
    char **merges;
    int32_t *hash_head, *hash_next;
    uint32_t HT;
    int32_t *mr_head, *mr_next;
    char **mr_key;
    int32_t *mr_rank;
    uint32_t MHT;
    uint32_t byte_id[256];
} aim_tok;

int  aim_tok_init(const aim_gguf *g, aim_tok *t);
void aim_tok_free(aim_tok *t);
int  aim_tok_encode(aim_tok *t, const char *text,
                    uint32_t **out, uint32_t *n_out);
int  aim_tok_decode(aim_tok *t, const uint32_t *ids, uint32_t n,
                    char **out, uint32_t *len_out);

#endif
