/* aimp probe CLI: metadata dump / gen / chat / plugin bus */
#include "gguf.h"
#include "forward.h"
#include "tokenizer.h"
#include "bus.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int run_tok(const char *path, int enc, const char *text);

static void print_arch(const aim_gguf *g) {
    const aim_kv *arch = aim_gguf_find(g, "general.architecture");
    const char *base = (arch && arch->vt == AIM_T_STR) ? arch->v.str : "";
    printf("[arch] %s\n", base);
    const char *keys[] = {
        "context_length", "embedding_length", "block_count",
        "attention.head_count", "attention.head_count_kv",
        "attention.key_length", "feed_forward_length",
        "rope.dimension_count", "vocab_size",
        "expert_count", "expert_used_count",
        "attention.layer_norm_rms_epsilon"
    };
    char full[256];
    for (size_t i = 0; i < sizeof(keys)/sizeof(keys[0]); i++) {
        snprintf(full, sizeof(full), "%s.%s", base, keys[i]);
        const aim_kv *kv = aim_gguf_find(g, full);
        if (!kv) continue;
        if (kv->vt == AIM_T_F32)
            printf("  %-46s = %.8g\n", full, kv->v.f);
        else
            printf("  %-46s = %llu\n", full, (unsigned long long)kv->v.u);
    }
}

static void print_vocab(const aim_gguf *g) {
    const aim_kv *toks = aim_gguf_find(g, "tokenizer.ggml.tokens");
    const aim_kv *mdl  = aim_gguf_find(g, "tokenizer.ggml.model");
    printf("[vocab] model=%s tokens=%llu\n",
           (mdl && mdl->vt == AIM_T_STR) ? mdl->v.str : "?",
           (toks && toks->vt == AIM_T_ARR) ? (unsigned long long)toks->elem_count : 0ull);
    if (toks && toks->vt == AIM_T_ARR && toks->elem_count >= 4)
        printf("  sample: [%s] [%s] [%s] [%s]\n", toks->v.strs[0],
               toks->v.strs[1], toks->v.strs[2], toks->v.strs[3]);
}

#ifdef _WIN32
/* platform layer: wmain -> utf8 argv (core stays portable) */
static char *g_argv8[64];
static char g_buf8[16384];
int real_main(int argc, char **argv);
int wmain(int argc, wchar_t **wargv) {
    char *p8 = g_buf8;
    size_t used = 0;
    if (argc > 64) argc = 64;
    for (int i = 0; i < argc; i++) {
        int need = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                       NULL, 0, NULL, NULL);
        if (need <= 0 || used + (size_t)need > sizeof(g_buf8)) return 10;
        g_argv8[i] = p8;
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, p8, need, NULL, NULL);
        p8 += need;
        used += (size_t)need;
    }
    return real_main(argc, g_argv8);
}
#define main real_main
#endif

/* 通用对话生成: prompt 已按 chat template 组装好; 计时含 prefill */
static int run_chat_prompt(const char *weights, const char *tokgguf,
                           const char *prompt, uint32_t n_next) {
    aim_gguf g;
    int rc = aim_gguf_open(tokgguf, &g);
    if (rc) { fprintf(stderr, "tok gguf FAIL rc=%d\n", rc); return rc; }
    aim_tok t;
    if (aim_tok_init(&g, &t)) { fprintf(stderr, "tok init FAIL\n"); return 1; }
    uint32_t *pids;
    uint32_t np;
    if (aim_tok_encode(&t, prompt, &pids, &np)) {
        fprintf(stderr, "enc FAIL\n"); return 1;
    }
    uint32_t *ids = (uint32_t *)malloc((np + 1) * 4);
    ids[0] = 128000u;                            /* <|begin_of_text|> BOS */
    memcpy(ids + 1, pids, np * 4);
    uint32_t n = np + 1;
    free(pids);
    aim_model m;
    rc = aim_model_open(weights, &m);            /* AIMW magic -> 自研加载器 */
    if (rc) { fprintf(stderr, "model open FAIL rc=%d\n", rc); return rc; }
    uint32_t *out = (uint32_t *)malloc(n_next * 4);
    clock_t c0 = clock();
    rc = aim_generate(&m, ids, n, n_next, out);
    clock_t c1 = clock();
    if (rc) { fprintf(stderr, "gen FAIL rc=%d\n", rc); return rc; }
    double dt = (double)(c1 - c0) / CLOCKS_PER_SEC;
    char *txt;
    uint32_t len;
    if (aim_tok_decode(&t, out, n_next, &txt, &len)) {
        fprintf(stderr, "dec FAIL\n"); return 1;
    }
    /* 对话结束符截断: <|eot_id|> 后是未停机的重复 (aim_generate 只认 EOS) */
    char *eot = strstr(txt, "<|eot_id|>");
    if (eot) *eot = 0;
    printf("PROMPT_TOKS:%u\n", n);
    printf("GEN_TEXT:%s\n", txt);
    printf("TIME %.2fs  %.2f tok/s  (gen %u tok, incl. prefill)\n",
           dt, n_next / dt, n_next);
    free(txt); free(out); free(ids);
    aim_tok_free(&t);
    aim_gguf_free(&g);
    aim_model_close(&m);
    return 0;
}

/* --gt: gen-text — 自研加载器全家桶 (AIMW 权重 + GGUF 分词器同进程):
 * aimp_probe <weights.aimw> <tokenizer.gguf> --gt "text" n_next
 * 计时只含生成段 (权重加载排除); llama3 BOS 手动前置 */
static int run_gt(const char *weights, const char *tokgguf,
                  const char *text, uint32_t n_next) {
    char *buf = (char *)malloc(strlen(text) + 64);
    sprintf(buf, "User: %s<|eot_id|>Assistant: ", text);
    int rc = run_chat_prompt(weights, tokgguf, buf, n_next);
    free(buf);
    return rc;
}

/* 结果 JSON 里取 "key": "value" 的字符串值 (p 指向开引号之后) */
static size_t json_str_after(const char *p, char *out, size_t cap) {
    size_t n = 0;
    while (*p && *p != '"') {
        char c = *p;
        if (c == '\\' && p[1]) {
            p++;
            c = *p;
            if (c == 'n') c = '\n';
            else if (c == 't') c = '\t';
        }
        if (n + 1 < cap) out[n++] = c;
        p++;
    }
    out[n] = 0;
    return n;
}

/* --ask: 全链 = 插件总线 kb.search -> 卡片注入上下文 -> 模型生成
 * aimp_probe <weights.aimw> <tokenizer.gguf> --ask "query" n_next */
static int run_ask(const char *weights, const char *tokgguf,
                   const char *query, uint32_t n_next) {
    if (aim_bus_init(NULL)) { fprintf(stderr, "bus init FAIL (plugins/caps.json)\n"); return 3; }
    char esc[2048], q[4096], resp[262144];
    aim_json_escape(query, esc, sizeof esc);
    sprintf(q, "{\"query\":\"%s\",\"topk\":3}", esc);
    double secs;
    int rc = aim_bus_call("kb.search", q, resp, sizeof resp, &secs);
    if (rc) { fprintf(stderr, "kb.search FAIL rc=%d\n", rc); return rc; }
    printf("BUS kb.search %.2fs\n", secs);

    /* 抽 top3 卡片 (title + text 截 150 字, 防 prefill 爆炸) */
    char ctx[4096];
    size_t cl = 0;
    int hits = 0;
    const char *p = resp;
    while (hits < 3 && (p = strstr(p, "\"id\": \"kb-")) != NULL) {
        const char *end = strstr(p, "\"origin\"");
        const char *tq = strstr(p, "\"title\": \"");
        const char *xq = strstr(p, "\"text\": \"");
        const char *sq = strstr(p, "\"score\": ");
        if (!end || !tq || !xq || tq >= end) break;
        char title[256], text[512];
        json_str_after(tq + 10, title, sizeof title);
        json_str_after(xq + 9, text, sizeof text);
        double sc = (sq && sq < end) ? strtod(sq + 9, NULL) : 0.0;
        if (strlen(text) > 150) text[150] = 0;
        int w = snprintf(ctx + cl, sizeof ctx - cl,
                         "[%d] %.1f %s\n%s\n", hits + 1, sc, title, text);
        if (w < 0 || cl + (size_t)w >= sizeof ctx) break;
        cl += (size_t)w;
        hits++;
        p = end;
    }
    printf("KB_HITS:%d\n", hits);

    /* 诚实边界 (确定性拦截, 不指望模型自觉): 检索零命中就不进模型,
     * 如实回答没有——绝不编造。只有真命中才允许生成。 */
    if (hits == 0) {
        printf("ANSWER:[未检索到] 知识库中没有与「%s」相关的内容，如实告知：不知道。\n",
               query);
        printf("HONEST_MISS:1\n");
        return 0;
    }

    char prompt[8192];
    snprintf(prompt, sizeof prompt,
             "User: 参考资料:\n%s问题: %s\n<|eot_id|>Assistant: ", ctx, query);
    return run_chat_prompt(weights, tokgguf, prompt, n_next);
}

/* --cap: 直调插件总线 (不加载模型) — aimp_probe x --cap NAME "query" [topk] */
static int run_cap(const char *cap, const char *query, uint32_t topk) {
    if (aim_bus_init(NULL)) { fprintf(stderr, "bus init FAIL (plugins/caps.json)\n"); return 3; }
    char esc[8192], q[16384], resp[262144];
    aim_json_escape(query, esc, sizeof esc);
    sprintf(q, "{\"query\":\"%s\",\"topk\":%u}", esc, topk);
    double secs;
    int rc = aim_bus_call(cap, q, resp, sizeof resp, &secs);
    if (rc) { fprintf(stderr, "bus call FAIL rc=%d cap=%s\n", rc, cap); return rc; }
    printf("CAP:%s  BUS %.2fs\n", cap, secs);
    printf("%s\n", resp);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: aimp_probe <model> [--all | --gen ids n | --enc text | --dec ids | --cap NAME QUERY [TOPK] | <w> <tok> --gt text n | <w> <tok> --ask QUERY n]\n"); return 10; }
#ifdef _WIN32
    SetConsoleOutputCP(65001);   /* 控制台直显 UTF-8 结果 */
#endif
    if (argc > 3 && !strcmp(argv[2], "--cap") && argc > 4)
        return run_cap(argv[3], argv[4], argc > 5 ? (uint32_t)atoi(argv[5]) : 5);
    if (argc > 3 && !strcmp(argv[3], "--ask") && argc >= 6)
        return run_ask(argv[1], argv[2], argv[4], (uint32_t)atoi(argv[5]));
    if (argc > 3 && !strcmp(argv[3], "--gt") && argc >= 6)
        return run_gt(argv[1], argv[2], argv[4], (uint32_t)atoi(argv[5]));
    if (argc > 2 && (!strcmp(argv[2], "--enc") || !strcmp(argv[2], "--dec")) && argc > 3)
        return run_tok(argv[1], !strcmp(argv[2], "--enc"), argv[3]);
    if (argc > 2 && !strcmp(argv[2], "--gen") && argc > 4)
        return run_gen(argv[1], argv[3], (uint32_t)atoi(argv[4]),
                       argc > 5 ? (uint32_t)atoi(argv[5]) : 0);
    aim_gguf g;
    int rc = aim_gguf_open(argv[1], &g);
    if (rc) { fprintf(stderr, "FAIL rc=%d (1=IO 2=FORMAT 3=MEM)\n", rc); return rc; }

    printf("=== aimp M1 GGUF probe ===\n");
    printf("version=%u align=%u kv=%llu tensors=%llu header_end=%llu data_start=%llu\n",
           g.version, g.align, (unsigned long long)g.n_kv,
           (unsigned long long)g.n_tensors,
           (unsigned long long)g.header_end, (unsigned long long)g.data_start);
    print_arch(&g);
    print_vocab(&g);

    struct { uint32_t qt; uint64_t n, bytes; } agg[64];
    size_t nagg = 0;
    uint64_t total = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        aim_tensor *t = &g.tensors[i];
        total += t->nbytes;
        size_t k;
        for (k = 0; k < nagg; k++) if (agg[k].qt == t->qtype) break;
        if (k == nagg) { agg[k].qt = t->qtype; agg[k].n = 0; agg[k].bytes = 0; nagg++; }
        agg[k].n++; agg[k].bytes += t->nbytes;
    }
    printf("[weights] by type:\n");
    for (size_t k = 0; k < nagg; k++)
        printf("  %-8s x%-5llu %10.2f MB\n", aim_gguf_qt_name(agg[k].qt),
               (unsigned long long)agg[k].n, agg[k].bytes / 1048576.0);
    printf("  total %.2f GB\n", total / 1073741824.0);

    int all = (argc > 2 && !strcmp(argv[2], "--all"));
    uint64_t show = all ? g.n_tensors : (g.n_tensors < 12 ? g.n_tensors : 12);
    printf("[tensors] %s:\n", all ? "ALL" : "first 12");
    for (uint64_t i = 0; i < show; i++) {
        aim_tensor *t = &g.tensors[i];
        printf("  %-52s %-6s [", t->name, aim_gguf_qt_name(t->qtype));
        for (uint32_t d = 0; d < t->n_dims; d++)
            printf("%llu%s", (unsigned long long)t->dims[d],
                   d + 1 < t->n_dims ? "," : "");
        printf("] %9.1f MB @+%llu\n", t->nbytes / 1048576.0,
               (unsigned long long)t->offset);
    }
    aim_gguf_free(&g);
    return 0;
}

/* --gen 模式: aimp_probe model.gguf --gen tok,tok,... n_next [spec_k]
 * spec_k > 1: 投机解码 (prompt-lookup 起草 + 批量验证) */
static int run_gen(const char *path, const char *ids_csv, uint32_t n_next,
                   uint32_t spec_k) {
    aim_model m;
    int rc = aim_model_open(path, &m);
    if (rc) { fprintf(stderr, "model open FAIL rc=%d\n", rc); return rc; }
    uint32_t ids[512], n = 0;
    const char *p = ids_csv;
    /* 容错 CSV: 数字串收 token, 其余字符 (空格/逗号/换行) 一律跳过 */
    while (*p && n < 512) {
        if (*p >= '0' && *p <= '9') {
            uint32_t v = 0;
            while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
            ids[n++] = v;
        } else {
            p++;
        }
    }
    uint32_t *out = (uint32_t *)malloc(n_next * 4);
    clock_t g0 = clock();
    if (spec_k > 1)
        rc = aim_generate_spec(&m, ids, n, n_next, spec_k, out);
    else
        rc = aim_generate(&m, ids, n, n_next, out);
    clock_t g1 = clock();
    if (rc) return rc;
    double gdt = (double)(g1 - g0) / CLOCKS_PER_SEC;
    printf("GEN_IDS:");
    for (uint32_t i = 0; i < n_next; i++)
        printf(" %u", out[i]);
    printf("\n");
    printf("TIME %.2fs  %.2f tok/s  (gen %u tok, load excluded)\n",
           gdt, n_next / gdt, n_next);
    free(out);
    aim_model_close(&m);
    return 0;
}
static int run_tok(const char *path, int enc, const char *text) {
    aim_gguf g;
    int rc = aim_gguf_open(path, &g);
    if (rc) { fprintf(stderr, "open FAIL rc=%d\n", rc); return rc; }
    aim_tok t;
    rc = aim_tok_init(&g, &t);
    if (rc) { fprintf(stderr, "tok init FAIL rc=%d\n", rc); return rc; }
    if (enc) {
        uint32_t *ids, n;
        if (aim_tok_encode(&t, text, &ids, &n)) { fprintf(stderr, "enc FAIL\n"); return 1; }
        printf("ENC_IDS:");
        for (uint32_t i = 0; i < n; i++) printf(" %u", ids[i]);
        printf("\n");
        free(ids);
    } else {
        uint32_t ids[1024], n = 0;
        const char *p = text;
        /* 容错 CSV: 同 run_gen, 非数字字符跳过 (空格分隔/逗号分隔都收) */
        while (*p && n < 1024) {
            if (*p >= '0' && *p <= '9') {
                uint32_t v = 0;
                while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
                ids[n++] = v;
            } else {
                p++;
            }
        }
        char *txt; uint32_t len;
        if (aim_tok_decode(&t, ids, n, &txt, &len)) { fprintf(stderr, "dec FAIL\n"); return 1; }
        printf("DEC_TEXT:%s\n", txt);
        free(txt);
    }
    aim_tok_free(&t);
    aim_gguf_free(&g);
    return 0;
}
