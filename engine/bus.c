/* aimp plugin bus — cap -> 外部插件进程
 * 协议: 查询 JSON 经临时文件 (路径走插件 argv[1]), 结果 JSON 走插件 stdout
 * 总线本体零依赖 (_popen + 临时文件), 插件自管语言/运行时
 * 纪律对齐: 内核纯 C, KB/搜索属外置数据支撑, 以插件形态挂总线 */
#include "bus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <process.h>

#define AIM_BUS_MAXCAPS 32
typedef struct { char name[64]; char cmd[512]; } aim_caprec;
static aim_caprec g_caps[AIM_BUS_MAXCAPS];
static int g_ncaps = 0;
static long g_callno = 0;

static char *bus_read_all(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    long n;
    if (fseek(f, 0, SEEK_END) || (n = ftell(f)) < 0 || n > (1 << 20)) {
        fclose(f); return NULL;
    }
    rewind(f);
    char *b = (char *)malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    b[n] = 0;
    fclose(f);
    return b;
}

/* caps.json 微扫描: "cap" : "cmd" 键值对 (值内不支持转义引号, 登记表自守约) */
static int parse_caps(const char *txt) {
    const char *p = txt;
    while (g_ncaps < AIM_BUS_MAXCAPS && (p = strchr(p, '"')) != NULL) {
        const char *k0 = p + 1;
        const char *k1 = strchr(k0, '"');
        if (!k1) break;
        const char *v = k1 + 1;
        while (*v == ' ' || *v == ':' || *v == '\t') v++;
        if (*v != '"') { p = k1 + 1; continue; }
        v++;
        const char *v1 = strchr(v, '"');
        if (!v1) break;
        size_t kn = (size_t)(k1 - k0), vn = (size_t)(v1 - v);
        if (kn && kn < sizeof(g_caps[0].name) && vn && vn < sizeof(g_caps[0].cmd)) {
            memcpy(g_caps[g_ncaps].name, k0, kn);
            g_caps[g_ncaps].name[kn] = 0;
            memcpy(g_caps[g_ncaps].cmd, v, vn);
            g_caps[g_ncaps].cmd[vn] = 0;
            g_ncaps++;
        }
        p = v1 + 1;
    }
    return g_ncaps > 0 ? 0 : 2;
}

int aim_bus_init(const char *caps_path) {
    char def[512];
    if (!caps_path) {
        const char *env = getenv("AIM_BUS_CAPS");
        if (env && *env) caps_path = env;
        else { sprintf(def, "plugins/caps.json"); caps_path = def; }
    }
    char *txt = bus_read_all(caps_path);
    if (!txt) return 1;
    int rc = parse_caps(txt);
    free(txt);
    return rc;
}

int aim_bus_ncaps(void) { return g_ncaps; }

const char *aim_bus_capname(int i) {
    return (i >= 0 && i < g_ncaps) ? g_caps[i].name : NULL;
}

void aim_json_escape(const char *in, char *out, size_t cap) {
    size_t n = 0;
    for (; *in && n + 6 < cap; in++) {
        unsigned char c = (unsigned char)*in;
        if (c == '"' || c == '\\') { out[n++] = '\\'; out[n++] = (char)c; }
        else if (c == '\n') { out[n++] = '\\'; out[n++] = 'n'; }
        else if (c == '\r' || c == '\t') { out[n++] = '\\'; out[n++] = 't'; }
        else out[n++] = (char)c;
    }
    out[n] = 0;
}

int aim_bus_call(const char *cap, const char *query_json, char *resp,
                 size_t resp_cap, double *secs) {
    int idx = -1;
    for (int i = 0; i < g_ncaps; i++)
        if (!strcmp(g_caps[i].name, cap)) { idx = i; break; }
    if (idx < 0) return 4;

    /* query -> 临时文件 (UTF-8 内容, ANSI 路径) */
    char qpath[1100];
    const char *td = getenv("TEMP");
    if (!td || !*td) td = ".";
    sprintf(qpath, "%s\\aim_bus_q_%ld_%ld.json", td, (long)_getpid(), ++g_callno);
    FILE *f = fopen(qpath, "wb");
    if (!f) return 1;
    fwrite(query_json, 1, strlen(query_json), f);
    fclose(f);

    /* cmd 模板 %s -> qpath */
    const char *s = strstr(g_caps[idx].cmd, "%s");
    if (!s) { remove(qpath); return 2; }
    char cmd[1300];
    snprintf(cmd, sizeof cmd, "%.*s%s%s", (int)(s - g_caps[idx].cmd),
             g_caps[idx].cmd, qpath, s + 2);

    clock_t c0 = clock();
    FILE *pf = _popen(cmd, "rb");
    if (!pf) { remove(qpath); return 3; }
    size_t rn = 0;
    while (rn + 1 < resp_cap) {
        size_t got = fread(resp + rn, 1, resp_cap - 1 - rn, pf);
        if (got == 0) break;
        rn += got;
    }
    resp[rn] = 0;
    _pclose(pf);
    clock_t c1 = clock();
    remove(qpath);
    if (secs) *secs = (double)(c1 - c0) / CLOCKS_PER_SEC;
    if (rn == 0) return 5;   /* 插件无输出 (崩溃/异常) */
    return 0;
}
