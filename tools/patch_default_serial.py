# -*- coding: utf-8 -*-
# patch_default_serial.py: 线程池默认关闭, AIM_WORKERS 显式开启
import io
P = r"H:\AimCode\aimp\src\forward.c"
s = io.open(P, encoding="utf-8").read()

old = """static void aim_pool_init(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    LONG nw = (LONG)si.dwNumberOfProcessors - 1;
    char eb[16];
    if (GetEnvironmentVariableA("AIM_WORKERS", eb, sizeof(eb)) > 0)
        nw = atol(eb);
    if (GetEnvironmentVariableA("AIM_PAR_DBG", eb, sizeof(eb)) > 0)
        g_dbg = atol(eb);
    if (nw > AIM_MAX_WORKERS) nw = AIM_MAX_WORKERS;"""
new = """static void aim_pool_init(void) {
    /* 默认串行: 弱核/带宽受限机器上多线程负收益 (实测 lm_head 并行慢 8 倍)。
     * AIM_WORKERS=n 显式开启; 自校准护栏会把不快的并行自动降级。 */
    LONG nw = 0;
    char eb[16];
    if (GetEnvironmentVariableA("AIM_WORKERS", eb, sizeof(eb)) > 0)
        nw = atol(eb);
    if (GetEnvironmentVariableA("AIM_PAR_DBG", eb, sizeof(eb)) > 0)
        g_dbg = atol(eb);
    if (nw > AIM_MAX_WORKERS) nw = AIM_MAX_WORKERS;"""
assert old in s, "old not found"
s = s.replace(old, new)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("PATCH_OK")
