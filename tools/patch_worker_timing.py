# -*- coding: utf-8 -*-
# patch_worker_timing.py: worker fetch loop 加 QPC 计时
import io
P = r"H:\AimCode\aimp\src\forward.c"
s = io.open(P, encoding="utf-8").read()

old = """        if (v != my) {
            if (g_dbg) fprintf(stderr, "[w%ld] start gen=%ld\\n", id, v);
            my = v;
            LONG got = 0;
            for (;;) {
                LONG i = InterlockedExchangeAdd(&g_next, g_chunk);
                if (i >= g_total) break;
                LONG e = i + g_chunk;
                if (e > g_total) e = g_total;
                got += e - i;
                for (LONG u = i; u < e; u++) g_fn(g_cx, (uint32_t)u);
            }
            if (g_dbg) fprintf(stderr, "[w%ld] done gen=%ld got=%ld\\n", id, v, got);"""
new = """        if (v != my) {
            if (g_dbg) fprintf(stderr, "[w%ld] start gen=%ld\\n", id, v);
            my = v;
            LONG got = 0;
            LARGE_INTEGER t0, t1, fq;
            QueryPerformanceCounter(&t0);
            for (;;) {
                LONG i = InterlockedExchangeAdd(&g_next, g_chunk);
                if (i >= g_total) break;
                LONG e = i + g_chunk;
                if (e > g_total) e = g_total;
                got += e - i;
                for (LONG u = i; u < e; u++) g_fn(g_cx, (uint32_t)u);
            }
            QueryPerformanceCounter(&t1);
            QueryPerformanceFrequency(&fq);
            if (g_dbg) fprintf(stderr, "[w%ld] done gen=%ld got=%ld %.1fms\\n", id, v, got,
                               (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / (double)fq.QuadPart);"""
assert old in s, "old not found"
s = s.replace(old, new)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("PATCH_OK")
