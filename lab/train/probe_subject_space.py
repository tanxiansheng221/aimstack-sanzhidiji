# -*- coding: utf-8 -*-
# probe_subject_space.py — 十科采样空间容量探针：每科生成 N 次统计唯一 q 数，判定扩产策略
# 判据：唯一率 <90% 的科必须先扩参数域，纯拉 n 会记忆化
import io, sys, importlib, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LABS = ["curriculum_fsm_lab", "curriculum_math_lab", "curriculum_plan_lab",
        "curriculum_causal_lab", "curriculum_csp_lab", "curriculum_progexec_lab",
        "curriculum_seqind_lab", "curriculum_logic_lab"]

N = 30000
report = {}
for name in LABS:
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        report[name] = {"error": str(e)[:100]}
        continue
    fams = getattr(mod, "FAMS", None)
    gen_sample = getattr(mod, "gen_sample", None)
    if not gen_sample:
        report[name] = {"error": "no gen_sample"}
        continue
    import random
    rng = random.Random(20261030)
    qs = set()
    ok = 0
    for i in range(N):
        fam = fams[i % len(fams)] if fams else None
        try:
            sm = gen_sample(rng, fam)
        except TypeError:
            try:
                sm = gen_sample(rng, fam, 64)
            except TypeError:
                sm = gen_sample(rng)
        if sm:
            ok += 1
            qs.add(sm["q"])
    uniq_rate = len(qs) / max(ok, 1)
    report[name] = {"gen_ok": ok, "unique_q": len(qs), "uniq_rate": round(uniq_rate, 4),
                    "strategy": "PULL_N" if uniq_rate >= 0.90 else "NEED_WIDEN"}
    print(name, report[name], flush=True)

print(json.dumps(report, ensure_ascii=False, indent=1))
json.dump(report, open("probe_space_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
