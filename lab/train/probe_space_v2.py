# -*- coding: utf-8 -*-
# probe_space_v2.py — 用各科 export() 生成 2 万条测唯一率（判定纯拉 n vs 扩参数域）
import io, sys, json, importlib, hashlib, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LABS = ["curriculum_math_lab", "curriculum_plan_lab", "curriculum_causal_lab",
        "curriculum_csp_lab", "curriculum_progexec_lab", "curriculum_seqind_lab",
        "curriculum_logic_lab", "curriculum_fsm_lab"]
N = 20000
report = {}
for name in LABS:
    mod = importlib.import_module(name)
    fn = mod.export
    import inspect
    kw = {}
    if "n_train" in inspect.signature(fn).parameters:
        kw = {"n_train": N, "n_val": 500}
    try:
        info = fn(**kw)
    except TypeError:
        info = fn(N)
    # 统计唯一率
    import os
    fname = [f for f in os.listdir(".") if f.startswith(name.replace("_lab", "")) and f.endswith(".jsonl")]
    path = sorted(fname)[-1] if fname else None
    qs = set()
    total = 0
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "train":
                total += 1
                qs.add(r["q"])
    ur = len(qs) / max(total, 1)
    report[name] = {"file": path, "train": total, "unique": len(qs),
                    "uniq_rate": round(ur, 4),
                    "strategy": "PULL_N" if ur >= 0.90 else "NEED_WIDEN"}
    print(name, report[name], flush=True)

json.dump(report, open("probe_space_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("=== DONE ===")
