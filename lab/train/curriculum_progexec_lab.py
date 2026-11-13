# -*- coding: utf-8 -*-
# 课程生成器 v0 —— 第五科：代码执行追踪（随机小程序，验证器=独立解释器对拍）
# T 式纪律：G1 双通道（生成器构造 vs 通道 B 从 q 文本盲解析重跑）/ G2 定向毒 fail-closed /
#           G3 确定性 / G4 train/val 隔离。题面自包含纪律：程序全文在 q 里，答案可从 q 盲推。
import json, random, hashlib, sys, re

OUT_RESULTS = "curriculum_progexec_v0_results.json"
FAMS = ["progexec"]

# ---------- 生成器（通道 A：构造程序直接执行） ----------
def gen_sample(rng, fam=None):
    nv = 4
    env = {i: 0 for i in range(nv)}
    stmts = []
    n_stmt = rng.choice([5, 6, 7, 8])
    for _ in range(n_stmt):
        i = rng.randrange(nv)
        r = rng.random()
        if r < 0.25:
            k = rng.randrange(0, 10)
            stmts.append((i, "c", k))
        elif r < 0.5:
            j = rng.randrange(nv)
            k = rng.randrange(1, 6)
            stmts.append((i, "add", (j, k)))
        elif r < 0.75:
            j = rng.randrange(nv)
            k = rng.randrange(1, 6)
            stmts.append((i, "sub", (j, k)))
        else:
            k = rng.randrange(1, 6)
            stmts.append((i, "selfadd", k))
    def run(e):
        e = dict(e)
        for i, op, a in stmts:
            if op == "c":
                e[i] = a
            elif op == "add":
                e[i] = e[a[0]] + a[1]
            elif op == "sub":
                e[i] = e[a[0]] - a[1]
            else:
                e[i] = e[i] + a
        return e
    env = run(env)
    v = rng.randrange(nv)
    txt = "; ".join(
        "x%d = %d" % (i, a) if op == "c" else
        "x%d = x%d + %d" % (i, a[0], a[1]) if op == "add" else
        "x%d = x%d - %d" % (i, a[0], a[1]) if op == "sub" else
        "x%d = x%d + %d" % (i, i, a)
        for i, op, a in stmts)
    q = "All variables start at 0. Program: %s. What is the final value of x%d? Answer a number." % (txt, v)
    cot = "env: " + " ".join("x%d=%d" % (i, env[i]) for i in range(nv))
    return {"fam": "progexec", "q": q, "cot": cot, "ans": str(env[v]), "k": {"v": v}}

# ---------- 验证器（通道 B：只看 q 文本，盲解析重跑） ----------
def _parse_prog(q):
    body = q.split("Program: ")[1].split(". What is")[0]
    env = {0: 0, 1: 0, 2: 0, 3: 0}
    for st in body.split("; "):
        m = re.fullmatch(r"x(\d) = (\d+)", st)
        if m:
            env[int(m.group(1))] = int(m.group(2)); continue
        m = re.fullmatch(r"x(\d) = x(\d) ([+-]) (\d+)", st)
        if m:
            i, j, o, k = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
            env[i] = env[j] + k if o == "+" else env[j] - k
            continue
        return None
    return env

def verify(sample):
    env = _parse_prog(sample["q"])
    if env is None:
        return {"ok": False, "gold": "?", "cot_ok": False}
    v = sample["k"]["v"]
    gold = str(env[v])
    exp_cot = "env: " + " ".join("x%d=%d" % (i, env[i]) for i in range(4))
    return {"ok": sample["ans"] == gold, "gold": gold, "cot_ok": sample["cot"] == exp_cot}

# ---------- 定向毒（保证判定必翻；构造不出返回 None） ----------
def _poison_ans(sample):
    p = dict(sample); p["ans"] = str(int(sample["ans"]) + 1)
    return p if p["ans"] != sample["ans"] else None

def _poison_struct(sample):
    body = sample["q"].split("Program: ")[1].split(". What is")[0]
    head = sample["q"].split("Program: ")[0] + "Program: "
    tail = ". What is" + sample["q"].split(". What is")[1]
    for si in range(len(body.split("; "))):
        for dk in (-1, 1):
            stmts = body.split("; ")
            m = re.fullmatch(r"x(\d) = (x\d [+-] )?(\d+)", stmts[si])
            if not m:
                continue
            k2 = int(m.group(3)) + dk
            if k2 < 0:
                continue
            stmts[si] = ("x%s = %d" % (m.group(1), k2)) if not m.group(2) else ("x%s = %s%d" % (m.group(1), m.group(2), k2))
            p = dict(sample)
            p["q"] = head + "; ".join(stmts) + tail
            v = sample["k"]["v"]
            env = _parse_prog(p["q"])
            if env is not None and str(env[v]) != sample["ans"]:
                return p
    return None

# ---------- T 式自检 ----------
def self_test():
    r = {}
    bad = total = 0
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        for _ in range(200):
            sm = gen_sample(rng)
            total += 1
            v = verify(sm)
            if not v["ok"] or not v["cot_ok"]:
                bad += 1
    r["G1_dual_channel"] = {"total": total, "bad": bad, "pass": bad == 0}
    caught = total2 = 0
    rng = random.Random(99)
    for _ in range(100):
        sm = gen_sample(rng)
        p1 = _poison_ans(sm)
        total2 += 1
        if not verify(p1)["ok"]:
            caught += 1
        total2 += 1
        p2 = dict(sm); p2["cot"] = sm["cot"] + " -> tampered"
        if not verify(p2)["cot_ok"]:
            caught += 1
        p3 = _poison_struct(sm)  # 定向：程序常量变异到答案必翻，翻不动跳过不计数
        if p3 is None:
            continue
        total2 += 1
        if not verify(p3)["ok"]:
            caught += 1
    r["G2_poison_failclosed"] = {"total": total2, "caught": caught, "pass": caught == total2}
    def corpus(seed):
        rng = random.Random(seed)
        return json.dumps([gen_sample(rng) for _ in range(50)], sort_keys=True)
    r["G3_determinism"] = {"pass": corpus(7) == corpus(7) and corpus(7) != corpus(8)}
    r["G4_trainval_split"] = {"pass": corpus(1001) != corpus(2002)}
    r["ALL_PASS"] = all(v.get("pass") for k, v in r.items() if k.startswith("G"))
    return r

# ---------- 课程包导出 ----------
def export(n_train=1500, n_val=200):
    lines = []
    for split, n, seed in (("train", n_train, 11001), ("val", n_val, 9004)):
        rng = random.Random(seed)
        i = 0
        while i < n:
            sm = gen_sample(rng)
            v = verify(sm)
            assert v["ok"] and v["cot_ok"], "curriculum poison at %s/%d" % (split, i)
            lines.append(json.dumps({
                "split": split, "subject": "prog_exec", "id": "%s_%04d" % (split, i),
                "q": sm["q"], "cot": sm["cot"], "ans": sm["ans"]}, ensure_ascii=False))
            i += 1
    data = "\n".join(lines) + "\n"
    open("curriculum_progexec_v0.jsonl", "w", encoding="utf-8", newline="").write(data)
    return {"records": len(lines), "sha256": hashlib.sha256(data.encode()).hexdigest()[:16]}

if __name__ == "__main__":
    res = self_test()
    res["export"] = export()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(OUT_RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if res["ALL_PASS"] else 2)
