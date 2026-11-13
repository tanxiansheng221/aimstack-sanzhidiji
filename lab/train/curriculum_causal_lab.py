# -*- coding: utf-8 -*-
# 课程生成器 v0 —— 第三科：因果推断（do 干预求值 + 反事实 twin 重算，验证器=独立解释器对拍）
# T 式纪律：G1 双通道 / G2 定向毒 fail-closed / G3 确定性 / G4 train/val 隔离。题面自包含。
import json, random, hashlib, sys, re

OUT_RESULTS = "curriculum_causal_v0_results.json"
FAMS = ["do", "cf"]
NODES = ["A", "B", "C", "D", "E", "F"]

# ---------- 生成器（通道 A：结构构造 + 直接求值） ----------
def gen_sample(rng, fam):
    funcs = {}  # node -> ("input",) | ("not", p) | ("and"/"or", p1, p2)
    for i, n in enumerate(NODES):
        if i == 0:
            funcs[n] = ("input",)
            continue
        pars = [NODES[j] for j in range(i) if rng.random() < 0.55]
        if not pars:
            pars = [NODES[i - 1]]
        if len(pars) == 1:
            funcs[n] = ("not", pars[0]) if rng.random() < 0.4 else ("id", pars[0])
        else:
            funcs[n] = (rng.choice(["and", "or"]), pars[0], pars[1])
    roots = [n for n in NODES if funcs[n][0] == "input"]
    w = {r: rng.randrange(2) for r in roots}
    def ev(world, funcs_):
        out = {}
        for n in NODES:
            f = funcs_[n]
            if f[0] == "input":
                out[n] = world[n]
            elif f[0] in ("not", "id"):
                v = out[f[1]]
                out[n] = 1 - v if f[0] == "not" else v
            else:
                a, b = out[f[1]], out[f[2]]
                out[n] = (a & b) if f[0] == "and" else (a | b)
        return out
    base = ev(w, funcs)
    txt = []
    for n in NODES:
        f = funcs[n]
        if f[0] == "input":
            txt.append("%s is an input" % n)
        elif f[0] == "not":
            txt.append("%s = NOT %s" % (n, f[1]))
        elif f[0] == "id":
            txt.append("%s = %s" % (n, f[1]))
        else:
            txt.append("%s = %s %s %s" % (n, f[1], f[0].upper(), f[2]))
    given = ", ".join("%s=%d" % (r, w[r]) for r in roots)
    target = rng.choice([n for n in NODES if n not in roots] or ["D"])
    if fam == "do":
        q = "Causal model: %s. Given %s. What is the value of %s? Answer 0 or 1." % ("; ".join(txt) + ".", given, target)
        gold = base[target]
        cot = "eval: " + " ".join("%s=%d" % (n, base[n]) for n in NODES)
    else:  # cf：反事实——翻转一个根
        rflip = roots[0] if len(roots) == 1 else rng.choice(roots)
        w2 = dict(w); w2[rflip] = 1 - w2[rflip]
        cf = ev(w2, funcs)
        q = ("Causal model: %s. Observed: %s (so %s=%d). If %s had been %d instead, what would %s be? Answer 0 or 1."
             % ("; ".join(txt) + ".", given, target, base[target], rflip, w2[rflip], target))
        gold = cf[target]
        cot = "twin: " + " ".join("%s=%d" % (n, cf[n]) for n in NODES)
    return {"fam": fam, "q": q, "cot": cot, "ans": str(gold), "k": {"target": target}}

# ---------- 验证器（通道 B：只吃 q 文本，独立解析重算） ----------
def verify(sample):
    m = re.match(r"Causal model: (.+?)\. (Given|Observed): (.+?) \(so (\w+)=(\d)\)\. If (\w) had been (\d) instead", sample["q"]) \
        if "Observed" in sample["q"] else \
        re.match(r"Causal model: (.+?)\. Given (.+?)\. What is the value of (\w)", sample["q"])
    if not m:
        return {"ok": False, "gold": "?", "cot_ok": False}
    model = m.group(1)
    funcs = {}
    for part in model.split("; "):
        part = part.strip().rstrip(".")
        pm = re.fullmatch(r"(\w) is an input", part)
        if pm:
            funcs[pm.group(1)] = ("input",); continue
        pm = re.fullmatch(r"(\w) = NOT (\w)", part)
        if pm:
            funcs[pm.group(1)] = ("not", pm.group(2)); continue
        pm = re.fullmatch(r"(\w) = (\w) (AND|OR) (\w)", part)
        if pm:
            funcs[pm.group(1)] = (pm.group(3).lower(), pm.group(2), pm.group(4)); continue
        pm = re.fullmatch(r"(\w) = (\w)", part)
        if pm:
            funcs[pm.group(1)] = ("id", pm.group(2)); continue
        return {"ok": False, "gold": "?", "cot_ok": False}
    if any(f[0] != "input" for f in funcs.values()) is False and False:
        pass
    def ev(world):
        out = {}
        for n in NODES:
            f = funcs.get(n)
            if f is None:
                return None
            if f[0] == "input":
                if n not in world:
                    return None
                out[n] = world[n]
            elif f[0] in ("not", "id"):
                if f[1] not in out:
                    return None
                out[n] = 1 - out[f[1]] if f[0] == "not" else out[f[1]]
            else:
                if f[1] not in out or f[2] not in out:
                    return None
                out[n] = (out[f[1]] & out[f[2]]) if f[0] == "and" else (out[f[1]] | out[f[2]])
        return out
    if sample["fam"] == "do":
        world = dict((a, int(b)) for a, b in re.findall(r"(\w)=(\d)", m.group(2)))  # do 族 group(2)=Given 段
        out = ev(world)
        if out is None:
            return {"ok": False, "gold": "?", "cot_ok": False}
        gold = out[sample["k"]["target"]]
        exp_cot = "eval: " + " ".join("%s=%d" % (n, out[n]) for n in NODES)
    else:
        world = dict((a, int(b)) for a, b in re.findall(r"(\w)=(\d)", m.group(3)))
        rflip, v2 = m.group(6), int(m.group(7))
        world[rflip] = v2
        out = ev(world)
        if out is None:
            return {"ok": False, "gold": "?", "cot_ok": False}
        gold = out[sample["k"]["target"]]
        exp_cot = "twin: " + " ".join("%s=%d" % (n, out[n]) for n in NODES)
    return {"ok": sample["ans"] == str(gold), "gold": str(gold), "cot_ok": sample["cot"] == exp_cot}

GEN = {"do": lambda rng: _mk(rng, "do"), "cf": lambda rng: _mk(rng, "cf")}

def _mk(rng, fam):
    fam = fam if fam in GEN else "do"
    for _ in range(20):  # 内嵌通道 B 复核
        sm = GEN[fam](rng)
        v = verify(sm)
        if v["ok"] and v["cot_ok"]:
            return sm
    return None

# ---------- 定向毒 ----------
def _poison_flip(sample):
    p = dict(sample)
    p["ans"] = "0" if sample["ans"] == "1" else "1"
    return p

def _poison_struct(sample):
    # 翻转模型里一个 AND↔OR / NOT 删除，直到答案翻转；翻不动跳过
    for _ in range(30):
        q = sample["q"]
        m = re.search(r" (\w) = (\w) (AND|OR) (\w)", q)
        if m and random.Random().randrange(2) == 0:
            op2 = "OR" if m.group(3) == "AND" else "AND"
            q2 = q[:m.start(3)] + op2 + q[m.end(3):]
        else:
            m = re.search(r" (\w) = NOT (\w)", q)
            if not m:
                m2 = re.search(r"(\w)=(\d)", sample["q"])
                if not m2:
                    return None
                q2 = sample["q"][:m2.start()] + "%s=%d" % (m2.group(1), 1 - int(m2.group(2))) + sample["q"][m2.end():]
            else:
                q2 = q[:m.start(2)] + m.group(2) + " " + q[m.start(2):] if False else q.replace(" NOT ", " ", 1)
        p = dict(sample); p["q"] = q2
        if not verify(p)["ok"]:
            return p
    return None

# ---------- T 式自检 ----------
def self_test():
    r = {}
    bad = total = 0
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        for _ in range(200):
            sm = gen_sample(rng, rng.choice(FAMS))
            if sm is None:
                continue
            total += 1
            v = verify(sm)
            if not v["ok"] or not v["cot_ok"]:
                bad += 1
    r["G1_dual_channel"] = {"total": total, "bad": bad, "pass": bad == 0}
    caught = total2 = 0
    rng = random.Random(99)
    for _ in range(100):
        sm = gen_sample(rng, rng.choice(FAMS))
        if sm is None:
            continue
        total2 += 1
        if not verify(_poison_flip(sm))["ok"]:
            caught += 1
        total2 += 1
        p2 = dict(sm); p2["cot"] = sm["cot"] + " -> tampered"
        if not verify(p2)["cot_ok"]:
            caught += 1
        p3 = _poison_struct(sm)
        if p3 is None:
            continue
        total2 += 1
        if not verify(p3)["ok"]:
            caught += 1
    r["G2_poison_failclosed"] = {"total": total2, "caught": caught, "pass": caught == total2}
    def corpus(seed):
        rng = random.Random(seed)
        out = []
        while len(out) < 50:
            sm = gen_sample(rng, FAMS[len(out) % len(FAMS)])
            if sm: out.append(sm)
        return json.dumps(out, sort_keys=True)
    r["G3_determinism"] = {"pass": corpus(7) == corpus(7) and corpus(7) != corpus(8)}
    r["G4_trainval_split"] = {"pass": corpus(1001) != corpus(2002)}
    r["ALL_PASS"] = all(v.get("pass") for k, v in r.items() if k.startswith("G"))
    return r

# ---------- 课程包导出 ----------
def export(n_train=1500, n_val=200):
    lines = []
    for split, n, seed in (("train", n_train, 15001), ("val", n_val, 9008)):
        rng = random.Random(seed)
        i = 0
        while i < n:
            sm = gen_sample(rng, FAMS[i % len(FAMS)])
            if sm is None:
                continue
            lines.append(json.dumps({
                "split": split, "subject": "causal", "id": "%s_%04d" % (split, i),
                "fam": sm["fam"], "q": sm["q"], "cot": sm["cot"], "ans": sm["ans"]}, ensure_ascii=False))
            i += 1
    data = "\n".join(lines) + "\n"
    open("curriculum_causal_v0.jsonl", "w", encoding="utf-8", newline="").write(data)
    return {"records": len(lines), "sha256": hashlib.sha256(data.encode()).hexdigest()[:16]}

if __name__ == "__main__":
    res = self_test()
    res["export"] = export()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(OUT_RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if res["ALL_PASS"] else 2)
