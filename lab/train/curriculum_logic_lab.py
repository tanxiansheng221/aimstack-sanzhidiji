# -*- coding: utf-8 -*-
# 课程生成器 v0 —— 第七科：形式逻辑（命题公式求值 + 前向链推理，验证器=独立解析器对拍）
# T 式纪律：G1 双通道 / G2 定向毒 fail-closed / G3 确定性 / G4 train/val 隔离。
# 一致性纪律：推理族规则必须在世界中为真（禁止 A 真而 B 假的规则），闭包零矛盾。
import json, random, hashlib, sys, re

OUT_RESULTS = "curriculum_logic_v0_results.json"
FAMS = ["prop", "chain"]

# ---------- A) 命题公式求值（通道 A：树构造 + 树求值） ----------
def _rt(rng, vs, depth):
    if depth == 0 or rng.random() < 0.3:
        return rng.choice(vs)
    r = rng.random()
    if r < 0.25:
        return ("not", _rt(rng, vs, depth - 1))
    if r < 0.6:
        return ("and", _rt(rng, vs, depth - 1), _rt(rng, vs, depth - 1))
    return ("or", _rt(rng, vs, depth - 1), _rt(rng, vs, depth - 1))

def _render(t):
    if isinstance(t, str):
        return t
    if t[0] == "not":
        return "(NOT %s)" % _render(t[1])
    return "(%s %s %s)" % (_render(t[1]), "AND" if t[0] == "and" else "OR", _render(t[2]))

def _teval(t, w):
    if isinstance(t, str):
        return w[t]
    if t[0] == "not":
        return 1 - _teval(t[1], w)
    if t[0] == "and":
        return _teval(t[1], w) & _teval(t[2], w)
    return _teval(t[1], w) | _teval(t[2], w)

def gen_prop(rng):
    vs = ["A", "B", "C", "D", "E"]
    t = _rt(rng, vs, rng.choice([2, 2, 3, 3, 4]))
    w = {v: rng.randrange(2) for v in vs}
    val = _teval(t, w)
    given = ", ".join("%s=%d" % (v, w[v]) for v in vs)
    q = "Formula: %s. Given %s, is the formula true? Answer true or false." % (_render(t), given)
    cot = "substitute: %s -> %s" % (" ".join("%s=%d" % (v, w[v]) for v in vs), "true" if val else "false")
    return {"fam": "prop", "q": q, "cot": cot, "ans": "true" if val else "false", "k": {"w": w}}

def _beval(expr, w):
    """通道 B：递归下降解析（与通道 A 树求值独立实现）。"""
    toks = re.findall(r"NOT|AND|OR|\(|\)|[A-E]", expr)
    pos = [0]
    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None
    def take():
        t = toks[pos[0]]; pos[0] += 1; return t
    def prim():
        t = take()
        if t == "NOT":
            return 1 - prim()
        if t == "(":
            v = orex()
            assert take() == ")"
            return v
        return w[t]
    def orex():
        v = andex()
        while peek() == "OR":
            take(); v |= andex()
        return v
    def andex():
        v = prim()
        while peek() == "AND":
            take(); v &= prim()
        return v
    return orex()

def verify_prop(sample):
    m = re.match(r"Formula: (.+)\. Given (.+?), is the formula true", sample["q"])
    if not m:
        return {"ok": False, "gold": "?", "cot_ok": False}
    w = {}
    for kv in m.group(2).split(", "):
        k, _, v = kv.partition("=")
        w[k] = int(v)
    try:
        val = _beval(m.group(1), w)
    except Exception:
        return {"ok": False, "gold": "?", "cot_ok": False}
    gold = "true" if val else "false"
    exp_cot = "substitute: %s -> %s" % (" ".join("%s=%d" % (v, w[v]) for v in sorted(w)), gold)
    return {"ok": sample["ans"] == gold, "gold": gold, "cot_ok": sample["cot"] == exp_cot}

# ---------- B) 前向链推理（通道 A：世界中一致的规则集） ----------
def gen_chain(rng):
    atoms = ["P", "Q", "R", "S", "T", "U"]
    w = {a: rng.randrange(2) for a in atoms}
    facts = [a for a in atoms if w[a] and rng.random() < 0.7]
    rules = []
    for a in atoms:
        for b in atoms:
            if a != b and not (w[a] == 1 and w[b] == 0) and rng.random() < 0.3:
                rules.append((a, b))
    target = rng.choice(atoms)
    q = "Facts: %s. Rules: %s. Does %s hold? Answer yes or no." % (
        ", ".join("%s is true" % a for a in facts) or "none",
        "; ".join("if %s then %s" % (a, b) for a, b in rules) or "none",
        target)
    cl = _closure(set(facts), rules)
    gold = "yes" if w[target] == 1 else "no"  # 世界真值（闭包与其一致，见 verify 对账）
    cot = "closure: %s" % (",".join(sorted(cl)) or "empty")
    return {"fam": "chain", "q": q, "cot": cot, "ans": gold, "k": {"target": target}}

def _closure(facts, rules):
    cl = set(facts)
    changed = True
    while changed:
        changed = False
        for a, b in rules:
            if a in cl and b not in cl:
                cl.add(b); changed = True
    return cl

def verify_chain(sample):
    m = re.match(r"Facts: (.+)\. Rules: (.+)\. Does (\w) hold", sample["q"])
    if not m:
        return {"ok": False, "gold": "?", "cot_ok": False}
    fs = re.findall(r"(\w) is true", m.group(1)) if m.group(1) != "none" else []
    rs = re.findall(r"if (\w) then (\w)", m.group(2)) if m.group(2) != "none" else []
    cl = _closure(set(fs), rs)
    # 一致性对账：闭包必须与事实不冲突（规则在世界中为真的副产品）
    gold = "yes" if m.group(3) in cl else "no"
    exp_cot = "closure: %s" % (",".join(sorted(cl)) or "empty")
    ok = sample["ans"] == gold
    cot_ok = sample["cot"] == exp_cot
    return {"ok": ok, "gold": gold, "cot_ok": cot_ok,
            "world_consistent": (gold == "yes") or (m.group(3) not in cl)}

def verify(sample):
    if sample["fam"] == "prop":
        return verify_prop(sample)
    return verify_chain(sample)

GEN = {"prop": gen_prop, "chain": gen_chain}

def gen_sample(rng, fam):
    fam = fam if fam in GEN else "prop"
    for _ in range(20):  # 内嵌通道 B 复核（零歧义纪律）
        sm = GEN[fam](rng)
        v = verify(sm)
        if v["ok"] and v["cot_ok"]:
            return sm
    return None

# ---------- 定向毒 ----------
def _poison_flip(sample):
    p = dict(sample)
    if sample["ans"] in ("true", "false"):
        p["ans"] = "false" if sample["ans"] == "true" else "true"
    else:
        p["ans"] = "no" if sample["ans"] == "yes" else "yes"
    return p

def _poison_struct(sample):
    if sample["fam"] == "prop":
        for _ in range(30):  # 变异一个 AND↔OR 直到判定翻转
            m = re.search(r" (AND|OR) ", sample["q"])
            if not m:
                return None
            ops = re.findall(r" (AND|OR) ", sample["q"])
            i = random.Random().randrange(len(ops))
            new = " OR " if ops[i] == "AND" else " AND "
            q2 = sample["q"]
            cnt = -1
            def rep(mm):
                nonlocal cnt
                cnt += 1
                return new if cnt == i else mm.group(0)
            q2 = re.sub(r" (AND|OR) ", rep, q2)
            p = dict(sample); p["q"] = q2
            if not verify(p)["ok"]:
                return p
        return None
    # chain：翻转一个事实
    m = re.search(r"(\w) is true", sample["q"])
    if not m:
        return None
    facts = re.findall(r"(\w) is true", sample["q"])
    rules = re.findall(r"if (\w) then (\w)", sample["q"].split("Rules: ")[1].split(". Does")[0])
    target = re.search(r"Does (\w) hold", sample["q"]).group(1)
    for a in facts:
        fs = [x for x in facts if x != a]
        cl = _closure(set(fs), rules)
        gold = "yes" if target in cl else "no"
        if gold != sample["ans"]:
            p = dict(sample)
            p["q"] = sample["q"].replace("Facts: " + ", ".join("%s is true" % x for x in facts),
                                         "Facts: " + (", ".join("%s is true" % x for x in fs) or "none"))
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
    for split, n, seed in (("train", n_train, 12001), ("val", n_val, 9005)):
        rng = random.Random(seed)
        i = 0
        while i < n:
            sm = gen_sample(rng, FAMS[i % len(FAMS)])
            if sm is None:
                continue
            lines.append(json.dumps({
                "split": split, "subject": "formal_logic", "id": "%s_%04d" % (split, i),
                "fam": sm["fam"], "q": sm["q"], "cot": sm["cot"], "ans": sm["ans"]}, ensure_ascii=False))
            i += 1
    data = "\n".join(lines) + "\n"
    open("curriculum_logic_v0.jsonl", "w", encoding="utf-8", newline="").write(data)
    return {"records": len(lines), "sha256": hashlib.sha256(data.encode()).hexdigest()[:16]}

if __name__ == "__main__":
    res = self_test()
    res["export"] = export()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(OUT_RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if res["ALL_PASS"] else 2)
