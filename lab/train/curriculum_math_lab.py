# -*- coding: utf-8 -*-
# 课程生成器 v0 —— 第八科：数学初阶（一元一次方程/算式求值，验证器=独立代数求解/AST 求值对拍）
# T 式纪律：G1 双通道 / G2 定向毒 fail-closed / G3 确定性 / G4 train/val 隔离。零歧义：解必须唯一且为整数。
import json, random, hashlib, sys, re, ast, operator

OUT_RESULTS = "curriculum_math_v0_results.json"
FAMS = ["lin1", "lin2", "arith"]

# ---------- A) 一元一次方程 ----------
def gen_lin(rng, two_sided):
    x0 = rng.randrange(-60, 61)
    a = rng.choice([x for x in range(-20, 21) if x != 0])
    b = rng.randrange(-99, 100)
    if not two_sided:
        c = a * x0 + b
        if c < -999 or c > 999:
            return None
        q = "Solve for x: %dx + %d = %d. Answer the value of x." % (a, b, c)
        s = c - b
        cot = "sub: %dx = %d; div: x = %d" % (a, s, x0)
        return {"fam": "lin1", "q": q, "cot": cot, "ans": str(x0), "k": {"a": a, "b": b, "c": c, "x0": x0}}
    c = rng.choice([x for x in range(-20, 21) if x != a])
    d = a * x0 + b - c * x0
    if abs(d) > 999:
        return None
    q = "Solve for x: %dx + %d = %dx + %d. Answer the value of x." % (a, b, c, d)
    s = d - b
    den = a - c
    if s % den != 0:
        return None
    cot = "sub: %dx = %d; div: x = %d" % (den, s, x0)
    return {"fam": "lin2", "q": q, "cot": cot, "ans": str(x0), "k": {"a": a, "b": b, "c": c, "d": d, "x0": x0}}

def _fmt_lin(a, b, c):
    return ("%d*x + %d" % (a, b)) if b >= 0 else ("%d*x - %d" % (a, -b))

def verify_lin(sample):
    q = sample["q"]
    m = re.match(r"Solve for x: (-?\d+)x ([+-]) (\d+) = (-?\d+)x ([+-]) (\d+)\.", q)
    if m:
        a, o1, b, c, o2, d = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5), int(m.group(6))
        b = b if o1 == "+" else -b
        d = d if o2 == "+" else -d
    else:
        m = re.match(r"Solve for x: (-?\d+)x ([+-]) (\d+) = (-?\d+)\.", q)
        if not m:
            return {"ok": False, "gold": "?", "cot_ok": False}
        a, o1, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        b = b if o1 == "+" else -b
        d = c; c = 0
    den = a - c
    s = d - b
    if den == 0 or s % den != 0:
        return {"ok": False, "gold": "?", "cot_ok": False}
    x0 = s // den
    # 代入回验（第三通道）
    if a * x0 + b != c * x0 + d:
        return {"ok": False, "gold": "?", "cot_ok": False}
    gold = str(x0)
    exp_cot = "sub: %dx = %d; div: x = %d" % (den, s, x0)
    return {"ok": sample["ans"] == gold, "gold": gold, "cot_ok": sample["cot"] == exp_cot}

# ---------- B) 算式求值 ----------
def _rt_a(rng, depth):
    if depth == 0:
        return rng.randrange(1, 30)
    op = rng.choice(["+", "-", "*"])
    return (op, _rt_a(rng, depth - 1), _rt_a(rng, depth - 1))

def _render_a(t):
    if isinstance(t, int):
        return str(t)
    return "(%s %s %s)" % (_render_a(t[1]), t[0], _render_a(t[2]))

def _teval_a(t):
    if isinstance(t, int):
        return t
    l, r = _teval_a(t[1]), _teval_a(t[2])
    return l + r if t[0] == "+" else l - r if t[0] == "-" else l * r

def gen_arith(rng):
    t = _rt_a(rng, rng.choice([2, 2, 3, 3, 4]))
    e = _render_a(t)
    v = _teval_a(t)
    if abs(v) > 99999:
        return None
    q = "Evaluate: %s. Answer a number." % e
    cot = "eval: %s = %d" % (e, v)
    return {"fam": "arith", "q": q, "cot": cot, "ans": str(v), "k": {}}

_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}

def verify_arith(sample):
    m = re.match(r"Evaluate: (.+)\. Answer", sample["q"])
    if not m:
        return {"ok": False, "gold": "?", "cot_ok": False}
    try:
        node = ast.parse(m.group(1), mode="eval")
        v = _aeval(node.body)
    except Exception:
        return {"ok": False, "gold": "?", "cot_ok": False}
    gold = str(v)
    exp_cot = "eval: %s = %s" % (m.group(1), gold)
    return {"ok": sample["ans"] == gold, "gold": gold, "cot_ok": sample["cot"] == exp_cot}

def _aeval(n):
    if isinstance(n, ast.Constant):
        return n.value
    return _OPS[type(n.op).__name__ == "Add" and "+" or type(n.op).__name__ == "Sub" and "-" or "*"](_aeval(n.left), _aeval(n.right))

def verify(sample):
    if sample["fam"] == "arith":
        return verify_arith(sample)
    return verify_lin(sample)

GEN = {"lin1": lambda rng: gen_lin(rng, False), "lin2": lambda rng: gen_lin(rng, True), "arith": gen_arith}

def gen_sample(rng, fam):
    fam = fam if fam in GEN else "arith"
    for _ in range(20):  # 内嵌通道 B 复核
        sm = GEN[fam](rng)
        if sm is None:
            continue
        v = verify(sm)
        if v["ok"] and v["cot_ok"]:
            return sm
    return None

# ---------- 定向毒 ----------
def _poison_ans(sample):
    g = int(sample["ans"])
    p = dict(sample); p["ans"] = str(g + 1)
    return p

def _poison_struct(sample):
    if sample["fam"] == "arith":
        for leaf_pos in range(8):
            m = re.match(r"Evaluate: (.+)\. Answer", sample["q"])
            leaves = [(mm.group(0), mm.start()) for mm in re.finditer(r"\d+", m.group(1))]
            if leaf_pos >= len(leaves):
                break
            old, start = leaves[leaf_pos]
            for dv in (1, -1):
                expr2 = m.group(1)[:start] + str(int(old) + dv) + m.group(1)[start + len(old):]
                p = dict(sample)
                p["q"] = "Evaluate: %s. Answer a number." % expr2
                if not verify(p)["ok"]:
                    return p
        return None
    # 线性方程：变异常数直到解变（lin1 变 RHS；lin2 变 RHS 常数项）
    m2 = re.match(r"Solve for x: (.+?) = (-?\d+)x ([+-]) (\d+)\.", sample["q"])
    if m2:
        for dv in (1, -1, 2, -2):
            k = int(m2.group(4))
            o = m2.group(3)
            k2 = k + dv
            if k2 < 0:
                o = "+" if o == "-" else "-"
                k2 = abs(k2)
            p = dict(sample)
            p["q"] = "Solve for x: %s = %sx %s %d." % (m2.group(1), m2.group(2), o, k2)
            if not verify(p)["ok"]:
                return p
        return None
    m = re.match(r"Solve for x: (.+?) = (-?\d+)\.", sample["q"])
    if not m:
        return None
    lhs, rhs = m.group(1), m.group(2)
    for dv in (1, -1, 2, -2):
        p = dict(sample)
        p["q"] = "Solve for x: %s = %d." % (lhs, int(rhs) + dv)
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
        if not verify(_poison_ans(sm))["ok"]:
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
    for split, n, seed in (("train", n_train, 14001), ("val", n_val, 9007)):
        rng = random.Random(seed)
        i = 0
        while i < n:
            sm = gen_sample(rng, FAMS[i % len(FAMS)])
            if sm is None:
                continue
            lines.append(json.dumps({
                "split": split, "subject": "math_basic", "id": "%s_%04d" % (split, i),
                "fam": sm["fam"], "q": sm["q"], "cot": sm["cot"], "ans": sm["ans"]}, ensure_ascii=False))
            i += 1
    data = "\n".join(lines) + "\n"
    open("curriculum_math_v0.jsonl", "w", encoding="utf-8", newline="").write(data)
    return {"records": len(lines), "sha256": hashlib.sha256(data.encode()).hexdigest()[:16]}

if __name__ == "__main__":
    res = self_test()
    res["export"] = export()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(OUT_RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if res["ALL_PASS"] else 2)
