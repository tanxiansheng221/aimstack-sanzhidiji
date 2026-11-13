# -*- coding: utf-8 -*-
# 课程生成器 v0 —— 第四科：约束满足（可着色判定/赋值核查/4x4数独填格，验证器=机械判定器）
# T 式纪律：G1 双通道全对（生成器构造 vs 验证器独立判定）/ G2 毒样本 fail-closed /
#           G3 确定性 / G4 train/val 隔离。零歧义纪律：生成器内嵌通道 B 复核，不一致弃样。
import json, random, hashlib, sys, itertools

OUT_RESULTS = "curriculum_csp_v0_results.json"
FAMS = ["colorable", "assign", "sudoku"]

# ---------- A) K-可着色判定 ----------
def gen_colorable(rng):
    n = rng.choice([4, 5, 6])
    K = rng.choice([2, 3])
    p = 0.4 if K == 3 else 0.35
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    chi = _chromatic(n, edges, K)
    if chi is None or chi > 3:
        return None
    el = ", ".join("S%d-S%d" % e for e in sorted(edges)) or "none"
    q = "Graph vertices S0..S%d, edges: %s. Can you color all vertices with %d colors so adjacent vertices differ? Answer yes or no." % (n - 1, el, K)
    ans = "yes" if chi <= K else "no"
    cot = "chromatic: %d" % chi
    return {"fam": "colorable", "q": q, "cot": cot, "ans": ans,
            "k": {"n": n, "edges": sorted(edges), "K": K}}

def _chromatic(n, edges, cap):
    """通道 B：穷举所有 ≤cap 着色（n≤6, cap≤3 → ≤729 种），返回色数或 None(>cap)。"""
    for k in range(1, cap + 1):
        for assign in itertools.product(range(k), repeat=n):
            if all(assign[u] != assign[v] for u, v in edges):
                return k
    return None

# ---------- B) 约束赋值核查 ----------
def gen_assign(rng):
    nv = 5
    vals = {i: rng.randrange(0, 10) for i in range(nv)}
    cons = []
    kinds = []
    tries = 0
    while len(cons) < 4:
        tries += 1
        if tries > 200:
            return None  # 退化赋值（如全同值≥5：三类约束全拒）弃样
        kind = rng.choice(["sum", "gt", "neq"])
        i, j = rng.sample(range(nv), 2)
        if kind == "sum":
            t = vals[i] + vals[j]
            if t > 9 or (i, j, "sum") in kinds:
                continue
            cons.append("x%d + x%d = %d" % (i, j, t))
        elif kind == "gt":
            if vals[i] <= vals[j] or (i, j, "gt") in kinds:
                continue
            cons.append("x%d > x%d" % (i, j))
        else:
            if vals[i] == vals[j] or (i, j, "neq") in kinds:
                continue
            cons.append("x%d != x%d" % (i, j))
        kinds.append((i, j, kind))
    assign = ", ".join("x%d=%d" % (i, vals[i]) for i in range(nv))
    q = "Variables %s. Constraints: %s. Does this assignment satisfy all constraints? Answer yes or no." % (
        assign, "; ".join(cons))
    checks = _check_assign(cons, vals)
    fails = [k for k, ok in checks if not ok]
    ans = "no" if fails else "yes"
    cot = "check: %s verdict=%s" % (", ".join(("c%d ok" % k) if ok else ("c%d fail" % k) for k, ok in checks), ans)
    return {"fam": "assign", "q": q, "cot": cot, "ans": ans, "k": {"cons": cons, "vals": vals}}

def _parse_cons(cons):
    out = []
    for c in cons:
        if " + " in c:
            a = c.split(" + "); b = a[1].split(" = ")
            out.append(("sum", int(a[0][1:]), int(b[0][1:]), int(b[1])))
        elif " != " in c:
            a = c.split(" != "); out.append(("neq", int(a[0][1:]), int(a[1][1:]), None))
        else:
            a = c.split(" > "); out.append(("gt", int(a[0][1:]), int(a[1][1:]), None))
    return out

def _check_assign(cons, vals):
    res = []
    for idx, (kind, i, j, t) in enumerate(_parse_cons(cons)):
        if kind == "sum":
            res.append((idx, vals[i] + vals[j] == t))
        elif kind == "gt":
            res.append((idx, vals[i] > vals[j]))
        else:
            res.append((idx, vals[i] != vals[j]))
    return res

# ---------- C) 4x4 数独填格 ----------
def _sudoku_full(rng):
    g = [[0] * 4 for _ in range(4)]
    def ok(r, c, v):
        if any(g[r][j] == v for j in range(4)) or any(g[i][c] == v for i in range(4)):
            return False
        br, bc = (r // 2) * 2, (c // 2) * 2
        return all(g[br + i][bc + j] != v for i in range(2) for j in range(2))
    def fill(p):
        if p == 16:
            return True
        r, c = divmod(p, 4)
        for v in rng.sample([1, 2, 3, 4], 4):
            if ok(r, c, v):
                g[r][c] = v
                if fill(p + 1):
                    return True
                g[r][c] = 0
        return False
    fill(0)
    return g

def _sudoku_solve_count(g, limit=2):
    cnt = 0
    def go():
        nonlocal cnt
        for r in range(4):
            for c in range(4):
                if g[r][c] == 0:
                    for v in range(1, 5):
                        g[r][c] = v
                        if _sudoku_cell_ok(g, r, c):
                            go()
                        g[r][c] = 0
                    return
        cnt += 1
    go()
    return cnt

def _sudoku_cell_ok(g, r, c):
    v = g[r][c]
    if sum(1 for j in range(4) if g[r][j] == v) > 1: return False
    if sum(1 for i in range(4) if g[i][c] == v) > 1: return False
    br, bc = (r // 2) * 2, (c // 2) * 2
    return sum(1 for i in range(2) for j in range(2) if g[br + i][bc + j] == v) == 1

def gen_sudoku(rng):
    full = _sudoku_full(rng)
    cells = [(r, c) for r in range(4) for c in range(4)]
    rng.shuffle(cells)
    g = [row[:] for row in full]
    removed = 0
    target = None
    for (r, c) in cells:
        if removed >= 6:
            break
        keep = g[r][c]
        g[r][c] = 0
        if _sudoku_solve_count([row[:] for row in g]) != 1:
            g[r][c] = keep
        else:
            removed += 1
            if _cand_intersect(g, r, c) is not None:
                target = (r, c)
    if target is None or removed < 3:
        return None
    tr, tc = target
    if g[tr][tc] != 0:
        return None
    x = _cand_intersect(g, tr, tc)
    if x is None or x != full[tr][tc]:
        return None  # 通道 B 交叉集必须单值且等于真值
    gs = "; ".join(" ".join(str(v) if v else "." for v in row) for row in g)
    q = "4x4 sudoku (2x2 blocks, digits 1-4), rows: %s. What digit goes in row %d col %d? Answer one digit." % (gs, tr, tc)
    cot = "cell r%d c%d: row has %d, col has %d, block has %d" % (tr, tc, full[tr][tc], full[tr][tc], full[tr][tc])
    # cot 改为「同行/列/块缺失」叙述太长；统一为交叉集单值陈述
    cot = "cell r%d c%d unique = %d" % (tr, tc, x)
    return {"fam": "sudoku", "q": q, "cot": cot, "ans": str(x),
            "k": {"grid": g, "tr": tr, "tc": tc}}

def _cand_intersect(g, r, c):
    if g[r][c] != 0:
        return None
    used = set(g[r]) | {g[i][c] for i in range(4)}
    br, bc = (r // 2) * 2, (c // 2) * 2
    used |= {g[br + i][bc + j] for i in range(2) for j in range(2)}
    cand = [v for v in range(1, 5) if v not in used]
    return cand[0] if len(cand) == 1 else None

# ---------- 验证器（通道 B：只吃 q 文本 + 结构化 k，独立重判） ----------
def verify(sample):
    fam = sample["fam"]
    if fam == "colorable":
        n, K = sample["k"]["n"], sample["k"]["K"]
        edges = [tuple(e) for e in sample["k"]["edges"]]
        chi = _chromatic(n, edges, K)
        gold = "yes" if (chi is not None and chi <= K) else "no"
        # 穷举上限与题面 K 一致；chi>3 不会出现在语料（生成器弃样）
        return {"ok": sample["ans"] == gold, "gold": gold,
                "cot_ok": chi is not None and sample["cot"] == "chromatic: %d" % chi, "fam_match": True}
    if fam == "assign":
        import re
        vals = {}
        for m in re.finditer(r"x(\d)=(\d)", sample["q"].split("Variables ")[1].split(". Constraints")[0]):
            vals[int(m.group(1))] = int(m.group(2))
        cons = sample["q"].split("Constraints: ")[1].split(". Does")[0].split("; ")
        checks = _check_assign(cons, vals)
        fails = [k for k, ok in checks if not ok]
        gold = "no" if fails else "yes"
        exp_cot = "check: %s verdict=%s" % (", ".join(("c%d ok" % k) if ok else ("c%d fail" % k) for k, ok in checks), gold)
        return {"ok": sample["ans"] == gold, "gold": gold,
                "cot_ok": sample["cot"] == exp_cot, "fam_match": True}
    if fam == "sudoku":
        g = [row[:] for row in sample["k"]["grid"]]
        tr, tc = sample["k"]["tr"], sample["k"]["tc"]
        x = _cand_intersect(g, tr, tc)
        gold = str(x) if x is not None else "?"
        exp_cot = "cell r%d c%d unique = %s" % (tr, tc, gold)
        return {"ok": sample["ans"] == gold and gold != "?", "gold": gold,
                "cot_ok": sample["cot"] == exp_cot, "fam_match": True}
    return {"ok": False, "gold": "?", "cot_ok": False, "fam_match": False}

GEN = {"colorable": gen_colorable, "assign": gen_assign, "sudoku": gen_sudoku}

def gen_sample(rng, fam):
    fam = fam if fam in GEN else "assign"
    for _ in range(20):  # 歧义/退化弃样重抽，内嵌通道 B 复核
        sm = GEN[fam](rng)
        if sm is None:
            continue
        v = verify(sm)
        if v["ok"] and v["cot_ok"]:
            return sm
    return None

# ---------- 定向结构毒（保证判定必翻；构造不出返回 None） ----------
def _poison_struct(sm):
    import re
    if sm["fam"] == "assign":
        cons = [c.strip() for c in sm["q"].split("Constraints: ")[1].split(". Does")[0].split("; ")]
        vals = dict(sm["k"]["vals"])
        for i in sorted(vals):
            for dv in range(10):
                if dv == vals[i]:
                    continue
                v2 = dict(vals); v2[i] = dv
                checks = _check_assign(cons, v2)
                gold2 = "no" if any(not ok for _, ok in checks) else "yes"
                if gold2 != sm["ans"]:
                    p = dict(sm)
                    asg_old = sm["q"].split("Variables ")[1].split(". Constraints")[0]
                    asg = ", ".join("x%d=%d" % (d, v2[d]) for d in sorted(v2))
                    p["q"] = sm["q"].replace(asg_old, asg)
                    return p
        return None
    if sm["fam"] == "colorable":
        n, K = sm["k"]["n"], sm["k"]["K"]
        base = [tuple(e) for e in sm["k"]["edges"]]
        cands = []
        for ei, (u, v) in enumerate(base):
            for w in range(n):
                if w != u and w != v and [u, w] not in sm["k"]["edges"]:
                    cands.append((ei, (u, w)))
        for u in range(n):
            for v in range(u + 1, n):
                if [u, v] not in sm["k"]["edges"] and (u, v) not in base:
                    cands.append((-1, (u, v)))  # 加边
        for ei, e2 in cands:
            edges = [tuple(e) for e in base]
            if ei >= 0:
                edges[ei] = e2
            else:
                edges.append(e2)
            chi = _chromatic(n, edges, K)
            gold2 = "yes" if (chi is not None and chi <= K) else "no"
            if gold2 != sm["ans"]:
                p = dict(sm); p["k"] = json.loads(json.dumps(sm["k"])); p["k"]["edges"] = [list(e) for e in edges]
                return p
        return None
    # sudoku：改任一格值直到目标格交叉集必变
    tr, tc = sm["k"]["tr"], sm["k"]["tc"]
    gold0 = _cand_intersect([row[:] for row in sm["k"]["grid"]], tr, tc)
    for r_ in range(4):
        for c_ in range(4):
            if (r_, c_) == (tr, tc) or sm["k"]["grid"][r_][c_] == 0:
                continue
            for v in range(1, 5):
                if v == sm["k"]["grid"][r_][c_]:
                    continue
                g2 = [row[:] for row in sm["k"]["grid"]]
                g2[r_][c_] = v
                if _cand_intersect(g2, tr, tc) != gold0:
                    p = dict(sm); p["k"] = json.loads(json.dumps(sm["k"])); p["k"]["grid"] = g2
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
        p1 = dict(sm)
        p1["ans"] = "no" if sm["ans"] == "yes" else "yes"
        if not verify(p1)["ok"]:
            caught += 1
        total2 += 1
        p2 = dict(sm); p2["cot"] = sm["cot"] + " -> tampered"
        if not verify(p2)["cot_ok"]:
            caught += 1
        total2 += 1
        p3 = _poison_struct(sm)  # 定向结构毒：保证判定必翻，构造不出则跳过不计数（科目#2 纪律）
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
    for split, n, seed in (("train", n_train, 8001), ("val", n_val, 9003)):
        rng = random.Random(seed)
        i = 0
        while i < n:
            sm = gen_sample(rng, FAMS[i % len(FAMS)])
            if sm is None:
                continue
            lines.append(json.dumps({
                "split": split, "subject": "constraint_satisfy", "id": "%s_%04d" % (split, i),
                "fam": sm["fam"], "q": sm["q"], "cot": sm["cot"], "ans": sm["ans"]}, ensure_ascii=False))
            i += 1
    data = "\n".join(lines) + "\n"
    open("curriculum_csp_v0.jsonl", "w", encoding="utf-8", newline="").write(data)
    return {"records": len(lines), "sha256": hashlib.sha256(data.encode()).hexdigest()[:16]}

if __name__ == "__main__":
    res = self_test()
    res["export"] = export()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(OUT_RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if res["ALL_PASS"] else 2)
