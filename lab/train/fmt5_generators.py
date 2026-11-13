# -*- coding: utf-8 -*-
# fmt5_generators.py — 趴零五科格式工程 v1（Abacus 位权 + 真 CoT 链）
# 处方（CPU_NATIVE_DEDUCTION.md L2405/L2429）：
#   math/mod_arith: 中间步全分解为「digit×number 部分积 + 单次取模」微步，杜绝大数直给
#   plan/progexec/seqind: cot 从结论快照改为真推理链（照 csp 逐约束样板）
# 纪律：G1 双通道（gen 从参数构造链 vs verify 从 q 盲重建链，独立实现）/ G2 定向毒 fail-closed /
#       G3 确定性 / G4 train/val 隔离 / 字节预算闸（q+cot+ans ≤ 480 < T=512）
# 题面 q 与 v0 完全同格式（只改答案侧链）；mod_arith inv 族限 p ≤ 43（链可控），余族全保留。
import json, random, re, ast, operator, sys, io, hashlib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CHAIN_CAP = 300   # 单链字节硬顶（超限弃样重抽）
BUDGET = 480      # len(q)+len(cot)+len(ans)+10 预算硬顶

# ============ 通用：乘法部分积分解（Abacus 核心） ============
def mult_steps(l, r):
    """l*r 的部分积微步（要求 l,r ≥ 10 且 ≥0）。返回 (steps, value)。
    例: 47*83 -> ['47*3=141', '47*80=3760', '141+3760=3901']"""
    ds = []
    for i, ch in enumerate(reversed(str(r))):
        if ch != "0":
            ds.append(int(ch) * (10 ** i))
    parts = ["%d*%d=%d" % (l, pv, l * pv) for pv in ds]
    run = l * ds[0]
    for pv in ds[1:]:
        nxt = run + l * pv
        parts.append("%d+%d=%d" % (run, l * pv, nxt))
        run = nxt
    return parts, run

def mod_step(v, p):
    """v mod p 的归约微步（v ≥ p 才有）。例: 5476 mod 59 -> '5476-(92*59)=48'"""
    q = v // p
    return "%d-(%d*%d)=%d" % (v, q, p, v - q * p)

# ============ 第一科：math（lin1/lin2/arith） ============
# v0 纪律固化：verify 正则吃不下负 b/d（'4x + -50' 不匹配），有效分布即 b,d ≥ 0，此处显式约束
def gen_math_lin1(rng):
    x0 = rng.randrange(-60, 61)
    a = rng.choice([x for x in range(-20, 21) if x != 0])
    b = rng.randrange(0, 100)
    c = a * x0 + b
    if c < -999 or c > 999:
        return None
    q = "Solve for x: %dx + %d = %d. Answer the value of x." % (a, b, c)
    s = c - b
    cot = "s1: %d-%d=%d; s2: %d*%d=%d; x=%d" % (c, b, s, a, x0, a * x0, x0)
    return {"fam": "lin1", "q": q, "cot": cot, "ans": str(x0)}

def gen_math_lin2(rng):
    x0 = rng.randrange(-60, 61)
    a = rng.choice([x for x in range(-20, 21) if x != 0])
    c = rng.choice([x for x in range(-20, 21) if x != a])
    den = a - c
    b = rng.randrange(0, 100)
    d = a * x0 + b - c * x0
    if d < 0 or abs(d) > 999:
        return None
    q = "Solve for x: %dx + %d = %dx + %d. Answer the value of x." % (a, b, c, d)
    s = d - b
    cot = "s1: %d-%d=%d; s2: %d-%d=%d; s3: %d*%d=%d; x=%d" % (
        a, c, den, d, b, s, den, x0, den * x0, x0)
    return {"fam": "lin2", "q": q, "cot": cot, "ans": str(x0)}

def _rt_a(rng, depth):
    if depth == 0:
        return rng.randrange(1, 30)
    op = rng.choice(["+", "-", "*"])
    return (op, _rt_a(rng, depth - 1), _rt_a(rng, depth - 1))

def _render_a(t):
    if isinstance(t, int):
        return str(t)
    return "(%s %s %s)" % (_render_a(t[1]), t[0], _render_a(t[2]))

def _arith_chain(t, steps):
    """后序遍历产微步；乘法双操作数 ≥10 且 ≥0 时部分积分解。返回 value。"""
    if isinstance(t, int):
        return t
    op, l, r = t[0], t[1], t[2]
    lv = _arith_chain(l, steps)
    rv = _arith_chain(r, steps)
    if op == "*" and lv >= 10 and rv >= 10:
        parts, run = mult_steps(lv, rv)
        head = "s%d: %d*%d: " % (len(steps) + 1, lv, rv)
        segs = [head + parts[0]]
        steps.extend(segs)
        for p_ in parts[1:]:
            steps.append("s%d: %s" % (len(steps) + 1, p_))
        return run
    v = lv + rv if op == "+" else lv - rv if op == "-" else lv * rv
    steps.append("s%d: %d%s%d=%d" % (len(steps) + 1, lv, op, rv, v))
    return v

def gen_math_arith(rng):
    t = _rt_a(rng, rng.choice([2, 2, 3, 3]))
    e = _render_a(t)
    steps = []
    v = _arith_chain(t, steps)
    if abs(v) > 99999:
        return None
    cot = "; ".join(steps)
    q = "Evaluate: %s. Answer a number." % e
    return {"fam": "arith", "q": q, "cot": cot, "ans": str(v)}

MATH_GEN = {"lin1": gen_math_lin1, "lin2": gen_math_lin2, "arith": gen_math_arith}

def _verify_math(sample):
    q = sample["q"]
    m = re.match(r"Solve for x: (-?\d+)x ([+-]) (\d+) = (-?\d+)x ([+-]) (\d+)\.", q)
    if m:
        a, o1, b, c, o2, d = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5), int(m.group(6))
        b = b if o1 == "+" else -b
        d = d if o2 == "+" else -d
        den = a - c
        s = d - b
        if den == 0 or s % den != 0:
            return {"ok": False, "gold": "?", "cot_ok": False}
        x0 = s // den
        if a * x0 + b != c * x0 + d:
            return {"ok": False, "gold": "?", "cot_ok": False}
        exp = "s1: %d-%d=%d; s2: %d-%d=%d; s3: %d*%d=%d; x=%d" % (
            a, c, den, d, b, s, den, x0, den * x0, x0)
        return {"ok": sample["ans"] == str(x0), "gold": str(x0), "cot_ok": sample["cot"] == exp}
    m = re.match(r"Solve for x: (-?\d+)x ([+-]) (\d+) = (-?\d+)\.", q)
    if m:
        a, o1, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        b = b if o1 == "+" else -b
        s = c - b
        if s % a != 0:
            return {"ok": False, "gold": "?", "cot_ok": False}
        x0 = s // a
        if a * x0 + b != c:
            return {"ok": False, "gold": "?", "cot_ok": False}
        exp = "s1: %d-%d=%d; s2: %d*%d=%d; x=%d" % (c, b, s, a, x0, a * x0, x0)
        return {"ok": sample["ans"] == str(x0), "gold": str(x0), "cot_ok": sample["cot"] == exp}
    m = re.match(r"Evaluate: (.+)\. Answer", q)
    if m:
        try:
            node = ast.parse(m.group(1), mode="eval")
        except Exception:
            return {"ok": False, "gold": "?", "cot_ok": False}
        steps = []
        v = _ast_chain(node.body, steps)
        exp = "; ".join(steps)
        return {"ok": sample["ans"] == str(v), "gold": str(v), "cot_ok": sample["cot"] == exp}
    return {"ok": False, "gold": "?", "cot_ok": False}

def _ast_chain(n, steps):
    """独立通道：从 AST 盲重建微步链（与 gen 的树遍历相互独立）。"""
    if isinstance(n, ast.Constant):
        return n.value
    op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}[type(n.op)]
    lv = _ast_chain(n.left, steps)
    rv = _ast_chain(n.right, steps)
    if op == "*" and lv >= 10 and rv >= 10:
        parts, run = mult_steps(lv, rv)
        steps.append("s%d: %d*%d: %s" % (len(steps) + 1, lv, rv, parts[0]))
        for p_ in parts[1:]:
            steps.append("s%d: %s" % (len(steps) + 1, p_))
        return run
    v = lv + rv if op == "+" else lv - rv if op == "-" else lv * rv
    steps.append("s%d: %d%s%d=%d" % (len(steps) + 1, lv, op, rv, v))
    return v

# ============ 第二科：mod_arith（mod/pow/inv） ============
MOD_PRIMES = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]
INV_P_CAP = 43   # inv 族链长可控上限

def _reduce_chain(v, p, steps):
    """v ≥ p 时产归约微步，返回残差。"""
    if v < p:
        return v
    steps.append("s%d: %s" % (len(steps) + 1, mod_step(v, p)))
    return v % p

def _mulmod_chain(l, r, p, steps):
    """l*r mod p 的微步（乘法分解 + 归约），返回残差。"""
    if l >= 10 and r >= 10:
        parts, run = mult_steps(l, r)
        steps.append("s%d: %d*%d: %s" % (len(steps) + 1, l, r, parts[0]))
        for p_ in parts[1:]:
            steps.append("s%d: %s" % (len(steps) + 1, p_))
        prod = run
    else:
        prod = l * r
        steps.append("s%d: %d*%d=%d" % (len(steps) + 1, l, r, prod))
    return _reduce_chain(prod, p, steps)

def _gen_mod_chain(a, op, b, p):
    steps = []
    if op == "+":
        mid = a + b
        steps.append("s1: %d+%d=%d" % (a, b, mid))
    elif op == "-":
        d = a - b
        steps.append("s1: %d-%d=%d" % (a, b, d))
        if d < 0:
            j = -(-d // p)  # 加 j*p 使非负
            m = d + j * p
            steps.append("s2: %d+%d=%d" % (d, j * p, m))
            d = m
        mid = d
    else:
        return _mulmod_chain(a, b, p, steps), steps
    r = _reduce_chain(mid, p, steps)
    return r, steps

def gen_mod_arith(rng):
    p = rng.choice(MOD_PRIMES)
    fam = rng.choice(["mod", "mod", "pow", "inv"])
    if fam == "mod":
        op = rng.choice(["+", "-", "*"])
        a, b = rng.randrange(0, p * 2), rng.randrange(0, p * 2)
        q = "Compute (%d %s %d) mod %d. Answer the residue." % (a, op, b, p)
        r, steps = _gen_mod_chain(a, op, b, p)
        chk = (a + b) % p if op == "+" else ((a - b) % p if op == "-" else (a * b) % p)
        if chk != r:
            return None
        return {"fam": "mod", "q": q, "cot": "; ".join(steps), "ans": str(r)}
    if fam == "pow":
        a, b = rng.randrange(1, p * 2), rng.randrange(2, 6)
        q = "Compute %d^%d mod %d. Answer the residue." % (a, b, p)
        steps = []
        r = a
        for bit in bin(b)[3:]:  # 从次高位起：平方，遇 1 乘底
            r = _mulmod_chain(r, r, p, steps)
            if bit == "1":
                r = _mulmod_chain(r, a, p, steps)
        chk = pow(a, b, p)
        if chk != r:
            return None
        return {"fam": "pow", "q": q, "cot": "; ".join(steps), "ans": str(r)}
    # inv：费马小定理 a^(p-2) mod p，平方乘链
    if p > INV_P_CAP:
        return None
    a = rng.randrange(1, p)
    q = "Compute the modular inverse of %d mod %d. Answer the residue." % (a, p)
    steps = []
    r = a
    for bit in bin(p - 2)[3:]:
        r = _mulmod_chain(r, r, p, steps)
        if bit == "1":
            r = _mulmod_chain(r, a, p, steps)
    chk = pow(a, p - 2, p)
    if chk != r:
        return None
    return {"fam": "inv", "q": q, "cot": "; ".join(steps), "ans": str(r)}

def _verify_mod_arith(sample):
    q = sample["q"]
    m = re.match(r"Compute \((-?\d+) ([+\-*]) (-?\d+)\) mod (\d+)\.", q)
    if m:
        a, op, b, p = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        r, steps = _gen_mod_chain(a, op, b, p)
        exp = "; ".join(steps)
        return {"ok": sample["ans"] == str(r), "gold": str(r), "cot_ok": sample["cot"] == exp}
    m = re.match(r"Compute (\d+)\^(\d+) mod (\d+)\.", q)
    if m:
        a, b, p = int(m.group(1)), int(m.group(2)), int(m.group(3))
        steps = []
        r = a
        for bit in bin(b)[3:]:
            r = _mulmod_chain(r, r, p, steps)
            if bit == "1":
                r = _mulmod_chain(r, a, p, steps)
        exp = "; ".join(steps)
        return {"ok": sample["ans"] == str(r), "gold": str(r), "cot_ok": sample["cot"] == exp}
    m = re.match(r"Compute the modular inverse of (\d+) mod (\d+)\.", q)
    if m:
        a, p = int(m.group(1)), int(m.group(2))
        steps = []
        r = a
        for bit in bin(p - 2)[3:]:
            r = _mulmod_chain(r, r, p, steps)
            if bit == "1":
                r = _mulmod_chain(r, a, p, steps)
        exp = "; ".join(steps)
        return {"ok": sample["ans"] == str(r), "gold": str(r), "cot_ok": sample["cot"] == exp}
    return {"ok": False, "gold": "?", "cot_ok": False}

# ============ 第三科：plan（exec/goal） ============
PLAN_LETTERS = "ABCDE"

def _plan_parse(q):
    m = re.match(r"Initial stacks: (.+?)\. Moves: (.+?)\. (.+)$", q)
    if not m:
        return None
    stacks = {}
    for part in m.group(1).split(" | "):
        km = re.match(r"S(\d): (.*)", part)
        if not km:
            return None
        stacks[int(km.group(1))] = [] if km.group(2) == "empty" else km.group(2).split(",")
    moves = [(int(s), int(d)) for s, d in re.findall(r"move top of S(\d) to S(\d)", m.group(2))]
    return stacks, moves, m.group(3)

def _plan_chain(stacks, moves, tail):
    st = {k: list(v) for k, v in stacks.items()}
    steps = []
    for i, (src, dst) in enumerate(moves):
        if not st[src]:
            return None
        st[dst].append(st[src].pop())
        fmt = lambda kk: ",".join(st[kk]) if st[kk] else "empty"
        steps.append("m%d: S%d->S%d: S%d=%s" % (i, src, dst, dst, fmt(dst)))
    full = " | ".join("S%d: %s" % (k, ",".join(st[k]) if st[k] else "empty") for k in sorted(st))
    steps.append("final: %s" % full)
    if tail.startswith("Which block"):
        km = re.search(r"top of S(\d) at the end", tail)
        k = int(km.group(1))
        gold = st[k][-1] if st[k] else "empty"
        steps.append("top: S%d=%s" % (k, gold))
    else:
        gm = re.search(r"is block (\w) directly on top of block (\w) at the end", tail)
        x, y = gm.group(1), gm.group(2)
        gold = "no"
        for stk in st.values():
            for i in range(1, len(stk)):
                if stk[i] == x and stk[i - 1] == y:
                    gold = "yes"
        steps.append("check: %s-on-%s=%s" % (x, y, gold))
    return steps, gold

def gen_plan(rng, fam):
    n_st = 3
    blocks = list(PLAN_LETTERS[: rng.choice([4, 5])])
    rng.shuffle(blocks)
    init = {i: [] for i in range(n_st)}
    for b in blocks:
        init[rng.randrange(n_st)].append(b)
    cur = {k: list(v) for k, v in init.items()}
    moves = []
    nm = rng.choice([3, 4, 5, 6])
    for _ in range(nm):
        srcs = [i for i in range(n_st) if cur[i]]
        if not srcs:
            break
        src = rng.choice(srcs)
        dst = rng.choice([i for i in range(n_st) if i != src])
        moves.append((src, dst))
        cur[dst].append(cur[src].pop())
    init_txt = " | ".join("S%d: %s" % (k, ",".join(init[k]) if init[k] else "empty") for k in range(n_st))
    moves_txt = "; ".join("move top of S%d to S%d" % (s, d) for s, d in moves) or "none"
    if fam == "exec":
        k = rng.randrange(n_st)
        gold = cur[k][-1] if cur[k] else "empty"
        q = "Initial stacks: %s. Moves: %s. Which block is on top of S%d at the end? Answer a letter or empty." % (init_txt, moves_txt, k)
    else:
        fin = cur
        pairs = [(x, y) for st in fin.values() for i, x in enumerate(st) for y in ([st[i - 1]] if i > 0 else [])]
        if rng.random() < 0.5 and pairs:
            x, y = rng.choice(pairs)
            gold = "yes"
        else:
            allb = list(PLAN_LETTERS[:5])
            cands = [(x, y) for x in allb for y in allb if x != y and (x, y) not in pairs]
            if not cands:
                return None
            x, y = rng.choice(cands)
            gold = "no"
        q = "Initial stacks: %s. Moves: %s. Goal check: is block %s directly on top of block %s at the end? Answer yes or no." % (init_txt, moves_txt, x, y)
    pr = _plan_chain(init, moves, q.split(". ", 2)[2])
    if pr is None:
        return None
    steps, gold2 = pr
    if gold2 != gold:
        return None
    return {"fam": fam, "q": q, "cot": "; ".join(steps), "ans": gold}

def _verify_plan(sample):
    pr = _plan_parse(sample["q"])
    if pr is None:
        return {"ok": False, "gold": "?", "cot_ok": False}
    stacks, moves, tail = pr
    pr2 = _plan_chain(stacks, moves, tail)
    if pr2 is None:
        return {"ok": False, "gold": "?", "cot_ok": False}
    steps, gold = pr2
    exp = "; ".join(steps)
    return {"ok": sample["ans"] == gold, "gold": gold, "cot_ok": sample["cot"] == exp}

# ============ 第四科：progexec ============
def gen_progexec(rng):
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
    trace = []
    def run(e, collect):
        e = dict(e)
        for si, (i, op, a) in enumerate(stmts):
            if op == "c":
                e[i] = a
            elif op == "add":
                e[i] = e[a[0]] + a[1]
            elif op == "sub":
                e[i] = e[a[0]] - a[1]
            else:
                e[i] = e[i] + a
            if collect is not None:
                collect.append("s%d: x%d=%d" % (si, i, e[i]))
        return e
    env = run(env, trace)
    v = rng.randrange(nv)
    txt = "; ".join(
        "x%d = %d" % (i, a) if op == "c" else
        "x%d = x%d + %d" % (i, a[0], a[1]) if op == "add" else
        "x%d = x%d - %d" % (i, a[0], a[1]) if op == "sub" else
        "x%d = x%d + %d" % (i, i, a)
        for i, op, a in stmts)
    q = "All variables start at 0. Program: %s. What is the final value of x%d? Answer a number." % (txt, v)
    cot = "; ".join(trace) + "; env: " + " ".join("x%d=%d" % (i, env[i]) for i in range(nv))
    return {"fam": "progexec", "q": q, "cot": cot, "ans": str(env[v])}

def _prog_parse_trace(q):
    body = q.split("Program: ")[1].split(". What is")[0]
    env = {0: 0, 1: 0, 2: 0, 3: 0}
    trace = []
    for si, st in enumerate(body.split("; ")):
        m = re.fullmatch(r"x(\d) = (\d+)", st)
        if m:
            env[int(m.group(1))] = int(m.group(2))
            trace.append("s%d: x%d=%d" % (si, int(m.group(1)), env[int(m.group(1))]))
            continue
        m = re.fullmatch(r"x(\d) = x(\d) ([+-]) (\d+)", st)
        if m:
            i, j, o, k = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
            env[i] = env[j] + k if o == "+" else env[j] - k
            trace.append("s%d: x%d=%d" % (si, i, env[i]))
            continue
        return None
    return env, trace

def _verify_progexec(sample):
    pr = _prog_parse_trace(sample["q"])
    if pr is None:
        return {"ok": False, "gold": "?", "cot_ok": False}
    env, trace = pr
    exp = "; ".join(trace) + "; env: " + " ".join("x%d=%d" % (i, env[i]) for i in range(4))
    v = int(re.search(r"final value of x(\d)", sample["q"]).group(1))
    gold = str(env[v])
    return {"ok": sample["ans"] == gold, "gold": gold, "cot_ok": sample["cot"] == exp}

# ============ 第五科：seqind ============
def _seq_parse(q):
    try:
        return [int(x) for x in q.split("Sequence: ")[1].split(".")[0].split(",")]
    except Exception:
        return None

def _seq_chain(fam, seq, d1, d2, ev, od, de, do, pat):
    """从验证器自己算出的统计量构造期望链（与生成器参数通道独立）。"""
    if fam == "ap":
        return "d1: %s; s1: %d+%d=%d" % (
            ",".join([str(d1[0])] * (len(seq) - 1)), seq[-1], d1[0], seq[-1] + d1[0])
    if fam == "gp":
        return "r1: %s; s1: %d*%d=%d" % (
            ",".join([str(seq[1] // seq[0])] * (len(seq) - 1)), seq[-1], seq[1] // seq[0],
            seq[-1] * (seq[1] // seq[0]))
    if fam == "quad":
        nx1 = seq[-1] + d1[-1]
        nx = nx1 + d2[0]
        return "d1: %s; d2: %s; s1: %d+%d=%d; s2: %d+%d=%d" % (
            ",".join(map(str, d1)), ",".join([str(d2[0])] * len(d2)),
            seq[-1], d1[-1], nx1, nx1, d2[0], nx)
    if fam == "fib":
        cs = ["c: %d=%d+%d" % (seq[i], seq[i - 1], seq[i - 2]) for i in range(2, len(seq))]
        nx = seq[-1] + seq[-2]
        cs.append("next: %d+%d=%d" % (seq[-1], seq[-2], nx))
        return "; ".join(cs)
    if fam == "alt":
        nx = od[-1] + do[0]
        return "ev: %s d=%d; od: %s d=%d; next: %d+%d=%d" % (
            ",".join(map(str, ev)), de[0], ",".join(map(str, od)), do[0], od[-1], do[0], nx)
    if fam == "cyc":
        p = len(pat)
        return "cyc: p%d: %s; next=%d" % (p, ",".join(map(str, pat)), pat[0])
    return None

def _seq_infer(seq):
    """盲反推：只看序列本身。返回 (fam, gold, chain) 或 None。"""
    n = len(seq)
    d1 = [seq[i + 1] - seq[i] for i in range(n - 1)]
    if len(set(d1)) == 1:
        return "ap", seq[-1] + d1[0], _seq_chain("ap", seq, d1, None, None, None, None, None, None)
    ok_gp = seq[0] != 0
    if ok_gp:
        rs = []
        for i in range(n - 1):
            if seq[i] == 0 or seq[i + 1] % seq[i] != 0:
                ok_gp = False
                break
            rs.append(seq[i + 1] // seq[i])
    if ok_gp and len(set(rs)) == 1 and all(seq[i + 1] == seq[i] * rs[0] for i in range(n - 1)):
        return "gp", seq[-1] * rs[0], _seq_chain("gp", seq, None, None, None, None, None, None, None)
    if n >= 5:
        d2 = [d1[i + 1] - d1[i] for i in range(n - 2)]
        if len(set(d2)) == 1:
            nx = seq[-1] + d1[-1] + d2[0]
            return "quad", nx, _seq_chain("quad", seq, d1, d2, None, None, None, None, None)
    if n >= 4 and all(seq[i] == seq[i - 1] + seq[i - 2] for i in range(2, n)):
        return "fib", seq[-1] + seq[-2], _seq_chain("fib", seq, None, None, None, None, None, None, None)
    if n >= 7 and n % 2 == 1:
        ev, od = seq[0::2], seq[1::2]
        de = [ev[i + 1] - ev[i] for i in range(len(ev) - 1)]
        do = [od[i + 1] - od[i] for i in range(len(od) - 1)]
        if len(set(de)) == 1 and len(set(do)) == 1:
            nx = od[-1] + do[0]
            return "alt", nx, _seq_chain("alt", seq, None, None, ev, od, de, do, None)
    for p in range(1, n // 2 + 1):
        if all(seq[i] == seq[i - p] for i in range(p, n)):
            return "cyc", seq[-p], _seq_chain("cyc", seq, None, None, None, None, None, None, seq[n - p:])
    return None

def gen_seqind(rng, fam):
    for _ in range(30):
        if fam == "ap":
            a, d = rng.randrange(0, 100), rng.choice([x for x in range(-25, 26) if x != 0])
            seq = [a + d * i for i in range(6)]
        elif fam == "gp":
            a, r = rng.randrange(1, 50), rng.choice([2, 3, 4])
            seq = [a * r ** i for i in range(6)]
            if seq[-1] >= 10 ** 6:
                continue
        elif fam == "quad":
            a2, b, c = rng.choice([1, 2, 3, -1, -2, -3]), rng.randrange(-15, 16), rng.randrange(-15, 16)
            seq = [a2 * i * i + b * i + c for i in range(7)]
        elif fam == "fib":
            x0, x1 = rng.randrange(0, 100), rng.randrange(0, 100)
            if x0 == 0 and x1 == 0:
                continue
            seq = [x0, x1]
            while len(seq) < 7:
                seq.append(seq[-1] + seq[-2])
            if seq[-1] >= 10 ** 5:
                continue
        elif fam == "alt":
            a1, d1_ = rng.randrange(0, 40), rng.choice([x for x in range(-12, 13) if x != 0])
            a2, d2_ = rng.randrange(0, 40), rng.choice([x for x in range(-12, 13) if x != 0])
            seq = [a1 + d1_ * (i // 2) if i % 2 == 0 else a2 + d2_ * (i // 2) for i in range(7)]
        else:  # cyc
            k = rng.choice([3, 4, 5, 6])
            pat = rng.sample(range(12), k)
            if len(set(pat[i + 1] - pat[i] for i in range(k - 1))) == 1:
                continue
            seq = pat * 2
        q = "Sequence: %s. What is the next term? Answer a number." % ",".join(map(str, seq))
        r = _seq_infer(seq)
        if r is None:
            continue
        fam2, gold, chain = r
        if fam2 != fam:
            continue  # 歧义弃样（如 AP 形 quad）
        return {"fam": fam, "q": q, "cot": chain, "ans": str(gold)}
    return None

def _verify_seqind(sample):
    seq = _seq_parse(sample["q"])
    if not seq:
        return {"ok": False, "gold": "?", "cot_ok": False, "fam_match": False}
    r = _seq_infer(seq)
    if r is None:
        return {"ok": False, "gold": "?", "cot_ok": False, "fam_match": False}
    fam, gold, chain = r
    return {"ok": sample["ans"] == str(gold), "gold": str(gold),
            "cot_ok": sample["cot"] == chain, "fam_match": fam == sample.get("fam", fam)}

# ============ 统一接口 ============
GENS = {
    "math": lambda rng: MATH_GEN[rng.choice(["lin1", "lin2", "arith"])](rng),
    "mod_arith": gen_mod_arith,
    "plan": lambda rng: gen_plan(rng, rng.choice(["exec", "goal"])),
    "progexec": gen_progexec,
    "seqind": lambda rng: gen_seqind(rng, rng.choice(["ap", "gp", "quad", "fib", "alt", "cyc"])),
}
VERIF = {"math": _verify_math, "mod_arith": _verify_mod_arith, "plan": _verify_plan,
         "progexec": _verify_progexec, "seqind": _verify_seqind}

def _poison_math_struct(sm):
    m = re.match(r"Evaluate: (.+)\. Answer", sm["q"])
    if m:
        leaves = [(mm.group(0), mm.start()) for mm in re.finditer(r"\d+", m.group(1))]
        if not leaves:
            return None
        old, start = leaves[0]
        expr2 = m.group(1)[:start] + str(int(old) + 1) + m.group(1)[start + len(old):]
        p = dict(sm)
        p["q"] = "Evaluate: %s. Answer a number." % expr2
        return p if not _verify_math(p)["ok"] else None
    m = re.match(r"Solve for x: (.+?) = (-?\d+)\.", sm["q"])
    if not m:
        return None
    p = dict(sm)
    p["q"] = "Solve for x: %s = %d." % (m.group(1), int(m.group(2)) + 1)
    return p if not _verify_math(p)["ok"] else None

def _poison_mod_struct(sm):
    m = re.match(r"Compute \((-?\d+) ([+\-*]) (-?\d+)\) mod (\d+)\.", sm["q"])
    if not m:
        return None
    for dv in (1, -1):
        p = dict(sm)
        p["q"] = "Compute (%d %s %d) mod %d. Answer the residue." % (
            int(m.group(1)) + dv, m.group(2), int(m.group(3)), int(m.group(4)))
        if not _verify_mod_arith(p)["ok"]:
            return p
    return None

def _poison_plan_struct(sm):
    m = re.match(r"(Initial stacks: .+?\. Moves: )(.+?)(\. .+)$", sm["q"])
    if not m:
        return None
    mv = re.findall(r"move top of S(\d) to S(\d)", m.group(2))
    for si in range(len(mv)):
        for d2 in range(3):
            if d2 == int(mv[si][0]):
                continue
            mv2 = list(mv)
            mv2[si] = (mv[si][0], str(d2))
            p = dict(sm)
            p["q"] = m.group(1) + "; ".join("move top of S%s to S%s" % t for t in mv2) + m.group(3)
            if not _verify_plan(p)["ok"]:
                return p
    return None

def _poison_prog_struct(sm):
    head = sm["q"].split("Program: ")[0] + "Program: "
    body = sm["q"].split("Program: ")[1].split(". What is")[0]
    tail = ". What is" + sm["q"].split(". What is")[1]
    for si, st in enumerate(body.split("; ")):
        m = re.fullmatch(r"x(\d) = (x\d [+-] )?(\d+)", st)
        if not m:
            continue
        for dk in (-1, 1):
            k2 = int(m.group(3)) + dk
            if k2 < 0:
                continue
            stmts = body.split("; ")
            stmts[si] = ("x%s = %d" % (m.group(1), k2)) if not m.group(2) else ("x%s = %s%d" % (m.group(1), m.group(2), k2))
            p = dict(sm)
            p["q"] = head + "; ".join(stmts) + tail
            if not _verify_progexec(p)["ok"]:
                return p
    return None

def _poison_seq_struct(sm):
    seq = _seq_parse(sm["q"])
    if not seq:
        return None
    seq2 = list(seq)
    seq2[2] = seq2[2] + 3
    p = dict(sm)
    p["q"] = "Sequence: %s. What is the next term? Answer a number." % ",".join(map(str, seq2))
    v = _verify_seqind(p)
    return p if not (v["ok"] and v["cot_ok"]) else None

POISON_STRUCT = {
    "math": _poison_math_struct,
    "mod_arith": _poison_mod_struct,
    "plan": _poison_plan_struct,
    "progexec": _poison_prog_struct,
    "seqind": _poison_seq_struct,
}

def gen_sample(rng, subject):
    """带预算闸 + 内嵌通道 B 复核的统一采样入口。"""
    for _ in range(30):
        sm = GENS[subject](rng)
        if sm is None:
            continue
        if len(sm["cot"]) > CHAIN_CAP or len(sm["q"]) + len(sm["cot"]) + len(sm["ans"]) + 10 > BUDGET:
            continue
        v = VERIF[subject](sm)
        if not (v["ok"] and v["cot_ok"]):
            continue
        if subject == "seqind" and not v.get("fam_match", True):
            continue
        return sm
    return None

def self_test(subject, n=200):
    r = {}
    bad = total = 0
    lens = []
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        for _ in range(n):
            sm = gen_sample(rng, subject)
            if sm is None:
                continue
            total += 1
            lens.append(len(sm["cot"]))
            v = VERIF[subject](sm)
            if not (v["ok"] and v["cot_ok"]):
                bad += 1
    r["G1_dual_channel"] = {"total": total, "bad": bad, "avg_cot_bytes": sum(lens) // max(1, len(lens)),
                            "max_cot_bytes": max(lens) if lens else 0, "pass": bad == 0}
    caught = total2 = 0
    rng = random.Random(99)
    for _ in range(100):
        sm = gen_sample(rng, subject)
        if sm is None:
            continue
        # 毒1：答案 ±1（或文本答案翻转）
        p1 = dict(sm)
        try:
            p1["ans"] = str(int(sm["ans"]) + 1)
        except ValueError:
            p1["ans"] = "no" if sm["ans"] == "yes" else "yes"
        total2 += 1
        if not VERIF[subject](p1)["ok"]:
            caught += 1
        # 毒2：链尾篡改
        total2 += 1
        p2 = dict(sm)
        p2["cot"] = sm["cot"] + " -> tampered"
        if not VERIF[subject](p2)["cot_ok"]:
            caught += 1
        # 毒3：结构变异（q 变 → 链必须跟着破）
        p3 = POISON_STRUCT[subject](sm)
        if p3 is None:
            continue
        total2 += 1
        v3 = VERIF[subject](p3)
        if not (v3["ok"] and v3["cot_ok"]):
            caught += 1
    r["G2_poison_failclosed"] = {"total": total2, "caught": caught, "pass": caught == total2}
    def corpus(seed):
        rng = random.Random(seed)
        out = []
        while len(out) < 50:
            sm = gen_sample(rng, subject)
            if sm:
                out.append(sm)
        return json.dumps(out, sort_keys=True)
    r["G3_determinism"] = {"pass": corpus(7) == corpus(7) and corpus(7) != corpus(8)}
    r["G4_trainval_split"] = {"pass": corpus(1001) != corpus(2002)}
    r["ALL_PASS"] = all(v.get("pass") for k, v in r.items() if k.startswith("G"))
    return r

def emit_samples(subject, n_total, val_ratio, seed):
    """产 n_total 条，val_ratio 比例进 val。返回 (train_rows, val_rows) 的 q/a 形态。"""
    rng = random.Random(seed)
    rows = []
    while len(rows) < n_total:
        sm = gen_sample(rng, subject)
        if sm is None:
            continue
        a = sm["cot"] + " || " + sm["ans"]
        rows.append({"subject": subject, "fam": sm.get("fam", ""),
                     "q": sm["q"], "a": a, "cot": sm["cot"], "ans": sm["ans"]})
    n_val = max(20, int(n_total * val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    return train_rows, val_rows

if __name__ == "__main__":
    allpass = True
    for subj in ("math", "mod_arith", "plan", "progexec", "seqind"):
        res = self_test(subj)
        allpass = allpass and res["ALL_PASS"]
        print("=====", subj, json.dumps(res, ensure_ascii=False))
        rng = random.Random(7)
        sm = gen_sample(rng, subj)
        print("  Q:", sm["q"][:120])
        print("  A:", sm["a"] if "a" in sm else (sm["cot"] + " || " + sm["ans"])[:260])
    print("FMT5 ALL_PASS:", allpass)
    sys.exit(0 if allpass else 2)
