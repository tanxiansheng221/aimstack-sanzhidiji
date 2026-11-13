# -*- coding: utf-8 -*-
# fmt2_generators.py — 五科破零 v2：课程分期（T1/T2/T3）+ 答案位反转 + 单素数钻攻
# 关键复用：fmt5 的链格式与盲验证器对数字范围不敏感（verify 从 q 盲重建），
#          fmt2 只改采样范围（tier）+ 链尾加位反转附注（verify 剥附注+独立验反转）。
# 纪律同 fmt5：G1 双通道 / G2 毒 fail-closed / G3 确定性 / G4 隔离 / 字节预算闸。
import json, random, re, sys, io

sys.path.insert(0, r"H:\AimCode\lab\cpu-native")
import fmt5_generators as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CHAIN_CAP = 300
BUDGET = 480

def _rev(ans_str):
    """答案位反转：-48 -> '84-'（符号挪尾，数字逆序）。"""
    s = ans_str
    sign = ""
    if s.startswith("-"):
        sign = "-"
        s = s[1:]
    return s[::-1] + sign

def _with_rev(sm):
    """math/mod_arith 链尾附位反转注（对齐进位生成方向，Lee et al. 处方）。"""
    a = sm["ans"]
    if re.fullmatch(r"-?\d+", a):
        sm = dict(sm)
        sm["cot"] = sm["cot"] + " (rev %s)" % _rev(a)
    return sm

def _strip_rev(cot):
    m = re.match(r"^(.*) \(rev (-?[\d-]+)\)$", cot)
    if m:
        return m.group(1), m.group(2)
    return cot, None

# ---------- T1/T2/T3 采样范围 ----------
def gen_math(rng, tier):
    if tier == "T1":
        x0 = rng.randrange(0, 10)
        a = rng.randrange(1, 10)
        b = rng.randrange(0, 10)
        c = a * x0 + b
        q = "Solve for x: %dx + %d = %d. Answer the value of x." % (a, b, c)
        s = c - b
        cot = "s1: %d-%d=%d; s2: %d*%d=%d; x=%d" % (c, b, s, a, x0, a * x0, x0)
        return {"fam": "lin1", "q": q, "cot": cot, "ans": str(x0)}
    if tier == "T2":
        x0 = rng.randrange(-30, 31)
        a = rng.choice([x for x in range(-12, 13) if x != 0])
        b = rng.randrange(0, 50)
        c = a * x0 + b
        if c < -999 or c > 999:
            return None
        q = "Solve for x: %dx + %d = %d. Answer the value of x." % (a, b, c)
        s = c - b
        cot = "s1: %d-%d=%d; s2: %d*%d=%d; x=%d" % (c, b, s, a, x0, a * x0, x0)
        return {"fam": "lin1", "q": q, "cot": cot, "ans": str(x0)}
    # T3：直接用 fmt5 原采样（含 lin2/arith 全深度）
    return F.MATH_GEN[rng.choice(["lin1", "lin2", "arith"])](rng)

def gen_mod(rng, tier, primes=None):
    p = rng.choice(primes if primes else F.MOD_PRIMES)
    if tier == "T1":
        a, b = rng.randrange(0, p), rng.randrange(0, p)   # 操作数 < p，商大多 0/1
    else:
        a, b = rng.randrange(0, p * 2), rng.randrange(0, p * 2)
    op = rng.choice(["+", "-", "*"])
    r, steps = F._gen_mod_chain(a, op, b, p)
    chk = (a + b) % p if op == "+" else ((a - b) % p if op == "-" else (a * b) % p)
    if chk != r:
        return None
    q = "Compute (%d %s %d) mod %d. Answer the residue." % (a, op, b, p)
    return {"fam": "mod", "q": q, "cot": "; ".join(steps), "ans": str(r)}

MOD_TIERS = {"T1": [7, 11], "T2": [13, 17, 19, 23], "T3": None}  # None=全素数池

def gen_plan(rng, tier):
    n_st = 3
    n_blocks = {"T1": 3, "T2": 4, "T3": 5}[tier]
    nm_choices = {"T1": [2, 3], "T2": [3, 4], "T3": [4, 5, 6]}[tier]
    blocks = list(F.PLAN_LETTERS[:n_blocks])
    rng.shuffle(blocks)
    init = {i: [] for i in range(n_st)}
    for b in blocks:
        init[rng.randrange(n_st)].append(b)
    cur = {k: list(v) for k, v in init.items()}
    moves = []
    nm = rng.choice(nm_choices)
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
    fam = rng.choice(["exec", "goal"])
    if fam == "exec":
        k = rng.randrange(n_st)
        gold = cur[k][-1] if cur[k] else "empty"
        q = "Initial stacks: %s. Moves: %s. Which block is on top of S%d at the end? Answer a letter or empty." % (init_txt, moves_txt, k)
    else:
        pairs = [(x, y) for st in cur.values() for i, x in enumerate(st) for y in ([st[i - 1]] if i > 0 else [])]
        if rng.random() < 0.5 and pairs:
            x, y = rng.choice(pairs); gold = "yes"
        else:
            allb = list(F.PLAN_LETTERS[:5])
            cands = [(x, y) for x in allb for y in allb if x != y and (x, y) not in pairs]
            if not cands:
                return None
            x, y = rng.choice(cands); gold = "no"
        q = "Initial stacks: %s. Moves: %s. Goal check: is block %s directly on top of block %s at the end? Answer yes or no." % (init_txt, moves_txt, x, y)
    pr = F._plan_chain(init, moves, q.split(". ", 2)[2])
    if pr is None:
        return None
    steps, gold2 = pr
    if gold2 != gold:
        return None
    return {"fam": fam, "q": q, "cot": "; ".join(steps), "ans": gold}

def gen_progexec(rng, tier):
    n_stmt = {"T1": rng.choice([3, 4]), "T2": rng.choice([5, 6]), "T3": rng.choice([7, 8])}[tier]
    nv = 4
    env = {i: 0 for i in range(nv)}
    stmts = []
    for _ in range(n_stmt):
        i = rng.randrange(nv)
        r = rng.random()
        if r < 0.25:
            stmts.append((i, "c", rng.randrange(0, 10)))
        elif r < 0.5:
            stmts.append((i, "add", (rng.randrange(nv), rng.randrange(1, 6))))
        elif r < 0.75:
            stmts.append((i, "sub", (rng.randrange(nv), rng.randrange(1, 6))))
        else:
            stmts.append((i, "selfadd", rng.randrange(1, 6)))
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

def gen_seqind(rng, tier):
    fams = {"T1": ["ap"], "T2": ["ap", "fib", "alt"], "T3": ["ap", "gp", "quad", "fib", "alt", "cyc"]}[tier]
    return F.gen_seqind(rng, rng.choice(fams))

def gen_sample(rng, subject, tier, mod_primes=None):
    """fmt2 统一采样：tier 约束范围 + math/mod 链尾位反转附注 + fmt5 盲验证。"""
    for _ in range(30):
        if subject == "math":
            sm = gen_math(rng, tier)
        elif subject == "mod_arith":
            sm = gen_mod(rng, tier, mod_primes)
        elif subject == "plan":
            sm = gen_plan(rng, tier)
        elif subject == "progexec":
            sm = gen_progexec(rng, tier)
        else:
            sm = gen_seqind(rng, tier)
        if sm is None:
            continue
        if len(sm["cot"]) > CHAIN_CAP or len(sm["q"]) + len(sm["cot"]) + len(sm["ans"]) + 10 > BUDGET:
            continue
        if subject in ("math", "mod_arith"):
            sm = _with_rev(sm)
        # 盲验证：剥附注后用 fmt5 验证器（从 q 重建链），再独立验反转一致性
        cot_body, rev = _strip_rev(sm["cot"])
        probe = dict(sm); probe["cot"] = cot_body
        v = F.VERIF[subject](probe)
        if not (v["ok"] and v["cot_ok"]):
            continue
        if rev is not None and rev != _rev(sm["ans"]):
            continue
        if subject == "seqind":
            v2 = F._verify_seqind(probe)
            if not v2.get("fam_match", True):
                continue
        return sm
    return None

def self_test():
    ok = True
    for subj, tiers in (("math", ["T1", "T2", "T3"]), ("mod_arith", ["T1", "T2", "T3"]),
                        ("plan", ["T1", "T2", "T3"]), ("progexec", ["T1", "T2", "T3"]),
                        ("seqind", ["T1", "T2", "T3"])):
        for tier in tiers:
            rng = random.Random(hash((subj, tier)) % (2**31))
            bad = total = 0
            for _ in range(120):
                sm = gen_sample(rng, subj, tier)
                if sm is None:
                    continue
                total += 1
                cot_body, rev = _strip_rev(sm["cot"])
                probe = dict(sm); probe["cot"] = cot_body
                v = F.VERIF[subj](probe)
                if not (v["ok"] and v["cot_ok"]):
                    bad += 1
                elif rev is not None and rev != _rev(sm["ans"]):
                    bad += 1
            st = "PASS" if bad == 0 and total >= 60 else "FAIL"
            if st == "FAIL":
                ok = False
            print("%-10s %s total=%d bad=%d %s" % (subj, tier, total, bad, st), flush=True)
            if total:
                print("   sample:", sm["q"][:90], "||", sm["cot"][:150])
    print("FMT2 SELFTEST", "ALL_PASS" if ok else "FAIL")
    return ok

if __name__ == "__main__":
    sys.exit(0 if self_test() else 2)
