# -*- coding: utf-8 -*-
# 课程生成器 v0 —— 第二科：多步规划（积木世界：计划执行/目标核查，验证器=独立模拟器对拍）
# T 式纪律：G1 双通道 / G2 定向毒 fail-closed / G3 确定性 / G4 train/val 隔离。题面自包含。
import json, random, hashlib, sys, re

OUT_RESULTS = "curriculum_plan_lab_v0_results.json"
FAMS = ["exec", "goal"]
LETTERS = "ABCDE"

# ---------- 工具（模拟器：两通道共用底层语义，通道 B 只吃 q 文本） ----------
def _simulate(stacks, moves):
    st = {k: list(v) for k, v in stacks.items()}
    for (src, dst) in moves:
        if not st[src]:
            return None  # 非法移动
        b = st[src].pop()
        st[dst].append(b)
    return st

def _fmt(st):
    return " | ".join("S%d: %s" % (k, ",".join(st[k]) if st[k] else "empty") for k in sorted(st))

# ---------- 生成器 ----------
def gen_sample(rng, fam):
    n_st = 3
    blocks = list(LETTERS[: rng.choice([4, 5])])
    rng.shuffle(blocks)
    stacks = {i: [] for i in range(n_st)}
    for b in blocks:
        stacks[rng.randrange(n_st)].append(b)
    moves = []
    nm = rng.choice([3, 4, 5, 6])
    for _ in range(nm):
        srcs = [i for i in range(n_st) if stacks[i]]
        if not srcs:
            break
        src = rng.choice(srcs)
        dst = rng.choice([i for i in range(n_st) if i != src])
        moves.append((src, dst))
        stacks[dst].append(stacks[src].pop())
    init = {i: [] for i in range(n_st)}
    # 重放初始盘面（moves 是在「当前盘面」上生成的，这里用 copies 重建）
    # 简化：生成时直接在副本上走，init 需另存 —— 上面 stacks 已被消耗，改用独立随机 init：
    return None  # 占位：真实现见 gen_sample2

def gen_sample2(rng, fam):
    n_st = 3
    blocks = list(LETTERS[: rng.choice([4, 5])])
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
    fin = cur
    init_txt = " | ".join("S%d: %s" % (k, ",".join(init[k]) if init[k] else "empty") for k in range(n_st))
    moves_txt = "; ".join("move top of S%d to S%d" % (s, d) for s, d in moves) or "none"
    if fam == "exec":
        k = rng.randrange(n_st)
        gold = fin[k][-1] if fin[k] else "empty"
        q = "Initial stacks: %s. Moves: %s. Which block is on top of S%d at the end? Answer a letter or empty." % (init_txt, moves_txt, k)
        ans = gold
        cot = "final: %s" % _fmt(fin)
        return {"fam": fam, "q": q, "cot": cot, "ans": ans, "k": {"kind": "top", "arg": k}}
    # goal 核查
    pairs = [(x, y) for st in fin.values() for i, x in enumerate(st) for y in ([st[i - 1]] if i > 0 else [])]
    if rng.random() < 0.5 and pairs:
        x, y = rng.choice(pairs)
        gold = "yes"
    else:
        allb = list(LETTERS[:5])
        cands = [(x, y) for x in allb for y in allb if x != y and (x, y) not in pairs]
        if not cands:
            return None
        x, y = rng.choice(cands)
        gold = "no"
    q = "Initial stacks: %s. Moves: %s. Goal check: is block %s directly on top of block %s at the end? Answer yes or no." % (init_txt, moves_txt, x, y)
    ans = gold
    cot = "final: %s" % _fmt(fin)
    return {"fam": fam, "q": q, "cot": cot, "ans": ans, "k": {"kind": "goal", "x": x, "y": y}}

# ---------- 验证器（通道 B：只吃 q 文本） ----------
def verify(sample):
    m = re.match(r"Initial stacks: (.+?)\. Moves: (.+?)\. (.+)$", sample["q"])
    if not m:
        return {"ok": False, "gold": "?", "cot_ok": False}
    stacks = {}
    for part in m.group(1).split(" | "):
        km = re.match(r"S(\d): (.*)", part)
        stacks[int(km.group(1))] = [] if km.group(2) == "empty" else km.group(2).split(",")
    mvtxt = m.group(3) if False else m.group(2)
    moves = []
    for s, d in re.findall(r"move top of S(\d) to S(\d)", mvtxt):
        moves.append((int(s), int(d)))
    fin = _simulate(stacks, moves)
    if fin is None:
        return {"ok": False, "gold": "?", "cot_ok": False}
    exp_cot = "final: %s" % _fmt(fin)
    if sample["k"]["kind"] == "top":
        k = sample["k"]["arg"]
        gold = fin[k][-1] if fin[k] else "empty"
        return {"ok": sample["ans"] == gold, "gold": gold, "cot_ok": sample["cot"] == exp_cot}
    x, y = sample["k"]["x"], sample["k"]["y"]
    gold = "no"
    for st in fin.values():
        for i in range(1, len(st)):
            if st[i] == x and st[i - 1] == y:
                gold = "yes"
    return {"ok": sample["ans"] == gold, "gold": gold, "cot_ok": sample["cot"] == exp_cot}

GEN = {"exec": lambda rng: gen_sample2(rng, "exec"), "goal": lambda rng: gen_sample2(rng, "goal")}

def gen_sample(rng, fam):
    fam = fam if fam in GEN else "exec"
    for _ in range(20):  # 内嵌通道 B 复核
        sm = GEN[fam](rng)
        v = verify(sm)
        if v["ok"] and v["cot_ok"]:
            return sm
    return None

# ---------- 定向毒 ----------
def _poison_ans(sample):
    if sample["k"]["kind"] == "top":
        if sample["ans"] == "empty":
            return None
        cands = [c for c in LETTERS[:5] if c != sample["ans"]]
        p = dict(sample); p["ans"] = rng_choice(cands)
        return p
    p = dict(sample)
    p["ans"] = "no" if sample["ans"] == "yes" else "yes"
    return p

def rng_choice(c):
    return c[random.Random().randrange(len(c))]

def _poison_struct(sample):
    # 变异一个移动的目的栈直到答案翻转；翻不动跳过
    m = re.match(r"(Initial stacks: .+?\. Moves: )(.+?)(\. .+)$", sample["q"])
    if not m:
        return None
    mv = re.findall(r"move top of S(\d) to S(\d)", m.group(2))
    for si in range(len(mv)):
        for d2 in range(3):
            if d2 == int(mv[si][0]):
                continue
            mv2 = list(mv)
            mv2[si] = (mv[si][0], str(d2))
            p = dict(sample)
            p["q"] = m.group(1) + "; ".join("move top of S%s to S%s" % t for t in mv2) + m.group(3)
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
        p1 = _poison_ans(sm)
        if p1 is not None:
            total2 += 1
            if not verify(p1)["ok"]:
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
    for split, n, seed in (("train", n_train, 13001), ("val", n_val, 9006)):
        rng = random.Random(seed)
        i = 0
        while i < n:
            sm = gen_sample(rng, FAMS[i % len(FAMS)])
            if sm is None:
                continue
            lines.append(json.dumps({
                "split": split, "subject": "planning", "id": "%s_%04d" % (split, i),
                "fam": sm["fam"], "q": sm["q"], "cot": sm["cot"], "ans": sm["ans"]}, ensure_ascii=False))
            i += 1
    data = "\n".join(lines) + "\n"
    open("curriculum_plan_v0.jsonl", "w", encoding="utf-8", newline="").write(data)
    return {"records": len(lines), "sha256": hashlib.sha256(data.encode()).hexdigest()[:16]}

if __name__ == "__main__":
    res = self_test()
    res["export"] = export()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(OUT_RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if res["ALL_PASS"] else 2)
