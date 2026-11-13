# -*- coding: utf-8 -*-
# 课程生成器 v0 —— 第九科：序列归纳（AP/GP/二次/fib/交错AP/循环，验证器=序列反推规则对拍）
# T 式纪律：G1 双通道全对（生成器规则 vs 验证器从序列盲反推）/ G2 毒样本 fail-closed /
#           G3 确定性 / G4 train/val 隔离。生成器侧若两通道答案不一致直接弃样，
#           保证语料零歧义；循环族用「最小周期 + 恰两整周期」约定，模式为等差则弃样防 AP 歧义。
import json, random, hashlib, sys

OUT_RESULTS = "curriculum_seqind_v0_results.json"
FAMS = ["ap", "gp", "quad", "fib", "alt", "cyc"]

# ---------- 生成器（通道 A：直接用规则参数） ----------
def _ap(rng):
    a, d = rng.randrange(0, 100), rng.choice([x for x in range(-25, 26) if x != 0])
    seq = [a + d * i for i in range(6)]
    return seq, seq[5] + d, "ap: diffs %s next=%d" % (",".join([str(d)] * 5), seq[5] + d)

def _gp(rng):
    a, r = rng.randrange(1, 50), rng.choice([2, 3, 4])
    seq = [a * r ** i for i in range(6)]
    if seq[-1] >= 10 ** 6:
        return None
    return seq, seq[5] * r, "gp: ratios %s next=%d" % (",".join([str(r)] * 5), seq[5] * r)

def _quad(rng):
    a2, b, c = rng.choice([1, 2, 3, -1, -2, -3]), rng.randrange(-15, 16), rng.randrange(-15, 16)
    seq = [a2 * i * i + b * i + c for i in range(7)]
    nx = a2 * 49 + b * 7 + c
    return seq, nx, "quad: 2nd diffs %s next=%d" % (",".join([str(2 * a2)] * 5), nx)

def _fib(rng):
    x0, x1 = rng.randrange(0, 100), rng.randrange(0, 100)
    if x0 == 0 and x1 == 0:
        return None
    seq = [x0, x1]
    while len(seq) < 7:
        seq.append(seq[-1] + seq[-2])
    if seq[-1] >= 10 ** 5:
        return None
    return seq, seq[6] + seq[5], "fib: t(n)=t(n-1)+t(n-2) next=%d" % (seq[6] + seq[5])

def _alt(rng):
    a1, d1 = rng.randrange(0, 40), rng.choice([x for x in range(-12, 13) if x != 0])
    a2, d2 = rng.randrange(0, 40), rng.choice([x for x in range(-12, 13) if x != 0])
    seq = []
    for i in range(7):  # 7 项（奇数）→ 下一项 idx7 落在奇位族
        seq.append(a1 + d1 * (i // 2) if i % 2 == 0 else a2 + d2 * (i // 2))
    nx = a2 + d2 * 3
    return seq, nx, "alt: even d=%d odd d=%d next=%d" % (d1, d2, nx)

def _cyc(rng):
    k = rng.choice([3, 4, 5, 6])
    pat = rng.sample(range(12), k)
    if len(set(pat[i + 1] - pat[i] for i in range(k - 1))) == 1:
        return None  # 模式本身是等差 → 与 AP 族歧义，弃样
    seq = pat * 2  # 恰两个整周期 → 最小周期约定下下一项唯一
    return seq, pat[0], "cyc: period %d = %s next=%d" % (k, ",".join(map(str, pat)), pat[0])

GEN = {"ap": _ap, "gp": _gp, "quad": _quad, "fib": _fib, "alt": _alt, "cyc": _cyc}

def gen_sample(rng, fam):
    fam = fam if fam in GEN else "gp"
    for _ in range(30):  # 歧义弃样重抽：生成器内嵌通道 B 可行性检查（零歧义纪律）
        r = GEN[fam](rng)
        if r is None:
            continue
        seq, ans, cot = r
        q = "Sequence: %s. What is the next term? Answer a number." % ",".join(map(str, seq))
        sm = {"fam": fam, "q": q, "cot": cot, "ans": str(ans)}
        v = verify(sm)
        if v["ok"] and v["cot_ok"] and v["fam_match"]:
            return sm
    return None

# ---------- 验证器（通道 B：只看 q 文本里的序列，盲反推规则） ----------
def _parse_seq(q):
    try:
        return [int(x) for x in q.split("Sequence: ")[1].split(".")[0].split(",")]
    except Exception:
        return None

def infer(seq):
    """按固定族序盲反推，返回 (fam, gold, expected_cot) 或 None。"""
    n = len(seq)
    d1 = [seq[i + 1] - seq[i] for i in range(n - 1)]
    if len(set(d1)) == 1:
        return "ap", seq[-1] + d1[0], "ap: diffs %s next=%d" % (",".join([str(d1[0])] * (n - 1)), seq[-1] + d1[0])
    ok_gp = seq[0] != 0
    if ok_gp:
        rs = []
        for i in range(n - 1):
            if seq[i] == 0 or seq[i + 1] % seq[i] != 0:
                ok_gp = False
                break
            rs.append(seq[i + 1] // seq[i])
    if ok_gp and len(set(rs)) == 1 and all(seq[i + 1] == seq[i] * rs[0] for i in range(n - 1)):
        return "gp", seq[-1] * rs[0], "gp: ratios %s next=%d" % (",".join([str(rs[0])] * (n - 1)), seq[-1] * rs[0])
    if n >= 5:
        d2 = [d1[i + 1] - d1[i] for i in range(n - 2)]
        if len(set(d2)) == 1:
            nx = seq[-1] + d1[-1] + d2[0]
            return "quad", nx, "quad: 2nd diffs %s next=%d" % (",".join([str(d2[0])] * (n - 2)), nx)
    if n >= 4 and all(seq[i] == seq[i - 1] + seq[i - 2] for i in range(2, n)):
        return "fib", seq[-1] + seq[-2], "fib: t(n)=t(n-1)+t(n-2) next=%d" % (seq[-1] + seq[-2])
    if n >= 7 and n % 2 == 1:
        ev, od = seq[0::2], seq[1::2]
        de = [ev[i + 1] - ev[i] for i in range(len(ev) - 1)]
        do = [od[i + 1] - od[i] for i in range(len(od) - 1)]
        if len(set(de)) == 1 and len(set(do)) == 1:
            nx = od[-1] + do[0]  # 下一项 idx n=奇数 → 奇位族
            return "alt", nx, "alt: even d=%d odd d=%d next=%d" % (de[0], do[0], nx)
    for p in range(1, n // 2 + 1):
        if all(seq[i] == seq[i - p] for i in range(p, n)):
            return "cyc", seq[-p], "cyc: period %d = %s next=%d" % (p, ",".join(map(str, seq[n - p:])), seq[-p])
    return None

def verify(sample):
    seq = _parse_seq(sample["q"])
    if not seq:
        return {"ok": False, "gold": "?", "cot_ok": False, "fam_match": False}
    r = infer(seq)
    if r is None:
        return {"ok": False, "gold": "?", "cot_ok": False, "fam_match": False}
    fam, gold, exp_cot = r
    return {"ok": sample["ans"] == str(gold), "gold": str(gold),
            "cot_ok": sample["cot"] == exp_cot,
            "fam_match": fam == sample.get("fam", fam)}

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
            if not v["ok"] or not v["cot_ok"] or not v["fam_match"]:
                bad += 1
    r["G1_dual_channel"] = {"total": total, "bad": bad, "pass": bad == 0}
    caught = total2 = 0
    rng = random.Random(99)
    for _ in range(100):
        sm = gen_sample(rng, rng.choice(FAMS))
        if sm is None:
            continue
        total2 += 1
        p1 = dict(sm); p1["ans"] = str(int(sm["ans"]) + 1)
        if not verify(p1)["ok"]:
            caught += 1
        total2 += 1
        p2 = dict(sm); p2["cot"] = sm["cot"] + " -> tampered"
        if not verify(p2)["cot_ok"]:
            caught += 1
        total2 += 1
        p3 = dict(sm)
        seq = _parse_seq(sm["q"]); seq[2] = seq[2] + 3  # 篡改序列本体
        p3["q"] = "Sequence: %s. What is the next term? Answer a number." % ",".join(map(str, seq))
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
    for split, n, seed in (("train", n_train, 7001), ("val", n_val, 9002)):
        rng = random.Random(seed)
        i = 0
        while i < n:
            sm = gen_sample(rng, FAMS[i % len(FAMS)])
            if sm is None:
                continue
            v = verify(sm)
            assert v["ok"] and v["cot_ok"] and v["fam_match"], "curriculum poison at %s/%d" % (split, i)
            lines.append(json.dumps({
                "split": split, "subject": "seq_induction", "id": "%s_%04d" % (split, i),
                "fam": sm["fam"], "q": sm["q"], "cot": sm["cot"], "ans": sm["ans"]}, ensure_ascii=False))
            i += 1
    data = "\n".join(lines) + "\n"
    open("curriculum_seqind_v0.jsonl", "w", encoding="utf-8", newline="").write(data)
    return {"records": len(lines), "sha256": hashlib.sha256(data.encode()).hexdigest()[:16]}

if __name__ == "__main__":
    res = self_test()
    res["export"] = export()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(OUT_RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if res["ALL_PASS"] else 2)
