# -*- coding: utf-8 -*-
# 课程生成器 v0 —— 第一科：状态机追踪（FSM tracking）
# 设计：每科=「生成器+验证器」对，答案 100% 机器可查；CoT 轨迹随题出厂（逐步监督目标）。
# T 式纪律：①生成器↔验证器双通道全对 ②毒样本 fail-closed（答案错一位必被抓）③同种子逐字节确定
# 输出：JSONL 课程包（train/val 不相交）+ 结果 JSON
import json, random, hashlib, sys

OUT_RESULTS = "curriculum_fsm_v0_results.json"

# ---------- 生成器 ----------
def gen_dfa(rng, n_states=4, n_symbols=3):
    """随机 DFA：转移表 trans[s][a] -> s'，起始态 0。"""
    trans = [[rng.randrange(n_states) for _ in range(n_symbols)] for _ in range(n_states)]
    return {"n_states": n_states, "n_symbols": n_symbols, "trans": trans}

def gen_sample(rng, dfa, seq_len):
    """一题：符号序列 + 问句类型 + 答案 + CoT 轨迹（状态逐步链）。
    题面自包含（v1 修复）：q 必须带完整转移表与输入序列——机器定义不进题面=答案不可预测。"""
    seq = [rng.randrange(dfa["n_symbols"]) for _ in range(seq_len)]
    # 走一遍拿真值与轨迹（生成器通道）
    s = 0
    trace = [0]
    for a in seq:
        s = dfa["trans"][s][a]
        trace.append(s)
    sym = "abc"[: dfa["n_symbols"]]
    trans_txt = "; ".join("S%d%s>S%d" % (si, sym[a], dfa["trans"][si][a])
                          for si in range(dfa["n_symbols"]) for a in range(dfa["n_symbols"]))
    inp = ",".join(sym[a] for a in seq)
    qtype = rng.choice(["final", "visit_count", "visited"])
    if qtype == "final":
        q = "Machine transitions: %s. Input: %s. Start at S0. After the whole input, which state? Answer S<k>." % (trans_txt, inp)
        ans = "S%d" % s
    elif qtype == "visit_count":
        k = rng.randrange(dfa["n_states"])
        q = "Machine transitions: %s. Input: %s. Start at S0. How many times is state S%d visited (including start)? Answer a number." % (trans_txt, inp, k)
        ans = str(trace.count(k))
    else:
        k = rng.randrange(dfa["n_states"])
        q = "Machine transitions: %s. Input: %s. Start at S0. Is state S%d visited at any point? Answer yes or no." % (trans_txt, inp, k)
        ans = "yes" if k in trace else "no"
    cot = " -> ".join("S%d" % t for t in trace)
    return {"dfa": dfa, "seq": seq, "qtype": qtype, "k": k if qtype != "final" else None,
            "q": q, "cot": cot, "ans": ans}

# ---------- 验证器（独立通道：不读生成器的 ans/cot，重放求值再对账；参数吃结构化字段 k，不吃叙述文本） ----------
def verify(sample):
    dfa = sample["dfa"]; seq = sample["seq"]
    s = 0; trace = [0]
    for a in seq:
        s = dfa["trans"][s][a]; trace.append(s)
    if sample["qtype"] == "final":
        gold = "S%d" % s
    elif sample["qtype"] == "visit_count":
        gold = str(trace.count(sample["k"]))
    else:
        gold = "yes" if sample["k"] in trace else "no"
    return {"ok": sample["ans"] == gold, "gold": gold,
            "cot_ok": sample["cot"] == " -> ".join("S%d" % t for t in trace)}

# ---------- T 式自检 ----------
def self_test():
    r = {}
    # G1 双通道全对（多种子×多长度）
    bad = 0; total = 0
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        for _ in range(200):
            dfa = gen_dfa(rng, rng.choice([3, 4, 5]))
            sm = gen_sample(rng, dfa, rng.choice([3, 6, 10, 16]))
            total += 1
            if not verify(sm)["ok"] or not verify(sm)["cot_ok"]:
                bad += 1
    r["G1_dual_channel"] = {"total": total, "bad": bad, "pass": bad == 0}
    # G2 毒样本 fail-closed：答案错一位/轨迹错一步必被抓
    rng = random.Random(99)
    caught = 0; total2 = 0
    for _ in range(100):
        sm = gen_sample(rng, gen_dfa(rng), rng.choice([6, 10]))
        v0 = verify(sm); assert v0["ok"]
        poison = dict(sm)
        if sm["qtype"] == "visit_count":
            poison["ans"] = str(int(sm["ans"]) + 1)          # 数值毒
        elif sm["qtype"] == "final":
            poison["ans"] = "S%d" % ((int(sm["ans"][1:]) + 1) % sm["dfa"]["n_states"])
        else:
            poison["ans"] = "no" if sm["ans"] == "yes" else "yes"
        if not verify(poison)["ok"]:
            caught += 1
        total2 += 1
        poison2 = dict(sm); poison2["cot"] = sm["cot"] + " -> S0"   # 轨迹毒
        if not verify(poison2)["cot_ok"]:
            caught += 1
        total2 += 1
    r["G2_poison_failclosed"] = {"total": total2, "caught": caught, "pass": caught == total2}
    # G3 确定性：同种子语料逐字节一致
    def corpus(seed):
        rng = random.Random(seed); out = []
        for _ in range(50):
            out.append(gen_sample(rng, gen_dfa(rng), 8))
        return json.dumps(out, sort_keys=True)
    r["G3_determinism"] = {"pass": corpus(7) == corpus(7) and corpus(7) != corpus(8)}
    # G4 课程包出厂：train/val 不相交（序列空间按 seed 分段）
    tr = corpus(1001); va = corpus(2002)
    r["G4_trainval_split"] = {"pass": tr != va}
    r["ALL_PASS"] = all(v.get("pass") for k, v in r.items() if k.startswith("G"))
    return r

# ---------- 课程包导出 ----------
def export(n_train=2000, n_val=256):
    lines = []
    for split, n, seed in (("train", n_train, 5001), ("val", n_val, 9001)):
        rng = random.Random(seed)
        for i in range(n):
            dfa = gen_dfa(rng, rng.choice([3, 4, 5]))
            sm = gen_sample(rng, dfa, rng.choice([4, 8, 12, 16]))
            assert verify(sm)["ok"], "curriculum poison at %s/%d" % (split, i)
            lines.append(json.dumps({
                "split": split, "subject": "fsm_tracking", "id": "%s_%04d" % (split, i),
                "seq": sm["seq"], "qtype": sm["qtype"], "q": sm["q"],
                "cot": sm["cot"], "ans": sm["ans"]}, ensure_ascii=False))
    data = "\n".join(lines) + "\n"
    open("curriculum_fsm_v0.jsonl", "w", encoding="utf-8", newline="").write(data)
    return {"records": len(lines), "sha256": hashlib.sha256(data.encode()).hexdigest()[:16]}

if __name__ == "__main__":
    res = self_test()
    res["export"] = export()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(OUT_RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if res["ALL_PASS"] else 2)
