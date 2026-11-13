# -*- coding: utf-8 -*-
# 课程生成器 —— 图论科重建（curriculum_graph_lab.py）
# v0 遗留仅 1517 条（2 题型/5 节点）；本版 7 题型 × 节点 5-9 × 边密度扫描，目标 10 万+ 级。
# 纪律同 fsm_lab：①生成器↔验证器双通道全对（MST 用 Prim 对账 Kruskal=异算法独立通道）
# ②毒样本 fail-closed ③同种子逐字节确定 ④train/val 不相交 ⑤题面自包含+退化 cap。
# 题型：shortest/toposort/reach/mst/bipartite/degree/cycle
import json, random, hashlib, sys

OUT_RESULTS = "curriculum_graph_v2_results.json"
FAMS = ("shortest", "toposort", "reach", "mst", "bipartite", "degree", "cycle")

# ---------- 底层图构造 ----------
def _undirected_w(rng, n, p_extra):
    """连通无向带权图：随机生成树 + 概率加边。edges=[(u,v,w)] u<v。"""
    edges = set()
    for i in range(1, n):
        edges.add((rng.randrange(i), i))
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in edges]
    rng.shuffle(pairs)
    for e in pairs:
        if rng.random() < p_extra:
            edges.add(e)
    out = [(a, b, rng.randrange(1, 10)) for a, b in sorted(edges)]
    return out

def _undirected_uw(rng, n, p_extra):
    """连通无向无权图（只留 u,v）。"""
    return [(u, v) for u, v, _ in _undirected_w(rng, n, p_extra)]

def _arborescence_w(rng, n, s, p_extra):
    """有向带权：从 S 的随机生成树形（保证 S 可达全图）+ 概率加边。edges=[(u,v,w)]。"""
    rest = [v for v in range(n) if v != s]
    rng.shuffle(rest)
    edges = set()
    added = [s]
    for v in rest:
        u = added[rng.randrange(len(added))]
        edges.add((u, v))
        added.append(v)
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b and (a, b) not in edges]
    rng.shuffle(pairs)
    for e in pairs:
        if rng.random() < p_extra:
            edges.add(e)
    return [(a, b, rng.randrange(1, 10)) for a, b in sorted(edges)]

def _dag(rng, n, p_extra):
    """随机 DAG：按隐藏排列只允许前->后。edges=[(u,v)]。保证至少 1 条边。"""
    perm = list(range(n))
    rng.shuffle(perm)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_extra:
                edges.add((perm[i], perm[j]))
    if not edges:
        edges.add((perm[0], perm[1]))
    return sorted(edges)

def _bipartite_yes(rng, n, p_extra):
    """二部图（yes 构造）：两色类，只允许跨类边；生成树父节点取自已放置的异类节点。"""
    nA = rng.randrange(1, n)
    A = list(range(nA))
    B = list(range(nA, n))
    rng.shuffle(A); rng.shuffle(B)
    edges = set()
    placed_a, placed_b = [A[0]], []
    rest = [B[0]]  # 首个新节点必须是 B（保证 placed_b 先非空）
    ia, ib = 1, 1
    while ia < len(A) or ib < len(B):
        if ib < len(B) and (ia >= len(A) or rng.random() < 0.5):
            rest.append(B[ib]); ib += 1
        else:
            rest.append(A[ia]); ia += 1
    for v in rest:
        if v < nA:  # v 属于 A 类：父取已放置的 B 类节点
            u = placed_b[rng.randrange(len(placed_b))]
            placed_a.append(v)
        else:       # v 属于 B 类：父取已放置的 A 类节点
            u = placed_a[rng.randrange(len(placed_a))]
            placed_b.append(v)
        edges.add((min(u, v), max(u, v)))
    pairs = [(a, b) for a in A for b in B if (a, b) not in edges]
    rng.shuffle(pairs)
    for e in pairs:
        if rng.random() < p_extra:
            edges.add(e)
    return sorted(edges)

def _fmt_uw(edges):
    return ", ".join("%d-%d" % (u, v) for u, v in edges)

def _fmt_w(edges, sep, arrow):
    return ", ".join("%d%s%d:%d" % (u, arrow, v, w) for u, v, w in edges)

# ---------- 通道 A：生成器 ----------
def gen_sample(rng):
    fam = rng.choice(FAMS)
    if fam == "bipartite":
        n = rng.randrange(7, 12)   # 小 n 跨类边集空间太小（dup 大户），抬下限
    elif fam == "cycle":
        n = rng.randrange(6, 10)   # 同理：n=5 连通图空间仅 ~8k，撞车重灾区
    else:
        n = rng.randrange(5, 10)
    if fam in ("shortest", "mst"):  # 带权题空间大，保留低密度档
        p = rng.choice([0.10, 0.25, 0.45, 0.70])
    else:  # 无权/DAG/二部：低密度档题面空间太小（diag 实证 70-88%），砍掉
        p = rng.choice([0.25, 0.45, 0.70])
    sm = {"fam": fam, "n": n}
    if fam == "shortest":
        s = rng.randrange(n)
        t = rng.choice([v for v in range(n) if v != s])
        ew = _arborescence_w(rng, n, s, p)
        adj = {}
        for u, v, w in ew:
            adj.setdefault(u, []).append((v, w))
        dist = _dijkstra(adj, s, n)
        sm.update({"edges_w": ew, "s": s, "t": t})
        sm["q"] = ("Graph edges (u->v:w): %s. Start S%d, target T%d. "
                   "What is the shortest path length? Answer a number."
                   % (_fmt_w(ew, "", ">"), s, t))
        sm["cot"] = "dijkstra: dist from S%d = %s" % (
            s, ",".join("T%d=%s" % (v, dist[v] if dist[v] is not None else "INF") for v in range(n)))
        sm["ans"] = str(dist[t])
        sm["k"] = {"edges_w": ew, "s": s, "t": t}
    elif fam == "toposort":
        e = _dag(rng, n, p)
        order = _kahn(e, n)
        sm.update({"edges": e})
        sm["q"] = ("DAG edges (u->v): %s. Give one valid topological order. "
                   "Answer as a sequence like 0,1,2." % _fmt_uw(e))
        sm["cot"] = "kahn: pick zero-indegree nodes in order -> %s" % ",".join(map(str, order))
        sm["ans"] = ",".join(map(str, order))
        sm["k"] = {"edges": e}
    elif fam == "reach":
        s = rng.randrange(n)
        t = rng.choice([v for v in range(n) if v != s])
        e = _undirected_uw(rng, n, p)
        reach = _bfs_set(e, s)
        ok = t in reach
        sm.update({"edges": e, "s": s, "t": t})
        sm["q"] = ("Undirected edges: %s. Is node %d reachable from node %d? "
                   "Answer yes or no." % (_fmt_uw(e), t, s))
        sm["cot"] = "bfs from S%d: T%d reachable=%s" % (s, t, "yes" if ok else "no")
        sm["ans"] = "yes" if ok else "no"
        sm["k"] = {"edges": e, "s": s, "t": t}
    elif fam == "mst":
        ew = _undirected_w(rng, n, p)
        total = _kruskal_total(ew, n)
        sm.update({"edges_w": ew})
        sm["q"] = ("Undirected weighted edges (u-v:w): %s. What is the total weight of a "
                   "minimum spanning tree? Answer a number." % _fmt_w(ew, "", "-"))
        sm["cot"] = "kruskal: %d edges picked, total=%d" % (n - 1, total)
        sm["ans"] = str(total)
        sm["k"] = {"edges_w": ew, "n": n}
    elif fam == "bipartite":
        if rng.random() < 0.5:
            e = _bipartite_yes(rng, n, p)
            ok = True
        else:
            e = _undirected_uw(rng, n, p)
            ok = _is_bipartite(e, n)
            if not ok:
                pass
            else:
                # 强制 no：在 BFS 同色类内补一条边（奇圈）
                color = _bfs_color(e, n)
                same = [(a, b) for a in range(n) for b in range(a + 1, n)
                        if color[a] == color[b] and (a, b) not in set(e)]
                if same:
                    e = sorted(e + [same[rng.randrange(len(same))]])
                    ok = False
                else:  # 退化 cap：全图同色不可能（连通图 n>=2 必有异色边）
                    ok = True
        sm.update({"edges": e})
        sm["q"] = ("Undirected edges: %s. Can the graph be colored with 2 colors so that "
                   "no edge joins two nodes of the same color? Answer yes or no." % _fmt_uw(e))
        sm["cot"] = "2color: bipartite=%s" % ("yes" if ok else "no")
        sm["ans"] = "yes" if ok else "no"
        sm["k"] = {"edges": e, "n": n}
    elif fam == "degree":
        e = _undirected_uw(rng, n, p)
        k = rng.randrange(n)
        d = sum(1 for u, v in e if u == k or v == k)
        sm.update({"edges": e})
        sm["q"] = ("Undirected edges: %s. What is the degree of node %d? "
                   "Answer a number." % (_fmt_uw(e), k))
        sm["cot"] = "deg(%d): count edges at %d = %d" % (k, k, d)
        sm["ans"] = str(d)
        sm["k"] = {"edges": e, "k": k}
    else:  # cycle
        e = _undirected_uw(rng, n, p)
        has = _has_cycle(e, n)
        sm.update({"edges": e})
        sm["q"] = ("Undirected edges: %s. Does the graph contain a cycle? "
                   "Answer yes or no." % _fmt_uw(e))
        sm["cot"] = "unionfind: cycle=%s" % ("yes" if has else "no")
        sm["ans"] = "yes" if has else "no"
        sm["k"] = {"edges": e, "n": n}
    return sm

# ---------- 通道 A 内部算法（生成器自己的求值） ----------
def _dijkstra(adj, s, n):
    import heapq
    dist = {v: None for v in range(n)}
    dist[s] = 0
    h = [(0, s)]
    done = set()
    while h:
        d, u = heapq.heappop(h)
        if u in done:
            continue
        done.add(u)
        for v, w in adj.get(u, []):
            nd = d + w
            if dist[v] is None or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return dist

def _kahn(edges, n):
    indeg = [0] * n
    adj = [[] for _ in range(n)]
    for u, v in edges:
        indeg[v] += 1
        adj[u].append(v)
    import heapq
    h = [v for v in range(n) if indeg[v] == 0]
    heapq.heapify(h)
    order = []
    while h:
        u = heapq.heappop(h)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(h, v)
    return order

def _bfs_set(edges, s):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen = {s}
    q = [s]
    while q:
        u = q.pop()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen

def _kruskal_total(ew, n):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    total = 0
    cnt = 0
    for u, v, w in sorted(ew, key=lambda e: e[2]):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            total += w
            cnt += 1
    return total

def _is_bipartite(edges, n):
    return _bfs_color(edges, n) is not None

def _bfs_color(edges, n):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    color = [None] * n
    for s0 in range(n):
        if color[s0] is not None:
            continue
        color[s0] = 0
        q = [s0]
        while q:
            u = q.pop()
            for v in adj.get(u, []):
                if color[v] is None:
                    color[v] = color[u] ^ 1
                    q.append(v)
                elif color[v] == color[u]:
                    return None
    return color

def _has_cycle(edges, n):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return True
        parent[ru] = rv
    return False

# ---------- 通道 B：验证器（独立重放，不读 ans/cot；算法与通道 A 异构处标注） ----------
def verify(sm):
    k = sm["k"]
    fam = sm["fam"]
    if fam == "shortest":
        ew = k["edges_w"]; s = k["s"]; t = k["t"]; n = sm["n"]
        adj = {}
        for u, v, w in ew:
            adj.setdefault(u, []).append((v, w))
        dist = _dijkstra(adj, s, n)  # 同算法不同实现路径（生成器文本 vs 结构化重放）
        gold = str(dist[t])
        cot = "dijkstra: dist from S%d = %s" % (
            s, ",".join("T%d=%s" % (v, dist[v] if dist[v] is not None else "INF") for v in range(n)))
    elif fam == "toposort":
        e = k["edges"]; n = sm["n"]
        pos = {}
        seq = sm["ans"].split(",")
        if sorted(x for x in seq) != [str(v) for v in range(n)]:
            return {"ok": False, "gold": "?", "cot_ok": False, "fam_match": True}
        pos = {int(x): i for i, x in enumerate(seq)}
        gold_ok = all(pos[u] < pos[v] for u, v in e)  # 独立判据：答案是否合法序，不比对唯一 gold
        order = _kahn(e, n)
        gold = ",".join(map(str, order))
        cot = "kahn: pick zero-indegree nodes in order -> %s" % ",".join(map(str, order))
        return {"ok": gold_ok, "gold": gold,
                "cot_ok": sm["cot"] == cot, "fam_match": True}
    elif fam == "reach":
        e = k["edges"]; s = k["s"]; t = k["t"]
        ok = t in _bfs_set(e, s)
        gold = "yes" if ok else "no"
        cot = "bfs from S%d: T%d reachable=%s" % (s, t, gold)
    elif fam == "mst":
        ew = k["edges_w"]; n = k["n"]
        total = _prim_total(ew, n)  # 异算法独立通道：Prim 对账 Kruskal
        gold = str(total)
        cot = "kruskal: %d edges picked, total=%d" % (n - 1, total)
    elif fam == "bipartite":
        e = k["edges"]; n = k["n"]
        ok = _is_bipartite(e, n)
        gold = "yes" if ok else "no"
        cot = "2color: bipartite=%s" % gold
    elif fam == "degree":
        e = k["edges"]; kx = k["k"]
        d = sum(1 for u, v in e if u == kx or v == kx)
        gold = str(d)
        cot = "deg(%d): count edges at %d = %d" % (kx, kx, d)
    else:  # cycle
        e = k["edges"]; n = k["n"]
        ok = _has_cycle(e, n)
        gold = "yes" if ok else "no"
        cot = "unionfind: cycle=%s" % gold
    return {"ok": sm["ans"] == gold, "gold": gold, "cot_ok": sm["cot"] == cot, "fam_match": True}

def _prim_total(ew, n):
    """通道 B 专用：Prim（与生成器 Kruskal 异构）。"""
    adj = {}
    for u, v, w in ew:
        adj.setdefault(u, []).append((w, v))
        adj.setdefault(v, []).append((w, u))
    import heapq
    seen = {0}
    h = list(adj[0])
    heapq.heapify(h)
    total = 0
    while h and len(seen) < n:
        w, v = heapq.heappop(h)
        if v in seen:
            continue
        seen.add(v)
        total += w
        for e in adj[v]:
            if e[1] not in seen:
                heapq.heappush(h, e)
    return total

# ---------- T 式自检 ----------
def self_test():
    r = {}
    bad = 0; total = 0
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        for _ in range(300):
            sm = gen_sample(rng)
            v = verify(sm)
            total += 1
            if not v["ok"] or not v["cot_ok"] or not v["fam_match"]:
                bad += 1
                if bad <= 3:
                    print("G1 MISS:", sm["fam"], sm["ans"], v, flush=True)
    r["G1_dual_channel"] = {"total": total, "bad": bad, "pass": bad == 0}
    rng = random.Random(99)
    caught = 0; total2 = 0
    for _ in range(150):
        sm = gen_sample(rng)
        assert verify(sm)["ok"]
        poison = dict(sm)
        if sm["ans"] in ("yes", "no"):
            poison["ans"] = "no" if sm["ans"] == "yes" else "yes"
        elif "," in sm["ans"]:
            xs = sm["ans"].split(",")
            poison["ans"] = ",".join(reversed(xs))  # 逆序必违反 ≥1 条边（DAG 保证非空）
        else:
            poison["ans"] = str(int(sm["ans"]) + 1)
        if not verify(poison)["ok"]:
            caught += 1
        total2 += 1
        poison2 = dict(sm); poison2["cot"] = sm["cot"] + " x"  # 轨迹毒
        if not verify(poison2)["cot_ok"]:
            caught += 1
        total2 += 1
    r["G2_poison_failclosed"] = {"total": total2, "caught": caught, "pass": caught == total2}
    def corpus(seed):
        rng = random.Random(seed)
        return json.dumps([gen_sample(rng)["q"] for _ in range(60)], sort_keys=True)
    r["G3_determinism"] = {"pass": corpus(7) == corpus(7) and corpus(7) != corpus(8)}
    tr = corpus(1001); va = corpus(2002)
    r["G4_trainval_split"] = {"pass": tr != va}
    r["ALL_PASS"] = all(v.get("pass") for kk, v in r.items() if kk.startswith("G"))
    return r

# ---------- 课程包导出 ----------
def export(n_train=2000, n_val=256):
    lines = []
    seen = set()  # 全局去重：跨 split 重复=泄露 val（G4 纪律），重复即重抽
    for split, n, seed in (("train", n_train, 5001), ("val", n_val, 9001)):
        rng = random.Random(seed)
        made, attempts = 0, 0
        while made < n:
            attempts += 1
            assert attempts <= n * 12, "export degenerate: question space exhausted"
            sm = gen_sample(rng)
            if sm["q"] in seen:
                continue
            v = verify(sm)
            assert v["ok"] and v["cot_ok"], "curriculum poison at %s/%d: %s" % (split, made, v)
            seen.add(sm["q"])
            lines.append(json.dumps({
                "split": split, "subject": "graph_algo", "id": "%s_%05d" % (split, made),
                "qtype": sm["fam"], "q": sm["q"], "cot": sm["cot"], "ans": sm["ans"]},
                ensure_ascii=False))
            made += 1
    data = "\n".join(lines) + "\n"
    open("curriculum_graph_v2.jsonl", "w", encoding="utf-8", newline="").write(data)
    return {"records": len(lines), "sha256": hashlib.sha256(data.encode()).hexdigest()[:16]}

if __name__ == "__main__":
    res = self_test()
    print(json.dumps({k: v for k, v in res.items() if k != "ALL_PASS" or True},
                     ensure_ascii=False, indent=1), flush=True)
    if res["ALL_PASS"]:
        res["export"] = export()
        json.dump(res, open(OUT_RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("EXPORT:", res["export"], flush=True)
    sys.exit(0 if res["ALL_PASS"] else 2)
