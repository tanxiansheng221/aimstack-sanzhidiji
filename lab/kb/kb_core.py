# -*- coding: utf-8 -*-
"""KB core: card store + chunker + dedup gate + retrieval. Stdlib only (断供免疫纪律).

Card schema: {id, title, text, source, origin(user|crawl|self), lang, ts, sha256}
Storage: append-only JSONL + in-memory BM25-lite index. Fail-closed: no card enters without passing gate.
"""
import io, json, hashlib, os, re, time, unicodedata

def _norm(text):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# ---------------- chunker: doc -> cards ----------------
_HEAD = re.compile(r"^#{1,4}\s+", re.M)

def chunk_document(text, title, source, origin, max_len=800, min_len=40):
    """Split into heading-aware chunks; each becomes a card (title -> chunk 键值对)."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    # split before headings (markdown) or blank-line blocks
    parts, cur = [], []
    for line in text.split("\n"):
        if _HEAD.match(line) and cur:
            parts.append("\n".join(cur)); cur = [line]
        else:
            cur.append(line)
    if cur:
        parts.append("\n".join(cur))
    cards, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) <= max_len:
            buf = (buf + "\n" + p).strip()
        else:
            if len(buf) >= min_len:
                cards.append(buf)
            buf = p if len(p) <= max_len else ""
            # hard-split oversize paragraph
            while len(p) > max_len:
                cards.append(p[:max_len]); p = p[max_len:]
            buf = p
    if len(buf) >= min_len:
        cards.append(buf)
    out = []
    for i, c in enumerate(cards):
        out.append({"title": title, "text": c, "source": source,
                    "origin": origin, "part": i})
    return out

# ---------------- gate: dedup ----------------
class DedupGate:
    def __init__(self, store_path):
        self.seen_sha = set()
        self.seen_titles = set()
        self.path = store_path
        if os.path.exists(store_path):
            for l in io.open(store_path, encoding="utf-8"):
                try:
                    c = json.loads(l)
                    self.seen_sha.add(c["sha256"])
                    self.seen_titles.add(_norm(c["title"]) + "|" + c["sha256"][:8])
                except Exception:
                    pass

    def check(self, card):
        """Return (ok, reason). Exact-text dedup + title+head dedup."""
        body = _norm(card["text"])
        if len(body) < 20:
            return False, "too_short"
        h = _sha(body)
        if h in self.seen_sha:
            return False, "dup_text"
        key = _norm(card["title"]) + "|" + _sha(body)[:8]
        if key in self.seen_titles:
            return False, "dup_title"
        return True, h

    def commit(self, card, h):
        self.seen_sha.add(h)

# ---------------- store ----------------
class KB:
    def __init__(self, root, build_index=True):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.path = os.path.join(root, "cards.jsonl")
        self.gate = DedupGate(self.path)
        self.cards = []
        self._no_index = not build_index  # huge-store mode: gate only, search via FTS5
        if os.path.exists(self.path):
            for l in io.open(self.path, encoding="utf-8"):
                try:
                    self.cards.append(json.loads(l))
                except Exception:
                    pass
        if build_index:
            self._build_index()

    # ---- retrieval: CJK bigram + ASCII word BM25-lite ----
    @staticmethod
    def _tokens(s):
        s = _norm(s)
        toks = re.findall(r"[a-z0-9]+", s)
        cjk = re.findall(r"[\u4e00-\u9fff]", s)
        toks += [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]  # bigram
        return toks

    def _build_index(self):
        self.df = {}
        self.tfs = []
        for c in self.cards:
            toks = set(self._tokens(c["title"])) , set(self._tokens(c["text"]))
            tset = toks[0] | toks[1]
            for t in tset:
                self.df[t] = self.df.get(t, 0) + 1
            self.tfs.append(toks)
        self.N = len(self.cards)

    def add(self, title, text, source, origin="user", tags=None):
        """Fail-closed add. Returns list of (id, status) for produced cards."""
        made, results = [], []
        for card in chunk_document(text, title, source, origin):
            ok, h = self.gate.check(card)
            if not ok:
                results.append((None, "rejected:" + h))
                continue
            rec = {"id": "kb-" + h, "title": card["title"], "text": card["text"],
                   "source": card["source"], "origin": origin,
                   "tags": tags or [], "ts": int(time.time()), "sha256": h,
                   "part": card["part"]}
            self.cards.append(rec)
            self.gate.commit(card, h)
            made.append(rec)
            results.append((rec["id"], "added"))
        if made:
            with io.open(self.path, "a", encoding="utf-8") as f:
                for r in made:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if not self._no_index:
                self._build_index()
        return results

    def search(self, query, topk=5):
        if not self.N:
            return []
        q = self._tokens(query)
        if not q:
            return []
        scored = []
        for i, c in enumerate(self.cards):
            tt, bt = self.tfs[i]
            s = 0.0
            for t in q:
                idf = 1.0 + (self.N / (1 + self.df.get(t, 0)))
                s += 3.0 * idf * (t in tt) + idf * bt.__contains__(t)
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [{"score": round(s, 2), "id": self.cards[i]["id"],
                 "title": self.cards[i]["title"], "origin": self.cards[i]["origin"],
                 "text": self.cards[i]["text"][:200]}
                for s, i in scored[:topk]]

    def stats(self):
        by = {}
        for c in self.cards:
            by[c["origin"]] = by.get(c["origin"], 0) + 1
        return {"cards": len(self.cards), "by_origin": by}
