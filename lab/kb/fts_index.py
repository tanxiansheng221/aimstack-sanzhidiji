# -*- coding: utf-8 -*-
"""SQLite FTS5 index for large KB (1M+ cards). Stdlib sqlite3 only.
cards.jsonl remains source of truth; this index is rebuildable."""
import io, json, os, re, sqlite3, sys, time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def _norm(s):
    return re.sub(r"\s+", " ", s).strip()

class FtsIndex:
    def __init__(self, root):
        self.db_path = os.path.join(root, "kb_fts.db")
        self.con = sqlite3.connect(self.db_path)
        self._ensure()

    def _ensure(self):
        c = self.con.cursor()
        # trigram tokenizer: substring matching for CJK (unicode61 keeps CJK runs
        # as single tokens -> phrase queries fail). Requires SQLite >= 3.34.
        try:
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS cards USING fts5("
                      "title, text, id UNINDEXED, origin UNINDEXED, source UNINDEXED, "
                      "tokenize='trigram')")
            self.con.commit()
        except sqlite3.OperationalError as e:
            raise RuntimeError("FTS5/trigram unavailable: %s" % e)

    def count(self):
        return self.con.execute("SELECT count(*) FROM cards").fetchone()[0]

    def add_jsonl(self, path, batch=5000, quiet=False):
        """Bulk load cards.jsonl (skips ids already present)."""
        c = self.con.cursor()
        have = set(r[0] for r in c.execute("SELECT id FROM cards"))
        buf = []
        n = added = 0
        t0 = time.time()
        for line in io.open(path, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            n += 1
            if d["id"] in have:
                continue
            buf.append((d["title"], d["text"], d["id"], d.get("origin", ""),
                        d.get("source", "")))
            if len(buf) >= batch:
                c.executemany("INSERT INTO cards VALUES (?,?,?,?,?)", buf)
                self.con.commit()
                added += len(buf); buf = []
                if not quiet and added % 50000 < batch:
                    print("  loaded %dk cards (%.0fs)" % (added // 1000, time.time() - t0), flush=True)
        if buf:
            c.executemany("INSERT INTO cards VALUES (?,?,?,?,?)", buf)
            self.con.commit()
            added += len(buf)
        return added, n

    def search(self, query, topk=5, origin=None):
        """FTS5 query; CJK works via trigram fallback or raw phrase (unicode61 tokenizes
        CJK per-char blocks — phrase match keeps contiguity)."""
        q = _norm(query)
        # quote each term to avoid FTS syntax injection
        terms = " ".join('"%s"' % t for t in q.split() if t)
        if not terms:
            return []
        sql = "SELECT id, title, text, origin, bm25(cards) FROM cards WHERE cards MATCH ?"
        args = [terms]
        if origin:
            sql += " AND origin = ?"
            args.append(origin)
        sql += " ORDER BY bm25(cards) LIMIT ?"
        args.append(topk)
        try:
            rows = self.con.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            # no space-separated terms (pure CJK) -> use phrase on whole string
            rows = self.con.execute(
                "SELECT id, title, text, origin, bm25(cards) FROM cards WHERE cards MATCH ? ORDER BY bm25(cards) LIMIT ?",
                ['"%s"' % q, topk]).fetchall()
        return [{"id": r[0], "title": r[1], "text": r[2], "origin": r[3],
                 "score": round(-r[4], 2)} for r in rows]

if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kb_store")
    idx = FtsIndex(root)
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        added, n = idx.add_jsonl(os.path.join(root, "cards.jsonl"))
        print("indexed %d/%d | total %d" % (added, n, idx.count()))
    elif len(sys.argv) > 1 and sys.argv[1] == "search":
        for r in idx.search(" ".join(sys.argv[2:]), topk=5):
            print("%.2f %s [%s] %s\n  %s" % (r["score"], r["id"], r["origin"], r["title"], r["text"][:100]))
    else:
        print("usage: fts_index.py rebuild | search <query>")
