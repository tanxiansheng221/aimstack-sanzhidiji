# -*- coding: utf-8 -*-
"""KB CLI: add files (rich formats) / search / stats.
Usage:
  python kb_cli.py add <file...>            # user manual feed (txt/md/html/docx/pdf)
  python kb_cli.py search <query> [--k 5]
  python kb_cli.py stats
"""
import io, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.kb_core import KB
from kb import ingest

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb_store")

def main():
    kb = KB(ROOT)
    args = sys.argv[1:]
    if not args:
        print(__doc__); return 2
    cmd = args[0]
    if cmd == "add":
        ok = 0
        for path in args[1:]:
            try:
                title, text = ingest.load_any(path)
            except Exception as e:
                print("SKIP %s: %s" % (path, e)); continue
            res = kb.add(title, text, source=path, origin="user", tags=["manual"])
            added = sum(1 for _, st in res if st == "added")
            rej = len(res) - added
            print("%s: +%d cards (%d rejected by gate)" % (os.path.basename(path), added, rej))
            ok += added
        print("TOTAL +%d | kb %s" % (ok, kb.stats()))
        return 0
    if cmd == "search":
        q = args[1]; k = 5
        if "--k" in args: k = int(args[args.index("--k") + 1])
        for r in kb.search(q, topk=k):
            print("%.1f %s [%s] %s\n   %s" % (r["score"], r["id"], r["origin"], r["title"], r["text"][:120]))
        return 0
    if cmd == "stats":
        print(json.dumps(kb.stats(), ensure_ascii=False)); return 0
    print(__doc__); return 2

if __name__ == "__main__":
    sys.exit(main())
