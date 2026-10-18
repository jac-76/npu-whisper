"""Which SearXNG engines actually answer: one search per query through the
JSON API, then per engine how many queries it returned results for and why it
failed on the others.

usage: evals/probe_searx.py [URL] [PACE]
       URL defaults to $NPU_CHAT_SEARX_URL or http://127.0.0.1:8888;
       PACE = seconds between queries (default 15; unpaced runs get engines
       suspended).
Neutral queries only: nothing about the person running it.
"""
import collections
import json
import os
import sys
import time
import urllib.parse
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else
       os.environ.get("NPU_CHAT_SEARX_URL", "http://127.0.0.1:8888")).rstrip("/")
PACE = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
QUERIES = [
    "kentucky derby 2026 winner",
    "boiling point of water at high altitude",
    "python 3.14 release notes",
    "how does a tokamak confine plasma",
    "tour de france 2026 winner",
    "who wrote the novel dune",
    "arch linux pacman mirrorlist",
    "helium-3 fusion fuel",
    "nba finals 2026 result",
    "speed of light in optical fiber",
]

answered = collections.Counter()
failed = collections.defaultdict(collections.Counter)
useful = 0
for i, q in enumerate(QUERIES):
    if i:
        time.sleep(PACE)
    url = f"{URL}/search?" + urllib.parse.urlencode({"q": q, "format": "json"})
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(url, timeout=30).read())
    except Exception as exc:
        print(f"{q[:40]:40}  request failed: {exc}")
        continue
    results = d.get("results") or []
    engines = {e for r in results for e in (r.get("engines") or [])}
    for e in engines:
        answered[e] += 1
    for e, reason in (x for x in d.get("unresponsive_engines") or []
                      if isinstance(x, (list, tuple)) and len(x) >= 2):
        failed[e][reason] += 1
    useful += len(results) >= 3
    print(f"{q[:40]:40}  {len(results):3d} results in {time.time() - t0:4.1f}s  "
          f"from {', '.join(sorted(engines)) or '-'}")

print(f"\n{useful}/{len(QUERIES)} queries returned 3+ results")
for e in sorted(set(answered) | set(failed), key=lambda e: -answered[e]):
    why = "; ".join(f"{r} x{n}" for r, n in failed[e].most_common())
    print(f"  {e:16} answered {answered[e]:2d}/{len(QUERIES)}  {('failed: ' + why) if why else ''}")
