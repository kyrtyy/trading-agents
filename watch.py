"""Follow the agents' conversations live: prints each decision as it is appended to the newest runs/*.jsonl.
Usage: python watch.py [arm-file-substring] [--full]   (default: newest file, each message cut to 600 chars)"""
import json, sys, time
from run import RUNS

full = "--full" in sys.argv
pat = next((a for a in sys.argv[1:] if not a.startswith("--")), "")
path = max((p for p in RUNS.glob("*.jsonl") if pat in p.name), key=lambda p: p.stat().st_mtime)
print(f"following {path.name}  (Ctrl+C to quit)\n")
f = open(path)
f.seek(0, 2)
while True:
    line = f.readline()
    if not line or not line.endswith("\n"):
        time.sleep(1); f.seek(f.tell() - len(line)); continue
    r = json.loads(line)
    print(f"\n{'=' * 100}\n{r['ticker']}  {r['date']}  window {r['window']}  ->  {r['action']}\n{'=' * 100}")
    for c in r["calls"]:
        text = c["text"] if full else c["text"][:600] + (" ..." if len(c["text"]) > 600 else "")
        print(f"\n--- {c['role']} ---\n{text}")
