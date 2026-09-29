"""Send a paraphrase-heavy workload to a running proxy and print the savings.  python scripts/simulate.py [url]"""
import sys
import time

import httpx

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000") + "/v1/chat/completions"
WORKLOAD = [
    "How do I reset my password?", "how can I reset my password", "reset my password please",
    "What is your refund policy?", "refund policy please", "Do you offer refunds?",
    "What is 15% of 200?", "What is 25% of 200?",
    "Is aspirin safe with alcohol?", "Is aspirin not safe with alcohol?",
] * 3
seen = {}
t0 = time.time()
with httpx.Client(timeout=60) as c:
    for q in WORKLOAD:
        r = c.post(URL, json={"model": "auto", "messages": [{"role": "user", "content": q}]})
        s = r.headers["X-Cache"]
        seen[s] = seen.get(s, 0) + 1
        print(f"{s:9} {r.headers['X-Latency-Ms']:>8} ms  {q}")
    print("\nsummary:", seen, f"in {time.time() - t0:.1f}s")
    print("stats:", c.get(URL.replace("/v1/chat/completions", "/stats")).json())
