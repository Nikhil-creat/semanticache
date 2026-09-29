"""Framework-free cache engine. main.py exposes it over HTTP; tests call it directly."""
import copy
import hashlib
import json
import time

from .agents import Governor, Verifier, route
from .config import settings
from .embeddings import make_embedder
from .policy import cacheable, guard, redact, ttl_for
from .providers import make_provider
from .store import Entry, MemoryStore, QdrantStore

TOOLS = [  # OpenAI function-calling / MCP-style manifest so other agents can operate the cache
    {"name": "cache_stats", "description": "Hit rate, tokens and USD saved, current threshold.", "parameters": {"type": "object", "properties": {}}},
    {"name": "agent_journal", "description": "Recent decisions of the Governor agent.", "parameters": {"type": "object", "properties": {}}},
    {"name": "invalidate", "description": "Delete cached answers whose prompt contains a phrase.",
     "parameters": {"type": "object", "properties": {"contains": {"type": "string"}}, "required": ["contains"]}},
    {"name": "tune_threshold", "description": "Set similarity threshold (0.80-0.97).",
     "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "reason": {"type": "string"}}, "required": ["value"]}},
]


class L1:
    """Exact-match cache: Redis when REDIS_URL is set, otherwise a local dict."""

    def __init__(self):
        self.d, self.r = {}, None
        if settings.redis_url:
            import redis.asyncio as redis

            self.r = redis.from_url(settings.redis_url)

    async def get(self, k):
        if self.r:
            v = await self.r.get(k)
            return json.loads(v) if v else None
        e = self.d.get(k)
        return e[1] if e and e[0] > time.time() else None

    async def set(self, k, v, ttl):
        if self.r:
            await self.r.set(k, json.dumps(v), ex=ttl)
        else:
            self.d[k] = (time.time() + ttl, v)

    async def clear(self):
        self.d.clear()
        if self.r:
            await self.r.flushdb()


class Engine:
    def __init__(self, embedder=None, store=None, provider=None):
        self.embedder = embedder or make_embedder()
        self.provider = provider or make_provider()
        self.store = store or (QdrantStore(settings.qdrant_url, self.embedder.dim) if settings.backend == "qdrant"
                               else MemoryStore(settings.max_items))
        self.l1, self.gov, self.verifier = L1(), Governor(settings.threshold), Verifier(self.provider)
        self.s = dict(requests=0, hits=0, misses=0, bypass=0, tokens_saved=0, cost_saved_usd=0.0)

    async def handle(self, body: dict, tenant="default"):
        t0 = time.perf_counter()
        msgs = body.get("messages")
        if not msgs:
            raise ValueError("messages is required")
        self.s["requests"] += 1
        requested = body.get("model", "auto")
        model = route(msgs, requested)
        user = next((str(m.get("content", "")) for m in reversed(msgs) if m.get("role") == "user"), "")
        system = " ".join(str(m.get("content", "")) for m in msgs if m.get("role") == "system")

        def meta(status, sim=0.0, matched=None, resp=None, saved=False):
            if saved and resp:
                tok = resp.get("usage", {}).get("total_tokens", 0)
                self.s["tokens_saved"] += tok
                self.s["cost_saved_usd"] += tok / 1000 * settings.usd_per_1k_tokens
            return {"status": status, "similarity": round(sim, 4), "matched_prompt": matched,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "threshold": self.gov.threshold}

        if not cacheable(body):
            self.s["bypass"] += 1
            return await self.provider.chat(body, model), meta("BYPASS")

        prompt = redact(user)  # PII never reaches the embedder, store or logs
        ns = f"{tenant}:{hashlib.sha1(system.encode()).hexdigest()[:8]}"
        key = hashlib.sha256(f"{ns}|{requested}|{prompt}".encode()).hexdigest()

        cached = await self.l1.get(key)
        if cached:
            self.s["hits"] += 1
            return cached, meta("HIT-EXACT", 1.0, prompt, cached, True)

        vec = await self.embedder.embed(prompt)
        entry, score = await self.store.search(vec, ns, requested)
        status, thr = "MISS", self.gov.threshold
        if entry:
            safe = guard(prompt, entry.prompt)
            if score >= thr:
                if safe:
                    status = "HIT"
                else:
                    self.gov.record("guard_reject", score)
            elif score >= thr - settings.margin and safe:
                if await self.verifier.judge(prompt, entry.prompt):
                    status = "VERIFIED"
                    self.gov.record("borderline_ok", score)
                else:
                    self.gov.record("borderline_no", score)

        if status != "MISS":
            entry.hits += 1
            entry.last_used = time.time()
            self.s["hits"] += 1
            resp = copy.deepcopy(entry.response)
            await self.l1.set(key, resp, ttl_for(prompt, settings.ttl))
            return resp, meta(status, score, entry.prompt, resp, True)

        self.s["misses"] += 1
        resp = await self.provider.chat(body, model)
        ttl = ttl_for(prompt, settings.ttl)
        await self.store.add(Entry(ns=ns, model=requested, prompt=prompt, response=resp, vec=vec, ttl=ttl))
        await self.l1.set(key, resp, ttl)
        return resp, meta("MISS", score, entry.prompt if entry else None)

    async def stats(self):
        s = dict(self.s)
        served = s["hits"] + s["misses"]
        s.update(hit_rate=round(s["hits"] / served, 4) if served else 0.0, threshold=self.gov.threshold,
                 entries=await self.store.size(), embedder=self.embedder.name, provider=self.provider.name,
                 cost_saved_usd=round(s["cost_saved_usd"], 6))
        return s

    async def call_tool(self, name, args):
        if name == "cache_stats":
            return await self.stats()
        if name == "agent_journal":
            return list(self.gov.journal)[:20]
        if name == "invalidate":
            return {"removed": await self.store.invalidate(contains=args["contains"])}
        if name == "tune_threshold":
            return {"threshold": self.gov.tune(args["value"], args.get("reason", "agent tool call"))}
        raise KeyError(name)
