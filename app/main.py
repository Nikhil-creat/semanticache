"""HTTP layer: OpenAI-compatible proxy + metrics + agent tools."""
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from .config import settings
from .core import TOOLS, Engine

app = FastAPI(title="SemantiCache", version="1.0.0",
              description="Semantic caching proxy for LLM APIs with self-tuning agents.")
engine = Engine()

REQ = Counter("semanticache_requests_total", "Requests by cache status", ["status"])
LAT = Histogram("semanticache_latency_seconds", "Latency by cache status", ["status"],
                buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2, 5, 10))
TOK = Counter("semanticache_tokens_saved_total", "Tokens not sent to the provider")
USD = Counter("semanticache_cost_saved_usd_total", "Estimated USD saved")
THR = Gauge("semanticache_threshold", "Current similarity threshold")
SIZE = Gauge("semanticache_entries", "Entries in the cache")


def admin(token: str):
    if settings.admin_token and token != settings.admin_token:
        raise HTTPException(401, "invalid admin token")


@app.post("/v1/chat/completions")
async def chat(req: Request, x_tenant: str = Header("default")):
    body = await req.json()
    before = engine.s["tokens_saved"], engine.s["cost_saved_usd"]
    try:
        resp, m = await engine.handle(body, x_tenant)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # upstream failure
        raise HTTPException(502, f"upstream error: {e}")
    REQ.labels(m["status"]).inc()
    LAT.labels(m["status"]).observe(m["latency_ms"] / 1000)
    TOK.inc(engine.s["tokens_saved"] - before[0])
    USD.inc(engine.s["cost_saved_usd"] - before[1])
    THR.set(m["threshold"])
    resp = {**resp, "semanticache": m}
    return JSONResponse(resp, headers={"X-Cache": m["status"], "X-Similarity": str(m["similarity"]),
                                       "X-Latency-Ms": str(m["latency_ms"])})


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/stats")
async def stats():
    return await engine.stats()


@app.get("/metrics")
async def metrics():
    SIZE.set(await engine.store.size())
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/agent/journal")
async def journal():
    return list(engine.gov.journal)


@app.get("/agent/tools")
async def tools():
    return TOOLS


@app.post("/agent/call")
async def call(req: Request, x_admin_token: str = Header("")):
    admin(x_admin_token)
    b = await req.json()
    try:
        return await engine.call_tool(b["tool"], b.get("args", {}))
    except KeyError:
        raise HTTPException(404, "unknown tool")
