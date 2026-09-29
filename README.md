# SemantiCache

A semantic caching proxy for LLM APIs. It sits in front of any OpenAI-compatible
endpoint, recognizes when a new prompt means the same thing as one it already
answered — even worded differently — and returns the stored answer instead of
paying for another model call. A small set of agents keep that safe and tuned
without a human watching a dashboard.

**Live demo (client-side simulator, no backend needed):** GitHub Pages, `docs/index.html`
**Full stack:** FastAPI + Redis (exact cache) + Qdrant (semantic cache) + Prometheus + Grafana

## Designed and Developed by 
# **NIKHIL CHARY SRIRAMOJU**
BTech CSE (Final Year)

- GitHub: [Nikhil-creat](https://github.com/Nikhil-creat)
- LinkedIn: [nikhil-chary-sriramoju](https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a)
- Email: sriramojunikhil66@gmail.com
- Instagram: [@nikhil__sriramoju](https://www.instagram.com/nikhil__sriramoju)
- Facebook: [Profile](https://www.facebook.com/profile.php?id=100079201124141)

## Why a cache isn't enough on its own

A plain vector-similarity cache is dangerous: "Is 15% too much interest?" and
"Is 25% too much interest?" embed almost identically, and so do "is this
medicine safe" and "is this medicine *not* safe." Serving the wrong one back
with confidence is worse than a cache miss. SemantiCache treats that as the
core problem, not an edge case:

- **Deterministic safety guard** — numbers and negations must match between
  a prompt and its candidate before a semantic hit is ever allowed, regardless
  of embedding similarity.
- **Verifier agent** — matches that land just under the similarity threshold
  go to a cheap model that judges whether the two prompts really want the same
  answer. Ties are broken by *not* serving the cache (fail closed).
- **Governor agent** — watches how often the guard blocks a should-have-been-fine
  match versus how often the verifier approves a near-miss, and nudges the
  threshold up or down (within hard bounds) over time. It keeps a plain-English
  journal of every change so the tuning is auditable, not a black box.
- **Cost-aware router** — `model: "auto"` sends short, simple prompts to a cheap
  model and anything that looks like code, math, or multi-step reasoning to a
  stronger one.
- **PII redaction & tenant isolation** — emails, card numbers, and phone numbers
  never reach the embedder or the store; each tenant and system prompt gets its
  own cache namespace so answers never leak across customers.
- **Agent tool interface** — `/agent/tools` publishes an OpenAI-style function
  schema (`cache_stats`, `agent_journal`, `invalidate`, `tune_threshold`) so
  another AI agent, not just a human, can operate the cache.

## Architecture

```
client ──▶ FastAPI proxy ──▶ L1 exact cache (Redis)
              │                    │ miss
              │              embed prompt (OpenAI embeddings, or local
              │              hash-embedder with zero API key)
              │                    │
              │              L2 semantic search (Qdrant)
              │                    │
              │        score ≥ threshold + guard? ──▶ HIT
              │        score in [threshold-margin, threshold)?
              │              └─ guard + Verifier agent ──▶ VERIFIED
              │        else ──▶ MISS, call the real model, store the answer
              │                    │
              └── Governor agent watches guard/verifier outcomes,
                  retunes threshold, writes to its journal
```

Everything runs with **zero external dependencies and no API key**: without
`OPENAI_API_KEY` set, a local hashed-embedding function and a mock model
provider stand in, so the whole demo, test suite, and simulator work offline.
Set the key and it talks to real OpenAI models.

## Quickstart

```bash
git clone https://github.com/YOUR-USERNAME/semanticache
cd semanticache
cp .env.example .env
docker compose up --build       # API on :8000, Grafana on :3000, Prometheus on :9090
python scripts/simulate.py      # fires a paraphrase-heavy workload, prints hit/miss per request
```

Point any OpenAI SDK at `http://localhost:8000/v1` and it works as a drop-in
proxy — same request/response shape, plus a `semanticache` field and
`X-Cache` / `X-Similarity` / `X-Latency-Ms` headers on every response.

No Docker? `pip install -r requirements.txt && uvicorn app.main:app --reload`
runs the API alone with the in-memory store (no Redis/Qdrant needed).

## Endpoints

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible; cached transparently |
| `GET /stats` | hit rate, tokens/USD saved, current threshold |
| `GET /metrics` | Prometheus exposition |
| `GET /agent/journal` | Governor's tuning history |
| `GET /agent/tools` | function-calling schema for the cache's own tools |
| `POST /agent/call` | invoke a tool (`ADMIN_TOKEN` header required if set) |

## Tests

```bash
python -m unittest discover -s tests -v
```

Ten tests cover exact/semantic hits, the negation and number guard, PII
redaction, tenant isolation, bypass rules (streaming, high temperature), the
Verifier, and the Governor's threshold tuning and clamping.

## Repo layout

```
app/         engine, agents, providers, embedders, vector stores, FastAPI layer
tests/       unittest suite (offline, mock provider)
scripts/     simulate.py — load-test / demo script against a running proxy
monitoring/  Prometheus config + Grafana datasource/dashboard provisioning
docs/        GitHub Pages site with an in-browser simulator (same algorithm, no backend)
.github/     CI (tests) + Pages deploy workflow
```

## Roadmap ideas

- Streaming responses replayed token-by-token from a cached completion
- Per-tenant threshold overrides and budget alerts
- A second Verifier vote (self-consistency) before high-stakes invalidations
