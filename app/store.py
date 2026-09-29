"""Vector stores: in-memory (default, zero-dependency) and Qdrant (production)."""
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Entry:
    ns: str
    model: str
    prompt: str
    response: dict
    vec: Optional[np.ndarray] = None
    created: float = field(default_factory=time.time)
    ttl: int = 3600
    hits: int = 0
    last_used: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class MemoryStore:
    kind = "memory"

    def __init__(self, max_items=10000):
        self.items: list[Entry] = []
        self.max_items = max_items

    def _purge(self):
        now = time.time()
        self.items = [e for e in self.items if now - e.created < e.ttl]

    async def search(self, vec, ns, model):
        self._purge()
        cands = [e for e in self.items if e.ns == ns and e.model == model]
        if not cands:
            return None, 0.0
        scores = np.stack([e.vec for e in cands]) @ vec
        i = int(scores.argmax())
        return cands[i], float(scores[i])

    async def add(self, e: Entry):
        self.items.append(e)
        if len(self.items) > self.max_items:  # LRU eviction
            self.items.sort(key=lambda x: x.last_used)
            self.items = self.items[len(self.items) - self.max_items:]

    async def invalidate(self, ns=None, contains=None) -> int:
        keep = [
            e for e in self.items
            if not ((ns is None or e.ns.startswith(ns)) and (contains is None or contains.lower() in e.prompt.lower()))
        ]
        n = len(self.items) - len(keep)
        self.items = keep
        return n

    async def size(self):
        self._purge()
        return len(self.items)


class QdrantStore:
    """Production backend. Requires `qdrant-client` and a running Qdrant (see docker-compose.yml)."""
    kind = "qdrant"

    def __init__(self, url, dim, collection="semanticache"):
        from qdrant_client import AsyncQdrantClient, models

        self.m, self.c, self.col, self.dim, self._ready = models, AsyncQdrantClient(url=url), collection, dim, False

    async def _init(self):
        if self._ready:
            return
        names = [c.name for c in (await self.c.get_collections()).collections]
        if self.col not in names:
            await self.c.create_collection(
                self.col, vectors_config=self.m.VectorParams(size=self.dim, distance=self.m.Distance.COSINE)
            )
        self._ready = True

    def _flt(self, ns=None, model=None):
        m = self.m
        must = [m.FieldCondition(key="expires", range=m.Range(gt=time.time()))]
        if ns:
            must.append(m.FieldCondition(key="ns", match=m.MatchValue(value=ns)))
        if model:
            must.append(m.FieldCondition(key="model", match=m.MatchValue(value=model)))
        return m.Filter(must=must)

    async def search(self, vec, ns, model):
        await self._init()
        r = await self.c.query_points(self.col, query=vec.tolist(), query_filter=self._flt(ns, model), limit=1)
        if not r.points:
            return None, 0.0
        p = r.points[0]
        pl = p.payload
        return Entry(ns=pl["ns"], model=pl["model"], prompt=pl["prompt"], response=pl["response"],
                     created=pl["created"], ttl=pl["ttl"], id=str(p.id)), float(p.score)

    async def add(self, e: Entry):
        await self._init()
        payload = dict(ns=e.ns, model=e.model, prompt=e.prompt, response=e.response,
                       created=e.created, ttl=e.ttl, expires=e.created + e.ttl)
        await self.c.upsert(self.col, points=[self.m.PointStruct(id=e.id, vector=e.vec.tolist(), payload=payload)])

    async def invalidate(self, ns=None, contains=None) -> int:
        await self._init()
        before = await self.size()
        await self.c.delete(self.col, points_selector=self.m.FilterSelector(filter=self._flt(ns)))
        return before - await self.size()

    async def size(self):
        await self._init()
        return (await self.c.count(self.col, exact=True)).count
