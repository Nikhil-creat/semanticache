"""Embedders. LocalEmbedder needs no API key (hashed word + char n-grams); OpenAIEmbedder is production."""
import re
import zlib

import numpy as np

from .config import settings

STOP = set(
    "a an the is are was were be to of in on for and or i you me my it its can could would should please "
    "tell explain what how do does did with about".split()
)


def tokens(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class LocalEmbedder:
    dim = 512
    name = "local-hash-512"

    async def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokens(text)
        feats = [(w, 1.0) for w in toks]
        feats += [(a + "_" + b, 0.5) for a, b in zip(toks, toks[1:])]
        for w in toks:
            p = f"<{w}>"
            feats += [(p[i:i + 3], 0.3) for i in range(len(p) - 2)]
        for f, wt in feats:
            h = zlib.crc32(f.encode())
            v[h % self.dim] += wt if (h >> 20) & 1 else -wt
        n = np.linalg.norm(v)
        return v / n if n else v


class OpenAIEmbedder:
    dim = 1536
    name = settings.embed_model

    async def embed(self, text: str) -> np.ndarray:
        import httpx

        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{settings.openai_base}/embeddings",
                headers={"Authorization": f"Bearer {settings.openai_key}"},
                json={"model": settings.embed_model, "input": text},
            )
        r.raise_for_status()
        v = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
        return v / np.linalg.norm(v)


def make_embedder():
    return OpenAIEmbedder() if settings.openai_key else LocalEmbedder()
