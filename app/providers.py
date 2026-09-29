"""LLM providers. MockProvider lets the whole stack run with no API key."""
import asyncio
import hashlib

from .config import settings
from .embeddings import tokens


def _last_user(body):
    return next((str(m.get("content", "")) for m in reversed(body["messages"]) if m.get("role") == "user"), "")


class MockProvider:
    name = "mock"

    async def chat(self, body, model):
        await asyncio.sleep(settings.mock_latency)  # simulate model latency
        q = _last_user(body)
        text = f"[mock:{model}] Simulated answer to: {q}"
        return {
            "id": "chatcmpl-" + hashlib.md5(text.encode()).hexdigest()[:12],
            "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(q) // 4 + 1, "completion_tokens": len(text) // 4 + 1,
                      "total_tokens": (len(q) + len(text)) // 4 + 2},
        }

    async def judge(self, a, b):
        A, B = set(tokens(a)), set(tokens(b))
        return len(A & B) / max(1, len(A | B)) >= 0.6


class OpenAIProvider:
    name = "openai"

    async def _post(self, payload):
        import httpx

        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{settings.openai_base}/chat/completions",
                             headers={"Authorization": f"Bearer {settings.openai_key}"}, json=payload)
        r.raise_for_status()
        return r.json()

    async def chat(self, body, model):
        return await self._post({**body, "model": model})

    async def judge(self, a, b):
        q = (f"Do these two user questions ask for the same answer? Reply YES or NO only.\n"
             f"A: {a}\nB: {b}")
        r = await self._post({"model": settings.cheap_model, "temperature": 0, "max_tokens": 3,
                              "messages": [{"role": "user", "content": q}]})
        return r["choices"][0]["message"]["content"].strip().upper().startswith("YES")


def make_provider():
    return OpenAIProvider() if settings.openai_key else MockProvider()
