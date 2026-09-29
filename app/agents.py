"""Agentic layer: Verifier (judges borderline hits), Governor (self-tunes the threshold), Router (cost-aware)."""
import re
import time
from collections import deque

from .config import settings

HARD = re.compile(r"(step[- ]by[- ]step|prove|derive|refactor|architecture|optimi[sz]e|```|\bdef |\bclass |\bsql\b)", re.I)


def route(messages, requested: str) -> str:
    """Model router: 'auto' sends short/simple prompts to the cheap tier and hard ones to the strong tier."""
    if requested != "auto":
        return requested
    text = " ".join(str(m.get("content", "")) for m in messages)
    return settings.strong_model if len(text) > 1500 or HARD.search(text) else settings.cheap_model


class Verifier:
    """Agent that decides if a borderline match (just under the threshold) really means the same thing."""

    def __init__(self, provider):
        self.provider = provider

    async def judge(self, a: str, b: str) -> bool:
        try:
            return await self.provider.judge(a, b)
        except Exception:
            return False  # fail closed: never serve a doubtful answer


class Governor:
    """Agent that watches evidence and tunes the similarity threshold within safe bounds, with a public journal.

    - guard_reject: a match above threshold was rejected by the safety guard -> threshold is too loose
    - borderline_ok: a match below threshold was approved by the Verifier   -> threshold is too tight
    """

    def __init__(self, base, lo=0.80, hi=0.97, every=5):
        self.threshold, self.lo, self.hi, self.every = base, lo, hi, every
        self.win, self.journal = [], deque(maxlen=200)
        self.log(f"Started with threshold {base:.2f}")

    def log(self, msg):
        self.journal.appendleft({"t": time.strftime("%H:%M:%S"), "msg": msg})

    def record(self, kind, score):
        self.win.append((kind, score))
        if len(self.win) >= self.every:
            self.review()

    def review(self):
        rej = sum(k == "guard_reject" for k, _ in self.win)
        ok = sum(k == "borderline_ok" for k, _ in self.win)
        no = sum(k == "borderline_no" for k, _ in self.win)
        old, new, why = self.threshold, self.threshold, f"holding: {rej} rejects, {ok} approvals, {no} denials"
        if rej > ok and rej >= 2:
            new, why = min(self.hi, old + 0.01), f"tighten: {rej} unsafe matches above threshold"
        elif ok > rej and ok >= 2 and no <= ok:
            new, why = max(self.lo, old - 0.01), f"loosen: verifier approved {ok} near-misses"
        self.threshold = round(new, 3)
        self.log(f"{old:.2f} -> {self.threshold:.2f} ({why})")
        self.win = []

    def tune(self, value: float, reason="manual"):
        self.threshold = round(min(self.hi, max(self.lo, float(value))), 3)
        self.log(f"set to {self.threshold:.2f} ({reason})")
        return self.threshold
