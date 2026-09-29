"""Cache policy: PII redaction, volatility-aware TTL, and the deterministic safety guard."""
import re

PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "<CARD>"),
    (re.compile(r"\+?\d[\d\s().-]{8,}\d"), "<PHONE>"),
]
VOLATILE = re.compile(r"\b(today|now|latest|current|currently|tonight|tomorrow|yesterday|price|weather|score|news)\b", re.I)
NEG = re.compile(r"\b(not|no|never|without|dont|don't|cannot|can't|isn't|won't)\b", re.I)
NUM = re.compile(r"\d+(?:\.\d+)?")


def redact(text: str) -> str:
    for rx, tag in PII:
        text = rx.sub(tag, text)
    return text


def ttl_for(prompt: str, base: int) -> int:
    return max(60, base // 20) if VOLATILE.search(prompt) else base


def cacheable(body: dict) -> bool:
    return not body.get("stream") and not body.get("tools") and float(body.get("temperature", 0) or 0) <= 0.7


def guard(a: str, b: str) -> bool:
    """Near-identical embeddings can hide opposite meaning ('is X safe' vs 'is X not safe', 15% vs 25%)."""
    return set(NUM.findall(a)) == set(NUM.findall(b)) and bool(NEG.search(a)) == bool(NEG.search(b))
