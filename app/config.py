import os
from dataclasses import dataclass, field


def _f(name, default):
    return type(default)(os.getenv(name, default))


@dataclass
class Settings:
    openai_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    cheap_model: str = field(default_factory=lambda: os.getenv("CHEAP_MODEL", "gpt-4o-mini"))
    strong_model: str = field(default_factory=lambda: os.getenv("STRONG_MODEL", "gpt-4o"))
    threshold: float = field(default_factory=lambda: _f("SIM_THRESHOLD", 0.88))
    margin: float = field(default_factory=lambda: _f("VERIFY_MARGIN", 0.06))
    ttl: int = field(default_factory=lambda: _f("CACHE_TTL", 3600))
    max_items: int = field(default_factory=lambda: _f("MAX_ITEMS", 10000))
    backend: str = field(default_factory=lambda: os.getenv("VECTOR_BACKEND", "memory"))  # memory | qdrant
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://qdrant:6333"))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", ""))
    mock_latency: float = field(default_factory=lambda: _f("MOCK_LATENCY", 0.9))
    admin_token: str = field(default_factory=lambda: os.getenv("ADMIN_TOKEN", ""))
    usd_per_1k_tokens: float = field(default_factory=lambda: _f("USD_PER_1K_TOKENS", 0.002))


settings = Settings()
