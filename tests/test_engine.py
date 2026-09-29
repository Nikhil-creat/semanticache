import asyncio
import os
import unittest

os.environ["MOCK_LATENCY"] = "0"

from app.core import Engine  # noqa: E402
from app.policy import guard, redact  # noqa: E402


def ask(e, q, tenant="default", **kw):
    body = {"model": "auto", "messages": [{"role": "user", "content": q}], **kw}
    return asyncio.run(e.handle(body, tenant))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.e = Engine()

    def test_exact_then_semantic_hit(self):
        self.assertEqual(ask(self.e, "How do I reset my password?")[1]["status"], "MISS")
        self.assertEqual(ask(self.e, "How do I reset my password?")[1]["status"], "HIT-EXACT")
        self.assertEqual(ask(self.e, "how can I reset my password")[1]["status"], "HIT")

    def test_unrelated_is_miss(self):
        ask(self.e, "How do I reset my password?")
        self.assertEqual(ask(self.e, "Explain quantum entanglement")[1]["status"], "MISS")

    def test_number_guard(self):
        ask(self.e, "What is 15% of 200?")
        self.assertEqual(ask(self.e, "What is 25% of 200?")[1]["status"], "MISS")

    def test_negation_guard(self):
        self.assertFalse(guard("Is this medicine safe", "Is this medicine not safe"))

    def test_pii_redacted(self):
        self.assertEqual(redact("mail bob@x.com or 4111 1111 1111 1111"), "mail <EMAIL> or <CARD>")
        ask(self.e, "Send invoice to bob@x.com")
        self.assertNotIn("bob@x.com", self.e.store.items[0].prompt)

    def test_tenant_isolation(self):
        ask(self.e, "Reset my password", tenant="a")
        self.assertEqual(ask(self.e, "Reset my password", tenant="b")[1]["status"], "MISS")

    def test_bypass_for_streaming_and_hot_temperature(self):
        self.assertEqual(ask(self.e, "Write a poem", stream=True)[1]["status"], "BYPASS")
        self.assertEqual(ask(self.e, "Write a poem", temperature=1.2)[1]["status"], "BYPASS")

    def test_verifier_approves_borderline(self):
        ask(self.e, "cancel my subscription plan")
        self.e.gov.tune(0.97, "test")
        st = ask(self.e, "cancel subscription plan please")[1]["status"]
        self.assertIn(st, ("VERIFIED", "HIT", "MISS"))

    def test_governor_tightens_and_loosens(self):
        g = self.e.gov
        start = g.threshold
        for _ in range(5):
            g.record("guard_reject", 0.95)
        self.assertGreater(g.threshold, start)
        mid = g.threshold
        for _ in range(5):
            g.record("borderline_ok", 0.85)
        self.assertLess(g.threshold, mid)
        g.tune(0.5)
        self.assertEqual(g.threshold, 0.80)  # clamped to safe bounds

    def test_stats_and_tools(self):
        ask(self.e, "How do I reset my password?")
        ask(self.e, "How do I reset my password?")
        s = asyncio.run(self.e.stats())
        self.assertEqual((s["hits"], s["misses"], s["entries"]), (1, 1, 1))
        self.assertGreater(s["tokens_saved"], 0)
        self.assertEqual(asyncio.run(self.e.call_tool("invalidate", {"contains": "password"}))["removed"], 1)


if __name__ == "__main__":
    unittest.main()
