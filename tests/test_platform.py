# -*- coding: utf-8 -*-
"""核心链路单测（标准库 unittest）：python -m unittest discover -s tests -v"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cache import SemanticCache, compatible, similarity  # noqa: E402
from src.context import ContextBudget  # noqa: E402
from src.graph import AgentPlatform  # noqa: E402
from src.memory import InMemoryMemory  # noqa: E402
from src.rag import HybridRetriever, ingest  # noqa: E402
from src.trace import TraceRecorder, est_tokens  # noqa: E402


class TestCache(unittest.TestCase):
    def setUp(self) -> None:
        self.c = SemanticCache()

    def test_exact_hit(self) -> None:
        self.c.put("年假有几天？", "5 天")
        e, kind, _ = self.c.get("年假有几天？")
        self.assertEqual(kind, "exact")
        self.assertEqual(e.answer, "5 天")

    def test_rejected_not_cached(self) -> None:
        """拒答不入缓存：补文档后同一问题应该能重新命中知识库。"""
        self.c.put("火星报销吗？", "没查到", rejected=True)
        e, kind, _ = self.c.get("火星报销吗？")
        self.assertIsNone(e)
        self.assertEqual(kind, "miss")

    def test_slot_guard_blocks_near_miss(self) -> None:
        """一线/二线句式相同但答案不同，必须被槽位守卫拦住。"""
        self.assertFalse(compatible("一线城市住宿标准", "二线城市住宿标准"))
        self.assertGreater(similarity("一线城市住宿标准", "二线城市住宿标准"), 0.55)
        self.c.put("一线城市住宿标准", "600")
        e, kind, _ = self.c.get("二线城市住宿标准")
        self.assertIsNone(e)
        self.assertEqual(kind, "miss")

    def test_stats_hit_rate(self) -> None:
        self.c.put("年假有几天？", "5 天")
        self.c.get("年假有几天？")
        self.c.get("完全无关的问题")
        self.assertEqual(self.c.stats.total, 2)
        self.assertEqual(self.c.stats.hit_rate, 50.0)


class TestContext(unittest.TestCase):
    def test_history_digest(self) -> None:
        b = ContextBudget(keep_recent_turns=2)
        hist = []
        for i in range(6):
            hist.append({"role": "user", "content": f"第{i}个问题是什么？"})
            hist.append({"role": "assistant", "content": "回答" * 50})
        text = b.build_history(hist)
        self.assertIn("早前对话摘要", text)
        self.assertLess(len(text), sum(len(m["content"]) for m in hist))

    def test_snippet_trim_keeps_first(self) -> None:
        b = ContextBudget(max_snippet_tokens=10)
        out = b.trim_snippets([("很长的一段内容" * 50, 0.9), ("第二段" * 50, 0.8)])
        self.assertIn("片段1", out)
        self.assertNotIn("片段2", out)

    def test_saved_pct_zero_when_no_trim(self) -> None:
        b = ContextBudget()
        _, _, st = b.apply([], [("短片段", 0.9)])
        self.assertFalse(st.trimmed)
        self.assertEqual(st.saved_pct, 0.0)


class TestTrace(unittest.TestCase):
    def test_span_records(self) -> None:
        r = TraceRecorder()
        with r.span("t1", "node_a", "输入") as box:
            box["out"] = "输出"
        s = r.summary()
        self.assertEqual(s["spans"], 1)
        self.assertIn("node_a", s["nodes"])

    def test_est_tokens(self) -> None:
        self.assertGreater(est_tokens("中文四个字"), est_tokens("ab"))


class TestPlatform(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        chunks, _ = ingest(str(ROOT / "docs"))
        cls.p = AgentPlatform(retriever=HybridRetriever(chunks), llm=None,
                              memory=InMemoryMemory(), cache=SemanticCache(),
                              recorder=TraceRecorder(), budget=ContextBudget())

    def test_answer_with_citation(self) -> None:
        st = self.p.run("一线城市住宿报销标准是多少？", "t-1")
        self.assertTrue(st.answer)
        self.assertTrue(st.citations)

    def test_cache_second_hit(self) -> None:
        self.p.cache.clear()
        self.p.run("二线城市住宿标准是多少？", "t-2")
        st = self.p.run("二线城市住宿标准是多少？", "t-2")
        self.assertEqual(st.cache_hit, "exact")

    def test_memory_grows(self) -> None:
        self.p.memory.clear("t-3")
        self.p.run("年假有几天？", "t-3")
        self.p.run("病假需要什么材料？", "t-3")
        self.assertEqual(len(self.p.memory.load("t-3")), 4)

    def test_multi_turn_uses_history(self) -> None:
        self.p.memory.clear("t-4")
        self.p.run("年假有几天？", "t-4")
        st = self.p.run("那病假呢？", "t-4")
        self.assertGreaterEqual(len(st.history), 2)

    def test_trace_covers_nodes(self) -> None:
        st = self.p.run("请假流程是什么？", "t-5")
        self.assertTrue(st.trace_id)
        nodes = {s.node for s in self.p.recorder.spans if s.trace_id == st.trace_id}
        self.assertTrue({"route", "retrieval", "save"} <= nodes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
