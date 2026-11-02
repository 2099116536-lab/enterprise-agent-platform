# -*- coding: utf-8 -*-
"""上下文预算管理：历史裁剪 + 抽取式摘要 + 检索片段截断。

问题：多轮对话越走越长，把全部历史和 5 条片段都塞进 Prompt，token 线性涨，
      而且模型容易被早期无关内容带偏（上下文污染）。
做法：设一个 token 预算（默认 1200），超了按"先砍片段、再压缩历史"的顺序降级：
      1) 检索片段按相似度分数从高到低取，直到预算用完；
      2) 历史保留最近 N 轮原文，更早的压成一句话摘要（抽取式，不调模型）。
取舍：摘要用抽取式而不是让模型总结 —— 总结要多一次模型调用（贵、慢、还可能编），
      而"用户早前问过差旅报销"这种信息，抽首句就够用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .trace import est_tokens


@dataclass
class ContextStats:
    history_in: int = 0
    history_kept: int = 0
    history_digested: int = 0
    snippets_in: int = 0
    snippets_kept: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def trimmed(self) -> bool:
        """是否真的发生了裁剪（没裁剪时压缩率无意义，记 0 而不是负数）。"""
        return self.history_digested > 0 or self.snippets_kept < self.snippets_in

    @property
    def saved_pct(self) -> float:
        if not self.trimmed or not self.tokens_before:
            return 0.0
        return round((1 - self.tokens_after / self.tokens_before) * 100, 1)

    def to_dict(self) -> dict:
        d = {
            "history_in": self.history_in, "history_kept": self.history_kept,
            "history_digested": self.history_digested,
            "snippets_in": self.snippets_in, "snippets_kept": self.snippets_kept,
            "tokens_before": self.tokens_before, "tokens_after": self.tokens_after,
            "trimmed": self.trimmed, "saved_pct": self.saved_pct,
        }
        return d


@dataclass
class ContextBudget:
    max_tokens: int = 1200
    keep_recent_turns: int = 3          # 最近 N 轮保留原文
    max_snippet_tokens: int = 700       # 片段最多占多少
    _last: ContextStats = field(default_factory=ContextStats)

    def build_history(self, history: list[dict[str, str]], stats: ContextStats | None = None) -> str:
        """返回可直接拼进 Prompt 的历史文本（含摘要段）。"""
        stats = stats or self._last
        stats.history_in = len(history)
        if not history:
            return ""

        recent = history[-self.keep_recent_turns * 2:]          # 一轮 = user + assistant
        older = history[:-len(recent)] if recent else history
        stats.history_kept = len(recent)

        lines: list[str] = []
        if older:
            digest = self._digest(older)
            if digest:
                lines.append(f"[早前对话摘要] {digest}")
                stats.history_digested = len(older) // 2
        for m in recent:
            role = "用户" if m.get("role") == "user" else "助手"
            text = (m.get("content") or "").strip().replace("\n", " ")
            if len(text) > 120:
                text = text[:120] + "…"
            lines.append(f"{role}：{text}")
        return "\n".join(lines)

    @staticmethod
    def _digest(history: list[dict[str, str]]) -> str:
        """抽取式摘要：取每个用户问题的主干，最多 3 条。"""
        qs: list[str] = []
        for m in history:
            if m.get("role") != "user":
                continue
            q = (m.get("content") or "").strip().replace("\n", " ")
            if not q:
                continue
            q = q.split("？")[0].split("?")[0][:24]
            if q and q not in qs:
                qs.append(q)
        if not qs:
            return ""
        head = "、".join(qs[:3])
        more = f" 等 {len(qs)} 个问题" if len(qs) > 3 else ""
        return head + more

    def trim_snippets(self, snippets: list[tuple[str, float]], stats: ContextStats | None = None) -> str:
        """snippets: [(文本, 相似度分数)]，按分数降序取到预算为止。"""
        stats = stats or self._last
        stats.snippets_in = len(snippets)
        used, kept = 0, []
        for i, (text, score) in enumerate(snippets):
            t = est_tokens(text)
            # 至少保留第一条（最相关的那条），否则这条链路没意义
            if used + t > self.max_snippet_tokens and kept:
                break
            kept.append(f"[片段{i + 1}｜相似度 {score:.3f}]\n{text}")
            used += t
        stats.snippets_kept = len(kept)
        return "\n\n".join(kept)

    def apply(self, history: list[dict[str, str]], snippets: list[tuple[str, float]]) -> tuple[str, str, ContextStats]:
        """一次性产出（历史文本, 片段文本, 统计），并把 token 前后对比写进统计。"""
        stats = ContextStats()
        before = sum(est_tokens(m.get("content") or "") for m in history) + \
            sum(est_tokens(t) for t, _ in snippets)
        hist_text = self.build_history(history, stats)
        snip_text = self.trim_snippets(snippets, stats)
        stats.tokens_before = before
        stats.tokens_after = est_tokens(hist_text) + est_tokens(snip_text)
        self._last = stats
        return hist_text, snip_text, stats

    @property
    def last_stats(self) -> ContextStats:
        return self._last
