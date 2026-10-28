"""评估模块：把"感觉变好了"变成可回归的数字。

评估集每条含：question / expected_answer_keywords / expected_source。
指标：
- 召回命中率：期望来源是否出现在 citations 里；
- 关键词覆盖率：标准答案关键词在生成答案中的命中比例；
- 拒答率：被认为知识库无依据而拒答的比例（过高说明检索不行，过低说明幻觉风险）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .graph import AgentPlatform, Intent


@dataclass
class EvalCase:
    question: str
    keywords: list[str]
    source: str = ""
    intent: str = ""


@dataclass
class EvalResult:
    total: int
    source_hit: int
    keyword_coverage: float
    rejected: int
    intent_correct: int

    def summary(self) -> dict[str, float]:
        return {
            "total": self.total,
            "source_hit_rate": round(self.source_hit / max(self.total, 1), 3),
            "keyword_coverage": round(self.keyword_coverage, 3),
            "reject_rate": round(self.rejected / max(self.total, 1), 3),
            "intent_accuracy": round(self.intent_correct / max(self.total, 1), 3),
        }


def load_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        cases.append(
            EvalCase(
                question=raw["question"],
                keywords=raw.get("keywords", []),
                source=raw.get("source", ""),
                intent=raw.get("intent", ""),
            )
        )
    return cases


def run_eval(platform: AgentPlatform, cases: list[EvalCase]) -> EvalResult:
    source_hit = 0
    rejected = 0
    intent_correct = 0
    coverage_sum = 0.0

    for case in cases:
        state = platform.run(case.question, session_id=f"eval_{abs(hash(case.question)) % 10000}")

        if state.rejected:
            rejected += 1

        if case.source and case.source in state.citations:
            source_hit += 1

        if case.intent and state.intent.value == case.intent:
            intent_correct += 1

        if case.keywords:
            hits = sum(1 for kw in case.keywords if kw in state.answer)
            coverage_sum += hits / len(case.keywords)

    result = EvalResult(
        total=len(cases),
        source_hit=source_hit,
        keyword_coverage=coverage_sum / max(len(cases), 1),
        rejected=rejected,
        intent_correct=intent_correct,
    )
    logger.info("评估完成：{}", result.summary())
    return result
