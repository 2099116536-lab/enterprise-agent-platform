# -*- coding: utf-8 -*-
"""节点级 Trace：每次请求记录每个节点的耗时、输入输出摘要、token 估算。

为什么单独做一层，而不是在节点里 print
- 31% 的目标 JD 明确要求"日志/监控/链路追踪"，面试会问"线上慢了怎么定位"，
  答案是"看每个节点耗时分布"，没有这层就只能猜；
- Trace 落 JSONL，可以直接被 jq/Excel 消费，也能聚合成 /metrics 给监控用；
- 用上下文管理器包节点，业务代码零侵入（加一行 with 即可）。

设计取舍：不引 OpenTelemetry。单机 Agent 链路只有 6-8 个节点，
自己记 span 就够，引 OTel 反而是为了简历好看而不是解决问题。
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


def est_tokens(text: str) -> int:
    """粗估 token：中文 1 字≈1 token，其他按 4 字符≈1 token。"""
    if not text:
        return 0
    cn = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return int(cn + (len(text) - cn) / 4) + 1


@dataclass
class Span:
    trace_id: str
    node: str
    ms: float
    in_tokens: int = 0
    out_tokens: int = 0
    detail: str = ""


@dataclass
class TraceRecorder:
    path: Path | None = None
    spans: list[Span] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(self, trace_id: str, node: str, input_text: str = "", detail: str = "") -> Iterator[dict[str, Any]]:
        """用法：
            with recorder.span(tid, "retrieval", query) as s:
                ...
                s["detail"] = "命中 5 条"
        """
        t0 = time.perf_counter()
        box: dict[str, Any] = {"detail": detail, "out": ""}
        try:
            yield box
        finally:
            sp = Span(
                trace_id=trace_id, node=node,
                ms=(time.perf_counter() - t0) * 1000,
                in_tokens=est_tokens(input_text),
                out_tokens=est_tokens(box.get("out", "")),
                detail=box.get("detail", ""),
            )
            with self._lock:
                self.spans.append(sp)
                if self.path:
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(asdict(sp), ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, Any]:
        """按节点聚合：次数、平均耗时、P95、总 token。给 /metrics 和压测报告用。"""
        with self._lock:
            spans = list(self.spans)
        by_node: dict[str, list[Span]] = {}
        for s in spans:
            by_node.setdefault(s.node, []).append(s)

        nodes = {}
        for node, items in by_node.items():
            ms = sorted(i.ms for i in items)
            nodes[node] = {
                "count": len(items),
                "avg_ms": round(statistics.mean(ms), 2),
                "p95_ms": round(ms[min(len(ms) - 1, int(len(ms) * 0.95))], 2),
                "avg_out_tokens": round(statistics.mean(i.out_tokens for i in items), 1),
            }
        return {
            "spans": len(spans),
            "traces": len({s.trace_id for s in spans}),
            "total_out_tokens": sum(s.out_tokens for s in spans),
            "nodes": dict(sorted(nodes.items(), key=lambda kv: -kv[1]["avg_ms"])),
        }

    def reset(self) -> None:
        with self._lock:
            self.spans.clear()


RECORDER = TraceRecorder(path=Path(__file__).resolve().parent.parent / "logs" / "trace.jsonl")
