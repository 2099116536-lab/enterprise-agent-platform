# -*- coding: utf-8 -*-
"""压测/基准：冷启动 vs 命中缓存，量化缓存与上下文压缩带来的收益。

    python -m src.bench            # 在项目根执行

三轮：
  cold        清空缓存跑一遍 60 条 → 基线延迟与 0 命中
  warm        同一批再跑一遍 → 精确缓存命中，看延迟下降
  paraphrase  换一种说法问同样的问题（10 条）→ 语义缓存命中率
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .context import ContextBudget
from .eval import load_cases
from .graph import AgentPlatform
from .memory import InMemoryMemory
from .rag import HybridRetriever, ingest

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "eval" / "qa_set.jsonl"

# 同一批问题的另一种说法（评测集之外的改写，用来测语义缓存）
PARAPHRASE = [
    ("一线城市住宿报销上限是多少？", "一线城市住宿最多能报多少"),
    ("年假有几天？", "年假多少天"),
    ("出差结束后多久内要提交报销？", "报销最晚什么时候交"),
    ("病假需要什么材料？", "请病假要提供什么证明"),
    ("请假流程是什么？", "怎么走请假流程"),
    ("二线城市住宿标准是多少？", "二线城市住宿能报多少"),
    ("高铁可以坐什么座位？", "坐高铁能报销哪些座位"),
    ("跨年度报销可以吗？", "去年的发票今年还能报吗"),
    ("请假三天以上找谁审批？", "三天以上请假谁批准"),
    ("试用期有年假吗？", "试用期能不能休年假"),
]


def build() -> AgentPlatform:
    chunks, _ = ingest(str(ROOT / "docs"))
    return AgentPlatform(retriever=HybridRetriever(chunks), llm=None,
                         memory=InMemoryMemory(max_turns=20),
                         budget=ContextBudget(max_tokens=1200))


def once(platform: AgentPlatform, queries: list[str], tag: str) -> dict:
    platform.cache.clear()
    platform.recorder.reset()
    lat = []
    for q in queries:
        t0 = time.perf_counter()
        platform.run(q, session_id=f"bench-{tag}")
        lat.append((time.perf_counter() - t0) * 1000)
    s = platform.recorder.summary()
    lat_sorted = sorted(lat)
    return {
        "round": tag, "n": len(queries),
        "total_ms": round(sum(lat), 1),
        "avg_ms": round(statistics.mean(lat), 2),
        "p95_ms": round(lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))], 2),
        "cache": platform.cache.stats.to_dict(),
        "slowest_nodes": dict(list(s["nodes"].items())[:3]),
    }


def main() -> None:
    cases = load_cases(CASES)
    queries = [c.question for c in cases]
    platform = build()

    cold = once(platform, queries, "cold")
    print(json.dumps(cold, ensure_ascii=False, indent=2))

    # warm：不清缓存，用同一批问题再问一遍
    platform.recorder.reset()
    lat = []
    for q in queries:
        t0 = time.perf_counter()
        platform.run(q, session_id="bench-warm")
        lat.append((time.perf_counter() - t0) * 1000)
    warm = {
        "round": "warm", "n": len(queries),
        "total_ms": round(sum(lat), 1),
        "avg_ms": round(statistics.mean(lat), 2),
        "p95_ms": round(sorted(lat)[min(len(lat) - 1, int(len(lat) * 0.95))], 2),
        "cache": platform.cache.stats.to_dict(),
        "slowest_nodes": dict(list(platform.recorder.summary()["nodes"].items())[:3]),
    }
    print(json.dumps(warm, ensure_ascii=False, indent=2))

    # paraphrase：换成另一种说法，测语义缓存
    platform.cache.clear()
    hits = 0
    detail = []
    for ask, other in PARAPHRASE:
        platform.run(ask, session_id="bench-para")
        st = platform.run(other, session_id="bench-para")
        hit = st.cache_hit
        hits += hit != "miss"
        detail.append({"first": ask, "again": other, "cache": hit})
    para = {
        "round": "paraphrase", "n": len(PARAPHRASE),
        "semantic_hit": hits,
        "semantic_hit_rate": round(hits / len(PARAPHRASE) * 100, 1),
        "detail": detail,
    }
    print(json.dumps(para, ensure_ascii=False, indent=2))

    # multi_turn：同一会话连问 10 轮，触发历史摘要压缩 + 片段裁剪
    platform.cache.clear()
    platform.recorder.reset()
    platform.memory.clear("bench-multi")
    for q in queries[:10]:
        platform.run(q, session_id="bench-multi")
    last = platform.budget.last_stats.to_dict()
    multi = {"round": "multi_turn", "turns": 10, "context": last,
             "memory": platform.memory.stats()}
    print(json.dumps(multi, ensure_ascii=False, indent=2))

    print("\n=== 结论 ===")
    print(f"冷启动 平均 {cold['avg_ms']}ms / P95 {cold['p95_ms']}ms（缓存命中 0）")
    print(f"重复提问 平均 {warm['avg_ms']}ms / P95 {warm['p95_ms']}ms"
          f"（精确命中 {warm['cache']['exact_hit']}/{warm['n']}）")
    print(f"换说法重问 语义命中 {hits}/{len(PARAPHRASE)} = {para['semantic_hit_rate']}%")
    st = platform.budget.last_stats
    if st.trimmed:
        print(f"多轮对话压缩 省 {st.saved_pct}% token"
              f"（{st.tokens_before}→{st.tokens_after}，历史 {st.history_in}→{st.history_kept} 条）")
    else:
        print("多轮对话压缩 未触发（历史/片段未超预算）")


if __name__ == "__main__":
    main()
