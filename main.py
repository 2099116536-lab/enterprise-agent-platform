"""CLI 入口。

    python main.py ingest --docs ./docs
    python main.py ask "差旅报销标准是多少？" --mock
    python main.py eval --docs ./docs --set eval/qa_set.jsonl --mock
"""

from __future__ import annotations

import argparse
import json
import sys

from loguru import logger

from src.eval import load_cases, run_eval
from src.graph import AgentPlatform
from src.rag import HybridRetriever, ingest

DEMO_DOCS = {
    "expense.md": """# 差旅报销标准

## 住宿标准
一线城市每晚不超过 600 元，二线城市不超过 400 元，其他城市不超过 300 元。
超标部分需部门负责人审批。

## 交通标准
高铁二等座，飞机经济舱。连续乘车超过 6 小时可报销卧铺。

## 报销时限
出差结束后 30 个工作日内提交，跨年度不予报销。
""",
    "leave.md": """# 休假制度

## 年假
入职满 1 年享有 5 天年假，每满 1 年增加 1 天，上限 15 天。

## 病假
需提供二级及以上医院证明，全年累计不超过 30 天。

## 请假流程
在 OA 系统提交申请，1 天内由直属主管审批，3 天以上需部门负责人审批。
""",
}


def build_platform(args) -> AgentPlatform:
    from pathlib import Path

    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        logger.warning("文档目录不存在，生成内置演示文档：{}", docs_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)
        for name, content in DEMO_DOCS.items():
            (docs_dir / name).write_text(content, encoding="utf-8")

    chunks, stats = ingest(str(docs_dir))
    logger.info("入库：{} 文件 / {} 切片", stats.files, stats.chunks)

    llm = None
    if not args.mock:
        try:
            from src.llm import get_llm

            llm = get_llm()
            logger.info("模型：{}", llm.name)
        except Exception as exc:
            logger.warning("模型初始化失败，进入离线模式：{}", exc)

    return AgentPlatform(retriever=HybridRetriever(chunks), llm=llm)


def main() -> int:
    parser = argparse.ArgumentParser(description="Enterprise Agent Platform")
    parser.add_argument("command", choices=["ingest", "ask", "eval"])
    parser.add_argument("query_pos", nargs="?", default="", help="ask 命令的问题，可直接跟在 ask 后面")
    parser.add_argument("--docs", default="./docs")
    parser.add_argument("--query", default="")
    parser.add_argument("--set", dest="qa_set", default="eval/qa_set.jsonl")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.command == "ingest":
        chunks, stats = ingest(args.docs)
        print(json.dumps({"files": stats.files, "chunks": stats.chunks, "dropped": stats.dropped}, ensure_ascii=False))
        return 0

    platform = build_platform(args)

    if args.command == "ask":
        query = args.query or args.query_pos
        if not query:
            logger.error("需要提供问题，例如：python main.py ask \"差旅报销标准是多少？\"")
            return 2
        state = platform.run(query)
        print(json.dumps(
            {
                "intent": state.intent.value,
                "answer": state.answer,
                "citations": state.citations,
                "rejected": state.rejected,
                "trace": state.trace,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    if args.command == "eval":
        from pathlib import Path

        qa_path = Path(args.qa_set)
        if not qa_path.exists():
            logger.warning("评估集不存在，使用 3 条内置样例：{}", qa_path)
            cases = [
                __import__("src.eval", fromlist=["EvalCase"]).EvalCase(
                    question=q, keywords=k, source=s, intent="retrieval"
                )
                for q, k, s in [
                    ("一线城市住宿报销标准是多少？", ["600"], "expense.md"),
                    ("年假有几天？", ["5", "15"], "leave.md"),
                    ("出差报销要多久之内提交？", ["30"], "expense.md"),
                ]
            ]
        else:
            cases = load_cases(qa_path)
        result = run_eval(platform, cases)
        print(json.dumps(result.summary(), ensure_ascii=False, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
