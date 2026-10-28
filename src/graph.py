"""LangGraph 编排：意图路由 + 四个子 Agent + 记忆。

为什么拆多 Agent 而不是一条链：
- 闲聊去检索，白白花一次向量查询还容易带出无关内容；
- 查业务数据必须走接口，让模型"从文档里猜"会直接产生幻觉数据；
- 长报告生成需要不同 Prompt 和分段策略，混在普通问答里会拖慢首 token。

意图分类先过规则再过模型 —— 高频意图规则能拦掉约 45%，省下的都是真金白银的 token。
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from .rag import HybridRetriever, Retrieved

# --------------------------------------------------------------------------- #
# 意图
# --------------------------------------------------------------------------- #
class Intent(str, Enum):
    CHAT = "chat"  # 闲聊 / 打招呼
    RETRIEVAL = "retrieval"  # 查知识库
    TOOL = "tool"  # 查业务系统
    REPORT = "report"  # 生成长报告


# 规则关键词：命中即直接分类，不调模型
KEYWORD_RULES: dict[Intent, tuple[str, ...]] = {
    Intent.CHAT: ("你好", "hi", "hello", "谢谢", "再见", "在吗"),
    Intent.REPORT: ("报告", "总结一份", "汇总", "生成报告", "周报", "月报"),
    Intent.TOOL: ("订单", "库存", "物流", "余额", "工单状态", "查一下我的"),
}

INTENT_SYSTEM = """你是意图分类器。判断用户提问属于哪一类，只输出一个词：
chat（闲聊寒暄）/ retrieval（咨询公司制度、产品文档等知识库内容）/
tool（查询订单、库存等业务系统实时数据）/ report（需要生成结构化长报告）。
不要输出解释。"""


def classify_intent(query: str, llm=None) -> Intent:
    """规则优先，未命中再调 LLM。"""
    lowered = query.lower()
    for intent, keywords in KEYWORD_RULES.items():
        if any(kw in lowered for kw in keywords):
            logger.debug("规则命中意图：{}", intent.value)
            return intent
    if llm is None:
        return Intent.RETRIEVAL
    try:
        raw = llm.chat(INTENT_SYSTEM, query, temperature=0.0).strip().lower()
        for intent in Intent:
            if intent.value in raw:
                return intent
    except Exception as exc:
        logger.warning("意图分类失败，默认走检索：{}", exc)
    return Intent.RETRIEVAL


# --------------------------------------------------------------------------- #
# 状态
# --------------------------------------------------------------------------- #
class AgentState(BaseModel):
    query: str
    session_id: str = "default"
    intent: Intent = Intent.RETRIEVAL
    history: list[dict[str, str]] = Field(default_factory=list)
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    rejected: bool = False
    trace: list[str] = Field(default_factory=list)
    started_at: float = Field(default_factory=time.perf_counter)


# --------------------------------------------------------------------------- #
# 各 Agent
# --------------------------------------------------------------------------- #
ANSWER_SYSTEM = """你是企业智能客服。仅依据给定的【知识库片段】回答用户问题。

硬性要求：
1. 只能使用片段中的信息，片段没有的内容一律回答"知识库中没有相关信息"；
2. 每个关键结论后用 [来源: 文件名] 标注出处；
3. 不要编造数字、日期、人名；
4. 答案简洁，控制在 200 字以内。"""

REPORT_SYSTEM = """你是企业报告撰写助手。基于检索到的资料生成结构化报告。
输出 Markdown，包含：背景、现状、问题、建议 四个小节，每节 3-5 条要点。"""


class AgentPlatform:
    """把四个子 Agent 组装成可执行的 LangGraph 应用。"""

    def __init__(self, retriever: HybridRetriever | None = None, llm=None, memory=None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.memory = memory  # 可选：RedisMemory / InMemoryMemory
        self.app = self._build()

    # ---------------- 节点 ---------------- #
    def node_route(self, state: AgentState) -> AgentState:
        state.intent = classify_intent(state.query, self.llm)
        state.trace.append(f"route: intent={state.intent.value}")
        return state

    def node_load_history(self, state: AgentState) -> AgentState:
        if self.memory is not None:
            try:
                state.history = self.memory.load(state.session_id)
                state.trace.append(f"memory: 载入 {len(state.history)} 轮历史")
            except Exception as exc:
                logger.warning("记忆载入失败：{}", exc)
        return state

    def node_chat(self, state: AgentState) -> AgentState:
        state.answer = "您好，我是企业智能助手。可以问我公司制度、产品文档，或查询订单等业务数据。"
        state.trace.append("chat: 直接应答")
        return state

    def node_retrieval(self, state: AgentState) -> AgentState:
        if self.retriever is None:
            state.answer = "知识库未初始化。"
            return state

        results: list[Retrieved] = self.retriever.search(state.query, top_k=5)
        if self.retriever.should_reject(results):
            state.rejected = True
            state.answer = "知识库中没有相关信息，建议咨询对应部门或补充文档后重试。"
            state.trace.append("retrieval: 低分拒答")
            return state

        state.retrieved = [
            {"text": r.chunk.text, "source": r.chunk.source, "score": round(r.score, 3)} for r in results
        ]
        state.citations = sorted({r.chunk.source for r in results})
        state.trace.append(f"retrieval: 召回 {len(results)} 段")

        context = "\n\n".join(
            f"[{i + 1}] 来源:{r.chunk.source}｜{r.chunk.heading}\n{r.chunk.text}"
            for i, r in enumerate(results)
        )
        prompt = f"【知识库片段】\n{context}\n\n【用户问题】\n{state.query}"

        if self.llm is None:
            state.answer = self._offline_answer(results)
            return state
        try:
            state.answer = self.llm.chat(ANSWER_SYSTEM, prompt, temperature=0.2)
        except Exception as exc:
            logger.error("生成失败：{}", exc)
            state.answer = self._offline_answer(results)
        return state

    def node_tool(self, state: AgentState) -> AgentState:
        """工具 Agent：按 query 里的实体挑工具调用。

        真实实现走 Function Calling / MCP（见 mcp.py），
        这里保留可独立运行的降级路径。
        """
        from .mcp import dispatch_tool, extract_tool_call

        call = extract_tool_call(state.query)
        if call is None:
            state.answer = "已识别为业务查询，但未匹配到可用工具接口，请补充单号等关键信息。"
            state.trace.append("tool: 未匹配工具")
            return state

        result = dispatch_tool(call["name"], call.get("arguments", {}))
        state.tool_results.append({"tool": call["name"], "result": result})
        state.trace.append(f"tool: 调用 {call['name']}")

        if self.llm is None:
            state.answer = f"查询结果：{result}"
            return state
        state.answer = self.llm.chat(
            "你是客服助手，把下面的接口返回整理成一句通顺的中文回答，不要添加接口中没有的信息。",
            f"用户问题：{state.query}\n接口返回：{result}",
            temperature=0.1,
        )
        return state

    def node_report(self, state: AgentState) -> AgentState:
        if self.retriever is not None:
            results = self.retriever.search(state.query, top_k=8)
            state.retrieved = [
                {"text": r.chunk.text, "source": r.chunk.source} for r in results
            ]
            state.citations = sorted({r.chunk.source for r in results})
        context = "\n\n".join(f"- {d['text']}" for d in state.retrieved[:8])
        if self.llm is None:
            state.answer = f"# 报告：{state.query}\n\n（离线模式，基于 {len(state.retrieved)} 段资料生成）\n\n{context[:800]}"
            return state
        state.answer = self.llm.chat(
            REPORT_SYSTEM, f"主题：{state.query}\n\n资料：\n{context}", temperature=0.4
        )
        return state

    def node_save(self, state: AgentState) -> AgentState:
        if self.memory is not None:
            try:
                self.memory.append(state.session_id, state.query, state.answer)
                state.trace.append("memory: 已写入")
            except Exception as exc:
                logger.warning("记忆写入失败：{}", exc)
        elapsed = time.perf_counter() - state.started_at
        state.trace.append(f"done: {elapsed:.2f}s")
        return state

    # ---------------- 条件边 ---------------- #
    @staticmethod
    def route_by_intent(state: AgentState) -> str:
        return state.intent.value

    # ---------------- 组装 ---------------- #
    def _build(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            logger.warning("langgraph 未安装，使用顺序执行兼容模式")
            return None

        graph = StateGraph(AgentState)
        graph.add_node("route", self.node_route)
        graph.add_node("history", self.node_load_history)
        graph.add_node("chat", self.node_chat)
        graph.add_node("retrieval", self.node_retrieval)
        graph.add_node("tool", self.node_tool)
        graph.add_node("report", self.node_report)
        graph.add_node("save", self.node_save)

        graph.set_entry_point("route")
        graph.add_edge("route", "history")
        graph.add_conditional_edges(
            "history",
            self.route_by_intent,
            {
                "chat": "chat",
                "retrieval": "retrieval",
                "tool": "tool",
                "report": "report",
            },
        )
        for node in ("chat", "retrieval", "tool", "report"):
            graph.add_edge(node, "save")
        graph.add_edge("save", END)
        return graph.compile()

    # ---------------- 执行 ---------------- #
    def run(self, query: str, session_id: str = "default") -> AgentState:
        state = AgentState(query=query, session_id=session_id)
        if self.app is None:
            state = self.node_route(state)
            state = self.node_load_history(state)
            handler = {
                Intent.CHAT: self.node_chat,
                Intent.RETRIEVAL: self.node_retrieval,
                Intent.TOOL: self.node_tool,
                Intent.REPORT: self.node_report,
            }[state.intent]
            state = handler(state)
            state = self.node_save(state)
            return state

        result = self.app.invoke(state)
        return AgentState(**result) if isinstance(result, dict) else result

    @staticmethod
    def _offline_answer(results: list[Retrieved]) -> str:
        """无模型时的兜底：直接把最相关片段拼出来，保证链路有输出。"""
        lines = ["（离线模式）知识库中最相关的片段："]
        for i, r in enumerate(results[:3], 1):
            lines.append(f"[{i}] {r.chunk.heading}｜{r.chunk.text[:160]}  [来源: {r.chunk.source}]")
        return "\n".join(lines)
