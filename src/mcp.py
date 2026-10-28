"""MCP（Model Context Protocol）适配层。

MCP 解决的是"每接一个新数据源就要手写一遍工具定义"的问题：
服务端按协议暴露工具清单，Agent 侧运行时发现、按需调用。
这里做成"能跑通的最小实现 + 可替换的适配器接口"：
- 有真实 MCP Server 时走 `langchain-mcp-adapters`；
- 没有时退化到本地工具注册表，接口形态保持一致，上层代码不用改。
"""

from __future__ import annotations

import re
from typing import Any, Callable

from loguru import logger

# --------------------------------------------------------------------------- #
# 本地工具注册表（无 MCP Server 时的等价实现）
# --------------------------------------------------------------------------- #
ToolFn = Callable[..., dict[str, Any]]

LOCAL_TOOLS: dict[str, dict[str, Any]] = {
    "query_order": {
        "schema": {
            "name": "query_order",
            "description": "根据订单号查询订单状态与物流信息",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "订单号"}},
                "required": ["order_id"],
            },
        },
        "keywords": ("订单", "物流", "快递", "发货"),
    },
    "query_inventory": {
        "schema": {
            "name": "query_inventory",
            "description": "根据 SKU 查询库存数量与仓库分布",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string", "description": "商品 SKU"}},
                "required": ["sku"],
            },
        },
        "keywords": ("库存", "存货", "还有多少", "SKU"),
    },
    "query_ticket": {
        "schema": {
            "name": "query_ticket",
            "description": "根据工单号查询工单处理进度",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string", "description": "工单号"}},
                "required": ["ticket_id"],
            },
        },
        "keywords": ("工单", "进度", "处理到哪"),
    },
}

ORDER_PATTERN = re.compile(r"[A-Z]{0,3}\d{6,}")
SKU_PATTERN = re.compile(r"\bSKU[:#]?\s*([A-Za-z0-9\-]{4,})", re.I)
TICKET_PATTERN = re.compile(r"[Tt]#?(\d{4,})")


def _query_order(order_id: str) -> dict[str, Any]:
    # 真实实现替换为内部 HTTP/RPC 调用
    return {"order_id": order_id, "status": "已发货", "carrier": "顺丰", "tracking": "SF1234567890"}


def _query_inventory(sku: str) -> dict[str, Any]:
    return {"sku": sku, "total": 128, "warehouses": {"华东仓": 80, "华南仓": 48}}


def _query_ticket(ticket_id: str) -> dict[str, Any]:
    return {"ticket_id": ticket_id, "status": "处理中", "owner": "客服二组", "eta_hours": 6}


IMPLEMENTATIONS: dict[str, ToolFn] = {
    "query_order": _query_order,
    "query_inventory": _query_inventory,
    "query_ticket": _query_ticket,
}


# --------------------------------------------------------------------------- #
# 工具发现
# --------------------------------------------------------------------------- #
def list_tools(source: str = "local") -> list[dict[str, Any]]:
    """返回工具 schema 列表（OpenAI Function Calling 格式）。"""
    if source == "mcp":
        try:
            return _list_tools_from_mcp()
        except Exception as exc:
            logger.warning("MCP 工具发现失败，回退本地注册表：{}", exc)
    return [tool["schema"] for tool in LOCAL_TOOLS.values()]


def _list_tools_from_mcp() -> list[dict[str, Any]]:
    """从 MCP Server 拉取工具清单。

    生产环境换成异步 client：
        from langchain_mcp_adapters.client import MultiServerMCPClient
        client = MultiServerMCPClient({...})
        return await client.get_tools()
    """
    raise NotImplementedError("未配置 MCP Server，请在 config 中提供 endpoint")


def extract_tool_call(query: str) -> dict[str, Any] | None:
    """业务查询场景下，从用户语句里挑工具并抽出参数。

    真实系统这一步应由模型的 Function Calling 完成；
    这里用规则实现，是为了让链路在没有 Key 时也能端到端演示。
    """
    if any(kw in query for kw in LOCAL_TOOLS["query_order"]["keywords"]):
        m = ORDER_PATTERN.search(query)
        if m:
            return {"name": "query_order", "arguments": {"order_id": m.group(0)}}

    if any(kw in query for kw in LOCAL_TOOLS["query_inventory"]["keywords"]):
        m = SKU_PATTERN.search(query)
        if m:
            return {"name": "query_inventory", "arguments": {"sku": m.group(1)}}

    if any(kw in query for kw in LOCAL_TOOLS["query_ticket"]["keywords"]):
        m = TICKET_PATTERN.search(query)
        if m:
            return {"name": "query_ticket", "arguments": {"ticket_id": m.group(1)}}

    return None


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """统一分发入口：本地注册表 → MCP。"""
    fn = IMPLEMENTATIONS.get(name)
    if fn is not None:
        try:
            return fn(**arguments)
        except TypeError as exc:
            return {"error": f"参数不匹配：{exc}"}
    return {"error": f"未注册的工具：{name}"}
