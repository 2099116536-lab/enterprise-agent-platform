"""FastAPI + SSE 服务层。

选 SSE 而不是 WebSocket：Agent 场景本质是"服务端单向持续吐字"，
SSE 走普通 HTTP，穿透代理更稳、断线浏览器自动重连，实现成本也低一个量级。

两个容易踩的坑：
1. 反向代理（nginx）默认会缓冲响应，必须关掉 `proxy_buffering`；
2. 长连接空闲被掐，需要定期发心跳注释行。
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .graph import AgentPlatform, Intent
from .rag import HybridRetriever, ingest

app = FastAPI(title="Enterprise Agent Platform", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_PLATFORM: AgentPlatform | None = None


def get_platform() -> AgentPlatform:
    global _PLATFORM
    if _PLATFORM is None:
        _PLATFORM = AgentPlatform(retriever=HybridRetriever())
    return _PLATFORM


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = "default"
    mock: bool = False


class IngestRequest(BaseModel):
    docs_dir: str
    persist_dir: str | None = None


@app.get("/health")
async def health() -> dict:
    platform = get_platform()
    return {
        "status": "ok",
        "has_llm": platform.llm is not None,
        "has_kb": platform.retriever is not None and bool(platform.retriever.chunks),
    }


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """非流式问答。"""
    platform = get_platform()
    state = platform.run(req.query, req.session_id)
    return {
        "answer": state.answer,
        "intent": state.intent.value,
        "citations": state.citations,
        "rejected": state.rejected,
        "trace": state.trace,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式问答：先推意图与引用，再逐字推送正文。"""

    async def generator() -> AsyncIterator[dict[str, str]]:
        platform = get_platform()
        try:
            state = platform.run(req.query, req.session_id)
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"message": str(exc)}, ensure_ascii=False)}
            return

        yield {"event": "intent", "data": json.dumps({"intent": state.intent.value}, ensure_ascii=False)}

        for cite in state.citations:
            yield {"event": "citation", "data": json.dumps({"source": cite}, ensure_ascii=False)}

        # 逐字推送（生产环境替换为模型 streaming 回调）
        for char in state.answer:
            yield {"event": "token", "data": json.dumps({"t": char}, ensure_ascii=False)}

        yield {"event": "done", "data": json.dumps({"trace": state.trace}, ensure_ascii=False)}

    return EventSourceResponse(generator(), ping=15)  # ping 保活，防止代理掐空闲连接


@app.post("/ingest")
async def ingest_docs(req: IngestRequest) -> dict:
    """触发文档入库。"""
    from pathlib import Path

    if not Path(req.docs_dir).exists():
        raise HTTPException(status_code=400, detail=f"目录不存在：{req.docs_dir}")
    chunks, stats = ingest(req.docs_dir, req.persist_dir)
    platform = get_platform()
    platform.retriever = HybridRetriever(chunks)
    return {"files": stats.files, "chunks": stats.chunks, "dropped": stats.dropped}
