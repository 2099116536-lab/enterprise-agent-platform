# -*- coding: utf-8 -*-
"""会话记忆：进程内滑动窗口（默认）+ Redis（可选，多实例部署用）。

为什么默认给进程内实现
- 评测、CI、本地演示都不该依赖 Redis 才能跑；
- 单实例部署下进程内记忆完全够用，Redis 只在多副本共享会话时才必要；
- 接口一致，切换只改构造参数，不动业务代码。

窗口大小默认 20 轮（user+assistant 各一条 = 40 条消息），
再长由 ContextBudget 在拼 Prompt 时压缩 —— 记忆只负责"存"，压缩是"取"的职责，两者分开。
"""
from __future__ import annotations

import json
import threading
from typing import Any

from .log import logger


class InMemoryMemory:
    """进程内滑动窗口记忆。线程安全（FastAPI 是多线程跑同步节点）。"""

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._data: dict[str, list[dict[str, str]]] = {}
        self._lock = threading.Lock()

    def load(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._data.get(session_id, []))

    def append(self, session_id: str, query: str, answer: str) -> None:
        with self._lock:
            hist = self._data.setdefault(session_id, [])
            hist.append({"role": "user", "content": query})
            hist.append({"role": "assistant", "content": answer})
            keep = self.max_turns * 2
            if len(hist) > keep:
                del hist[: len(hist) - keep]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"sessions": len(self._data),
                    "messages": sum(len(v) for v in self._data.values())}


class RedisMemory:
    """Redis 版记忆（需要 redis 包 + 可用实例；失败自动回落进程内）。"""

    def __init__(self, url: str = "redis://localhost:6379/0", max_turns: int = 20, ttl: int = 86400) -> None:
        self.max_turns = max_turns
        self.ttl = ttl
        try:
            import redis  # type: ignore

            self.client = redis.Redis.from_url(url, decode_responses=True)
            self.client.ping()
        except Exception as exc:
            logger.warning("Redis 不可用，回落进程内记忆：{}", exc)
            self._fallback = InMemoryMemory(max_turns)
            self.client = None

    def _key(self, session_id: str) -> str:
        return f"agent:hist:{session_id}"

    def load(self, session_id: str) -> list[dict[str, str]]:
        if self.client is None:
            return self._fallback.load(session_id)
        raw = self.client.lrange(self._key(session_id), 0, -1) or []
        return [json.loads(x) for x in raw]

    def append(self, session_id: str, query: str, answer: str) -> None:
        if self.client is None:
            return self._fallback.append(session_id, query, answer)
        key = self._key(session_id)
        self.client.rpush(key, json.dumps({"role": "user", "content": query}, ensure_ascii=False))
        self.client.rpush(key, json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False))
        self.client.ltrim(key, -self.max_turns * 2, -1)
        self.client.expire(key, self.ttl)

    def clear(self, session_id: str) -> None:
        if self.client is not None:
            self.client.delete(self._key(session_id))
