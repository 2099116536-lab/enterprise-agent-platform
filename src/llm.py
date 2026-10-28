"""统一 LLM 接口。与主项目保持一致的设计：业务层不依赖具体厂商。"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    name: str = "base"

    @abstractmethod
    def chat(self, system: str, user: str, temperature: float = 0.2) -> str: ...


class OpenAICompatLLM(BaseLLM):
    """OpenAI 兼容接口：OpenAI / 通义 / DeepSeek / 本地 vLLM 通用。"""

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or os.getenv("LLM_MODEL", "qwen-plus")
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "sk-placeholder"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.name = self.model

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class MockLLM(BaseLLM):
    """离线模型：按 system 提示返回对应形态的结果，保证无 Key 也能跑通链路。"""

    name = "mock-offline"

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        if "意图分类器" in system:
            return "retrieval"
        if "报告" in system:
            return "# 报告\n\n（离线模式生成的结构化报告占位）"
        # 客服回答：把片段里跟问题最相关的一句摘出来
        lines = [ln for ln in user.splitlines() if ln.strip().startswith("[")]
        if lines:
            return "（离线模式）" + lines[0][:200]
        return "（离线模式）未检索到可用片段。"


def get_llm(force_mock: bool = False) -> BaseLLM:
    if force_mock:
        return MockLLM()
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAICompatLLM()
        except Exception:
            return MockLLM()
    return MockLLM()
