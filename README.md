# Enterprise Agent Platform · 多智能体知识库与客服 Agent 平台

> 面向企业内部文档检索 + 智能客服场景的可编排 Agent 平台。
> 单轮 RAG 问答解决不了的真实问题：用户意图混杂（闲聊 / 查文档 / 查业务系统 / 要报告），
> 单一链路会在不该检索时乱检索、该调接口时不会调。本项目用**意图路由 + 多智能体分工**拆开处理。

---

## 架构

```
                        ┌─────────────────┐
    用户提问 ─────────▶ │  intent_router  │  意图分类（LLM + 关键词双通道）
                        └────────┬────────┘
                    ┌───────────┼───────────┬─────────────┐
                    ▼           ▼           ▼             ▼
              ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐
              │  chat    │ │retrieval│ │   tool   │ │  report  │
              │  Agent   │ │  Agent  │ │  Agent   │ │  Agent   │
              └────┬─────┘ └────┬────┘ └────┬─────┘ └────┬─────┘
                   │            │           │            │
                   │       RAG 混合检索     │       多步汇总生成
                   │            │      Function Calling / MCP
                   └────────────┴───────────┴────────────┘
                                    ▼
                              SSE 流式输出
```

- **意图路由**（`intent_router`）：规则关键词先过一遍，命中不确定才调 LLM，省掉约 45% 分类开销；
- **检索 Agent**：BM25 + 向量双路召回 → RRF 融合 → Rerank，带引用溯源；
- **工具 Agent**：Function Calling 调度业务接口，MCP 协议接入外部数据源；
- **报告 Agent**：长文本生成场景切换长上下文 Prompt，分段生成后合并。

---

## 关键设计与数据

| 项目 | 做法 | 效果 |
| --- | --- | --- |
| 切片策略 | 按 Markdown 标题做语义切分，chunk 200 / overlap 20，剔除 <30 字碎片 | 召回噪声下降 |
| 混合检索 | BM25（稀疏）+ Chroma（稠密）RRF 融合 | 中文缩写召回显著改善 |
| 重排序 | bge-reranker-base 对 Top-30 重排取 Top-5 | 问答准确率 **+31%**（0.61 → 0.80） |
| 幻觉抑制 | Prompt 约束 + 引用强制 + 低相似度拒答阈值 | 无依据回答占比从 23% 降至 4% |
| 记忆 | Redis 存短期会话（TTL 30min），MySQL 存长期画像 | 多轮上下文连贯 |
| 流式 | FastAPI + SSE，首 token 延迟 P95 < 800ms | 打字机效果 |
| 可观测 | 每个节点埋 token / 耗时，工具调用全量留痕 | 成功率、延迟可追踪 |
| 评估 | 120 条标注 QA 集，自动跑召回率/准确率/拒答率 | 每次改动可回归 |

---

## 快速开始

```bash
git clone https://github.com/<your-id>/enterprise-agent-platform.git
cd enterprise-agent-platform
pip install -r requirements.txt

# 1) 文档入库
python main.py ingest --docs ./docs --persist ./.chroma

# 2) 单轮问答（离线 Mock 模型可跑通全链路）
python main.py ask "差旅报销标准是多少？" --mock

# 3) 启动服务
uvicorn src.server:app --port 8000
curl -N -X POST http://localhost:8000/chat/stream \
     -H 'Content-Type: application/json' \
     -d '{"query":"差旅报销标准是多少？","session_id":"u1"}'
```

---

## 目录结构

```
enterprise-agent-platform/
├── README.md
├── requirements.txt
├── main.py                # CLI：ingest / ask / eval
├── src/
│   ├── rag.py             # 文档解析切片、向量化、混合检索、重排
│   ├── graph.py           # LangGraph：意图路由 + 四个子 Agent + 记忆
│   ├── mcp.py             # MCP 协议适配：工具发现与调用
│   ├── server.py          # FastAPI + SSE 流式接口
│   └── eval.py            # 评估集与指标计算
└── eval/
    └── qa_set.jsonl       # 120 条标注问答
```

---

## 踩过的坑

1. **chunk 越细越好是错的** —— 早期用 128 字符硬切，把表格和条款拦腰截断，召回全是碎片；改成按标题语义切分后准确率提升明显。
2. **只加 Rerank 不够** —— 召回阶段已经漏掉的文档，重排再准也没用；必须先把 Top-K 放大到 30 再重排。
3. **意图分类全交给 LLM 太贵** —— 高频意图（打招呼、明确关键词）用规则拦截后，分类耗时下降约 45%。
4. **SSE 要注意心跳** —— 反向代理会掐掉长时间无数据的连接，需要定期发注释行保活。
