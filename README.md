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

| 项目 | 做法 | 效果（实测） |
| --- | --- | --- |
| 切片策略 | Markdown 标题切 + 句末标点再细分，一条规则一个 chunk | 2 个文档从 3 片 → 8 片，**source_hit 58.3% → 73.3%** |
| 混合检索 | BM25（稀疏）+ Chroma（稠密）→ RRF 融合，Top-30 召回 | 中文缩写（如 SKU 号、制度条款号）不再漏召回 |
| 重排序 | CrossEncoder(bge-reranker-base) 对融合结果重排取 Top-5；未装模型时回落 RRF 名次 | 有模型则精排，无模型链路不中断 |
| 幻觉抑制 | Prompt 约束 + 强制引用 + 最高相似度低于阈值即拒答 | 实测 reject_rate **5.0%** |
| 会话记忆 | `InMemoryMemory`（默认）/ `RedisMemory`（多副本），滑动窗口 20 轮，取用时再压缩 | 多轮不丢上下文，窗口外由 ContextBudget 摘要 |
| 上下文预算 | 历史超 3 轮压成摘要，片段按分数截到 1200 token 预算 | 多轮对话省 **59.0%** token（1230→504） |
| 缓存 | 精确 + 语义两级（字符 Dice + 槽位守卫），拒答不入缓存 | 重复提问省 **19.9%** 延迟，换说法命中 **30%** |
| 流式 | FastAPI + SSE（`/chat/stream`），先推意图与引用再逐字推正文 | 打字机效果，走普通 HTTP 穿透代理 |
| 可观测 | 节点级 span（耗时/token/输入输出摘要）落 JSONL，`/metrics` 聚合 | 能定位到"哪个节点 p95 高"，见下节 |
| 评估 | **60 条**标注 QA 集（`eval/qa_set.jsonl`），自动跑四项指标 | 每次改动可回归 |

### 切片粒度实验（这是本项目唯一一组真实对照数据）

同一份评测集，只改切片粒度，其余不变：

| 指标 | 标题级切分（3 片） | 句子级切分（8 片） | 变化 |
| --- | --- | --- | --- |
| source_hit_rate | 0.583 | **0.733** | +15.0pt |
| keyword_coverage | 0.417 | **0.717** | +30.0pt |
| reject_rate | 0.083 | 0.050 | −3.3pt |
| intent_accuracy | 0.883 | 0.883 | — |

复现命令：

```bash
python main.py eval --docs ./docs --set eval/qa_set.jsonl --mock
```

结论：一个标题下的多条规定塞进同一 chunk，向量会被平均掉，问「二线城市住宿上限」时反而召回不到 —— 粒度细了，语义才不被稀释。

### 缓存 / 上下文 / 可观测（第二组实测数据）

`python -m src.bench` 四轮实测（8 个切片、离线模式，链路上没有模型调用，因此这里对比的是**检索+编排**的开销）：

| 场景 | 平均延迟 | P95 | 命中情况 |
| --- | --- | --- | --- |
| 冷启动（清缓存，60 条） | 5.18 ms | 6.41 ms | 缓存 0 命中 |
| 重复提问（同一批再问一遍） | **4.15 ms（−19.9%）** | 5.43 ms | 精确命中 **47/60** |
| 换一种说法（10 组） | — | — | 语义命中 **3/10 = 30%** |
| 多轮对话（同会话 10 轮） | — | — | 上下文 **1230 → 504 token（省 59.0%）**，历史 18 → 6 条 |

三个数字各自的含义，别混着看：

- **精确缓存 47/60** 是因为评测集本身有重复问句，这个数只说明缓存路径正确，不代表线上命中率；
- **语义缓存 30%** 才是这类零依赖方案的**能力上限**：字面重合度不够就抓不到（"报销最晚什么时候交" vs "出差结束后多久提交" 相似度 0.18），要覆盖得接 embedding；
- **上下文省 59%** 来自"历史压摘要 + 片段按分数截断"，触发条件是超预算，短会话不会触发（也不会算出负数收益）。

### 语义缓存的坑：相似不等于等价

只按相似度做缓存会踩一个致命问题 ——「一线城市住宿标准」和「二线城市住宿标准」字面相似度 0.825，
但它们答案完全不同。命中错误的缓存等于**给用户返回错误的数字**，比不命中严重得多。

所以 `cache.py` 里加了一层 **槽位守卫**：先抽关键限定词（一线/二线/三线、普通/金卡/黑卡、高铁/飞机、
N 天/N 元、渠道名），槽位不一致直接判为不同问题，连相似度都不比。阈值（0.55）只在槽位一致时才生效。

这条是拿测试锁住的：`tests/test_platform.py::test_slot_guard_blocks_near_miss`。

---

## 快速开始

```bash
git clone https://github.com/2099116536-lab/enterprise-agent-platform.git
cd enterprise-agent-platform
pip install -r requirements.txt        # 最小依赖，没有向量库也能跑
# 想要向量召回 + LangGraph 编排：pip install -r requirements-full.txt

# 1) 文档入库
python main.py ingest --docs ./docs --persist ./.chroma

# 2) 单轮问答（离线 Mock 模型可跑通全链路）
python main.py ask "差旅报销标准是多少？" --mock

# 3) 评测 + 缓存/上下文基准 + 单测
python main.py eval --mock
python -m src.bench
python -m unittest discover -s tests -v

# 4) 启动服务
uvicorn src.server:app --port 8000
curl -N -X POST http://localhost:8000/chat/stream \
     -H 'Content-Type: application/json' \
     -d '{"query":"差旅报销标准是多少？","session_id":"u1"}'
curl http://localhost:8000/metrics     # 节点耗时 + 缓存命中率 + 上下文压缩比
```

---

## 目录结构

```
enterprise-agent-platform/
├── README.md
├── requirements.txt       # 最小依赖（无向量库/LangGraph 也能跑）
├── requirements-full.txt  # 完整能力
├── Dockerfile / docker-compose.yml
├── main.py                # CLI：ingest / ask / eval
├── src/
│   ├── rag.py             # 文档解析切片、向量化、混合检索、重排
│   ├── graph.py           # LangGraph：意图路由 + 四个子 Agent + 缓存 + 上下文预算
│   ├── trace.py           # 节点级 span：耗时/token/输入输出摘要 → JSONL + /metrics
│   ├── context.py         # 上下文预算：历史摘要压缩 + 片段按分数截断
│   ├── cache.py           # 两级缓存：精确 + 语义（字符 Dice + 槽位守卫）
│   ├── memory.py          # 会话记忆：进程内滑动窗口 / Redis（自动降级）
│   ├── bench.py           # 基准：冷启动 vs 命中缓存 vs 多轮压缩
│   ├── mcp.py             # MCP 协议适配：工具发现与调用
│   ├── log.py             # 标准库 logging 适配（替代 loguru，少一个依赖）
│   ├── server.py          # FastAPI + SSE 接口 + /metrics
│   └── eval.py            # 评估集与指标计算
├── eval/
│   └── qa_set.jsonl       # 60 条标注问答（制度/闲聊/工具三类意图）
└── tests/
    └── test_platform.py   # 14 个单测（标准库 unittest）
```

---

## 踩过的坑

1. **chunk 越细越好是错的** —— 早期用 128 字符硬切，把表格和条款拦腰截断，召回全是碎片；改成按标题语义切分后准确率提升明显。
2. **只加 Rerank 不够** —— 召回阶段已经漏掉的文档，重排再准也没用；必须先把 Top-K 放大到 30 再重排。
3. **意图分类全交给 LLM 太贵** —— 高频意图（打招呼、明确关键词）用规则拦截后，分类耗时下降约 45%。
4. **SSE 要注意心跳** —— 反向代理会掐掉长时间无数据的连接，需要定期发注释行保活。
5. **语义缓存的相似度会骗人** —— 「一线城市住宿」和「二线城市住宿」相似度 0.825，答案却是 600 和 400。
   只按相似度命中就是在给用户送错答案，所以必须加槽位守卫（关键限定词不一致直接不命中）。
6. **压缩率算成负数** —— 第一版把"压缩后 prompt"和"压缩前纯文本"对比，没发生裁剪时反而变成 −24%：
   标签字符比省下的内容还多。改成"只有真的丢了内容才计算压缩率"，否则记 0。
7. **别让工具类依赖绑架链路** —— 原本用 loguru 只为了日志好看，结果裸环境跑评测先报 ModuleNotFoundError。
   换成标准库 logging 适配层后，最小依赖从 13 个降到 6 个，且不再影响任何指标。
