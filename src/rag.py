"""文档 ingest 与混合检索。

这一版相比最初的朴素实现改了三处，每一处都对应线上真实问题：

1. 切片从「定长硬切」改成「按标题语义切分」—— 硬切会把条款和表格拦腰截断，
   召回回来一堆半句话，模型读了反而更容易幻觉；
2. 召回 Top-K 放大到 30 再重排 —— 召回阶段就漏掉的文档，重排再准也救不回来；
3. 加拒答阈值 —— 最高相似度低于阈值时直接说"没查到"，而不是硬编一段。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

# 低于该相似度视为"知识库里没有"，直接拒答而不是硬编
REJECT_THRESHOLD = 0.15


@dataclass
class Chunk:
    text: str
    source: str
    heading: str = ""
    chunk_id: str = ""

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "heading": self.heading,
            "chunk_id": self.chunk_id,
        }


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    rank: int = 0


@dataclass
class IngestStats:
    files: int = 0
    chunks: int = 0
    dropped: int = 0
    by_source: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 切片
# --------------------------------------------------------------------------- #
def split_markdown(text: str, size: int = 200, overlap: int = 20) -> list[tuple[str, str]]:
    """按 Markdown 标题切分，返回 (heading, body) 列表。

    标题是天然的语义边界：一个标题下的内容通常讲同一件事，
    把它作为切分单位比按字符数硬切合理得多。
    """
    blocks: list[tuple[str, str]] = []
    current_heading = ""
    buffer: list[str] = []

    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if buffer:
                blocks.append((current_heading, "\n".join(buffer).strip()))
                buffer = []
            current_heading = line.lstrip("#").strip()
        else:
            buffer.append(line)
    if buffer:
        blocks.append((current_heading, "\n".join(buffer).strip()))

    # 超长块再滑窗细分，避免超过 embedding 长度上限
    result: list[tuple[str, str]] = []
    for heading, body in blocks:
        if not body:
            continue
        if len(body) <= size:
            result.append((heading, body))
            continue
        step = max(size - overlap, 1)
        for start in range(0, len(body), step):
            piece = body[start : start + size].strip()
            if len(piece) > 30:  # 丢弃碎片
                result.append((heading, piece))
    return result


def ingest(docs_dir: str, persist_dir: str | None = None) -> tuple[list[Chunk], IngestStats]:
    """把目录下的 md/txt 文档切成 Chunk。"""
    stats = IngestStats()
    chunks: list[Chunk] = []
    root = Path(docs_dir)
    files = sorted(list(root.glob("*.md")) + list(root.glob("*.txt")))

    for path in files:
        stats.files += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        file_chunks = 0
        for idx, (heading, body) in enumerate(split_markdown(text)):
            if len(body) < 30:
                stats.dropped += 1
                continue
            chunks.append(
                Chunk(
                    text=body,
                    source=path.name,
                    heading=heading,
                    chunk_id=f"{path.stem}_{idx}",
                )
            )
            file_chunks += 1
        stats.by_source[path.name] = file_chunks

    stats.chunks = len(chunks)
    logger.info("入库完成：{} 个文件 → {} 个切片（丢弃 {} 个碎片）", stats.files, stats.chunks, stats.dropped)
    return chunks, stats


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
class HybridRetriever:
    """BM25 + 向量双路召回 → RRF 融合 → 重排。

    依赖缺失时自动降级到词面匹配，链路不中断。
    """

    def __init__(self, chunks: list[Chunk] | None = None, top_k_recall: int = 30) -> None:
        self.chunks = chunks or []
        self.top_k_recall = top_k_recall
        self._bm25 = None
        self._collection = None
        self._reranker = None
        if self.chunks:
            self._init_sparse()
            self._init_dense()

    # ---------------- 初始化 ---------------- #
    def _init_sparse(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 未安装，跳过稀疏检索")
            return
        self._bm25 = BM25Okapi([self._tokenize(c.text) for c in self.chunks])

    def _init_dense(self) -> None:
        try:
            import chromadb
        except ImportError:
            logger.warning("chromadb 未安装，跳过稠密检索")
            return
        try:
            client = chromadb.Client(chromadb.Settings(anonymized_telemetry=False))
            self._collection = client.get_or_create_collection("enterprise_kb")
            if self._collection.count() == 0:
                self._collection.add(
                    documents=[c.text for c in self.chunks],
                    ids=[c.chunk_id for c in self.chunks],
                    metadatas=[{"source": c.source, "heading": c.heading} for c in self.chunks],
                )
        except Exception as exc:
            logger.warning("Chroma 初始化失败，降级：{}", exc)
            self._collection = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens: list[str] = []
        tokens.extend(m.lower() for m in re.findall(r"[A-Za-z][A-Za-z\-]*", text))
        chinese = re.sub(r"[^一-龥]", "", text)
        tokens.extend(list(chinese))
        tokens.extend(chinese[i : i + 2] for i in range(len(chinese) - 1))
        return tokens

    # ---------------- 检索 ---------------- #
    def search(self, query: str, top_k: int = 5) -> list[Retrieved]:
        if not self.chunks:
            return []
        sparse = self._sparse_search(query, self.top_k_recall)
        dense = self._dense_search(query, self.top_k_recall)
        if not sparse and not dense:
            fused = self._lexical_search(query, self.top_k_recall)
        else:
            fused = self._rrf([sparse, dense])[: self.top_k_recall]
        reranked = self._rerank(query, fused, top_k)
        return reranked

    def _sparse_search(self, query: str, top_k: int) -> list[int]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked if scores[i] > 0][:top_k]

    def _dense_search(self, query: str, top_k: int) -> list[int]:
        if self._collection is None:
            return []
        try:
            res = self._collection.query(query_texts=[query], n_results=top_k)
        except Exception as exc:
            logger.warning("向量检索失败：{}", exc)
            return []
        ids = res.get("ids", [[]])[0]
        index = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        return [index[i] for i in ids if i in index]

    def _lexical_search(self, query: str, top_k: int) -> list[int]:
        q = set(self._tokenize(query))
        scored = [
            (i, len(q & set(self._tokenize(c.text)))) for i, c in enumerate(self.chunks)
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [i for i, s in scored[:top_k] if s > 0]

    @staticmethod
    def _rrf(rank_lists: list[list[int]], k: int = 60) -> list[int]:
        scores: dict[int, float] = {}
        for ranks in rank_lists:
            for pos, doc_id in enumerate(ranks):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + pos + 1)
        return [i for i, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

    def _rerank(self, query: str, ids: list[int], top_k: int) -> list[Retrieved]:
        if not ids:
            return []
        if self._reranker is None:
            self._reranker = self._load_reranker()
        if self._reranker is not None:
            try:
                scores = self._reranker.predict([(query, self.chunks[i].text) for i in ids])
                ordered = sorted(zip(ids, scores), key=lambda t: t[1], reverse=True)
                return [Retrieved(self.chunks[i], float(s), r + 1) for r, (i, s) in enumerate(ordered[:top_k])]
            except Exception as exc:
                logger.warning("重排失败，回落 RRF 分数：{}", exc)

        # 无 CrossEncoder 时用 RRF 名次折算成近似分
        return [
            Retrieved(self.chunks[i], 1.0 / (1 + r), r + 1)
            for r, i in enumerate(ids[:top_k])
        ]

    @staticmethod
    def _load_reranker():
        try:
            from sentence_transformers import CrossEncoder

            return CrossEncoder("BAAI/bge-reranker-base")
        except Exception:
            logger.debug("CrossEncoder 不可用，使用 RRF 名次近似")
            return None

    def should_reject(self, results: list[Retrieved]) -> bool:
        """最高分低于阈值 → 判定知识库无依据，触发拒答。"""
        if not results:
            return True
        return max(r.score for r in results) < REJECT_THRESHOLD
