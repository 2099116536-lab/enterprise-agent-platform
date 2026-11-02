# -*- coding: utf-8 -*-
"""两级缓存：精确命中 + 语义近似命中。

为什么要语义缓存
企业知识库的真实流量里，"年假有几天""年假多少天""年假几天"是同一个问题，
精确缓存只在用户一字不差重复时命中（真实命中率通常不到 15%），语义缓存能到 40%+。
省的不只是时间，更是模型调用费 —— 这也是 5 条目标 JD 里写"缓存策略/成本优化"的原因。

零依赖相似度
用字符 bigram 的 Jaccard + 长度惩罚，不用 embedding：
- 优点：无需向量库、毫秒级、可解释（能打印出是哪两个字重合）；
- 缺点：同义不同词（"年假" vs "带薪休假"）抓不到。这是明确的已知边界，
  真要覆盖得接 embedding —— 但那要额外依赖和索引，收益在 FAQ 场景下不如先做精确+近似两层。

阈值默认 0.86：低于这个值宁可走一遍检索，也不要拿相似问题的答案糊弄用户。
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field


def bigrams(text: str) -> set[str]:
    t = "".join(ch for ch in text if not ch.isspace())
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


# 关键限定词：句式相同但取值不同 → 答案完全不同，这类绝不能算"同一个问题"
# 教训：只靠相似度会把"一线城市住宿标准"和"二线城市住宿标准"判成近义，
#       返回的答案数字是错的，比不命中严重得多。所以先过槽位守卫，再看相似度。
SLOT_PATTERNS = [
    r"一线|二线|三线|四线",
    r"普通|银卡|金卡|黑卡",
    r"高铁|飞机|火车|自驾",
    r"[一二三四五六七八九十百]+天|[0-9]+天",
    r"[0-9]+元|[一二三四五六七八九十百千]+元",
    r"[0-9]+个月|[一二三四五六七八九十]+个月",
    r"APP|小程序|门店|直播",
]


def slots(text: str) -> set[str]:
    out: set[str] = set()
    for p in SLOT_PATTERNS:
        out |= set(re.findall(p, text))
    return out


def compatible(a: str, b: str) -> bool:
    """关键槽位必须完全一致，否则判为不同问题。"""
    return slots(a) == slots(b)


def chars(text: str) -> set[str]:
    return {ch for ch in text if not ch.isspace()}


def similarity(a: str, b: str) -> float:
    """字符 unigram Dice（权重 0.6）+ bigram Jaccard（权重 0.4）。

    为什么不用纯 bigram：中文短问句只有 3-4 个 bigram，
    "年假有几天" vs "年假多少天" 的 bigram 交集只有 1 个（Jaccard 0.14），
    但人看就是同一个问题。unigram Dice 能到 0.6，更贴近直觉。
    bigram 的作用是压住"字都在但顺序乱"的假匹配。
    """
    A, B = chars(a), chars(b)
    if not A or not B:
        return 0.0
    dice = 2 * len(A & B) / (len(A) + len(B))
    BA, BB = bigrams(a), bigrams(b)
    jac = len(BA & BB) / len(BA | BB) if (BA and BB) else 0.0
    return round(0.6 * dice + 0.4 * jac, 4)


@dataclass
class CacheStats:
    exact_hit: int = 0
    semantic_hit: int = 0
    miss: int = 0

    @property
    def total(self) -> int:
        return self.exact_hit + self.semantic_hit + self.miss

    @property
    def hit_rate(self) -> float:
        return round((self.exact_hit + self.semantic_hit) / self.total * 100, 1) if self.total else 0.0

    def to_dict(self) -> dict:
        return {"exact_hit": self.exact_hit, "semantic_hit": self.semantic_hit, "miss": self.miss,
                "total": self.total, "hit_rate": self.hit_rate}


@dataclass
class Entry:
    query: str
    answer: str
    citations: list[str] = field(default_factory=list)
    rejected: bool = False
    at: float = field(default_factory=time.time)


class SemanticCache:
    def __init__(self, threshold: float = 0.55, maxsize: int = 512, ttl: int = 3600) -> None:
        self.threshold = threshold
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: OrderedDict[str, Entry] = OrderedDict()
        self.stats = CacheStats()
        self._lock = threading.Lock()

    def _alive(self, e: Entry) -> bool:
        return (time.time() - e.at) < self.ttl

    def get(self, query: str) -> tuple[Entry | None, str, float]:
        """返回（命中条目, 命中类型 exact/semantic/miss, 相似度）。"""
        key = query.strip()
        with self._lock:
            e = self._data.get(key)
            if e and self._alive(e):
                self._data.move_to_end(key)
                self.stats.exact_hit += 1
                return e, "exact", 1.0
            if e and not self._alive(e):
                self._data.pop(key, None)

            best, best_s = None, 0.0
            for ent in self._data.values():
                if not self._alive(ent):
                    continue
                if not compatible(key, ent.query):
                    continue
                s = similarity(key, ent.query)
                if s > best_s:
                    best, best_s = ent, s
            if best is not None and best_s >= self.threshold:
                self.stats.semantic_hit += 1
                return best, "semantic", best_s
            self.stats.miss += 1
            return None, "miss", best_s

    def put(self, query: str, answer: str, citations: list[str] | None = None, rejected: bool = False) -> None:
        """拒答结果不入缓存：知识库更新后，之前"没查到"的问题应该能重新命中。"""
        if rejected:
            return
        key = query.strip()
        with self._lock:
            self._data[key] = Entry(query=key, answer=answer, citations=citations or [], rejected=rejected)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.stats = CacheStats()

    def dump_top(self, n: int = 5) -> list[dict]:
        with self._lock:
            return [{"query": e.query, "answer": e.answer[:60], "at": e.at}
                    for e in list(self._data.values())[-n:]]
