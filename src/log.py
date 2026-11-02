# -*- coding: utf-8 -*-
"""极简日志适配层：用标准库 logging 复刻 loguru 的 logger.xxx("模板 {}", v) 调用风格。

为什么要换掉 loguru
这个项目的评测和 CI 都在裸环境跑，loguru 只是为了让日志好看一点，
却让"跑个评测先装依赖"变成额外成本。标准库 logging 足够，且零安装。

兼容性：只实现用到的 info/warning/error/debug/exception，占位符仍是 {}，
所以调用处一行都不用改，只把 import 换掉。
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def _setup() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    root = logging.getLogger("agent")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


class _Logger:
    def __init__(self) -> None:
        _setup()
        self._log = logging.getLogger("agent")

    @staticmethod
    def _fmt(msg, args) -> str:
        if not args:
            return str(msg)
        try:
            return str(msg).format(*args)
        except (IndexError, KeyError):
            return f"{msg} {' '.join(map(str, args))}"

    def debug(self, msg, *a) -> None:
        self._log.debug(self._fmt(msg, a))

    def info(self, msg, *a) -> None:
        self._log.info(self._fmt(msg, a))

    def warning(self, msg, *a) -> None:
        self._log.warning(self._fmt(msg, a))

    def error(self, msg, *a) -> None:
        self._log.error(self._fmt(msg, a))

    def exception(self, msg, *a) -> None:
        self._log.exception(self._fmt(msg, a))


logger = _Logger()
