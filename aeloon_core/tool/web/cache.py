from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Generic, TypeVar

T = TypeVar("T")


class WebCache(Generic[T]):
    def __init__(self, ttl_s: int, size: int) -> None:
        self.ttl_s = ttl_s
        self.size = size
        self._values: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> T | None:
        if self.ttl_s <= 0 or self.size <= 0:
            return None
        async with self._lock:
            item = self._values.get(key)
            if item is None or time.monotonic() - item[0] > self.ttl_s:
                self._values.pop(key, None)
                return None
            self._values.move_to_end(key)
            return item[1]

    async def put(self, key: str, value: T) -> None:
        if self.ttl_s <= 0 or self.size <= 0:
            return
        async with self._lock:
            self._values[key] = (time.monotonic(), value)
            self._values.move_to_end(key)
            while len(self._values) > self.size:
                self._values.popitem(last=False)
