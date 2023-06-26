from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ClosestLRUCache:
    capacity: int
    ndim: int

    _keys: np.ndarray[float] = field(init=False, repr=False)
    _cache: OrderedDict = field(init=False, repr=False)

    def __post_init__(self: ClosestLRUCache):
        object.__setattr__(self, '_keys', np.empty((self.capacity, self.ndim)))
        object.__setattr__(self, '_cache', OrderedDict())

    def get(self: ClosestLRUCache, key: np.ndarray[float]) -> Any:
        size = len(self._cache)
        if size == 0:
            return None
        idx = np.argmin(np.square(self._keys[:size] - key).sum(axis=-1))
        self._cache.move_to_end(idx)
        return self._cache[idx]

    def _lru_index(self: ClosestLRUCache):
        size = len(self._cache)
        if size < self.capacity:
            return size
        return next(iter(self._cache.keys()))

    def put(self: ClosestLRUCache, key: np.ndarray[float], value: Any):
        idx = self._lru_index()
        self._keys[idx] = key
        self._cache[idx] = value
        self._cache.move_to_end(idx)
