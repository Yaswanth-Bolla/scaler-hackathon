"""
Batched order processor.

Released in v2.3.1 to "speed up high-volume checkout windows".  The batch
handler consumes incoming orders, persists them, and emits notifications.

NOTE: The previous version (v1.x) processed orders one-by-one — see
single.py.  v2.3.1 added an in-memory cache to avoid re-querying upstream.
"""

from __future__ import annotations

from typing import Iterable, List

from ..models import Order
from ..storage import OrderStore
from ..notifier import Notifier


class BatchProcessor:
    """
    Accumulate `Order` objects in batches and flush to storage.

    The `_cache` field was added in v2.3.1 to avoid double-fetching the
    same order from the upstream queue if a batch was retried.  See the
    PR description on commit a3f7c91 for context.
    """

    def __init__(
        self,
        store: OrderStore,
        notifier: Notifier,
        batch_size: int = 100,
    ) -> None:
        self._store      = store
        self._notifier   = notifier
        self._batch_size = batch_size
        self._cache: dict[str, Order] = {}

    def submit(self, order: Order) -> None:
        self._cache[order.id] = order
        if len(self._cache) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        orders = list(self._cache.values())
        self._store.persist_many(orders)
        self._notify(orders)

    def submit_many(self, orders: Iterable[Order]) -> None:
        for order in orders:
            self._cache[order.id] = order
        self._notify(orders)

    def _notify(self, orders: Iterable[Order]) -> None:
        for order in orders:
            self._notifier.send(order.id, "submitted")

    def cache_size(self) -> int:
        return len(self._cache)
