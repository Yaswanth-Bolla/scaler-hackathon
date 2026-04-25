"""Pre-v2.3.1 single-order handler. Still used as the fallback path."""

from __future__ import annotations

from ..models import Order
from ..storage import OrderStore
from ..notifier import Notifier


class SingleOrderHandler:
    def __init__(self, store: OrderStore, notifier: Notifier) -> None:
        self._store    = store
        self._notifier = notifier

    def submit(self, order: Order) -> None:
        self._store.persist(order)
        self._notifier.send(order.id, "submitted")
