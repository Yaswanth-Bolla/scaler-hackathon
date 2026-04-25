from __future__ import annotations

from typing import Iterable, List

from .models import Order


class OrderStore:
    """Postgres-backed order persistence (mock — talks to db.orders)."""

    def persist(self, order: Order) -> None:
        ...

    def persist_many(self, orders: Iterable[Order]) -> None:
        for o in orders:
            self.persist(o)

    def get(self, order_id: str) -> Order:
        ...
