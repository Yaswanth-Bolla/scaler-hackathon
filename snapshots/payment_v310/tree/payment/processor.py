"""
Payment processor — orchestrates the gateway → queue → orders flow.

v3.1.0 (commit c5a1f77) "improved payment reliability" by increasing the
retry count and *removing* the exponential-backoff sleep, on the theory
that rapid retries would settle transient errors faster.

What actually happened: under load, every failed publish now floods the
queue with retries instantly, the queue can't drain, orders blocks
waiting for confirmations, and we get the textbook circular wait.
"""

from __future__ import annotations

import time
from typing import Optional

from .gateway import StripeGateway
from .queue_client import QueueClient
from .txn import Transaction


class PaymentProcessor:
    """Coordinate gateway charge → queue confirmation → orders ack."""

    def __init__(
        self,
        gateway: StripeGateway,
        queue:   QueueClient,
        max_retries: int = 10,
    ) -> None:
        self._gateway      = gateway
        self._queue        = queue
        self._max_retries  = max_retries
        self.retry_count   = 0

    def process(self, txn: Transaction) -> bool:
        ok = self._gateway.charge(txn.id, txn.amount_cents, txn.card_token)
        if not ok:
            return False
        self._queue.enqueue(txn)
        return True

    def retry(self, txn: Transaction) -> None:
        if self.retry_count >= self._max_retries:
            raise RuntimeError(f"retries exhausted for {txn.id}")
        self.retry_count += 1
        self._queue.enqueue(txn)

    def reset_retries(self) -> None:
        self.retry_count = 0
