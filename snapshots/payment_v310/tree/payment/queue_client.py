from __future__ import annotations

from .txn import Transaction


class QueueClient:
    """SQS-backed enqueue + ack-poll client."""

    def enqueue(self, txn: Transaction) -> None:
        ...

    def ack(self, txn_id: str) -> bool:
        ...
