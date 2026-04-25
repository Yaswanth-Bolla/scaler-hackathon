from __future__ import annotations


class Notifier:
    """Outbound notification client — pushes status to downstream queue."""

    def send(self, order_id: str, event: str) -> None:
        ...
