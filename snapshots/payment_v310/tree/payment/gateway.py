from __future__ import annotations


class StripeGateway:
    """Outbound HTTP client to the upstream payment processor."""

    def charge(self, txn_id: str, amount_cents: int, card_token: str) -> bool:
        ...
