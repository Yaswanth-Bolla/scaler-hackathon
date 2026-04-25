from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Transaction:
    id: str
    amount_cents: int
    currency: str
    card_token: str
    customer_id: str
