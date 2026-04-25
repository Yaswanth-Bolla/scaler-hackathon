from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Order:
    id: str
    customer_id: str
    line_items: Dict[str, int] = field(default_factory=dict)
    total_cents: int = 0
    coupon_code: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
