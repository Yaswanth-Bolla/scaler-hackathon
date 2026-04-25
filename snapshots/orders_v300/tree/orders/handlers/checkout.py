"""Checkout endpoint — wraps the payment call in the circuit breaker."""

from __future__ import annotations

import time

from ..circuit_breaker import CircuitBreaker


class Checkout:
    def __init__(self, payment_client, breaker: CircuitBreaker | None = None) -> None:
        self._payment = payment_client
        self._breaker = breaker or CircuitBreaker()

    def submit(self, order_id: str, amount_cents: int) -> str:
        if not self._breaker.allow():
            return "503 — payment temporarily unavailable, please retry"
        start = time.time()
        try:
            ok = self._payment.charge(order_id, amount_cents)
        except Exception:
            self._breaker.record_failure()
            raise
        latency_ms = (time.time() - start) * 1000
        if ok:
            self._breaker.record_success(latency_ms)
            return "201 ok"
        self._breaker.record_failure()
        return "402 declined"
