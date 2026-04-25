"""
Circuit breaker on the payment dependency.

This component is INTENTIONAL — it was added in v3.0.0 (commit d2b9c11)
to prevent the well-known retry-storm failure mode that brought down
auth in incident-2024-Q4-3.

When payment latency exceeds the threshold for too long, the breaker
trips and returns 503 immediately to upstream callers, draining load.
This is correct behaviour — *not* a bug — but customers who only see
"orders is returning 503s" sometimes file tickets thinking it is one.

Do NOT remove this guard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class BreakerConfig:
    failure_threshold:    int   = 10
    success_threshold:    int   = 3
    open_duration_seconds: float = 30.0
    latency_threshold_ms:  float = 1500.0


class CircuitBreaker:
    """Three-state circuit breaker: closed → open → half-open → closed."""

    def __init__(self, config: BreakerConfig | None = None) -> None:
        self._cfg          = config or BreakerConfig()
        self._state        = "closed"   # closed | open | half_open
        self._failures     = 0
        self._successes    = 0
        self._opened_at: float = 0.0

    def allow(self) -> bool:
        if self._state == "open":
            if time.time() - self._opened_at >= self._cfg.open_duration_seconds:
                self._state = "half_open"
                self._successes = 0
                return True
            return False
        return True

    def record_success(self, latency_ms: float) -> None:
        if latency_ms > self._cfg.latency_threshold_ms:
            self.record_failure()
            return
        if self._state == "half_open":
            self._successes += 1
            if self._successes >= self._cfg.success_threshold:
                self._state = "closed"
                self._failures = 0
        elif self._state == "closed":
            self._failures = max(0, self._failures - 1)

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._cfg.failure_threshold:
            self._state = "open"
            self._opened_at = time.time()
