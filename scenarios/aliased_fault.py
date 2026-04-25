"""
Phase-B scenario 1 — Aliased fault patterns.

Two distinct faults produce *identical initial observations*:
  - Memory leak in `orders`             (real cause)
  - Cache thrashing in `queue` (saturated upstream that orders depends on)

Both surface as "high memory + degraded orders" on the dashboard.

The agent's default prior is "investigate the loudest service" — it heads
straight to `check_metrics(orders)` and `check_deploy_history(orders)`,
finds the recent batch-processing deploy, and rolls it back.  In the
aliased version, the recent deploy on `orders` is innocuous; the *real*
cause is in `queue`'s flush worker.

The diagnostic that disambiguates is `check_dependencies(orders)` followed
by `check_metrics(queue)` — only that ordering reveals queue is saturated.

This scenario is designed to *break* the base model's prior.  An RL agent
with the right credit assignment learns to interleave a dependency check
*before* committing to the loudest service.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set

from .base import BaseScenario
from .code_context_builder import ALIASED_FAULT_CODE_CONTEXT
from ..models import CodeContext
from ..simulation.infrastructure import Infrastructure
from ..simulation.service import Deploy
from ..simulation.metrics import (
    generate_memory_leak_history,
    generate_high_latency_history,
)


class AliasedFaultScenario(BaseScenario):

    @property
    def task_name(self) -> str:
        return "aliased_fault"

    @property
    def code_context(self) -> CodeContext:
        return ALIASED_FAULT_CODE_CONTEXT

    @property
    def fault_class(self) -> str:
        return "cache_thrash"

    @property
    def display_name(self) -> str:
        return "Aliased Fault — Cache Thrash Disguised as Memory Leak"

    @property
    def incident_summary(self) -> str:
        return (
            "INCIDENT: Orders service showing high memory usage (84%), elevated latency, "
            "and intermittent OOM-like errors. Recent orders deploy v2.4.0 was just "
            "rolled out 18 minutes ago. On-call SRE paged."
        )

    @property
    def severity(self) -> str:
        return "SEV2"

    @property
    def correct_root_cause(self) -> str:
        return "queue cache thrashing — saturated worker overflows shared cache used by orders"

    @property
    def root_cause_keywords(self) -> List[str]:
        return ["queue", "cache", "thrash", "worker"]

    @property
    def involved_services(self) -> Set[str]:
        return {"queue", "orders"}

    @property
    def root_cause_service(self) -> str:
        return "queue"

    @property
    def correct_remediation_actions(self) -> List[Dict[str, str]]:
        return [
            {"action_type": "rollback_deploy", "target_service": "queue"},
            {"action_type": "restart_service", "target_service": "orders"},
        ]

    def inject(self, infra: Infrastructure) -> None:
        orders = infra.get_service("orders")
        queue  = infra.get_service("queue")
        if orders is None or queue is None:
            return

        # --- Innocuous orders deploy (the red herring) ---
        orders.deploy_history.append(Deploy(
            version="v2.4.0", timestamp_minutes=-18, author="alice",
            commit_hash="b7d291", description="Refactor: extract pricing helper",
            is_bad=False,
        ))

        # --- Real bad deploy: queue worker (the hidden root cause) ---
        queue.deploy_history.append(Deploy(
            version="v2.1.0", timestamp_minutes=-22, author="dan",
            commit_hash="e1f4a02", description="Optimization: bulk flush via shared cache",
            is_bad=True,
        ))

        # --- Symptoms on orders look IDENTICAL to a memory leak ---
        orders.memory_percent = 84.0 + random.gauss(0, 2)
        orders.cpu_percent    = 41.0 + random.gauss(0, 3)
        orders.error_rate_percent = 14.0 + random.gauss(0, 2)
        orders.latency_p95_ms = 380.0 + random.gauss(0, 30)
        orders.latency_p99_ms = 920.0 + random.gauss(0, 50)
        orders.status = "degraded"
        orders.metric_history = generate_memory_leak_history(
            minutes=30, start_minute=0, leak_start_offset=12, rate=1.4)

        orders.logs = [
            {"timestamp": "2025-01-15T14:18:00Z", "level": "WARN", "service": "orders",
             "message": "GC pressure: heap usage at 78%, GC pause 410ms", "trace_id": None},
            {"timestamp": "2025-01-15T14:22:00Z", "level": "ERROR", "service": "orders",
             "message": "Allocation failure: cache eviction backlog growing",
             "trace_id": "trace-554301"},
            {"timestamp": "2025-01-15T14:25:00Z", "level": "ERROR", "service": "orders",
             "message": "OutOfMemoryError-like behaviour: cache write blocked >2s",
             "trace_id": "trace-554404"},
        ]

        # --- Real fault on queue (subtle: only visible if you check it) ---
        queue.inject_fault("high_latency", p99=4500)
        queue.metric_history = generate_high_latency_history(
            minutes=30, start_minute=0, latency_start_offset=10, target_p99=4500)
        queue.cpu_percent          = 88.0 + random.gauss(0, 3)
        queue.memory_percent       = 71.0 + random.gauss(0, 2)
        queue.error_rate_percent   = 6.0  + random.gauss(0, 1)
        queue.latency_p99_ms       = 4500.0 + random.gauss(0, 100)
        queue.requests_per_sec     = 90.0  + random.gauss(0, 5)
        queue.status               = "degraded"
        queue.logs = [
            {"timestamp": "2025-01-15T14:14:00Z", "level": "INFO", "service": "queue",
             "message": "Deploy v2.1.0 complete — bulk-flush worker active", "trace_id": None},
            {"timestamp": "2025-01-15T14:19:00Z", "level": "WARN", "service": "queue",
             "message": "Worker backlog: 8400 messages awaiting flush, eviction rate 320/s",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:24:00Z", "level": "ERROR", "service": "queue",
             "message": "Cache eviction storm: orders-cache shard 3 evicted 84% of keys",
             "trace_id": "trace-771204"},
        ]
