"""
Phase-A "no-change" scenario.

Symptoms look real (orders is intermittently slow; user filed an issue
claiming "the new release broke checkout"), but on inspection the only
recent deploy is a documentation comment update — the slowness is the
service's normal weekly batch backup window kicking off.

The correct behaviour:
  - Phase 1: investigate, find that no fault is attributable
  - Phase 2: read the snapshot and emit `declare_no_change`

The scenario's `code_context.is_valid_issue == False`, so any proposed
diff scores 0 and `grade_no_change(True)` scores 1.0.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set

from .base import BaseScenario
from .code_context_builder import CIRCUIT_BREAKER_CODE_CONTEXT
from ..models import CodeContext
from ..simulation.infrastructure import Infrastructure
from ..simulation.service import Deploy
from ..simulation.metrics import generate_healthy_history


class CircuitBreakerNoopScenario(BaseScenario):

    @property
    def task_name(self) -> str:
        return "circuit_breaker_noop"

    @property
    def code_context(self) -> CodeContext:
        return CIRCUIT_BREAKER_CODE_CONTEXT

    @property
    def fault_class(self) -> str:
        return "none"

    @property
    def display_name(self) -> str:
        return "Spurious Issue — No Code Change Required"

    @property
    def incident_summary(self) -> str:
        return (
            "INCIDENT: User report — 'orders deploy v3.0.0 broke checkout, latency spiked'. "
            "On-call paged. The orders service is slightly elevated on latency but no "
            "alert fired."
        )

    @property
    def severity(self) -> str:
        return "SEV3"

    @property
    def correct_root_cause(self) -> str:
        return "no fault — orders is in its normal weekly backup window"

    @property
    def root_cause_keywords(self) -> List[str]:
        return ["no", "fault", "backup", "normal"]

    @property
    def involved_services(self) -> Set[str]:
        return {"orders"}

    @property
    def root_cause_service(self) -> str:
        return "orders"

    @property
    def correct_remediation_actions(self) -> List[Dict[str, str]]:
        # No remediation is "correct" — declaring no-change in P2 is the goal.
        return []

    def inject(self, infra: Infrastructure) -> None:
        orders = infra.get_service("orders")
        if orders is None:
            return

        orders.deploy_history.append(Deploy(
            version="v3.0.0", timestamp_minutes=-30, author="ed",
            commit_hash="d2b9c11",
            description="Docs: clarify checkout API contract in comments",
            is_bad=False,
        ))

        orders.metric_history = generate_healthy_history(minutes=30, start_minute=0)
        orders.cpu_percent          = 18.0 + random.gauss(0, 2)
        orders.memory_percent       = 41.0 + random.gauss(0, 2)
        orders.error_rate_percent   = 0.6  + random.gauss(0, 0.2)
        orders.latency_p95_ms       = 220.0 + random.gauss(0, 20)   # slightly elevated
        orders.latency_p99_ms       = 380.0 + random.gauss(0, 25)
        orders.status               = "healthy"

        orders.logs = [
            {"timestamp": "2025-01-15T14:00:00Z", "level": "INFO", "service": "orders",
             "message": "Weekly backup window started — minor latency overhead expected",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:12:00Z", "level": "INFO", "service": "orders",
             "message": "Backup snapshot 47% complete (12.4 GB / 26.5 GB)",
             "trace_id": None},
        ]
