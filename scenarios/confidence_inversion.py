"""
Phase-B scenario 3 — Confidence inversion.

The symptom pattern most strongly resembles a memory leak (smoothly
climbing memory, OOM-flavoured logs, p99 latency drift) but the
deploy history shows nothing memory-related and the *real* root cause
is a distributed deadlock — accumulating threads pinned waiting on a
lock cycle.  RAM grows because thread stacks accumulate, not because
of a heap leak.

Why this scenario matters:
  - The ops agent's belief should be LOW confidence even though the
    surface evidence is "high confidence memory leak".
  - The orchestrator must learn to *not* transition with high
    confidence based on apparent-evidence — it must actively check
    deploys, then thread metrics, then escalate.

This directly stresses the stopping criterion: keep investigating
despite clean-looking symptoms.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set

from .base import BaseScenario
from .code_context_builder import CONFIDENCE_INVERSION_CODE_CONTEXT
from ..models import CodeContext
from ..simulation.infrastructure import Infrastructure
from ..simulation.service import Deploy
from ..simulation.metrics import generate_memory_leak_history


class ConfidenceInversionScenario(BaseScenario):

    @property
    def task_name(self) -> str:
        return "confidence_inversion"

    @property
    def code_context(self) -> CodeContext:
        return CONFIDENCE_INVERSION_CODE_CONTEXT

    @property
    def fault_class(self) -> str:
        return "deadlock"

    @property
    def display_name(self) -> str:
        return "Confidence Inversion — Deadlock Masquerading as Memory Leak"

    @property
    def incident_summary(self) -> str:
        return (
            "INCIDENT: Payment service memory has climbed from 38% to 81% over the "
            "last 25 minutes. Latency p99 drifting upward. OOM-flavoured exceptions "
            "appearing in logs. Looks like a textbook memory leak."
        )

    @property
    def severity(self) -> str:
        return "SEV2"

    @property
    def correct_root_cause(self) -> str:
        return "payment threadpool deadlock — accumulating pinned threads, not a heap leak"

    @property
    def root_cause_keywords(self) -> List[str]:
        return ["payment", "deadlock", "thread", "lock"]

    @property
    def involved_services(self) -> Set[str]:
        return {"payment"}

    @property
    def root_cause_service(self) -> str:
        return "payment"

    @property
    def correct_remediation_actions(self) -> List[Dict[str, str]]:
        return [
            {"action_type": "rollback_deploy", "target_service": "payment"},
            {"action_type": "restart_service", "target_service": "payment"},
        ]

    def inject(self, infra: Infrastructure) -> None:
        payment = infra.get_service("payment")
        if payment is None:
            return

        # --- The "innocent" recent deploy: a config change unrelated to memory.
        #     The agent who only checks deploy_history will see this and
        #     correctly note: nothing memory-shaped here.  That's the trap —
        #     the absence of a memory deploy is *evidence* the symptom is
        #     misleading, but the base-model will dismiss it.
        payment.deploy_history.append(Deploy(
            version="v3.1.2", timestamp_minutes=-25, author="dave",
            commit_hash="11abf04",
            description="Refactor: lock acquisition order in PoolWorker",
            is_bad=True,
        ))

        payment.inject_fault("memory_leak", rate=1.7)
        payment.metric_history = generate_memory_leak_history(
            minutes=30, start_minute=0, leak_start_offset=8, rate=1.7)
        payment.memory_percent      = 81.0 + random.gauss(0, 2)
        payment.cpu_percent         = 22.0 + random.gauss(0, 2)   # ← cpu LOW (deadlock signature)
        payment.error_rate_percent  = 9.0  + random.gauss(0, 1)
        payment.latency_p95_ms      = 1800.0 + random.gauss(0, 100)
        payment.latency_p99_ms      = 3400.0 + random.gauss(0, 200)
        payment.requests_per_sec    = 110.0 + random.gauss(0, 8)  # ← throughput collapsed
        payment.status              = "degraded"

        payment.logs = [
            {"timestamp": "2025-01-15T14:09:00Z", "level": "INFO", "service": "payment",
             "message": "Deploy v3.1.2 complete — pool worker refactor live", "trace_id": None},
            {"timestamp": "2025-01-15T14:14:00Z", "level": "WARN", "service": "payment",
             "message": "GC pressure: heap usage at 64%, GC pause 290ms (mostly old-gen)",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:18:00Z", "level": "WARN", "service": "payment",
             "message": "Thread pool active count: 198/200 (CPU usage low — threads blocked)",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:23:00Z", "level": "ERROR", "service": "payment",
             "message": "Lock contention: PoolWorker.acquire blocked for 8400ms on lock_b",
             "trace_id": "trace-883011"},
            {"timestamp": "2025-01-15T14:25:00Z", "level": "WARN", "service": "payment",
             "message": "OutOfMemoryError-like: thread stack pool exhausted",
             "trace_id": None},
        ]
