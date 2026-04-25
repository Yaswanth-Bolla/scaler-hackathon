"""
Pool D — held-out scenarios.

These combine fault families the agent saw individually during training
into novel compound scenarios that test whether the *strategy* generalized
(rather than just memorising scenario fingerprints).

Each held-out scenario reuses the same code-context infrastructure (real
mini-repo snapshots, ground-truth diffs) but in a configuration the agent
has *never* seen during training.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set

from .base import BaseScenario
from .code_context_builder import (
    HELDOUT_ALIASED_SEVERITY_CODE_CONTEXT,
    HELDOUT_CONFIDENCE_ORDERING_CODE_CONTEXT,
)
from ..models import CodeContext
from ..simulation.infrastructure import Infrastructure
from ..simulation.service import Deploy
from ..simulation.metrics import (
    generate_memory_leak_history,
    generate_error_spike_history,
    generate_high_latency_history,
)


# ──────────────────────────────────────────────────────────────────────
# 1. aliased + severity_inversion combo
#
# A retry storm in `orders` saturates `auth` (severity inversion), AND the
# saturation causes `auth`'s memory to climb because thread queues build up
# (aliased symptom — looks like memory leak in auth).
#
# Diagnostic strategy must combine: skip the loud service AND check
# dependencies.  Neither aliased_fault nor severity_inversion alone teach
# both rules simultaneously.
# ──────────────────────────────────────────────────────────────────────


class HeldoutAliasedSeverityScenario(BaseScenario):

    @property
    def task_name(self) -> str:
        return "heldout_aliased_severity"

    @property
    def code_context(self) -> CodeContext:
        return HELDOUT_ALIASED_SEVERITY_CODE_CONTEXT

    @property
    def fault_class(self) -> str:
        return "retry_storm"

    @property
    def display_name(self) -> str:
        return "[Held-out] Aliased + Severity Inversion"

    @property
    def incident_summary(self) -> str:
        return (
            "INCIDENT: Auth service is firing CRITICAL alerts (HighMemoryUsage, "
            "HighErrorRate). It looks like a memory leak in auth. Orders shows "
            "only a WARN-level retry-counter alert."
        )

    @property
    def severity(self) -> str:
        return "SEV1"

    @property
    def correct_root_cause(self) -> str:
        return "orders auth-client retry storm — auth is the victim, memory growth is queue build-up"

    @property
    def root_cause_keywords(self) -> List[str]:
        return ["orders", "retry", "auth-client", "storm"]

    @property
    def involved_services(self) -> Set[str]:
        return {"orders", "auth"}

    @property
    def root_cause_service(self) -> str:
        return "orders"

    @property
    def correct_remediation_actions(self) -> List[Dict[str, str]]:
        return [
            {"action_type": "rollback_deploy", "target_service": "orders"},
            {"action_type": "restart_service", "target_service": "auth"},
        ]

    def inject(self, infra: Infrastructure) -> None:
        orders = infra.get_service("orders")
        auth   = infra.get_service("auth")
        if orders is None or auth is None:
            return

        orders.deploy_history.append(Deploy(
            version="v2.6.0", timestamp_minutes=-14, author="frankie",
            commit_hash="d09a4f1",
            description="Resilience: bump auth-client retries 3 → 25 with no jitter",
            is_bad=True,
        ))
        # Auth has no recent change
        auth.deploy_history.append(Deploy(
            version="v1.7.4", timestamp_minutes=-86400, author="bob",
            commit_hash="a01122", description="Routine: bump TLS cert", is_bad=False,
        ))

        # Auth shows BOTH high memory and high error rate (aliased pattern!)
        auth.inject_fault("memory_leak", rate=0.9)
        auth.inject_fault("high_error_rate", rate=58.0)
        auth.metric_history = generate_memory_leak_history(
            minutes=30, start_minute=0, leak_start_offset=14, rate=0.9)
        auth.memory_percent      = 79.0 + random.gauss(0, 2)
        auth.error_rate_percent  = 58.0 + random.gauss(0, 3)
        auth.latency_p95_ms      = 3500 + random.gauss(0, 200)
        auth.latency_p99_ms      = 6200 + random.gauss(0, 400)
        auth.requests_per_sec    = 4100 + random.gauss(0, 200)
        auth.status              = "down"
        auth.logs = [
            {"timestamp": "2025-01-15T14:13:00Z", "level": "WARN", "service": "auth",
             "message": "Throughput surged 1300 → 4100 RPS in 4 min", "trace_id": None},
            {"timestamp": "2025-01-15T14:18:00Z", "level": "ERROR", "service": "auth",
             "message": "OOM-like: token validation queue at 11400 entries",
             "trace_id": "trace-771233"},
        ]

        orders.error_rate_percent = 1.8 + random.gauss(0, 0.4)
        orders.latency_p95_ms     = 130.0 + random.gauss(0, 10)
        orders.status             = "degraded"
        orders.logs = [
            {"timestamp": "2025-01-15T14:14:00Z", "level": "INFO", "service": "orders",
             "message": "Deploy v2.6.0 complete — auth-client retry policy updated",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:18:00Z", "level": "WARN", "service": "orders",
             "message": "auth-client retry counter elevated: avg 17 retries per validation",
             "trace_id": None},
        ]


# ──────────────────────────────────────────────────────────────────────
# 2. confidence_inversion + info_ordering combo
#
# Symptoms scream "memory leak in payment" but the cause is a shared
# dependency downgrade that happened to interact badly with payment's
# threadpool — so the fix is in `requirements.txt`, not in payment's
# service code.  The base model with high confidence on memory_leak
# will rollback payment, fail to fix it, then look at payment's deploy
# history (which is clean), and run out of time.
# ──────────────────────────────────────────────────────────────────────


class HeldoutConfidenceOrderingScenario(BaseScenario):

    @property
    def task_name(self) -> str:
        return "heldout_confidence_ordering"

    @property
    def code_context(self) -> CodeContext:
        return HELDOUT_CONFIDENCE_ORDERING_CODE_CONTEXT

    @property
    def fault_class(self) -> str:
        return "shared_dependency"

    @property
    def display_name(self) -> str:
        return "[Held-out] Confidence + Info-Ordering Inversion"

    @property
    def incident_summary(self) -> str:
        return (
            "INCIDENT: Payment service memory at 83%, climbing. Latency p99 4200ms "
            "and rising. Looks like a clear memory leak. No payment deploys in 24h."
        )

    @property
    def severity(self) -> str:
        return "SEV2"

    @property
    def correct_root_cause(self) -> str:
        return "shared-serializer dependency downgrade interacting with payment threadpool"

    @property
    def root_cause_keywords(self) -> List[str]:
        return ["shared", "dependency", "serializer", "thread"]

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
        ]

    def inject(self, infra: Infrastructure) -> None:
        payment = infra.get_service("payment")
        if payment is None:
            return

        payment.deploy_history.append(Deploy(
            version="img-payment-732", timestamp_minutes=-9, author="lib-bot",
            commit_hash="9d2e7af",
            description="Image rebuild: pulled latest shared-serializer (downgraded)",
            is_bad=True,
        ))

        payment.inject_fault("memory_leak", rate=1.6)
        payment.inject_fault("high_latency", p99=4200)
        payment.metric_history = generate_memory_leak_history(
            minutes=30, start_minute=0, leak_start_offset=8, rate=1.6)
        payment.memory_percent       = 83.0 + random.gauss(0, 2)
        payment.cpu_percent          = 24.0 + random.gauss(0, 2)
        payment.error_rate_percent   = 11.0 + random.gauss(0, 1)
        payment.latency_p99_ms       = 4200.0 + random.gauss(0, 200)
        payment.requests_per_sec     = 95.0 + random.gauss(0, 5)
        payment.status               = "degraded"

        payment.logs = [
            {"timestamp": "2025-01-15T14:08:00Z", "level": "INFO", "service": "payment",
             "message": "Image rebuild complete (shared-serializer pulled)",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:14:00Z", "level": "WARN", "service": "payment",
             "message": "Heap usage at 71%, GC pause 380ms",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:19:00Z", "level": "ERROR", "service": "payment",
             "message": "Deserialization failure: UnknownFieldError "
                        "(check shared-serializer version)",
             "trace_id": "trace-441312"},
            {"timestamp": "2025-01-15T14:22:00Z", "level": "WARN", "service": "payment",
             "message": "Thread pool saturated: 196/200 threads blocked on retry",
             "trace_id": None},
        ]
