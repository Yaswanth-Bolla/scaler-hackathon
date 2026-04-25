"""
Phase-B scenario 4 — Information ordering dependency.

Three services (`orders`, `payment`, `queue`) all degrade *simultaneously*
because a shared dependency (`shared-serializer`) was downgraded in
`requirements.txt` by an unrelated PR.  None of the three service-local
deploy histories show anything related — `check_deploy_history(orders)`
returns clean for each service.

The ONLY way to find the cause is to ask: "what was the most recent
commit that touched a file *every degraded service depends on*?"
That requires checking the **shared dependency file's git log**, which
the base model (following per-service deploy-history priors) won't do.

This is the canonical "the base model has the wrong prior on which
artifact to inspect first" scenario.  It's the one most likely to
expose a difference between prompt-engineered baselines and an
RL-trained agent that has internalized "correlated multi-service
degradation → look at shared dependency".

In Phase 2, the diff is in `requirements.txt` (not in any service's
own source tree), so the code agent must learn to inspect dependency
manifests as well as service-local code.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set

from .base import BaseScenario
from .code_context_builder import INFO_ORDERING_CODE_CONTEXT
from ..models import CodeContext
from ..simulation.infrastructure import Infrastructure
from ..simulation.service import Deploy
from ..simulation.metrics import (
    generate_error_spike_history,
    generate_high_latency_history,
)


class InfoOrderingScenario(BaseScenario):

    @property
    def task_name(self) -> str:
        return "info_ordering"

    @property
    def code_context(self) -> CodeContext:
        return INFO_ORDERING_CODE_CONTEXT

    @property
    def fault_class(self) -> str:
        return "shared_dependency"

    @property
    def display_name(self) -> str:
        return "Info Ordering — Shared-Library Downgrade Hits Three Services"

    @property
    def incident_summary(self) -> str:
        return (
            "INCIDENT: Three independent services — orders, payment, queue — all "
            "started erroring simultaneously 8 minutes ago. Errors are deserialization "
            "failures (UnknownFieldError, version mismatch). No service was deployed "
            "in the last 24 hours."
        )

    @property
    def severity(self) -> str:
        return "SEV1"

    @property
    def correct_root_cause(self) -> str:
        return "shared-serializer dependency downgrade in requirements.txt — affects all three services"

    @property
    def root_cause_keywords(self) -> List[str]:
        return ["shared", "serializer", "dependency", "requirements"]

    @property
    def involved_services(self) -> Set[str]:
        return {"orders", "payment", "queue"}

    @property
    def root_cause_service(self) -> str:
        # No single service deploy is to blame — but for remediation purposes
        # we treat `orders` as the canonical target (rolling its image will
        # pull the correct shared-serializer back in).
        return "orders"

    @property
    def correct_remediation_actions(self) -> List[Dict[str, str]]:
        return [
            {"action_type": "rollback_deploy", "target_service": "orders"},
            {"action_type": "rollback_deploy", "target_service": "payment"},
            {"action_type": "rollback_deploy", "target_service": "queue"},
        ]

    def inject(self, infra: Infrastructure) -> None:
        orders  = infra.get_service("orders")
        payment = infra.get_service("payment")
        queue   = infra.get_service("queue")
        if not all([orders, payment, queue]):
            return

        # --- A *single* shared-deps PR was merged — it touches no service
        # individually but affects all three image rebuilds.
        # We attach the same Deploy stamp to each so the agent can see "all
        # three have a deploy at the same minute" only after exhaustively
        # checking each one.
        for svc in (orders, payment, queue):
            svc.deploy_history.append(Deploy(
                version=f"img-{svc.name}-{random.randint(100, 999)}",
                timestamp_minutes=-8, author="lib-bot",
                commit_hash="9d2e7af",
                description="Image rebuild: pulled latest shared-serializer (downgraded)",
                is_bad=True,
            ))

        # --- Symptoms: all three show high error rate, similar log message
        for svc, p99 in [(orders, 1500), (payment, 2200), (queue, 1800)]:
            svc.inject_fault("high_error_rate", rate=24.0)
            svc.error_rate_percent  = 24.0 + random.gauss(0, 3)
            svc.latency_p95_ms      = p99 * 0.6 + random.gauss(0, 80)
            svc.latency_p99_ms      = p99 + random.gauss(0, 100)
            svc.status              = "degraded"
            svc.metric_history      = generate_error_spike_history(
                minutes=30, start_minute=0, spike_start_offset=22,
                error_rate_target=24.0)

        # --- Each service's logs say the same deserialisation error ---
        for svc in (orders, payment, queue):
            svc.logs = [
                {"timestamp": "2025-01-15T14:21:30Z", "level": "INFO", "service": svc.name,
                 "message": "Image rebuild complete — container restarted",
                 "trace_id": None},
                {"timestamp": "2025-01-15T14:22:30Z", "level": "ERROR", "service": svc.name,
                 "message": ("Deserialization failed: UnknownFieldError 'event_v2.idempotency_key' "
                             "(shared-serializer mismatch?)"),
                 "trace_id": "trace-441100"},
                {"timestamp": "2025-01-15T14:24:00Z", "level": "ERROR", "service": svc.name,
                 "message": ("Schema version mismatch: expected 1.4.x, got 1.3.0 "
                             "(check shared-serializer version)"),
                 "trace_id": "trace-441101"},
            ]
