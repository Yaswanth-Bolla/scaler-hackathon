"""
Task 3 — Hard: Distributed Deadlock.

A deploy to the payment service changed its retry logic to be aggressive.
This creates a circular wait:
  orders → payment (waiting on ack)
  payment → queue (retrying aggressively, flooding)
  queue → orders (backpressure, orders can't consume)

No single service crashes — all three show high latency and scattered
timeout errors.  The agent must correlate cross-service logs with trace IDs
and check deploy history to find that payment's retry change is the root cause.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set

from .base import BaseScenario
from ..simulation.infrastructure import Infrastructure
from ..simulation.service import Deploy
from ..simulation.metrics import generate_high_latency_history, generate_healthy_history


class DistributedDeadlockScenario(BaseScenario):

    @property
    def task_name(self) -> str:
        return "distributed_deadlock"

    @property
    def display_name(self) -> str:
        return "Distributed Deadlock — Payment/Orders/Queue Circular Wait"

    @property
    def incident_summary(self) -> str:
        return (
            "INCIDENT: Order processing latency has spiked dramatically. "
            "Customers report orders 'stuck in processing' for 5+ minutes. "
            "No single service appears to be down — all health checks pass "
            "but with high latency. Payment confirmations are severely delayed."
        )

    @property
    def severity(self) -> str:
        return "SEV1"

    @property
    def correct_root_cause(self) -> str:
        return "payment service deploy changed retry logic creating circular deadlock with orders and queue"

    @property
    def root_cause_keywords(self) -> List[str]:
        return ["payment", "retry", "deadlock", "deploy"]

    @property
    def involved_services(self) -> Set[str]:
        return {"orders", "payment", "queue"}

    @property
    def root_cause_service(self) -> str:
        return "payment"

    @property
    def correct_remediation_actions(self) -> List[Dict[str, str]]:
        return [
            {"action_type": "rollback_deploy", "target_service": "payment"},
            {"action_type": "scale_service", "target_service": "queue"},
            {"action_type": "restart_service", "target_service": "orders"},
        ]

    def inject(self, infra: Infrastructure) -> None:
        """
        Set up distributed deadlock:
        1. Payment gets bad deploy (aggressive retry) → circular_wait fault
        2. Orders and queue also get circular_wait
        3. All three show high latency but no crashes
        4. Logs show scattered timeouts with correlated trace IDs
        5. Red herring: cache shows a brief latency spike too
        """
        payment = infra.get_service("payment")
        orders = infra.get_service("orders")
        queue = infra.get_service("queue")
        cache = infra.get_service("cache")

        if not all([payment, orders, queue]):
            return

        # --- Bad deploy on payment (root cause) ---
        bad_deploy = Deploy(
            version="v3.1.0",
            timestamp_minutes=-12,
            author="charlie",
            commit_hash="f7g8h9",
            description="Improve payment reliability: increase retry count and reduce backoff",
            is_bad=True,
        )
        payment.deploy_history.append(bad_deploy)

        # --- Inject circular_wait on all three services ---
        payment.inject_fault("circular_wait", peers=["queue", "orders"])
        orders.inject_fault("circular_wait", peers=["payment"])
        queue.inject_fault("circular_wait", peers=["orders", "payment"])

        # --- High latency metric histories ---
        payment.metric_history = generate_high_latency_history(
            minutes=30, start_minute=0, latency_start_offset=18, target_p99=8000)
        orders.metric_history = generate_high_latency_history(
            minutes=30, start_minute=0, latency_start_offset=20, target_p99=6000)
        queue.metric_history = generate_high_latency_history(
            minutes=30, start_minute=0, latency_start_offset=19, target_p99=10000)

        # --- Current metrics ---
        for svc, p99 in [(payment, 7000), (orders, 5500), (queue, 9000)]:
            svc.latency_p50_ms = 400 + random.gauss(0, 30)
            svc.latency_p95_ms = p99 * 0.6 + random.gauss(0, 100)
            svc.latency_p99_ms = p99 + random.gauss(0, 200)
            svc.error_rate_percent = 12.0 + random.gauss(0, 3)
            svc.requests_per_sec = max(30, 150 + random.gauss(0, 20))
            svc.status = "degraded"

        # --- Shared trace IDs to enable cross-service correlation ---
        shared_traces = [f"trace-{random.randint(800000, 899999)}" for _ in range(5)]

        # Payment logs
        payment.logs = [
            {"timestamp": "2025-01-15T14:18:00Z", "level": "INFO", "service": "payment",
             "message": "Deploy v3.1.0 started — retry improvements", "trace_id": None},
            {"timestamp": "2025-01-15T14:18:30Z", "level": "INFO", "service": "payment",
             "message": "Deploy v3.1.0 complete — retry_count=10, backoff=100ms",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:20:00Z", "level": "WARN", "service": "payment",
             "message": f"Retrying queue publish: attempt 5/10, waiting 100ms — {shared_traces[0]}",
             "trace_id": shared_traces[0]},
            {"timestamp": "2025-01-15T14:21:00Z", "level": "ERROR", "service": "payment",
             "message": f"Timeout waiting for queue acknowledgment: blocked 12000ms",
             "trace_id": shared_traces[1]},
            {"timestamp": "2025-01-15T14:23:00Z", "level": "WARN", "service": "payment",
             "message": "Thread pool: 195/200 threads blocked waiting on downstream calls",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:25:00Z", "level": "ERROR", "service": "payment",
             "message": f"Payment processing stuck: round-trip to queue exceeded 25000ms",
             "trace_id": shared_traces[2]},
        ]

        # Orders logs
        orders.logs = [
            {"timestamp": "2025-01-15T14:20:00Z", "level": "WARN", "service": "orders",
             "message": f"Waiting on payment confirmation: no response after 8000ms",
             "trace_id": shared_traces[0]},
            {"timestamp": "2025-01-15T14:21:30Z", "level": "ERROR", "service": "orders",
             "message": f"Order {random.randint(10000, 99999)} stuck in PROCESSING state: "
                        f"payment callback not received after 15000ms",
             "trace_id": shared_traces[1]},
            {"timestamp": "2025-01-15T14:23:00Z", "level": "WARN", "service": "orders",
             "message": "Cannot consume from queue: consumer threads blocked waiting on payment",
             "trace_id": shared_traces[2]},
            {"timestamp": "2025-01-15T14:25:00Z", "level": "ERROR", "service": "orders",
             "message": "Timeout calling payment-service: deadline exceeded after 30000ms",
             "trace_id": shared_traces[3]},
        ]

        # Queue logs
        queue.logs = [
            {"timestamp": "2025-01-15T14:20:00Z", "level": "WARN", "service": "queue",
             "message": "Queue depth increasing: 5,420 messages pending (threshold: 1,000)",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:21:00Z", "level": "WARN", "service": "queue",
             "message": f"Consumer lag: orders consumer 3,200 messages behind",
             "trace_id": None},
            {"timestamp": "2025-01-15T14:22:00Z", "level": "ERROR", "service": "queue",
             "message": f"Publish flood detected: payment-service publishing at 500 msg/s "
                        f"(normal: 50 msg/s)", "trace_id": shared_traces[2]},
            {"timestamp": "2025-01-15T14:24:00Z", "level": "WARN", "service": "queue",
             "message": "Backpressure applied to orders consumer: cannot keep up with "
                        "publish rate", "trace_id": shared_traces[3]},
            {"timestamp": "2025-01-15T14:26:00Z", "level": "ERROR", "service": "queue",
             "message": "Queue depth critical: 12,840 messages pending — memory pressure",
             "trace_id": None},
        ]

        # --- Red herring: cache shows a brief latency spike ---
        if cache:
            cache.logs.append({
                "timestamp": "2025-01-15T14:22:00Z", "level": "WARN", "service": "cache",
                "message": "Redis SLOWLOG: KEYS pattern='order:pending:*' took 450ms",
                "trace_id": None})
            cache.logs.append({
                "timestamp": "2025-01-15T14:24:00Z", "level": "WARN", "service": "cache",
                "message": "Redis memory usage: 78% — evicting LRU keys",
                "trace_id": None})
