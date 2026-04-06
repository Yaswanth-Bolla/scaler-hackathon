"""
Alert firing engine.

Alerts fire based on metric thresholds — the agent sees what fired
but must investigate to find why.  Alert correlation (multiple alerts
from a cascading failure) is represented by shared source timestamps.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from .service import ServiceState


# ------------------------------------------------------------------
# Threshold definitions
# ------------------------------------------------------------------

_ALERT_RULES = [
    {
        "name": "HighErrorRate",
        "field": "error_rate_percent",
        "threshold": 10.0,
        "severity": "critical",
        "description": "{service}: error rate {value:.1f}% exceeds threshold 10%",
    },
    {
        "name": "HighMemoryUsage",
        "field": "memory_percent",
        "threshold": 80.0,
        "severity": "critical",
        "description": "{service}: memory usage {value:.0f}% exceeds threshold 80%",
    },
    {
        "name": "HighLatencyP99",
        "field": "latency_p99_ms",
        "threshold": 1000.0,
        "severity": "warning",
        "description": "{service}: p99 latency {value:.0f}ms exceeds threshold 1000ms",
    },
    {
        "name": "HighLatencyP95",
        "field": "latency_p95_ms",
        "threshold": 500.0,
        "severity": "warning",
        "description": "{service}: p95 latency {value:.0f}ms exceeds threshold 500ms",
    },
    {
        "name": "HighCPU",
        "field": "cpu_percent",
        "threshold": 80.0,
        "severity": "warning",
        "description": "{service}: CPU usage {value:.0f}% exceeds threshold 80%",
    },
    {
        "name": "ServiceDown",
        "field": "status",
        "threshold": "down",
        "severity": "critical",
        "description": "{service}: service is DOWN — health check failing",
    },
    {
        "name": "ServiceDegraded",
        "field": "status",
        "threshold": "degraded",
        "severity": "warning",
        "description": "{service}: service is DEGRADED — partial failures detected",
    },
    {
        "name": "LowRequestRate",
        "field": "requests_per_sec",
        "threshold": 100.0,
        "severity": "warning",
        "description": "{service}: request rate {value:.0f} rps dropped below threshold 100 rps",
        "below": True,
    },
]


def evaluate_alerts(
    services: Dict[str, ServiceState],
    current_minute: int,
) -> List[Dict[str, Any]]:
    """
    Evaluate all alert rules against current service states.
    Returns list of firing alert dicts.
    """
    alerts = []
    alert_counter = 0

    for svc_name, svc in services.items():
        for rule in _ALERT_RULES:
            field = rule["field"]
            threshold = rule["threshold"]

            # Status-based alerts
            if field == "status":
                if svc.status == threshold:
                    alert_counter += 1
                    alerts.append({
                        "alert_id": f"alert-{alert_counter:03d}",
                        "severity": rule["severity"],
                        "source_service": svc_name,
                        "description": rule["description"].format(
                            service=svc_name, value=0),
                        "firing_since": f"2025-01-15T14:{max(0, current_minute - svc.ticks_in_down):02d}:00Z"
                            if threshold == "down"
                            else f"2025-01-15T14:{max(0, current_minute - svc.ticks_in_degraded):02d}:00Z",
                        "rule_name": rule["name"],
                    })
                continue

            # Numeric threshold alerts
            value = getattr(svc, field, 0)
            is_below = rule.get("below", False)
            triggered = value < threshold if is_below else value > threshold

            if triggered:
                alert_counter += 1
                ticks_firing = max(1, len([
                    h for h in svc.metric_history[-10:]
                    if (h.get(field.replace("_percent", "").replace("_ms", ""),
                              h.get(field, 0))
                        < threshold if is_below
                        else h.get(field.replace("_percent", "").replace("_ms", ""),
                                   h.get(field, 0))
                        > threshold)
                ]))
                alerts.append({
                    "alert_id": f"alert-{alert_counter:03d}",
                    "severity": rule["severity"],
                    "source_service": svc_name,
                    "description": rule["description"].format(
                        service=svc_name, value=value),
                    "firing_since": f"2025-01-15T14:{max(0, current_minute - ticks_firing):02d}:00Z",
                    "rule_name": rule["name"],
                })

    # Sort by severity: critical first
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: severity_order.get(a["severity"], 9))

    return alerts
