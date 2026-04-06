"""
Log stream generator.

Produces realistic structured log entries — both signal and noise.
Red herring logs are mixed in so the agent must filter real evidence
from routine chatter.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Noise logs — routine operational chatter
# ------------------------------------------------------------------

_NOISE_TEMPLATES = [
    ("INFO", "Processed {n} requests in last 60 seconds"),
    ("INFO", "Health check passed — all dependencies reachable"),
    ("INFO", "Connection pool stats: active={a}, idle={i}, max={m}"),
    ("DEBUG", "Cache hit ratio: {r:.1%} — {h} hits, {m} misses"),
    ("INFO", "Scheduled job 'metrics_export' completed in {d}ms"),
    ("DEBUG", "TLS handshake completed with upstream in {d}ms"),
    ("INFO", "Config reload: no changes detected"),
    ("WARN", "Slow query detected: SELECT * FROM sessions took {d}ms"),
    ("INFO", "Garbage collection: freed {n}MB in {d}ms"),
    ("DEBUG", "Rate limiter: {n} requests allowed, 0 throttled"),
]


def generate_noise_logs(
    service_name: str,
    current_minute: int,
    count: int = 3,
) -> List[Dict[str, Any]]:
    """Generate routine noise logs for a service."""
    logs = []
    for _ in range(count):
        template_level, template_msg = random.choice(_NOISE_TEMPLATES)
        msg = template_msg.format(
            n=random.randint(100, 5000),
            a=random.randint(5, 20),
            i=random.randint(0, 10),
            m=random.randint(20, 50),
            r=random.uniform(0.85, 0.99),
            h=random.randint(1000, 9000),
            d=random.randint(1, 500),
        )
        logs.append({
            "timestamp": f"2025-01-15T14:{current_minute:02d}:{random.randint(0,59):02d}Z",
            "level": template_level,
            "service": service_name,
            "message": msg,
            "trace_id": None,
        })
    return logs


# ------------------------------------------------------------------
# Scenario-specific log generators (signal)
# ------------------------------------------------------------------

def generate_memory_leak_logs(
    service_name: str,
    current_minute: int,
    memory_percent: float,
) -> List[Dict[str, Any]]:
    """Logs that indicate a memory leak is in progress."""
    logs = []
    trace = f"trace-{random.randint(100000, 999999)}"

    if memory_percent > 90:
        logs.append(_log(current_minute, "FATAL", service_name,
            f"OutOfMemoryError: Java heap space — requested 256MB, "
            f"available 12MB", trace))
        logs.append(_log(current_minute, "ERROR", service_name,
            f"Container {service_name}-{random.randint(0,2)} killed by OOM killer "
            f"(exit code 137)", trace))
    elif memory_percent > 80:
        logs.append(_log(current_minute, "ERROR", service_name,
            f"Memory allocation failed: unable to allocate {random.randint(64, 256)}MB "
            f"for request processing", trace))
        logs.append(_log(current_minute, "WARN", service_name,
            f"GC overhead limit exceeded: spent {random.randint(80, 97)}% of time in GC"))
    elif memory_percent > 70:
        logs.append(_log(current_minute, "WARN", service_name,
            f"Heap usage warning: {memory_percent:.0f}% — approaching limit. "
            f"Consider increasing -Xmx or investigating leaks"))

    return logs


def generate_auth_failure_logs(
    service_name: str,
    current_minute: int,
    is_auth_service: bool = False,
) -> List[Dict[str, Any]]:
    """Logs for auth-related failures (used in cascading failure scenario)."""
    logs = []
    trace = f"trace-{random.randint(100000, 999999)}"

    if is_auth_service:
        logs.append(_log(current_minute, "ERROR", service_name,
            "NullPointerException: configuration key 'auth.jwt.secret' is null "
            "— cannot validate tokens", trace))
        logs.append(_log(current_minute, "ERROR", service_name,
            f"Authentication failed for {random.randint(50, 200)} requests in "
            f"last 60s — returning HTTP 500"))
    else:
        logs.append(_log(current_minute, "ERROR", service_name,
            f"Call to auth-service failed: HTTP 500 Internal Server Error "
            f"— retrying ({random.randint(1,3)}/3)", trace))
        logs.append(_log(current_minute, "WARN", service_name,
            f"Circuit breaker for auth-service: state=HALF_OPEN, "
            f"failures={random.randint(5, 20)}, threshold=10"))

    return logs


def generate_deadlock_logs(
    service_name: str,
    current_minute: int,
    waiting_on: str,
) -> List[Dict[str, Any]]:
    """Logs for distributed deadlock / circular wait."""
    logs = []
    trace = f"trace-{random.randint(100000, 999999)}"

    logs.append(_log(current_minute, "WARN", service_name,
        f"Request {trace} waiting on {waiting_on}: blocked for "
        f"{random.randint(5000, 25000)}ms — no response", trace))

    if random.random() < 0.4:
        logs.append(_log(current_minute, "ERROR", service_name,
            f"Timeout calling {waiting_on}: deadline exceeded after 30000ms. "
            f"Retry attempt {random.randint(3, 8)} of 10", trace))

    if random.random() < 0.2:
        logs.append(_log(current_minute, "ERROR", service_name,
            f"Thread pool exhausted: all {random.randint(50, 200)} threads blocked "
            f"waiting on downstream calls"))

    return logs


# ------------------------------------------------------------------
# Red herring logs — plausible but misleading
# ------------------------------------------------------------------

_RED_HERRING_TEMPLATES = [
    ("WARN", "DNS resolution for {svc}.internal took {d}ms (threshold: 100ms)"),
    ("WARN", "TLS certificate for {svc}.internal expires in {n} days"),
    ("WARN", "Disk usage on /var/log: {n}% — consider log rotation"),
    ("ERROR", "Failed to export metrics to Prometheus: connection timeout after {d}ms"),
    ("WARN", "Background job 'cleanup_sessions' took {d}ms (expected <500ms)"),
    ("ERROR", "Redis SLOWLOG: KEYS pattern='session:*' took {d}ms"),
]


def generate_red_herring_logs(
    service_name: str,
    current_minute: int,
    count: int = 1,
) -> List[Dict[str, Any]]:
    """Generate plausible but misleading log entries."""
    logs = []
    services = ["api_gateway", "auth", "orders", "payment", "cache", "database", "queue"]
    for _ in range(count):
        level, tmpl = random.choice(_RED_HERRING_TEMPLATES)
        msg = tmpl.format(
            svc=random.choice(services),
            d=random.randint(100, 3000),
            n=random.randint(3, 85),
        )
        logs.append(_log(current_minute, level, service_name, msg))
    return logs


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _log(
    minute: int,
    level: str,
    service: str,
    message: str,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "timestamp": f"2025-01-15T14:{minute:02d}:{random.randint(0,59):02d}Z",
        "level": level,
        "service": service,
        "message": message,
        "trace_id": trace_id,
    }
