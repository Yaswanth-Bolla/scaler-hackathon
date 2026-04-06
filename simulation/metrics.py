"""
Metric time-series generator.

Produces plausible metric history for services — both the healthy baseline
and the anomaly window. Used to populate metric_history on reset so the
agent sees a 30-minute lookback, not just the current point.
"""

from __future__ import annotations

import random
from typing import Dict, List


def generate_healthy_history(
    minutes: int = 30,
    start_minute: int = 0,
) -> List[Dict[str, float]]:
    """Generate 'minutes' worth of normal baseline metrics."""
    history = []
    for m in range(start_minute, start_minute + minutes):
        noise = lambda: random.gauss(0, 1)
        history.append({
            "minute": m,
            "cpu": round(max(5, min(40, 15 + noise() * 3)), 1),
            "memory": round(max(20, min(55, 35 + noise() * 3)), 1),
            "error_rate": round(max(0, min(2, 0.1 + abs(noise()) * 0.1)), 2),
            "latency_p50": round(max(5, 12 + noise() * 2), 1),
            "latency_p95": round(max(20, 45 + noise() * 5), 1),
            "latency_p99": round(max(50, 120 + noise() * 10), 1),
            "rps": round(max(200, 500 + noise() * 30), 1),
        })
    return history


def generate_memory_leak_history(
    minutes: int = 30,
    start_minute: int = 0,
    leak_start_offset: int = 10,
    rate: float = 1.5,
) -> List[Dict[str, float]]:
    """
    Generate metric history with a memory leak starting partway through.
    First 'leak_start_offset' minutes are normal, then memory climbs.
    """
    history = []
    mem = 35.0
    for m in range(start_minute, start_minute + minutes):
        noise = lambda: random.gauss(0, 1)
        elapsed = m - start_minute
        if elapsed >= leak_start_offset:
            mem = min(99.0, mem + rate + noise() * 0.3)
            cpu = min(95, 15 + (elapsed - leak_start_offset) * 0.3 + noise() * 2)
            error_rate = max(0, min(100, (mem - 75) * 2 + noise() * 2)) if mem > 75 else 0.1
            lat_p95 = max(45, 45 + (mem - 70) * 8 + noise() * 10) if mem > 70 else 45 + noise() * 5
            lat_p99 = max(120, lat_p95 * 2.5 + noise() * 20)
        else:
            cpu = max(5, min(40, 15 + noise() * 3))
            mem = max(20, min(55, 35 + noise() * 3))
            error_rate = max(0, 0.1 + abs(noise()) * 0.1)
            lat_p95 = max(20, 45 + noise() * 5)
            lat_p99 = max(50, 120 + noise() * 10)

        history.append({
            "minute": m,
            "cpu": round(cpu, 1),
            "memory": round(mem, 1),
            "error_rate": round(error_rate, 2),
            "latency_p50": round(max(5, 12 + noise() * 2), 1),
            "latency_p95": round(lat_p95, 1),
            "latency_p99": round(lat_p99, 1),
            "rps": round(max(100, 500 - max(0, mem - 80) * 15 + noise() * 20), 1),
        })
    return history


def generate_error_spike_history(
    minutes: int = 30,
    start_minute: int = 0,
    spike_start_offset: int = 5,
    error_rate_target: float = 60.0,
) -> List[Dict[str, float]]:
    """Metric history where error rate jumps suddenly (e.g. bad config push)."""
    history = []
    for m in range(start_minute, start_minute + minutes):
        noise = lambda: random.gauss(0, 1)
        elapsed = m - start_minute
        if elapsed >= spike_start_offset:
            error_rate = min(100, error_rate_target + noise() * 5)
            lat_p95 = max(100, 500 + noise() * 50)
            lat_p99 = max(200, 1500 + noise() * 100)
            cpu = max(5, min(80, 40 + noise() * 5))
        else:
            error_rate = max(0, 0.1 + abs(noise()) * 0.1)
            lat_p95 = max(20, 45 + noise() * 5)
            lat_p99 = max(50, 120 + noise() * 10)
            cpu = max(5, min(40, 15 + noise() * 3))

        history.append({
            "minute": m,
            "cpu": round(cpu, 1),
            "memory": round(max(20, min(55, 35 + noise() * 3)), 1),
            "error_rate": round(error_rate, 2),
            "latency_p50": round(max(5, 12 + noise() * 2), 1),
            "latency_p95": round(lat_p95, 1),
            "latency_p99": round(lat_p99, 1),
            "rps": round(max(100, 500 - error_rate * 3 + noise() * 20), 1),
        })
    return history


def generate_high_latency_history(
    minutes: int = 30,
    start_minute: int = 0,
    latency_start_offset: int = 8,
    target_p99: float = 8000,
) -> List[Dict[str, float]]:
    """Metric history with gradually increasing latency (deadlock/contention)."""
    history = []
    for m in range(start_minute, start_minute + minutes):
        noise = lambda: random.gauss(0, 1)
        elapsed = m - start_minute
        if elapsed >= latency_start_offset:
            progress = min(1.0, (elapsed - latency_start_offset) / 15)
            lat_p50 = max(12, 12 + progress * 400 + noise() * 20)
            lat_p95 = max(45, 45 + progress * target_p99 * 0.6 + noise() * 80)
            lat_p99 = max(120, 120 + progress * target_p99 + noise() * 200)
            error_rate = max(0, progress * 15 + noise() * 2)
            rps = max(20, 500 * (1 - progress * 0.7) + noise() * 15)
        else:
            lat_p50 = max(5, 12 + noise() * 2)
            lat_p95 = max(20, 45 + noise() * 5)
            lat_p99 = max(50, 120 + noise() * 10)
            error_rate = max(0, 0.1 + abs(noise()) * 0.1)
            rps = max(200, 500 + noise() * 30)

        history.append({
            "minute": m,
            "cpu": round(max(5, min(60, 15 + noise() * 3)), 1),
            "memory": round(max(20, min(55, 35 + noise() * 3)), 1),
            "error_rate": round(error_rate, 2),
            "latency_p50": round(lat_p50, 1),
            "latency_p95": round(lat_p95, 1),
            "latency_p99": round(lat_p99, 1),
            "rps": round(rps, 1),
        })
    return history
