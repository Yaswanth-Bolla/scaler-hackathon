"""
Behavioral metrics for Phase D ablations.

These four metrics answer the question "did RL actually *change* the
policy, or did it just shift the score?"  Without a behavioral signal,
a fixed-threshold orchestrator could match the trained orchestrator
on aggregate score by luck — these metrics force them to behave
differently to score differently.

All metrics consume a flat list of episode-result dicts (the structure
returned by `rollouts.run_episode`).  No torch/numpy/pandas — just
stdlib, so this module is friendly to dump-and-replay analysis on
saved trajectories.

  1. stopping_distribution(...)
       Histogram of (Phase-1 length) buckets per condition; KL between
       conditions is the headline number.

  2. action_ordering_position(action, ...)
       Per-condition mean *position-in-episode* at which a given action
       first appears.  Use to show the trained policy interleaves
       `check_dependencies` earlier than fixed-threshold baselines.

  3. confidence_calibration(...)
       Per-bucket reliability curve: declared `service_confidence` vs
       actual root-cause-correctness rate.  Returns ECE.

  4. p2_search_breadth_correlation(...)
       Pearson correlation between Phase-1 fault_confidence and the
       number of distinct files inspected during Phase 2.
       (Negative correlation = "low confidence -> wider search" is
       the desired learned behaviour.)
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple


EpisodeResult = Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _p1_length(ep: EpisodeResult) -> int:
    """Number of Phase-1 steps (excluding transition)."""
    return sum(1 for s in ep.get("p1_trajectory", []) if s.get("phase", 1) == 1)


def _p2_length(ep: EpisodeResult) -> int:
    return sum(1 for s in ep.get("p2_trajectory", []) if s.get("phase", 1) == 2)


def _action_at_step(ep: EpisodeResult, action_type: str) -> Optional[int]:
    """Index of first occurrence of `action_type` in the merged trajectory."""
    merged = ep.get("p1_trajectory", []) + ep.get("p2_trajectory", [])
    for i, s in enumerate(merged):
        atype = s.get("action", {}).get("action_type", "")
        if atype == action_type:
            return i
    return None


def _files_inspected_p2(ep: EpisodeResult) -> int:
    """Distinct file paths read or diffed in P2."""
    seen = set()
    for s in ep.get("p2_trajectory", []):
        params = s.get("action", {}).get("parameters", {}) or {}
        path = params.get("path") or ""
        if path:
            seen.add(path)
    return len(seen)


def _final_belief(ep: EpisodeResult) -> Dict[str, Any]:
    """The belief snapshot recorded at the transition step (if any)."""
    for s in ep.get("p2_trajectory", []):
        if s.get("belief_state_snapshot"):
            return s["belief_state_snapshot"]
    return {}


def _patch_correct(ep: EpisodeResult, threshold: float = 0.5) -> bool:
    """Did the proposed patch pass the threshold?"""
    return float(
        ep.get("score_breakdown", {}).get("patch_quality", 0.0)
    ) >= threshold


# ──────────────────────────────────────────────────────────────────────
# 1. Stopping distribution shift
# ──────────────────────────────────────────────────────────────────────


def stopping_distribution(
    episodes_by_condition: Dict[str, List[EpisodeResult]],
    bucket_edges: Tuple[int, ...] = (0, 4, 6, 8, 10, 13, 17, 25),
) -> Dict[str, Any]:
    """
    Per-condition histogram of Phase-1 lengths and pairwise KL.

    Returns:
        {
          condition_a: { bucket_string: prob, ... },
          condition_b: { ... },
          ...,
          "kl_pairwise": { "trained_vs_threshold_0.5": kl, ... }
        }
    """
    def _bucketize(lengths: List[int]) -> Dict[str, float]:
        counts = Counter()
        for L in lengths:
            for i in range(len(bucket_edges) - 1):
                if bucket_edges[i] <= L < bucket_edges[i + 1]:
                    counts[f"[{bucket_edges[i]},{bucket_edges[i + 1]})"] += 1
                    break
            else:
                counts[f"[{bucket_edges[-1]},inf)"] += 1
        total = sum(counts.values()) or 1
        # fill missing buckets with eps for KL stability
        out = {}
        labels = ([f"[{a},{b})" for a, b in zip(bucket_edges, bucket_edges[1:])]
                  + [f"[{bucket_edges[-1]},inf)"])
        for b in labels:
            p = counts.get(b, 0) / total
            out[b] = p + 1e-6
        # renormalise
        s = sum(out.values())
        return {k: v / s for k, v in out.items()}

    distributions = {
        name: _bucketize([_p1_length(ep) for ep in eps])
        for name, eps in episodes_by_condition.items()
    }
    kl_pairwise = {}
    names = list(distributions.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            P, Q = distributions[a], distributions[b]
            kl = sum(p * math.log(p / Q[k]) for k, p in P.items())
            kl_pairwise[f"{a}_vs_{b}"] = round(kl, 4)
    return {**distributions, "kl_pairwise": kl_pairwise}


# ──────────────────────────────────────────────────────────────────────
# 2. Action-ordering position
# ──────────────────────────────────────────────────────────────────────


def action_ordering_position(
    episodes_by_condition: Dict[str, List[EpisodeResult]],
    action_type:           str = "check_dependencies",
) -> Dict[str, Dict[str, float]]:
    """
    For each condition, the mean step index at which `action_type` first
    occurs (only across episodes where it occurs at all).
    Smaller is "earlier" = more proactive use of the action.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, eps in episodes_by_condition.items():
        positions = [
            _action_at_step(ep, action_type) for ep in eps
            if _action_at_step(ep, action_type) is not None
        ]
        if not positions:
            out[name] = {"frequency": 0.0,
                         "mean_position": float("inf"),
                         "median_position": float("inf"),
                         "n": len(eps)}
            continue
        out[name] = {
            "frequency":       round(len(positions) / len(eps), 3),
            "mean_position":   round(statistics.mean(positions), 2),
            "median_position": round(statistics.median(positions), 2),
            "n":               len(eps),
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# 3. Confidence calibration (ECE)
# ──────────────────────────────────────────────────────────────────────


def confidence_calibration(
    episodes:    List[EpisodeResult],
    n_buckets:   int = 10,
    confidence_field: str = "service_confidence",
    correctness_fn: Optional[Callable[[EpisodeResult], bool]] = None,
) -> Dict[str, Any]:
    """
    Reliability curve of declared confidence vs. actual correctness.
    Returns ECE (Expected Calibration Error) and per-bucket numbers.

    `confidence_field` reads from the recorded BeliefState snapshot
    (the one captured at transition).  `correctness_fn(ep)` -> bool
    defaults to "RCA component >= 0.6".
    """
    correctness_fn = correctness_fn or (
        lambda ep: float(ep.get("score_breakdown", {}).get("p1_rca", 0.0)) >= 0.6
    )

    buckets: List[List[Tuple[float, bool]]] = [[] for _ in range(n_buckets)]
    for ep in episodes:
        belief = _final_belief(ep)
        conf = float(belief.get(confidence_field, 0.0))
        correct = bool(correctness_fn(ep))
        idx = min(int(conf * n_buckets), n_buckets - 1)
        buckets[idx].append((conf, correct))

    rows = []
    total = sum(len(b) for b in buckets) or 1
    ece = 0.0
    for i, b in enumerate(buckets):
        if not b:
            rows.append({"bucket_low": i / n_buckets,
                         "bucket_high": (i + 1) / n_buckets,
                         "n": 0, "mean_conf": 0.0, "accuracy": 0.0})
            continue
        mean_conf = statistics.mean(c for c, _ in b)
        acc       = sum(1 for _, c in b if c) / len(b)
        ece      += (len(b) / total) * abs(mean_conf - acc)
        rows.append({
            "bucket_low":  round(i / n_buckets, 2),
            "bucket_high": round((i + 1) / n_buckets, 2),
            "n":           len(b),
            "mean_conf":   round(mean_conf, 3),
            "accuracy":    round(acc, 3),
        })
    return {"ece": round(ece, 4), "buckets": rows, "n": total}


# ──────────────────────────────────────────────────────────────────────
# 4. P2 search breadth correlation
# ──────────────────────────────────────────────────────────────────────


def p2_search_breadth_correlation(
    episodes:           List[EpisodeResult],
    confidence_field:   str = "fault_confidence",
) -> Dict[str, Any]:
    """
    Pearson correlation between Phase-1 fault_confidence and the number
    of distinct files inspected during Phase 2.

    Desired sign: negative — low confidence drives wider search.
    """
    pairs: List[Tuple[float, int]] = []
    for ep in episodes:
        belief = _final_belief(ep)
        if not belief:
            continue
        conf = float(belief.get(confidence_field, 0.0))
        breadth = _files_inspected_p2(ep)
        pairs.append((conf, breadth))

    if len(pairs) < 3:
        return {"pearson_r": 0.0, "n": len(pairs)}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return {"pearson_r": 0.0, "n": len(pairs)}
    r = num / (den_x * den_y)
    return {"pearson_r": round(r, 4), "n": len(pairs),
            "mean_breadth": round(my, 2), "mean_confidence": round(mx, 3)}


# ──────────────────────────────────────────────────────────────────────
# Convenience: compute all four for a labelled set of conditions
# ──────────────────────────────────────────────────────────────────────


def compute_all(
    episodes_by_condition: Dict[str, List[EpisodeResult]],
    interleave_action:     str = "check_dependencies",
) -> Dict[str, Any]:
    """One-shot computation of all four behavioral metrics."""
    flat = [e for eps in episodes_by_condition.values() for e in eps]
    return {
        "stopping_distribution": stopping_distribution(episodes_by_condition),
        f"action_position[{interleave_action}]":
            action_ordering_position(episodes_by_condition, interleave_action),
        "confidence_calibration": {
            name: confidence_calibration(eps)
            for name, eps in episodes_by_condition.items()
        },
        "p2_search_breadth_correlation": {
            name: p2_search_breadth_correlation(eps)
            for name, eps in episodes_by_condition.items()
        },
        "n_episodes_per_condition": {
            name: len(eps) for name, eps in episodes_by_condition.items()
        },
        "n_total": len(flat),
    }
