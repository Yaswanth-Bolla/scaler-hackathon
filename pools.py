"""
Pool registry for the four-stage curriculum.

A Pool is a named subset of TASK_REGISTRY plus an *episode mode* that controls
how the environment behaves for that pool.  The pool name is passed to
`/reset` via the `pool` field; the server then samples a task from that pool
and switches the environment into the matching mode.

Stages and pools (per the brief):

    Stage 2  bootstrap ops agent           → Pool A   mode = "p1_only"
    Stage 3  bootstrap code agent          → Pool B   mode = "p2_only"
                                                       (P1 context = ground truth)
    Stage 4  joint training with r_cross   → Pool C   mode = "joint"
    Final    held-out generalization eval  → Pool D   mode = "joint"

Pool A and Pool C reuse the same scenarios — only the mode differs.
Pool B is a *bootstrapping* mode where the orchestrator's belief is *synthesized*
from the scenario's ground truth, so the code agent never trains on garbage
Phase-1 context.  This implements the brief's "P2-only with ground-truth
P1 context injected" semantics exactly.

Pool D consists of *held-out* scenarios that never appear during training,
used to measure whether the learned stopping criterion generalizes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .models import BeliefState
from .scenarios.base import BaseScenario


# ──────────────────────────────────────────────────────────────────────
# Pool definitions
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pool:
    """A named pool: training scenarios + episode mode."""
    name:          str                # "A" | "B" | "C" | "D"
    description:   str
    task_names:    List[str]
    mode:          str                # "p1_only" | "p2_only" | "joint"
    # Stage-3 hints used when mode == "p2_only" (Pool B): if True, the env
    # auto-injects ground-truth context at handoff so the code agent never
    # sees a noisy P1 trajectory.
    inject_oracle_belief: bool = False


# Training scenarios (seen during all four training stages).  Phase-A
# scenarios + the brief's four research scenarios.
_TRAIN_TASKS = [
    "memory_leak",
    "cascading_failure",
    "distributed_deadlock",
    "circuit_breaker_noop",
    "aliased_fault",
    "severity_inversion",
    "confidence_inversion",
    "info_ordering",
]


# Held-out scenarios — same fault families, but combined in ways the agent
# never trained on.  Defined in scenarios/heldout.py and registered lazily.
_HELDOUT_TASKS = [
    "heldout_aliased_severity",      # aliased + severity-inversion combo
    "heldout_confidence_ordering",   # confidence-inversion + info-ordering combo
]


POOLS: Dict[str, Pool] = {
    "A": Pool(
        name        = "A",
        description = "Stage-2 ops bootstrap — P1 only, declare_root_cause terminates.",
        task_names  = _TRAIN_TASKS,
        mode        = "p1_only",
    ),
    "B": Pool(
        name        = "B",
        description = "Stage-3 code bootstrap — P2 only with oracle P1 context injected.",
        task_names  = _TRAIN_TASKS,
        mode        = "p2_only",
        inject_oracle_belief = True,
    ),
    "C": Pool(
        name        = "C",
        description = "Stage-4 joint training — full P1 → P2 with r_cross.",
        task_names  = _TRAIN_TASKS,
        mode        = "joint",
    ),
    "D": Pool(
        name        = "D",
        description = "Held-out generalization — never seen during training.",
        task_names  = _HELDOUT_TASKS,
        mode        = "joint",
    ),
}


def get_pool(name: str) -> Pool:
    name = (name or "").upper()
    if name not in POOLS:
        raise ValueError(f"Unknown pool {name!r}. Available: {list(POOLS)}")
    return POOLS[name]


def sample_task(pool_name: str, rng: Optional[random.Random] = None) -> str:
    """Sample one task from a pool."""
    rng = rng or random
    return rng.choice(get_pool(pool_name).task_names)


# ──────────────────────────────────────────────────────────────────────
# Oracle belief synthesis (for Pool B)
# ──────────────────────────────────────────────────────────────────────


def oracle_belief(scenario: BaseScenario) -> BeliefState:
    """
    Synthesize a *ground-truth* belief from the scenario's static config.

    Used by Pool B (Stage 3) so the code agent sees a perfect Phase-1
    handoff — its training signal is then purely Phase-2 quality, not
    Phase-1 errors.  This is the cleanest possible code-agent bootstrap.
    """
    return BeliefState(
        suspected_service     = scenario.root_cause_service,
        suspected_fault_class = scenario.fault_class,
        service_confidence    = 1.0,
        fault_confidence      = 1.0,
        evidence_gaps         = [],
        estimated_p2_cost     = "low",
        decision              = "transition",
        reasoning             = "[oracle] ground-truth belief synthesized for Pool B bootstrap",
    )
