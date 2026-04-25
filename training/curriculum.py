"""
Stage-wise curriculum runner.

Drives the four-stage training schedule from the brief:

    Stage 1 (offline)     :  not orchestrated here — bootstrap SFT data
                              prep is independent.
    Stage 2 (Pool A)      :  Ops bootstrap.  Run rollouts in p1_only mode,
                              save trajectories, hand them to the trainer
                              loop (caller-supplied) for SFT or warm-start
                              GRPO.
    Stage 3 (Pool B)      :  Code bootstrap.  Pool-B baseline is also
                              measured here so r_cross is calibrated.
    Stage 4 (Pool C)      :  Joint training with r_cross *gated* by the
                              variance gate and *warmed up* over the first
                              `warmup_steps` updates.

The runner itself is policy/trainer-agnostic — it provides:

  - `iter_stage_episodes(stage, n)` : a generator of episodes for that
                                       stage, fully scored
  - `record_for_variance(...)`      : update the variance gate
  - `should_enter_stage4()`         : check the bootstrapping constraint
  - `r_cross_weight(step)`          : query the warmup schedule

The caller (a torch / JAX / vLLM trainer) loops over these and computes
gradients itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from ..pools import POOLS, sample_task
from ..server.incident_environment import IncidentEnvironment
from .behavioral_metrics import compute_all
from .pool_b_baseline import run_all as measure_pool_b
from .policies import (
    NullPhase1Policy,
    OracleP1ThenRandomP2Policy,
    Policy,
    RandomPhase2Policy,
)
from .rollouts import run_episode
from .trajectory_dataset import TrajectoryStore, save_episode
from .variance_gate import RCrossWarmup, VarianceGate


# ──────────────────────────────────────────────────────────────────────
# Stages
# ──────────────────────────────────────────────────────────────────────


STAGE_TO_POOL = {2: "A", 3: "B", 4: "C", 5: "D"}     # 5 = held-out eval


@dataclass
class CurriculumConfig:
    storage_root:    str   = "trajectories"
    n_per_stage:     int   = 32
    warmup_steps:    int   = 500
    variance_window: int   = 64
    variance_thresh: float = 0.15


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────


class CurriculumRunner:
    """
    Stateful driver for the four-stage curriculum.

    Parameters
    ----------
    p1_policy / p2_policy
        Concrete `Policy` callables that drive Phase 1 / Phase 2 actions.
        For Stages 2 + 3 one of them is replaced with a synthetic policy
        (NullPhase1Policy for Stage 3, the trained policy for Stage 2).
    """

    def __init__(
        self,
        p1_policy: Optional[Policy] = None,
        p2_policy: Optional[Policy] = None,
        cfg:       Optional[CurriculumConfig] = None,
    ):
        self.cfg       = cfg or CurriculumConfig()
        self.env       = IncidentEnvironment()
        self.store     = TrajectoryStore(self.cfg.storage_root)
        self.gate      = VarianceGate(
            window=self.cfg.variance_window,
            max_acceptable_stdev=self.cfg.variance_thresh,
        )
        self.warmup    = RCrossWarmup(warmup_steps=self.cfg.warmup_steps)
        self.p1_policy = p1_policy
        self.p2_policy = p2_policy
        self.stage4_step_counter = 0

    # ----- helpers -----

    def _policy_for_stage(self, stage: int) -> Policy:
        """Pick the right policy for the current stage."""
        if stage == 2:
            # Stage 2: train Phase 1.  P2 doesn't run (mode=p1_only).
            if self.p1_policy is None:
                raise RuntimeError("Stage 2 requires a p1_policy")
            return self.p1_policy
        if stage == 3:
            # Stage 3: train Phase 2 only.  Pool B auto-injects oracle belief
            # at reset, so the policy only sees Phase 2.
            if self.p2_policy is None:
                return RandomPhase2Policy(seed=1)
            return self.p2_policy
        # Stages 4 / 5: joint — combine p1_policy for P1, p2_policy for P2
        p1 = self.p1_policy or OracleP1ThenRandomP2Policy(seed=2)
        p2 = self.p2_policy or RandomPhase2Policy(seed=3)
        def joint(obs, phase, task_name):
            return (p1 if phase == 1 else p2)(obs, phase, task_name)
        if hasattr(p2, "reset"):
            joint.reset = p2.reset                  # type: ignore[attr-defined]
        return joint

    # ----- API -----

    def measure_pool_b_baselines(self, samples: int = 5) -> Dict[str, Any]:
        """
        Run the Pool-B null-context baseline pass (Stage-3 prerequisite).

        Updates `null_context_p2_score` for every Phase-2-enabled task
        in-process, so subsequent r_cross calls return calibrated values.
        """
        return measure_pool_b(n_samples=samples, policy_name="random")

    def iter_stage_episodes(
        self,
        stage: int,
        n:     Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield `n` scored episodes for the given stage, persisting each
        to the trajectory store.
        """
        n = n or self.cfg.n_per_stage
        pool_name = STAGE_TO_POOL[stage]
        pool = POOLS[pool_name]
        policy = self._policy_for_stage(stage)
        for i in range(n):
            task = sample_task(pool_name)
            ep = run_episode(
                self.env, policy,
                task_name=task,
                pool=pool_name,
                mode=pool.mode,
                seed=10_000 * stage + i,
                max_steps=40,
            )
            ep["stage"] = stage
            self.gate.record(task, float(ep.get("r_cross", 0.0)
                                        + ep.get("score_breakdown", {})
                                              .get("patch_quality", 0.0)
                                        + ep.get("score_breakdown", {})
                                              .get("no_change_detection", 0.0)))
            save_episode(self.store.root, ep)
            if stage == 4:
                self.stage4_step_counter += 1
            yield ep

    def should_enter_stage4(self, n_required_open_tasks: int = 4) -> bool:
        """Variance gate check: at least N tasks with stable r_code variance."""
        return len(self.gate.open_tasks()) >= n_required_open_tasks

    def r_cross_weight(self) -> float:
        return self.warmup.weight(self.stage4_step_counter)

    # ----- evaluation -----

    def evaluate_held_out(self, n_per_task: int = 8) -> Dict[str, Any]:
        """Run Pool D and compute behavioral metrics broken down by task."""
        pool = POOLS["D"]
        policy = self._policy_for_stage(5)
        episodes_by_task: Dict[str, List[Dict[str, Any]]] = {}
        for task in pool.task_names:
            episodes_by_task[task] = []
            for i in range(n_per_task):
                ep = run_episode(
                    self.env, policy,
                    task_name=task, pool="D", mode=pool.mode,
                    seed=99_000 + i, max_steps=40,
                )
                ep["stage"] = 5
                save_episode(self.store.root, ep)
                episodes_by_task[task].append(ep)
        return {
            "per_task": {
                t: {"mean_final": _mean(ep.get("score_breakdown", {}).get("final", 0.0)
                                         for ep in eps),
                    "mean_r_cross": _mean(ep.get("r_cross", 0.0) for ep in eps)}
                for t, eps in episodes_by_task.items()
            },
            "metrics": compute_all(
                episodes_by_condition={"heldout": [
                    e for eps in episodes_by_task.values() for e in eps
                ]},
            ),
        }


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / max(len(xs), 1), 4)
