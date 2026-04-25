"""
Episode rollout utility — drives a Policy through `IncidentEnvironment`
without going through the HTTP server.  Used by every other training
module (baseline runner, ablations, dataset builder).

Usage:
    from incident_env.training.rollouts import run_episode
    from incident_env.training.policies   import RandomPhase2Policy
    from incident_env.server.incident_environment import IncidentEnvironment

    env = IncidentEnvironment()
    result = run_episode(env, RandomPhase2Policy(), task_name="memory_leak",
                         pool="B", max_steps=30)
    print(result["score_breakdown"])
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ..models import StepRecord
from ..tasks import compute_r_cross
from ..server.incident_environment import IncidentEnvironment
from .policies import Policy


def _trajectory_to_dicts(traj: List[StepRecord]) -> List[Dict[str, Any]]:
    """Serialize a List[StepRecord] to plain dicts (for JSON dumps)."""
    out: List[Dict[str, Any]] = []
    for r in traj:
        d = asdict(r)
        # IncidentAction inside dataclass — already nested-asdict'd
        out.append(d)
    return out


def run_episode(
    env:        IncidentEnvironment,
    policy:     Policy,
    task_name:  Optional[str] = None,
    pool:       Optional[str] = None,
    mode:       Optional[str] = None,
    seed:       Optional[int] = None,
    max_steps:  int = 40,
) -> Dict[str, Any]:
    """
    Drive `policy` through one episode against `env`.

    Returns a dict with:
        task_name, pool, mode, steps_taken,
        p1_trajectory, p2_trajectory, declared_patch, declared_no_change,
        score_breakdown (the /score response), r_cross,
        per_step_rewards.
    """
    info_reset = env.reset(task_name=task_name, pool=pool, mode=mode, seed=seed)
    obs = info_reset["observation"]
    initial_phase = info_reset.get("info", {}).get("phase", obs.get("current_phase", 1))
    actual_task = info_reset.get("info", {}).get("task_name", task_name)
    actual_pool = info_reset.get("info", {}).get("pool", pool)
    actual_mode = info_reset.get("info", {}).get("mode", mode or "joint")

    # Optional reset hook on policy
    if hasattr(policy, "reset"):
        try:
            policy.reset()
        except TypeError:
            policy.reset(actual_task)

    rewards: List[float] = []
    for _ in range(max_steps):
        phase = obs.get("current_phase", initial_phase)
        action = policy(obs, phase, actual_task)
        step_out = env.step(action)
        obs = step_out["observation"]
        rewards.append(float(step_out.get("reward", 0.0)))
        if step_out.get("done"):
            break

    state = env.get_state()
    breakdown = env.score_unified()
    r_cross = 0.0
    try:
        r_cross = compute_r_cross(
            task_name          = actual_task,
            declared_patch     = state.get("declared_patch"),
            declared_no_change = bool(state.get("declared_no_change")),
            p2_trajectory      = env.get_p2_trajectory(),
        )
    except Exception:
        pass

    return {
        "task_name":          actual_task,
        "pool":               actual_pool,
        "mode":               actual_mode,
        "steps_taken":        state.get("step_count", 0),
        "p1_trajectory":      _trajectory_to_dicts(env.get_p1_trajectory()),
        "p2_trajectory":      _trajectory_to_dicts(env.get_p2_trajectory()),
        "declared_patch":     state.get("declared_patch"),
        "declared_no_change": bool(state.get("declared_no_change")),
        "score_breakdown":    breakdown,
        "r_cross":            float(r_cross),
        "per_step_rewards":   rewards,
        "phase_transition_at": state.get("phase_transition_at"),
    }
