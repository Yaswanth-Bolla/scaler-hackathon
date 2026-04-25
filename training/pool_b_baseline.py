"""
Pool-B null-context baseline runner.

For each Phase-2-enabled task, run the code agent with an *empty*
BeliefState injected via NullPhase1Policy.  Average the patch/no-change
score across `n_samples` episodes and write the result back into
`code_context_builder.update_null_baseline(...)`.

The resulting `null_context_p2_score` is the denominator of the
counterfactual cross-phase reward:

    r_cross = max(0,  r_code(τ_2 | context(τ_1))
                       - r_code(τ_2 | context(∅)))

This is the *only* operation that needs to run before Stage 4 — without
a calibrated null baseline, `r_cross` is meaningless and Stage 4 cannot
start.

CLI:
    python -m incident_env.training.pool_b_baseline \
        --policy random --samples 8 --output baselines.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from ..pools import POOLS
from ..scenarios.code_context_builder import CODE_CONTEXTS, update_null_baseline
from ..scenarios.grader_p2 import grade_no_change, grade_patch_quality
from ..server.incident_environment import IncidentEnvironment
from ..tasks import get_scenario
from .policies import NullPhase1Policy, RandomPhase2Policy
from .rollouts import run_episode


def _make_policy(name: str):
    """Wrap a policy with the NullPhase1 prefix so P1 always emits empty handoff."""
    if name == "random":
        p2 = RandomPhase2Policy(seed=42)
        # Combined policy: P1=null, P2=random
        def policy(obs, phase, task_name):
            if phase == 1:
                return NullPhase1Policy()(obs, phase, task_name)
            return p2(obs, phase, task_name)
        # Forward .reset() so per-episode state is cleared
        policy.reset = p2.reset       # type: ignore[attr-defined]
        return policy
    if name == "openai":
        from .policies import OpenAIChatPolicy
        chat = OpenAIChatPolicy()
        def policy(obs, phase, task_name):
            if phase == 1:
                return NullPhase1Policy()(obs, phase, task_name)
            return chat(obs, phase, task_name)
        policy.reset = lambda: chat.reset(task_name="")  # type: ignore
        return policy
    raise ValueError(f"Unknown policy {name}")


def measure_null_baseline(
    task_name: str,
    n_samples: int = 5,
    policy_name: str = "random",
    seed_start: int = 0,
) -> Dict[str, float]:
    """
    Run `n_samples` episodes of `task_name` in Pool B mode with an empty
    Phase-1 belief and return aggregate stats.
    """
    env = IncidentEnvironment()
    policy = _make_policy(policy_name)
    scenario = get_scenario(task_name)
    ctx = scenario.code_context
    if ctx is None:
        return {"task": task_name, "skipped": True, "reason": "no code_context"}

    scores: List[float] = []
    for i in range(n_samples):
        result = run_episode(
            env, policy,
            task_name = task_name,
            mode      = "p2_only",
            seed      = seed_start + i,
            max_steps = 30,
        )
        # The score we care about is the *raw* P2 component
        if ctx.is_valid_issue:
            s = grade_patch_quality(result["declared_patch"] or "", ctx)
        else:
            s = grade_no_change(bool(result["declared_no_change"]), ctx)
        scores.append(float(s))

    return {
        "task":         task_name,
        "samples":      n_samples,
        "mean":         round(statistics.mean(scores), 4),
        "stdev":        round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "min":          round(min(scores), 4),
        "max":          round(max(scores), 4),
    }


def run_all(
    n_samples: int = 5,
    policy_name: str = "random",
    output_path: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Measure baselines for every task that has a code_context."""
    results: Dict[str, Dict[str, float]] = {}
    for task in CODE_CONTEXTS.keys():
        try:
            stats = measure_null_baseline(task, n_samples, policy_name)
        except Exception as e:
            stats = {"task": task, "error": str(e)}
        results[task] = stats
        # Update the in-process baseline so subsequent r_cross calls use it
        if "mean" in stats:
            update_null_baseline(task, stats["mean"])
        print(f"  {task:38s} → mean={stats.get('mean', '—'):>6}  "
              f"stdev={stats.get('stdev', '—'):>6}")

    if output_path is not None:
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nWrote baselines to {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5,
                        help="episodes per task (default 5)")
    parser.add_argument("--policy", choices=["random", "openai"], default="random",
                        help="policy to drive Phase 2 (default: random)")
    parser.add_argument("--output", type=Path, default=Path("pool_b_baselines.json"),
                        help="output JSON file (default: pool_b_baselines.json)")
    args = parser.parse_args()

    print(f"Running Pool-B null-context baseline:")
    print(f"  policy   = {args.policy}")
    print(f"  samples  = {args.samples}")
    print(f"  tasks    = {list(CODE_CONTEXTS)}")
    print()
    run_all(n_samples=args.samples, policy_name=args.policy, output_path=args.output)


if __name__ == "__main__":
    main()
