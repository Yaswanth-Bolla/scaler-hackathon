"""
Trajectory dataset — load/save episode rollouts as JSONL on disk and
optionally project them into HuggingFace `Dataset` rows for downstream
RL/SFT pipelines.

Layout on disk:
    trajectories/
        <pool>/<task>/
            ep_<uuid>.json     ← one episode each (output of run_episode)

A `TrajectoryStore` indexes a directory tree and can yield episodes
filtered by pool / task / quality (final score above a threshold).

A `to_segment_dataset` adapter converts each episode into TWO rows
(P1 segment, P2 segment) suitable for the segment-level GRPO loss.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional


# ──────────────────────────────────────────────────────────────────────
# Disk layout
# ──────────────────────────────────────────────────────────────────────


def _episode_path(root: Path, pool: str, task: str, ep_id: Optional[str] = None) -> Path:
    ep_id = ep_id or uuid.uuid4().hex[:12]
    p = root / pool / task
    p.mkdir(parents=True, exist_ok=True)
    return p / f"ep_{ep_id}.json"


def save_episode(
    root:    Path,
    episode: Dict[str, Any],
) -> Path:
    """Persist one rollout dict.  Returns the file path written."""
    pool = episode.get("pool") or "_unspecified"
    task = episode.get("task_name") or "_unspecified"
    path = _episode_path(root, pool, task)
    path.write_text(json.dumps(episode, indent=2, default=_default_serializer))
    return path


def _default_serializer(o: Any) -> Any:
    """Fallback for non-JSON-native objects (Enum, Path, etc.)."""
    try:
        return asdict(o)
    except Exception:
        pass
    if hasattr(o, "value"):  # Enum
        return o.value
    return str(o)


# ──────────────────────────────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────────────────────────────


class TrajectoryStore:
    """
    Read-side index over a trajectories directory.  Pure stdlib — does
    not load anything until you call `iter_episodes(...)`.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

    def list_pools(self) -> List[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def list_tasks(self, pool: Optional[str] = None) -> List[str]:
        out = set()
        for pool_dir in (self.root.iterdir() if not pool else [self.root / pool]):
            if not pool_dir.exists() or not pool_dir.is_dir():
                continue
            for task_dir in pool_dir.iterdir():
                if task_dir.is_dir():
                    out.add(task_dir.name)
        return sorted(out)

    def iter_episodes(
        self,
        pool:           Optional[str]   = None,
        task:           Optional[str]   = None,
        min_final:      Optional[float] = None,
        max_final:      Optional[float] = None,
        predicate:      Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield episode dicts matching the filter."""
        for pool_dir in self.root.iterdir():
            if not pool_dir.is_dir() or (pool and pool_dir.name != pool):
                continue
            for task_dir in pool_dir.iterdir():
                if not task_dir.is_dir() or (task and task_dir.name != task):
                    continue
                for fp in sorted(task_dir.glob("ep_*.json")):
                    try:
                        ep = json.loads(fp.read_text())
                    except json.JSONDecodeError:
                        continue
                    final = float(ep.get("score_breakdown", {}).get("final", 0.0))
                    if min_final is not None and final < min_final:
                        continue
                    if max_final is not None and final > max_final:
                        continue
                    if predicate is not None and not predicate(ep):
                        continue
                    yield ep

    def count(self, **filters) -> int:
        return sum(1 for _ in self.iter_episodes(**filters))


# ──────────────────────────────────────────────────────────────────────
# Segment projection (for segment-level GRPO)
# ──────────────────────────────────────────────────────────────────────


def to_segments(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Decompose a unified episode into one or two `Segment` rows.

    For Pool C (joint) this returns:
        [
          { phase: 1, trajectory: [...], terminal_reward: r_p1 + r_cross_stopgrad },
          { phase: 2, trajectory: [...], terminal_reward: r_p2_components },
        ]

    For Pool A (p1_only) returns just the Phase-1 segment.
    For Pool B (p2_only) returns just the Phase-2 segment.

    The `terminal_reward` field is the per-segment GRPO advantage
    *target*; the segment_grpo loss normalizes within a group at training
    time.  `r_cross` is added to the Phase-1 reward but flagged as
    `stop_gradient_through_p2 = True` so the trainer doesn't backprop
    Phase-2 quality through Phase-2 parameters.
    """
    pool = episode.get("pool") or ""
    mode = episode.get("mode") or "joint"
    breakdown = episode.get("score_breakdown") or {}
    r_cross = float(episode.get("r_cross", 0.0))

    out: List[Dict[str, Any]] = []

    if mode != "p2_only":
        p1_terminal = (
            float(breakdown.get("p1_rca", 0.0)) * 0.6
            + float(breakdown.get("p1_efficiency", 0.0)) * 0.4
            + r_cross  # cross-phase information value, stop-gradient w.r.t. P2 params
        )
        out.append({
            "segment_id":      f"{episode.get('task_name')}::p1",
            "phase":           1,
            "task_name":       episode.get("task_name"),
            "pool":            pool,
            "trajectory":      episode.get("p1_trajectory", []),
            "terminal_reward": round(p1_terminal, 4),
            "components":      {
                "p1_rca":         breakdown.get("p1_rca"),
                "p1_efficiency":  breakdown.get("p1_efficiency"),
                "r_cross":        r_cross,
            },
            "stop_gradient_through_p2": True,
        })

    if mode != "p1_only":
        p2_terminal = (
            float(breakdown.get("patch_quality", 0.0)) * 0.7
            + float(breakdown.get("no_change_detection", 0.0)) * 0.7   # mutually exclusive
            + float(breakdown.get("p2_efficiency", 0.0)) * 0.3
        )
        out.append({
            "segment_id":      f"{episode.get('task_name')}::p2",
            "phase":           2,
            "task_name":       episode.get("task_name"),
            "pool":            pool,
            "trajectory":      episode.get("p2_trajectory", []),
            "terminal_reward": round(p2_terminal, 4),
            "components":      {
                "patch_quality":       breakdown.get("patch_quality"),
                "no_change_detection": breakdown.get("no_change_detection"),
                "p2_efficiency":       breakdown.get("p2_efficiency"),
            },
            "stop_gradient_through_p2": False,
        })
    return out


def to_segment_dataset(store: TrajectoryStore, **filters) -> List[Dict[str, Any]]:
    """Eagerly materialize all segments matching filters."""
    rows: List[Dict[str, Any]] = []
    for ep in store.iter_episodes(**filters):
        rows.extend(to_segments(ep))
    return rows


# ──────────────────────────────────────────────────────────────────────
# Optional HuggingFace adapter
# ──────────────────────────────────────────────────────────────────────


def to_hf_dataset(rows: List[Dict[str, Any]]):
    """
    Convert a list of segment dicts to a HuggingFace `Dataset` if `datasets`
    is installed. Trainers can then call `.shuffle().select(range(N))`.
    """
    try:
        from datasets import Dataset       # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "`datasets` not installed. `pip install datasets`") from e
    return Dataset.from_list(rows)
