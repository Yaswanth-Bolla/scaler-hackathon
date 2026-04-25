"""
Segment-level GRPO loss — framework-agnostic core.

We optimise Phase 1 and Phase 2 as TWO separate GRPO problems, joined
only by the cross-phase reward `r_cross` which is added to the Phase-1
return *with stop-gradient on the Phase-2 path*.

Why segment-level?  In typical GRPO, one trajectory of length L tokens
gets one scalar reward — but our episodes are 8-16k tokens (P1 + P2)
and the credit structure is fundamentally bimodal: a single bad P1
choice should bias every P1 token's update, but should not propagate
through the (already-trained) P2 policy.  Segment-level GRPO solves
this exactly: each segment is its own group, advantages are normalized
within-segment within-group, and `r_cross` is bolted onto the P1 group
return as an additive constant per trajectory (with no gradient flowing
back through P2 — the trainer enforces this by simply not letting P2
parameters appear in the P1 group's loss graph).

This module provides:

  - Segment        : dataclass describing one (phase, trajectory, return)
  - GRPOGroup      : a group of K segments collected for the same prompt
  - grpo_advantages: per-step advantages within a group
  - grpo_loss      : final scalar loss given log-probs from the model

The trainer wires this up like:

    for batch in dataloader:                         # batch = list[GRPOGroup]
        for group in batch:
            adv = grpo_advantages(group)
            logp = model.logp_per_token(group)       # framework-specific
            loss = grpo_loss(logp, adv,
                             ref_logp=ref.logp_per_token(group),
                             beta=0.04)
            loss.backward()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class Segment:
    """One (phase, trajectory) pair with a scalar return."""
    segment_id:       str
    phase:            int                   # 1 or 2
    trajectory:       list                  # list of step dicts (tokens implicit downstream)
    terminal_reward:  float
    # Cross-phase credit: r_cross is *added* to terminal_reward for phase-1
    # segments only; the trainer must arrange so no gradient flows from
    # this term through phase-2 model parameters.
    r_cross:          float = 0.0
    stop_gradient_through_p2: bool = True


@dataclass
class GRPOGroup:
    """K segments drawn for the same prompt (same task, same phase)."""
    prompt_id: str
    segments:  List[Segment] = field(default_factory=list)

    @property
    def returns(self) -> List[float]:
        return [s.terminal_reward + s.r_cross for s in self.segments]


# ──────────────────────────────────────────────────────────────────────
# Group-relative advantage normalization (the GRPO step)
# ──────────────────────────────────────────────────────────────────────


def grpo_advantages(group: GRPOGroup, eps: float = 1e-6) -> List[float]:
    """
    Standard within-group standardization:
        A_i = (R_i - mean(R)) / (std(R) + eps)
    """
    R = group.returns
    if not R:
        return []
    mu = sum(R) / len(R)
    var = sum((r - mu) ** 2 for r in R) / max(len(R) - 1, 1)
    sigma = math.sqrt(var) + eps
    return [(r - mu) / sigma for r in R]


# ──────────────────────────────────────────────────────────────────────
# Loss (numerical core; trainer wraps in tensor ops)
# ──────────────────────────────────────────────────────────────────────


def grpo_loss(
    logps:      Sequence[Sequence[float]],
    advantages: Sequence[float],
    ref_logps:  Optional[Sequence[Sequence[float]]] = None,
    beta:       float = 0.04,
    clip:       float = 0.2,
) -> float:
    """
    Scalar GRPO objective (negated for minimization).

    logps[i]      : per-token log-prob sequence for segment i under the
                    current policy
    ref_logps[i]  : same sequence under the reference policy (for KL)
    advantages[i] : segment-level advantage from `grpo_advantages`

    The clipped objective:
        L_pg(i,t) = -min(ratio_{i,t} * A_i,
                         clip(ratio_{i,t}, 1-c, 1+c) * A_i)
    plus a per-token KL penalty `beta * KL[ref || policy]`.

    This implementation runs on plain floats so unit tests can verify the
    numerics; the trainer reimplements the same arithmetic in tensors.
    """
    if not logps:
        return 0.0
    n_segments = len(logps)
    total = 0.0
    n_tokens = 0
    for i, seq in enumerate(logps):
        if not seq:
            continue
        adv = advantages[i] if i < len(advantages) else 0.0
        ref_seq = ref_logps[i] if (ref_logps is not None and i < len(ref_logps)) else seq
        for t, lp in enumerate(seq):
            ref_lp = ref_seq[t] if t < len(ref_seq) else lp
            ratio = math.exp(lp - ref_lp)
            unclipped = ratio * adv
            clipped   = max(min(ratio, 1 + clip), 1 - clip) * adv
            policy_term = -min(unclipped, clipped)
            kl_term = beta * (ref_lp - lp)        # forward KL approximation
            total += policy_term + kl_term
            n_tokens += 1
    return total / max(n_tokens, 1)


# ──────────────────────────────────────────────────────────────────────
# Cross-phase wiring
# ──────────────────────────────────────────────────────────────────────


def attach_r_cross(
    p1_segments: List[Segment],
    r_cross_per_episode: List[float],
    weight: float = 1.0,
) -> List[Segment]:
    """
    Add `r_cross` to each Phase-1 segment with the configured weight.
    `weight` is the curriculum-driven warmup factor (0 → 1 across the
    first ~500 Stage-4 steps).
    """
    if len(p1_segments) != len(r_cross_per_episode):
        raise ValueError(
            f"r_cross length mismatch: segs={len(p1_segments)} "
            f"r_cross={len(r_cross_per_episode)}")
    out: List[Segment] = []
    for s, rc in zip(p1_segments, r_cross_per_episode):
        s2 = Segment(
            segment_id      = s.segment_id,
            phase           = s.phase,
            trajectory      = s.trajectory,
            terminal_reward = s.terminal_reward,
            r_cross         = float(rc) * float(weight),
            stop_gradient_through_p2 = True,
        )
        out.append(s2)
    return out


def group_by_prompt(segments: List[Segment], group_size: int) -> List[GRPOGroup]:
    """
    Bucket segments into groups of `group_size`.  In production, the
    sampler arranges that all segments in a group came from the same
    prompt (same task seed) so within-group standardization is meaningful.
    """
    groups: List[GRPOGroup] = []
    bucket: List[Segment] = []
    for s in segments:
        bucket.append(s)
        if len(bucket) >= group_size:
            groups.append(GRPOGroup(prompt_id=bucket[0].segment_id, segments=bucket))
            bucket = []
    if bucket:
        groups.append(GRPOGroup(prompt_id=bucket[0].segment_id, segments=bucket))
    return groups
