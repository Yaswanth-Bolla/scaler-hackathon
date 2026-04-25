"""
Training infrastructure for the two-phase RL system.

Modules:
    policies                — pluggable policy interface (random / LLM API)
    pool_b_baseline         — null-context P2 baseline runner (Stage-3 prereq)
    behavioral_metrics      — stopping/ordering/calibration/breadth metrics
    trajectory_dataset      — load/save trajectories, HF-Dataset adapter
    belief_aux_loss         — calibration regression + consistency loss
    segment_grpo            — framework-agnostic segment-level GRPO loss
    curriculum              — Stage-2 → Stage-3 → Stage-4 driver
    variance_gate           — Pool-B variance check + r_cross weight warmup
    ablations               — four ablation runners (claims 1-4)
    report                  — paper tables + behavioral plots
"""
