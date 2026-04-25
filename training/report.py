"""
Report generator — turns `ablations.json` into the four paper tables
and four behavioral metric plots.

Tables (markdown — copy/pasteable into a paper or report):

  Table 1: Orchestrator vs fixed thresholds (Claim 1)
  Table 2: r_cross on/off (Claim 2)
  Table 3: Stage 2+3 only vs full Stage 4 (Claim 3)
  Table 4: Pool-D held-out generalization (Claim 4)

Plots (matplotlib, optional — module is usable without matplotlib too):

  Plot 1: Stopping-distribution histogram per condition (Claim 1)
  Plot 2: P2 steps to correct patch — bar plot (Claim 2)
  Plot 3: Cumulative running mean (convergence proxy) (Claim 3)
  Plot 4: Confidence calibration curve, trained vs PE baseline (Claim 4)

CLI:
    python -m incident_env.training.report \
        --input ablation_results.json --out report/

Outputs:
    report/tables.md
    report/stopping_distribution.png
    report/p2_steps_to_correct.png
    report/convergence.png
    report/calibration.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────


def _md_row(cells: List[Any]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def table_claim1(claim1: Dict[str, Any]) -> str:
    """Aggregate scores + KL of stopping distribution + interleave position."""
    agg = claim1["aggregate"]
    behav = claim1.get("behavioral", {})
    kl_pairs = behav.get("stopping_distribution", {}).get("kl_pairwise", {})
    pos = behav.get("action_position[check_dependencies]", {})
    rows = [
        "## Table 1 — Orchestrator vs fixed thresholds (Claim 1)",
        "",
        _md_row(["condition", "n", "mean_final", "mean_p1_steps", "mean_p2_steps",
                 "check_deps_position (median)"]),
        _md_row(["---"] * 6),
    ]
    for name, a in agg.items():
        rows.append(_md_row([
            name, a["n"], a["mean_final"], a["mean_p1_steps"], a["mean_p2_steps"],
            pos.get(name, {}).get("median_position", "—"),
        ]))
    if kl_pairs:
        rows.append("")
        rows.append("**Pairwise KL between stopping-length distributions:**")
        rows.append("")
        for k, v in kl_pairs.items():
            rows.append(f"- `{k}` → {v}")
    return "\n".join(rows)


def table_claim2(claim2: Dict[str, Any]) -> str:
    agg = claim2["aggregate"]
    rows = [
        "## Table 2 — r_cross ablation (Claim 2)",
        "",
        _md_row(["condition", "n", "mean_final", "mean_r_cross",
                 "mean_p2_steps", "p2_steps_to_correct_patch"]),
        _md_row(["---"] * 6),
    ]
    for name, a in agg.items():
        rows.append(_md_row([
            name, a["n"], a["mean_final"], a["mean_r_cross"],
            a["mean_p2_steps"], a.get("p2_steps_to_correct_patch", "—"),
        ]))
    return "\n".join(rows)


def table_claim3(claim3: Dict[str, Any]) -> str:
    agg = claim3["aggregate"]
    curves = claim3.get("convergence_curve", {})
    rows = [
        "## Table 3 — Stage 2+3 only vs Full Stage 4 (Claim 3)",
        "",
        _md_row(["condition", "n", "mean_final", "stdev_final", "mean_p1_steps"]),
        _md_row(["---"] * 5),
    ]
    for name, a in agg.items():
        rows.append(_md_row([
            name, a["n"], a["mean_final"], a["stdev_final"], a["mean_p1_steps"],
        ]))
    if curves:
        rows.append("")
        rows.append("**Cumulative running-mean curves (early-vs-late convergence proxy):**")
        rows.append("")
        for name, vals in curves.items():
            rows.append(f"- `{name}` → {vals}")
    return "\n".join(rows)


def table_claim4(claim4: Dict[str, Any]) -> str:
    agg = claim4["aggregate"]
    behav = claim4.get("behavioral", {})
    cal = behav.get("confidence_calibration", {})
    rows = [
        "## Table 4 — Pool-D held-out generalization (Claim 4)",
        "",
        _md_row(["condition", "n", "mean_final", "stdev_final", "ECE"]),
        _md_row(["---"] * 5),
    ]
    for name, a in agg.items():
        ece = cal.get(name, {}).get("ece", "—")
        rows.append(_md_row([name, a["n"], a["mean_final"], a["stdev_final"], ece]))
    return "\n".join(rows)


def render_tables(report: Dict[str, Any]) -> str:
    parts = []
    if "claim1" in report:
        parts.append(table_claim1(report["claim1"]))
    if "claim2" in report:
        parts.append(table_claim2(report["claim2"]))
    if "claim3" in report:
        parts.append(table_claim3(report["claim3"]))
    if "claim4" in report:
        parts.append(table_claim4(report["claim4"]))
    return "\n\n".join(parts) + "\n"


# ──────────────────────────────────────────────────────────────────────
# Plots (optional — matplotlib import gated)
# ──────────────────────────────────────────────────────────────────────


def _try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt   # noqa
        return plt
    except ImportError:
        return None


def plot_stopping_distribution(claim1: Dict[str, Any], out: Path) -> Optional[Path]:
    plt = _try_matplotlib()
    if plt is None:
        return None
    sd = claim1.get("behavioral", {}).get("stopping_distribution", {})
    cond_dists = {k: v for k, v in sd.items() if k != "kl_pairwise"}
    if not cond_dists:
        return None
    buckets = list(next(iter(cond_dists.values())).keys())
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(len(cond_dists), 1)
    for i, (name, dist) in enumerate(cond_dists.items()):
        ys = [dist.get(b, 0.0) for b in buckets]
        xs = [j + i * width for j in range(len(buckets))]
        ax.bar(xs, ys, width=width, label=name)
    ax.set_xticks([j + 0.4 for j in range(len(buckets))])
    ax.set_xticklabels(buckets, rotation=30, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Phase-1 length distribution per condition (Claim 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_p2_steps_bar(claim2: Dict[str, Any], out: Path) -> Optional[Path]:
    plt = _try_matplotlib()
    if plt is None:
        return None
    agg = claim2["aggregate"]
    names = list(agg.keys())
    vals = [agg[n].get("p2_steps_to_correct_patch", 0.0) for n in names]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(names, vals, color=["#3a7", "#a73"])
    ax.set_ylabel("Mean P2 steps to correct patch")
    ax.set_title("Claim 2 — r_cross reduces P2 effort")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.1f}", ha="center")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_convergence(claim3: Dict[str, Any], out: Path) -> Optional[Path]:
    plt = _try_matplotlib()
    if plt is None:
        return None
    curves = claim3.get("convergence_curve", {})
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ys in curves.items():
        ax.plot(range(len(ys)), ys, marker="o", label=name)
    ax.set_xlabel("rollout block (4 episodes each)")
    ax.set_ylabel("running mean(final score)")
    ax.set_title("Claim 3 — convergence curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_calibration(claim4: Dict[str, Any], out: Path) -> Optional[Path]:
    plt = _try_matplotlib()
    if plt is None:
        return None
    cal = claim4.get("behavioral", {}).get("confidence_calibration", {})
    if not cal:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, c in cal.items():
        xs = [b["mean_conf"] for b in c.get("buckets", []) if b.get("n")]
        ys = [b["accuracy"]  for b in c.get("buckets", []) if b.get("n")]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=f"{name} (ECE={c.get('ece', 0):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="ideal")
    ax.set_xlabel("Declared confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Claim 4 — calibration on held-out (Pool D)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────────────
# Top-level
# ──────────────────────────────────────────────────────────────────────


def render(report: Dict[str, Any], outdir: Path) -> Dict[str, Any]:
    """Render tables + (best-effort) plots into outdir.  Returns manifest."""
    outdir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"outdir": str(outdir), "files": {}}

    # Tables
    md = render_tables(report)
    table_path = outdir / "tables.md"
    table_path.write_text(md)
    manifest["files"]["tables"] = str(table_path)

    # Plots
    plot_jobs = [
        ("stopping_distribution.png", "claim1", plot_stopping_distribution),
        ("p2_steps_to_correct.png",   "claim2", plot_p2_steps_bar),
        ("convergence.png",           "claim3", plot_convergence),
        ("calibration.png",           "claim4", plot_calibration),
    ]
    for fname, claim_key, fn in plot_jobs:
        if claim_key not in report:
            continue
        out = outdir / fname
        result = fn(report[claim_key], out)
        if result:
            manifest["files"][fname] = str(out)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("ablation_results.json"))
    parser.add_argument("--out",   type=Path, default=Path("report"))
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    report = json.loads(args.input.read_text())
    manifest = render(report, args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
