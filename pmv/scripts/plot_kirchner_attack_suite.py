#!/usr/bin/env python3
"""
Generate Kirchner-style attack plots from per-objective eval JSON files.

Expected inputs:
  - eval_*_kirchner_src_8k.json
  - eval_*_kirchner_cgc_8k.json
  - eval_*_kirchner_goodhart_8k.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


OBJECTIVE_ORDER = ["src", "cgc", "goodhart"]
OBJECTIVE_COLOR = {
    "src": "#1f77b4",
    "cgc": "#2ca02c",
    "goodhart": "#ff7f0e",
}


def _load_attack(json_path: Path, objective: str) -> Dict:
    data = json.loads(json_path.read_text())
    suite = (data.get("kirchner_attack_suite") or {})
    attacks = (suite.get("attacks") or {})
    atk = attacks.get(objective)
    if atk is None:
        # Fallback for older single-objective output key.
        atk = data.get("kirchner_attack_training") or {}
    return {
        "json_path": str(json_path),
        "checkpoint": data.get("checkpoint_used"),
        "run_metadata": data.get("run_metadata") or {},
        "config_snapshot": data.get("config_snapshot") or {},
        "probe_rates": (data.get("probe_metrics") or {}).get("rates", {}),
        "oversight_quality": data.get("oversight_quality") or {},
        "goal_summary": ((data.get("goal_robustness") or {}).get("summary") or {}),
        "attack": atk or {},
    }


def _common_value(values: List[object], fallback: str = "mixed") -> object:
    vals = [v for v in values if v is not None]
    if not vals:
        return fallback
    head = vals[0]
    if all(v == head for v in vals):
        return head
    return fallback


def _short_path(p: Optional[str], max_len: int = 90) -> str:
    if not p:
        return "N/A"
    p = str(p)
    if len(p) <= max_len:
        return p
    return "..." + p[-(max_len - 3):]


def _build_plot_context(loaded: Dict[str, Dict]) -> Dict[str, str]:
    entries = [loaded[o] for o in OBJECTIVE_ORDER if o in loaded]
    cfgs = [e.get("config_snapshot") or {} for e in entries]
    ckpts = [e.get("checkpoint") for e in entries]
    jsons = [e.get("json_path") for e in entries]

    prover_model = _common_value([c.get("prover_model") for c in cfgs])
    verifier_model = _common_value([c.get("verifier_model") for c in cfgs])
    num_verifiers = _common_value([c.get("num_verifiers") for c in cfgs])
    oversight_rule = _common_value([c.get("oversight_rule") for c in cfgs])
    checkpoint = _common_value(ckpts)
    dataset_mode = _common_value([c.get("dataset_mode") for c in cfgs], fallback="unknown")

    line_model = (
        f"Dataset={dataset_mode} | Oversight={oversight_rule} | "
        f"Num verifiers={num_verifiers}"
    )
    line_agents = (
        f"Prover={prover_model} | Verifier={verifier_model}"
    )
    line_ckpt = f"Checkpoint={_short_path(checkpoint)}"
    line_sources = (
        "Sources="
        + ", ".join(os.path.basename(j or "") for j in jsons if j)
    )
    return {
        "line_model": line_model,
        "line_agents": line_agents,
        "line_ckpt": line_ckpt,
        "line_sources": line_sources,
        "prover_model": str(prover_model),
        "verifier_model": str(verifier_model),
        "num_verifiers": str(num_verifiers),
        "oversight_rule": str(oversight_rule),
        "dataset_mode": str(dataset_mode),
        "checkpoint": str(checkpoint),
    }


def _add_context_footer(fig, context: Dict[str, str]):
    fig.text(0.01, 0.01, context["line_model"], ha="left", va="bottom", fontsize=8, color="#333333")
    fig.text(0.01, 0.028, context["line_agents"], ha="left", va="bottom", fontsize=8, color="#333333")
    fig.text(0.01, 0.046, context["line_ckpt"], ha="left", va="bottom", fontsize=8, color="#333333")


def _extract_history(atk: Dict) -> Tuple[List[float], List[float], List[float], List[Optional[float]], Optional[float]]:
    history = atk.get("history") or []
    x = [float(h.get("attack_updates_done", 0.0)) for h in history]
    incorrect_rate = [float(h.get("incorrect_rate", 0.0)) for h in history]
    accuracy = [float(h.get("accuracy", 1.0)) for h in history]
    avg_incorrect_score = [
        (None if h.get("avg_sneaky_incorrect_score") is None else float(h.get("avg_sneaky_incorrect_score")))
        for h in history
    ]
    helpful_ref = None
    ref = atk.get("reference_helpful_correct") or {}
    if ref.get("avg_helpful_correct_score") is not None:
        helpful_ref = float(ref.get("avg_helpful_correct_score"))
    return x, incorrect_rate, accuracy, avg_incorrect_score, helpful_ref


def _moving_average(values: List[float], window: int = 3) -> List[float]:
    if window <= 1 or not values:
        return list(values)
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(float(sum(chunk) / len(chunk)))
    return out


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = float(successes) / float(n)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _extract_binomial_ci_series(
    atk: Dict,
    metric: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Return x, p, lo, hi for binomial metrics in attack history.
    metric: "incorrect_rate" or "accuracy"
    """
    history = atk.get("history") or []
    xs, ps, lo, hi = [], [], [], []
    for h in history:
        n = int(h.get("episodes") or 0)
        k_bad = int(h.get("incorrect_count") or 0)
        if n <= 0:
            continue
        if metric == "accuracy":
            k = max(0, n - k_bad)
        else:
            k = k_bad
        p = float(k) / float(n)
        l, u = _wilson_interval(k, n)
        xs.append(float(h.get("attack_updates_done", 0.0)))
        ps.append(p)
        lo.append(l)
        hi.append(u)
    return xs, ps, lo, hi


def _plot_steps_until_exploit(loaded: Dict[str, Dict], out: Path, context: Dict[str, str]):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    labels, vals, colors, success_flags, budgets = [], [], [], [], []
    for obj in OBJECTIVE_ORDER:
        atk = loaded[obj]["attack"]
        labels.append(obj.upper())
        budget = atk.get("updates_budget")
        uts = atk.get("updates_to_success")
        success = bool(atk.get("success"))
        success_flags.append(success)
        if success and uts is not None:
            vals.append(float(uts))
        else:
            vals.append(float(budget if budget is not None else 0.0))
        colors.append(OBJECTIVE_COLOR[obj])
        budgets.append(float(budget if budget is not None else 0.0))

    bars = ax.bar(labels, vals, color=colors, alpha=0.9)
    for i, b in enumerate(bars):
        if not success_flags[i]:
            b.set_hatch("//")
            b.set_alpha(0.5)
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + max(80.0, budgets[i] * 0.01),
                f">={int(budgets[i])}\n(no exploit)",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + max(80.0, budgets[i] * 0.01),
                f"{int(vals[i])}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    max_budget = max(vals) if vals else 8000
    ax.set_ylim(0, max_budget * 1.10 if max_budget > 0 else 1.0)
    ax.set_ylabel("RL Updates Until Exploit (capped at budget)")
    ax.set_title("Figure 6a-style: Dedicated Attack Difficulty by Objective")
    ax.text(
        0.99,
        0.97,
        "Hatched bar = no exploit found within update budget",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#333333",
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend(
        handles=[
            Patch(facecolor="#bbbbbb", hatch="//", label="No exploit within budget"),
            Patch(facecolor="#666666", label="Exploit found"),
        ],
        fontsize=8,
        loc="upper left",
    )
    ax.text(
        0.01,
        -0.22,
        context["line_model"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#333333",
    )
    ax.text(
        0.01,
        -0.30,
        context["line_agents"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#333333",
    )
    ax.text(
        0.01,
        -0.38,
        context["line_ckpt"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#333333",
    )
    fig.subplots_adjust(bottom=0.34)
    plt.savefig(out, dpi=180)
    plt.close(fig)


def _plot_attack_dynamics(loaded: Dict[str, Dict], out: Path, context: Dict[str, str]):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    ax_ir = axes[0][0]
    ax_acc = axes[0][1]
    ax_si = axes[1][0]
    ax_gap = axes[1][1]

    for obj in OBJECTIVE_ORDER:
        atk = loaded[obj]["attack"]
        x, ir, acc, si, ref = _extract_history(atk)
        color = OBJECTIVE_COLOR[obj]
        label = obj.upper()

        if x and ir:
            ax_ir.plot(x, ir, marker="o", ms=2, color=color, alpha=0.25, linewidth=1.0)
            ax_ir.plot(x, _moving_average(ir, window=3), color=color, linewidth=2.0, label=f"{label} (smoothed)")
            x_ci, _, lo_ci, hi_ci = _extract_binomial_ci_series(atk, metric="incorrect_rate")
            if x_ci:
                ax_ir.fill_between(x_ci, lo_ci, hi_ci, color=color, alpha=0.12, linewidth=0)

        if x and acc:
            ax_acc.plot(x, acc, marker="o", ms=2, color=color, alpha=0.25, linewidth=1.0)
            ax_acc.plot(x, _moving_average(acc, window=3), color=color, linewidth=2.0, label=f"{label} (smoothed)")
            x_ci, _, lo_ci, hi_ci = _extract_binomial_ci_series(atk, metric="accuracy")
            if x_ci:
                ax_acc.fill_between(x_ci, lo_ci, hi_ci, color=color, alpha=0.12, linewidth=0)

        if any(v is not None for v in si):
            x_si = [xx for xx, vv in zip(x, si) if vv is not None]
            y_si = [vv for vv in si if vv is not None]
            ax_si.plot(x_si, y_si, marker="o", ms=2, color=color, alpha=0.25, linewidth=1.0)
            ax_si.plot(x_si, _moving_average(y_si, window=3), color=color, linewidth=2.0, label=f"{label} (smoothed)")
            if ref is not None:
                gaps = [vv - ref for vv in y_si]
                ax_gap.plot(x_si, gaps, marker="o", ms=2, color=color, alpha=0.25, linewidth=1.0)
                ax_gap.plot(x_si, _moving_average(gaps, window=3), color=color, linewidth=2.0, label=f"{label} (smoothed)")

    ax_ir.axhline(0.95, linestyle="--", color="red", alpha=0.7, label="SRC/CGC success threshold")
    ax_ir.set_title("Incorrect Rate vs Updates")
    ax_ir.set_ylabel("Incorrect Rate")
    ax_ir.set_xlabel("Attack Updates")
    ax_ir.set_ylim(0, 1.0)
    ax_ir.grid(alpha=0.3)
    ax_ir.legend(fontsize=8, loc="lower right")

    ax_acc.axhline(0.2, linestyle="--", color="red", alpha=0.7, label="Goodhart success threshold")
    ax_acc.set_title("Attacker Accuracy vs Updates")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Attack Updates")
    ax_acc.set_ylim(0, 1.0)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=8, loc="upper right")

    ax_si.set_title("Avg Sneaky Incorrect Score vs Updates")
    ax_si.set_ylabel("Avg Sneaky Incorrect Score")
    ax_si.set_xlabel("Attack Updates")
    ax_si.grid(alpha=0.3)
    ax_si.legend(fontsize=8, loc="best")

    ax_gap.axhline(0.0, linestyle="--", color="black", alpha=0.5)
    ax_gap.set_title("Score Gap: Sneaky Incorrect - Helpful Correct Ref")
    ax_gap.set_ylabel("Score Gap")
    ax_gap.set_xlabel("Attack Updates")
    ax_gap.grid(alpha=0.3)
    ax_gap.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Attack Trajectories (SRC / CGC / Goodhart)\n"
        f"{context['line_model']}\n"
        f"{context['line_agents']}\n"
        "Raw points (faint), 3-step moving-average trend (solid), 95% Wilson CI band for rate panels",
        y=0.995,
        fontsize=11,
    )
    _add_context_footer(fig, context)
    fig.subplots_adjust(top=0.84, bottom=0.12)
    plt.savefig(out, dpi=180)
    plt.close(fig)


def _plot_cgc_dynamics(loaded: Dict[str, Dict], out: Path, context: Dict[str, str]):
    atk = loaded["cgc"]["attack"]
    x, ir, acc, si, ref = _extract_history(atk)
    _, _, _, _, _ = loaded["cgc"]["attack"], None, None, None, None

    # Use avg_sneaky_score as a rough proxy for "verifier score" top panel.
    history = atk.get("history") or []
    avg_sneaky_score = [float(h.get("avg_sneaky_score", 0.0)) for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.8), sharex=True)
    ax_top, ax_bottom = axes

    ax_top.plot(x, avg_sneaky_score, color=OBJECTIVE_COLOR["cgc"], marker="o", ms=2, alpha=0.25, linewidth=1.0)
    ax_top.plot(x, _moving_average(avg_sneaky_score, window=3), color=OBJECTIVE_COLOR["cgc"], linewidth=2.0, label="Sneaky avg score (smoothed)")
    if ref is not None:
        ax_top.axhline(ref, linestyle="--", color="black", alpha=0.7, label="Helpful-correct ref score")
    ax_top.set_ylabel("Verifier / Oversight Score")
    ax_top.set_title(
        "Figure 9-style CGC Dynamics\n"
        f"{context['line_model']}\n"
        f"{context['line_agents']}"
    )
    ax_top.grid(alpha=0.3)
    ax_top.legend(fontsize=9)

    ax_bottom.plot(x, ir, color="#8c564b", marker="o", ms=2, alpha=0.25, linewidth=1.0)
    ax_bottom.plot(x, _moving_average(ir, window=3), color="#8c564b", linewidth=2.0, label="Incorrect rate (smoothed)")
    ax_bottom.plot(x, acc, color="#d62728", marker="o", ms=2, alpha=0.25, linewidth=1.0)
    ax_bottom.plot(x, _moving_average(acc, window=3), color="#d62728", linewidth=2.0, label="Accuracy (smoothed)")
    x_ir_ci, _, ir_lo, ir_hi = _extract_binomial_ci_series(atk, metric="incorrect_rate")
    if x_ir_ci:
        ax_bottom.fill_between(x_ir_ci, ir_lo, ir_hi, color="#8c564b", alpha=0.10, linewidth=0)
    x_acc_ci, _, acc_lo, acc_hi = _extract_binomial_ci_series(atk, metric="accuracy")
    if x_acc_ci:
        ax_bottom.fill_between(x_acc_ci, acc_lo, acc_hi, color="#d62728", alpha=0.10, linewidth=0)
    ax_bottom.axhline(0.95, linestyle="--", color="#8c564b", alpha=0.4)
    ax_bottom.set_ylabel("Rate")
    ax_bottom.set_xlabel("Attack Updates")
    ax_bottom.set_ylim(0, 1.0)
    ax_bottom.grid(alpha=0.3)
    ax_bottom.legend(fontsize=9)

    _add_context_footer(fig, context)
    fig.subplots_adjust(top=0.86, bottom=0.14)
    plt.savefig(out, dpi=180)
    plt.close(fig)


def _write_summary_md(loaded: Dict[str, Dict], out: Path):
    context = _build_plot_context(loaded)
    lines = []
    lines.append("# Kirchner Attack Plot Summary")
    lines.append("")
    lines.append("## Run Context")
    lines.append("")
    lines.append(f"- dataset: `{context['dataset_mode']}`")
    lines.append(f"- oversight_rule: `{context['oversight_rule']}`")
    lines.append(f"- num_verifiers: `{context['num_verifiers']}`")
    lines.append(f"- prover_model: `{context['prover_model']}`")
    lines.append(f"- verifier_model: `{context['verifier_model']}`")
    lines.append(f"- checkpoint: `{context['checkpoint']}`")
    lines.append(f"- source_jsons: `{context['line_sources'].replace('Sources=', '')}`")
    lines.append("- plot_style: `raw traces (faint) + 3-step moving-average trend; 95% Wilson CI on incorrect/accuracy panels`")
    lines.append("")
    for obj in OBJECTIVE_ORDER:
        atk = loaded[obj]["attack"]
        summ = loaded[obj]
        lines.append(f"## {obj.upper()}")
        lines.append(f"- source: `{summ['json_path']}`")
        lines.append(f"- success: `{atk.get('success')}`")
        lines.append(f"- updates_to_success: `{atk.get('updates_to_success')}`")
        lines.append(f"- updates_budget: `{atk.get('updates_budget')}`")
        hist = atk.get("history") or []
        if hist:
            last = hist[-1]
            lines.append(f"- last incorrect_rate: `{last.get('incorrect_rate')}`")
            lines.append(f"- last accuracy: `{last.get('accuracy')}`")
            lines.append(f"- last score_match: `{last.get('score_match')}`")
        lines.append("")

    lines.append("## Generated Figures")
    lines.append("")
    lines.append("- `fig6a_steps_until_exploit.png`")
    lines.append("- `fig_attack_dynamics_src_cgc_goodhart.png`")
    lines.append("- `fig9_style_cgc_dynamics.png`")
    out.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Plot Kirchner attack suite JSONs")
    ap.add_argument("--src-json", required=True)
    ap.add_argument("--cgc-json", required=True)
    ap.add_argument("--goodhart-json", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = {
        "src": _load_attack(Path(args.src_json), "src"),
        "cgc": _load_attack(Path(args.cgc_json), "cgc"),
        "goodhart": _load_attack(Path(args.goodhart_json), "goodhart"),
    }
    context = _build_plot_context(loaded)

    _plot_steps_until_exploit(loaded, out_dir / "fig6a_steps_until_exploit.png", context)
    _plot_attack_dynamics(loaded, out_dir / "fig_attack_dynamics_src_cgc_goodhart.png", context)
    _plot_cgc_dynamics(loaded, out_dir / "fig9_style_cgc_dynamics.png", context)
    _write_summary_md(loaded, out_dir / "README.md")

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
