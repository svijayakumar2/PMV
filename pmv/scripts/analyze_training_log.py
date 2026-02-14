#!/usr/bin/env python3
"""
Parse PMV training .txt logs and generate plots + written analysis.

Usage:
  python scripts/analyze_training_log.py /path/to/171606.txt
  python scripts/analyze_training_log.py /path/to/log1.txt /path/to/log2.txt --out-dir results/log_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROUND_RE = re.compile(r"ROUND\s+(\d+)/(\d+)")
PHASE1_RE = re.compile(r"Phase 1 complete\.\s+Oversight loss:\s+([\-0-9.eE]+)")
MEAN_REWARD_RE = re.compile(r"Mean reward:\s+([\-0-9.eE]+)")
HELPFUL_RE = re.compile(r"Helpful correctness:\s+(\d+)/(\d+)")
SNEAKY_FOOL_RE = re.compile(r"Sneaky fool rate@([0-9.]+):\s+(\d+)/(\d+)")
PPO_RE = re.compile(r"PPO epoch\s+(\d+)/(\d+),\s+loss:\s+([\-0-9.eE]+)")
COLL_RE = re.compile(
    r"Collection diagnostics:\s+FINAL format\s+(\d+)/(\d+)\s+\(([0-9.]+)%\),\s+"
    r"sneaky incorrect\s+(\d+)/(\d+),\s+sneaky fooled@([0-9.]+)\s+(\d+)/(\d+),\s+"
    r"mean f \(helpful/sneaky\)=\(([0-9.\-eE]+)/([0-9.\-eE]+)\)"
)
COLL_SIMPLE_RE = re.compile(
    r"Collection diagnostics:\s+FINAL format\s+(\d+)/(\d+)\s+\(([0-9.]+)%\),\s+"
    r"sneaky incorrect\s+(\d+)/(\d+),\s+sneaky fooled@([0-9.]+)\s+(\d+)/(\d+)"
)
JOB_RE = re.compile(r"Job ID:\s+(\S+)")
EXP_RE = re.compile(r"Experiment:\s+(\S+)")
CFG_RE = re.compile(r"Config:\s+(.+)")


@dataclass
class RoundMetrics:
    round_idx: int
    total_rounds: Optional[int] = None
    phase1_loss: Optional[float] = None
    mean_reward: Optional[float] = None
    helpful_correct: Optional[int] = None
    helpful_total: Optional[int] = None
    sneaky_fooled: Optional[int] = None
    sneaky_total: Optional[int] = None
    sneaky_fool_threshold: Optional[float] = None
    final_format_count: Optional[int] = None
    final_format_total: Optional[int] = None
    final_format_rate: Optional[float] = None
    sneaky_incorrect: Optional[int] = None
    sneaky_incorrect_total: Optional[int] = None
    sneaky_f_mean: Optional[float] = None
    helpful_f_mean: Optional[float] = None
    ppo_losses: Optional[List[float]] = None

    def helpful_rate(self) -> Optional[float]:
        if self.helpful_total and self.helpful_correct is not None:
            return self.helpful_correct / max(1, self.helpful_total)
        return None

    def sneaky_fool_rate(self) -> Optional[float]:
        if self.sneaky_total and self.sneaky_fooled is not None:
            return self.sneaky_fooled / max(1, self.sneaky_total)
        return None

    def sneaky_incorrect_rate(self) -> Optional[float]:
        if self.sneaky_incorrect_total and self.sneaky_incorrect is not None:
            return self.sneaky_incorrect / max(1, self.sneaky_incorrect_total)
        return None

    def ppo_mean(self) -> Optional[float]:
        if self.ppo_losses:
            return sum(self.ppo_losses) / len(self.ppo_losses)
        return None


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _clean_field(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    return val.strip().strip('"').strip("'")


def parse_log(log_path: Path) -> Dict:
    lines = log_path.read_text(errors="ignore").splitlines()
    rounds: Dict[int, RoundMetrics] = {}

    current_round: Optional[int] = None
    total_rounds: Optional[int] = None
    job_id = None
    experiment = None
    config_path = None
    training_complete = False

    for line in lines:
        if "Training complete!" in line:
            training_complete = True

        m = JOB_RE.search(line)
        if m:
            job_id = _clean_field(m.group(1))
        m = EXP_RE.search(line)
        if m:
            experiment = _clean_field(m.group(1))
        m = CFG_RE.search(line)
        if m:
            config_path = _clean_field(m.group(1).strip())

        m = ROUND_RE.search(line)
        if m:
            current_round = int(m.group(1))
            total_rounds = int(m.group(2))
            rounds.setdefault(current_round, RoundMetrics(round_idx=current_round))
            rounds[current_round].total_rounds = total_rounds
            continue

        if current_round is None:
            continue

        rd = rounds.setdefault(current_round, RoundMetrics(round_idx=current_round))

        m = PHASE1_RE.search(line)
        if m:
            rd.phase1_loss = _to_float(m.group(1))
            continue

        m = MEAN_REWARD_RE.search(line)
        if m:
            rd.mean_reward = _to_float(m.group(1))
            continue

        m = HELPFUL_RE.search(line)
        if m:
            rd.helpful_correct = int(m.group(1))
            rd.helpful_total = int(m.group(2))
            continue

        m = SNEAKY_FOOL_RE.search(line)
        if m:
            rd.sneaky_fool_threshold = _to_float(m.group(1))
            rd.sneaky_fooled = int(m.group(2))
            rd.sneaky_total = int(m.group(3))
            continue

        m = COLL_RE.search(line)
        if m:
            rd.final_format_count = int(m.group(1))
            rd.final_format_total = int(m.group(2))
            rd.final_format_rate = _to_float(m.group(3))
            rd.sneaky_incorrect = int(m.group(4))
            rd.sneaky_incorrect_total = int(m.group(5))
            rd.sneaky_fool_threshold = _to_float(m.group(6))
            rd.sneaky_fooled = int(m.group(7))
            rd.sneaky_total = int(m.group(8))
            rd.helpful_f_mean = _to_float(m.group(9))
            rd.sneaky_f_mean = _to_float(m.group(10))
            continue

        m = COLL_SIMPLE_RE.search(line)
        if m:
            rd.final_format_count = int(m.group(1))
            rd.final_format_total = int(m.group(2))
            rd.final_format_rate = _to_float(m.group(3))
            rd.sneaky_incorrect = int(m.group(4))
            rd.sneaky_incorrect_total = int(m.group(5))
            rd.sneaky_fool_threshold = _to_float(m.group(6))
            rd.sneaky_fooled = int(m.group(7))
            rd.sneaky_total = int(m.group(8))
            continue

        m = PPO_RE.search(line)
        if m:
            if rd.ppo_losses is None:
                rd.ppo_losses = []
            rd.ppo_losses.append(float(m.group(3)))

    rounds_sorted = [rounds[k] for k in sorted(rounds.keys())]
    parsed = {
        "log_path": str(log_path),
        "job_id": job_id,
        "experiment": experiment,
        "config_path": config_path,
        "training_complete": training_complete,
        "rounds": rounds_sorted,
        "total_rounds_declared": total_rounds,
        "rounds_seen": len(rounds_sorted),
    }
    return parsed


def _trend(values: List[float]) -> str:
    if len(values) < 2:
        return "insufficient data"
    if values[-1] < values[0]:
        return "down"
    if values[-1] > values[0]:
        return "up"
    return "flat"


def build_analysis(parsed: Dict) -> List[str]:
    rounds: List[RoundMetrics] = parsed["rounds"]
    if not rounds:
        return ["No rounds parsed from this log."]

    phase1 = [r.phase1_loss for r in rounds if r.phase1_loss is not None]
    helpful = [r.helpful_rate() for r in rounds if r.helpful_rate() is not None]
    fool = [r.sneaky_fool_rate() for r in rounds if r.sneaky_fool_rate() is not None]
    sneaky_incorrect = [r.sneaky_incorrect_rate() for r in rounds if r.sneaky_incorrect_rate() is not None]
    final_fmt = [r.final_format_rate / 100.0 for r in rounds if r.final_format_rate is not None]

    lines = []
    lines.append(f"Rounds parsed: {parsed['rounds_seen']}")
    if parsed.get("total_rounds_declared") is not None:
        lines.append(f"Declared total rounds: {parsed['total_rounds_declared']}")
    lines.append(f"Training complete marker found: {parsed['training_complete']}")

    if phase1:
        lines.append(
            f"Oversight loss: start={phase1[0]:.4f}, end={phase1[-1]:.4f}, trend={_trend(phase1)}"
        )
    if helpful:
        lines.append(
            f"Helpful correctness rate: start={helpful[0]:.3f}, end={helpful[-1]:.3f}, trend={_trend(helpful)}"
        )
    if fool:
        lines.append(
            f"Sneaky fool rate: start={fool[0]:.3f}, end={fool[-1]:.3f}, trend={_trend(fool)}"
        )
    if sneaky_incorrect:
        lines.append(
            f"Sneaky incorrect rate: start={sneaky_incorrect[0]:.3f}, end={sneaky_incorrect[-1]:.3f}, trend={_trend(sneaky_incorrect)}"
        )
    if final_fmt:
        lines.append(
            f"Final-format compliance: start={final_fmt[0]:.3f}, end={final_fmt[-1]:.3f}, trend={_trend(final_fmt)}"
        )

    # Diagnostics and likely failure mode flags
    if fool and all(abs(x) < 1e-9 for x in fool):
        if sneaky_incorrect and any(x < 0.2 for x in sneaky_incorrect):
            lines.append(
                "Flag: fool rate is zero mostly because sneaky outputs are often not incorrect (attack not realized)."
            )
        else:
            lines.append(
                "Flag: fool rate is zero while sneaky outputs are incorrect, suggesting a conservative oversight threshold or collapsed low scores."
            )
    if final_fmt and sum(1 for x in final_fmt if x < 0.8) >= max(1, math.ceil(0.5 * len(final_fmt))):
        lines.append("Flag: final-answer format compliance is low in many rounds.")

    return lines


def save_plots(parsed: Dict, out_dir: Path) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    rounds: List[RoundMetrics] = parsed["rounds"]
    if not rounds:
        return []

    xs = [r.round_idx for r in rounds]
    phase1 = [r.phase1_loss for r in rounds]
    helpful = [r.helpful_rate() for r in rounds]
    fool = [r.sneaky_fool_rate() for r in rounds]
    sneaky_incorrect = [r.sneaky_incorrect_rate() for r in rounds]
    mean_reward = [r.mean_reward for r in rounds]
    final_fmt = [r.final_format_rate / 100.0 if r.final_format_rate is not None else None for r in rounds]
    ppo_mean = [r.ppo_mean() for r in rounds]

    def _plot_line(yvals: List[Optional[float]], title: str, ylabel: str, filename: str, ylim: Optional[Tuple[float, float]] = None):
        xv = [x for x, y in zip(xs, yvals) if y is not None]
        yv = [y for y in yvals if y is not None]
        if not yv:
            return None
        plt.figure(figsize=(7, 4))
        plt.plot(xv, yv, marker="o")
        plt.title(title)
        plt.xlabel("Round")
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.grid(alpha=0.3)
        out = out_dir / filename
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out

    outputs: List[Path] = []
    for item in [
        _plot_line(phase1, "Oversight Loss by Round", "Loss", "oversight_loss.png"),
        _plot_line(mean_reward, "Mean Reward by Round", "Reward", "mean_reward.png"),
        _plot_line(helpful, "Helpful Correctness by Round", "Rate", "helpful_correctness.png", (0, 1)),
        _plot_line(fool, "Sneaky Fool Rate by Round", "Rate", "sneaky_fool_rate.png", (0, 1)),
        _plot_line(sneaky_incorrect, "Sneaky Incorrect Rate by Round", "Rate", "sneaky_incorrect_rate.png", (0, 1)),
        _plot_line(final_fmt, "Final Format Compliance by Round", "Rate", "final_format_rate.png", (0, 1)),
        _plot_line(ppo_mean, "Mean PPO Loss by Round", "Loss", "ppo_loss_mean.png"),
    ]:
        if item is not None:
            outputs.append(item)

    # Combined diagnostic plot
    if any(v is not None for v in helpful) and any(v is not None for v in fool):
        xv_h = [x for x, y in zip(xs, helpful) if y is not None]
        yv_h = [y for y in helpful if y is not None]
        xv_f = [x for x, y in zip(xs, fool) if y is not None]
        yv_f = [y for y in fool if y is not None]
        plt.figure(figsize=(7, 4))
        if yv_h:
            plt.plot(xv_h, yv_h, marker="o", label="Helpful correctness")
        if yv_f:
            plt.plot(xv_f, yv_f, marker="o", label="Sneaky fool rate")
        plt.ylim(0, 1)
        plt.title("Helpful vs Sneaky Outcomes")
        plt.xlabel("Round")
        plt.ylabel("Rate")
        plt.grid(alpha=0.3)
        plt.legend()
        out = out_dir / "helpful_vs_sneaky.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        outputs.append(out)

    return outputs


def write_outputs(parsed: Dict, out_root: Path) -> Dict:
    log_path = Path(parsed["log_path"])
    run_name = log_path.stem
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds_json = []
    for r in parsed["rounds"]:
        row = asdict(r)
        row["helpful_rate"] = r.helpful_rate()
        row["sneaky_fool_rate"] = r.sneaky_fool_rate()
        row["sneaky_incorrect_rate"] = r.sneaky_incorrect_rate()
        row["ppo_mean_loss"] = r.ppo_mean()
        rounds_json.append(row)

    analysis_lines = build_analysis(parsed)
    summary = {
        "log_path": parsed["log_path"],
        "job_id": parsed["job_id"],
        "experiment": parsed["experiment"],
        "config_path": parsed["config_path"],
        "training_complete": parsed["training_complete"],
        "rounds_seen": parsed["rounds_seen"],
        "total_rounds_declared": parsed["total_rounds_declared"],
        "analysis_lines": analysis_lines,
        "rounds": rounds_json,
    }

    plots = save_plots(parsed, out_dir)
    summary["plots"] = [str(p) for p in plots]

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    md_lines = [
        f"# Log Analysis: {log_path.name}",
        "",
        f"- Job ID: {parsed['job_id']}",
        f"- Experiment: {parsed['experiment']}",
        f"- Config: {parsed['config_path']}",
        f"- Training complete marker: {parsed['training_complete']}",
        "",
        "## Key Findings",
    ]
    for ln in analysis_lines:
        md_lines.append(f"- {ln}")
    md_lines.extend(["", "## Generated Plots"])
    if plots:
        for p in plots:
            md_lines.append(f"- `{p.name}`")
    else:
        md_lines.append("- No plots generated (matplotlib not available in this environment).")
    summary_md = out_dir / "analysis.md"
    summary_md.write_text("\n".join(md_lines))
    return {
        "run_name": run_name,
        "out_dir": str(out_dir),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "plots": [str(p) for p in plots],
    }


def write_multi_run_index(results: List[Dict], out_root: Path) -> Optional[Path]:
    if len(results) <= 1:
        return None
    lines = ["# Multi-run Log Analysis", "", "| Run | Output Directory |", "|---|---|"]
    for item in results:
        lines.append(f"| `{item['run_name']}` | `{item['out_dir']}` |")
    out = out_root / "index.md"
    out.write_text("\n".join(lines))
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze PMV training .txt logs and plot metrics.")
    parser.add_argument("logs", nargs="+", help="Path(s) to .txt training logs")
    parser.add_argument("--out-dir", default="results/log_analysis", help="Output directory root")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for p in args.logs:
        log_path = Path(p)
        if not log_path.exists():
            print(f"[skip] missing file: {log_path}")
            continue
        parsed = parse_log(log_path)
        item = write_outputs(parsed, out_root)
        results.append(item)
        print(f"[ok] analyzed {log_path} -> {item['out_dir']}")

    idx = write_multi_run_index(results, out_root)
    if idx is not None:
        print(f"[ok] multi-run index: {idx}")

    if not results:
        raise SystemExit("No valid log files were analyzed.")


if __name__ == "__main__":
    main()
