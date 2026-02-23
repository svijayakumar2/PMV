#!/usr/bin/env python3
"""
Generate a meeting-ready experiment pack from saved eval JSONs and run TXT logs.

Outputs:
  - CSV tables
  - XLSX workbook (if pandas/openpyxl available)
  - PNG plots (if matplotlib available)
  - Slide bullets markdown
  - MANIFEST.txt
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import yaml


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _basename(path: str) -> str:
    return os.path.basename(path.rstrip("/"))


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _find_eval_jsons(repo_root: str) -> List[str]:
    candidates = glob.glob(os.path.join(repo_root, "results", "**", "*.json"), recursive=True)
    out = []
    for p in candidates:
        try:
            d = _load_json(p)
        except Exception:
            continue
        if isinstance(d, dict) and "probe_metrics" in d and "scorecard" in d:
            out.append(p)
    return sorted(out)


def _find_txt_logs(repo_root: str) -> List[str]:
    txts = glob.glob(os.path.join(repo_root, "*.txt"))
    out = []
    for p in sorted(txts):
        try:
            t = open(p, "r").read()
        except Exception:
            continue
        if "Helpful correctness:" in t:
            out.append(p)
    return out


def _resolve_config(config_path: Optional[str], repo_root: str) -> Optional[Dict]:
    if not config_path:
        return None
    direct = os.path.join(repo_root, config_path)
    candidates = [direct]
    base = os.path.basename(config_path)
    candidates.extend(
        glob.glob(
            os.path.join(repo_root, "pmv", "configs", "experiments", "**", base),
            recursive=True,
        )
    )
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r") as f:
                    return yaml.safe_load(f)
            except Exception:
                continue
    return None


def _extract_eval_rows(eval_paths: List[str]) -> List[Dict]:
    rows = []
    for p in eval_paths:
        d = _load_json(p)
        cs = d.get("config_snapshot", {})
        pm = d.get("probe_metrics", {})
        rates = pm.get("rates", {})
        counts = pm.get("counts", {})
        oq = d.get("oversight_quality", {})
        scorecard = d.get("scorecard", {}).get("checks", {})
        fool_key = next((k for k in rates.keys() if k.startswith("sneaky_fool_rate@")), None)
        cond_fool_key = next(
            (k for k in rates.keys() if k.startswith("sneaky_conditional_fool_rate@")),
            None,
        )
        rows.append(
            {
                "path": p,
                "run_name": _basename(p),
                "job_id": d.get("run_metadata", {}).get("job_id"),
                "config": d.get("config"),
                "checkpoint_used": d.get("checkpoint_used"),
                "dataset": cs.get("resolved_dataset"),
                "dataset_mode": cs.get("dataset_mode"),
                "oversight_rule": cs.get("oversight_rule"),
                "prover_model": cs.get("prover_model"),
                "verifier_model": cs.get("verifier_model"),
                "num_verifiers": cs.get("num_verifiers"),
                "helpful_correctness": _safe_float(rates.get("helpful_correctness")),
                "sneaky_incorrect_rate": _safe_float(rates.get("sneaky_incorrect_rate")),
                "sneaky_fool_rate": _safe_float(rates.get(fool_key)) if fool_key else None,
                "sneaky_conditional_fool_rate": (
                    _safe_float(rates.get(cond_fool_key)) if cond_fool_key else None
                ),
                "fool_metric_key": fool_key,
                "final_format_before_fix_rate": _safe_float(rates.get("final_format_before_fix_rate")),
                "mean_helpful_oversight_score": _safe_float(rates.get("mean_helpful_oversight_score")),
                "mean_sneaky_oversight_score": _safe_float(rates.get("mean_sneaky_oversight_score")),
                "n_correct": counts.get("total_correct", oq.get("n_correct")),
                "n_incorrect": counts.get("total_incorrect", oq.get("n_incorrect")),
                "binary_accuracy": _safe_float(oq.get("binary_accuracy")),
                "separation": _safe_float(oq.get("separation")),
                "fool_rate_oversight_quality": _safe_float(oq.get("fool_rate")),
                "overall_pass": scorecard.get("overall_pass"),
                "helpful_correctness_ok": scorecard.get("helpful_correctness_ok"),
                "oversight_separation_ok": scorecard.get("oversight_separation_ok"),
                "binary_accuracy_ok": scorecard.get("binary_accuracy_ok"),
            }
        )
    return rows


def _parse_txt_rounds(path: str) -> Tuple[Dict, List[Dict]]:
    text = open(path, "r").read()
    config_match = re.search(r"Config:\s*(\S+)", text)
    config_path = config_match.group(1) if config_match else None

    helpful = []
    for m in re.finditer(r"Helpful correctness:\s*(\d+)\/(\d+)", text):
        num, den = int(m.group(1)), int(m.group(2))
        helpful.append((num, den, num / den if den > 0 else None))

    sneaky = []
    for m in re.finditer(r"Sneaky fool rate(?:@0\.50)?:\s*(\d+)\/(\d+)", text):
        num, den = int(m.group(1)), int(m.group(2))
        sneaky.append((num, den, num / den if den > 0 else None))

    n = max(len(helpful), len(sneaky))
    rounds = []
    for i in range(n):
        h = helpful[i] if i < len(helpful) else (None, None, None)
        s = sneaky[i] if i < len(sneaky) else (None, None, None)
        rounds.append(
            {
                "path": path,
                "run_name": _basename(path),
                "config": config_path,
                "round_idx": i + 1,
                "helpful_num": h[0],
                "helpful_den": h[1],
                "helpful_rate": h[2],
                "sneaky_fool_num": s[0],
                "sneaky_fool_den": s[1],
                "sneaky_fool_rate": s[2],
            }
        )

    summary = {
        "path": path,
        "run_name": _basename(path),
        "config": config_path,
        "rounds_with_helpful": len(helpful),
        "rounds_with_sneaky_fool": len(sneaky),
        "helpful_rate_mean": (sum(x[2] for x in helpful if x[2] is not None) / len(helpful)) if helpful else None,
        "helpful_rate_max": max((x[2] for x in helpful if x[2] is not None), default=None),
        "helpful_rate_min": min((x[2] for x in helpful if x[2] is not None), default=None),
        "sneaky_fool_rate_mean": (sum(x[2] for x in sneaky if x[2] is not None) / len(sneaky)) if sneaky else None,
        "sneaky_fool_rate_max": max((x[2] for x in sneaky if x[2] is not None), default=None),
        "sneaky_fool_rate_min": min((x[2] for x in sneaky if x[2] is not None), default=None),
    }
    return summary, rounds


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _try_write_xlsx(path: str, sheets: Dict[str, List[Dict]]) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        for sheet, rows in sheets.items():
            if rows:
                df = pd.DataFrame(rows)
            else:
                df = pd.DataFrame()
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return True


def _try_make_plots(out_dir: str, eval_rows: List[Dict], txt_round_rows: List[Dict]) -> List[str]:
    produced = []
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return produced

    # Plot 1: helpful correctness by eval run
    e = [r for r in eval_rows if r.get("helpful_correctness") is not None]
    if e:
        names = [r["run_name"] for r in e]
        vals = [r["helpful_correctness"] for r in e]
        plt.figure(figsize=(12, 5))
        plt.bar(range(len(names)), vals)
        plt.xticks(range(len(names)), names, rotation=75, ha="right", fontsize=8)
        plt.ylabel("Helpful Correctness")
        plt.title("Eval Runs: Helpful Correctness")
        plt.tight_layout()
        p = os.path.join(out_dir, "plot_eval_helpful_correctness.png")
        plt.savefig(p, dpi=180)
        plt.close()
        produced.append(p)

    # Plot 2: helpful vs sneaky fool scatter
    s = [
        r
        for r in eval_rows
        if r.get("helpful_correctness") is not None and r.get("sneaky_fool_rate") is not None
    ]
    if s:
        plt.figure(figsize=(6, 6))
        x = [r["helpful_correctness"] for r in s]
        y = [r["sneaky_fool_rate"] for r in s]
        plt.scatter(x, y)
        for r in s:
            plt.annotate(r["run_name"][:24], (r["helpful_correctness"], r["sneaky_fool_rate"]), fontsize=6)
        plt.xlabel("Helpful Correctness")
        plt.ylabel("Sneaky Fool Rate")
        plt.title("Eval Runs: Helpful vs Sneaky Fool")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        p = os.path.join(out_dir, "plot_eval_helpful_vs_fool.png")
        plt.savefig(p, dpi=180)
        plt.close()
        produced.append(p)

    # Plot 3: round-level helpful correctness from TXT logs
    by_run = defaultdict(list)
    for r in txt_round_rows:
        if r.get("helpful_rate") is not None:
            by_run[r["run_name"]].append(r)
    if by_run:
        plt.figure(figsize=(10, 6))
        for run_name, rows in by_run.items():
            rows = sorted(rows, key=lambda x: x["round_idx"])
            xs = [x["round_idx"] for x in rows]
            ys = [x["helpful_rate"] for x in rows]
            plt.plot(xs, ys, marker="o", linewidth=1.2, label=run_name.replace(".txt", ""))
        plt.xlabel("Round")
        plt.ylabel("Helpful Correctness")
        plt.title("Training Logs: Helpful Correctness by Round")
        plt.legend(fontsize=7, ncol=2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        p = os.path.join(out_dir, "plot_txt_helpful_by_round.png")
        plt.savefig(p, dpi=180)
        plt.close()
        produced.append(p)

    return produced


def _write_slide_bullets(
    path: str,
    eval_rows: List[Dict],
    txt_summary_rows: List[Dict],
    produced_plots: List[str],
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Best/worst quick picks
    eval_with_helpful = [r for r in eval_rows if r.get("helpful_correctness") is not None]
    best_helpful = max(eval_with_helpful, key=lambda x: x["helpful_correctness"]) if eval_with_helpful else None
    worst_helpful = min(eval_with_helpful, key=lambda x: x["helpful_correctness"]) if eval_with_helpful else None

    # Model + dataset groups
    group_stats = defaultdict(list)
    for r in eval_rows:
        key = (r.get("prover_model"), r.get("dataset"))
        group_stats[key].append(r)

    lines = []
    lines.append("# Meeting Bullets")
    lines.append("")
    lines.append("## What We Ran")
    lines.append(f"- Parsed eval JSON runs: {len(eval_rows)}")
    lines.append(f"- Parsed training TXT logs: {len(txt_summary_rows)}")
    lines.append("")
    lines.append("## Top-Line Results (Eval JSON)")
    if best_helpful:
        lines.append(
            f"- Best helpful correctness in saved eval JSONs: `{best_helpful['helpful_correctness']:.3f}` "
            f"({best_helpful['run_name']}, rule={best_helpful.get('oversight_rule')}, model={best_helpful.get('prover_model')})."
        )
    if worst_helpful:
        lines.append(
            f"- Worst helpful correctness in saved eval JSONs: `{worst_helpful['helpful_correctness']:.3f}` "
            f"({worst_helpful['run_name']})."
        )
    for (model, dataset), rows in sorted(group_stats.items(), key=lambda x: str(x[0])):
        hs = [r["helpful_correctness"] for r in rows if r.get("helpful_correctness") is not None]
        fs = [r["sneaky_fool_rate"] for r in rows if r.get("sneaky_fool_rate") is not None]
        bs = [r["binary_accuracy"] for r in rows if r.get("binary_accuracy") is not None]
        if hs and fs and bs:
            lines.append(
                f"- `{model}` on `{dataset}` (n={len(rows)} evals): "
                f"helpful mean `{sum(hs)/len(hs):.3f}`, "
                f"sneaky fool mean `{sum(fs)/len(fs):.3f}`, "
                f"binary acc mean `{sum(bs)/len(bs):.3f}`."
            )
    lines.append("")
    lines.append("## Historical Training-Log Signal (TXT)")
    strong = [
        r for r in txt_summary_rows if (r.get("helpful_rate_mean") is not None and r.get("helpful_rate_mean") >= 0.8)
    ]
    weak = [
        r for r in txt_summary_rows if (r.get("helpful_rate_mean") is not None and r.get("helpful_rate_mean") < 0.2)
    ]
    if strong:
        lines.append(
            "- There are historical logs with very high helpful correctness "
            f"(>=0.8 mean): {', '.join(r['run_name'] for r in strong[:5])}."
        )
    if weak:
        lines.append(
            "- There are also weak/collapsed logs (<0.2 mean helpful), showing instability across runs."
        )
    lines.append("")
    lines.append("## Slide Narrative Suggestions")
    lines.append("- We can demonstrate that capability exists in prior runs, but current pipeline is unstable.")
    lines.append("- Recent tiny/stage evals are mostly collapsed on helpful correctness, so robustness comparisons are premature in those checkpoints.")
    lines.append("- Immediate plan: lock one reproducible 3B baseline checkpoint, then run supervised vs PE-Min vs PE-Margin on identical eval settings.")
    lines.append("")
    if produced_plots:
        lines.append("## Included Plots")
        for p in produced_plots:
            lines.append(f"- `{os.path.basename(p)}`")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/Users/saranya/Desktop/PMV",
        help="Repository root containing results/ and pmv/",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: results/meeting_pack_<timestamp>).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(args.repo_root, "results", f"meeting_pack_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    eval_paths = _find_eval_jsons(args.repo_root)
    txt_paths = _find_txt_logs(args.repo_root)

    eval_rows = _extract_eval_rows(eval_paths)

    txt_summary_rows = []
    txt_round_rows = []
    for p in txt_paths:
        summary, rounds = _parse_txt_rounds(p)
        cfg = _resolve_config(summary.get("config"), args.repo_root)
        if cfg:
            summary["prover_model"] = cfg.get("model", {}).get("prover_model")
            summary["verifier_model"] = cfg.get("model", {}).get("verifier_model")
            summary["dataset_type"] = cfg.get("dataset", {}).get("type")
            summary["oversight_rule"] = cfg.get("training", {}).get("oversight_rule")
            for r in rounds:
                r["prover_model"] = summary["prover_model"]
                r["verifier_model"] = summary["verifier_model"]
                r["dataset_type"] = summary["dataset_type"]
                r["oversight_rule"] = summary["oversight_rule"]
        txt_summary_rows.append(summary)
        txt_round_rows.extend(rounds)

    # CSV outputs
    eval_cols = [
        "path",
        "run_name",
        "job_id",
        "config",
        "checkpoint_used",
        "dataset",
        "dataset_mode",
        "oversight_rule",
        "prover_model",
        "verifier_model",
        "num_verifiers",
        "helpful_correctness",
        "sneaky_incorrect_rate",
        "sneaky_fool_rate",
        "sneaky_conditional_fool_rate",
        "final_format_before_fix_rate",
        "mean_helpful_oversight_score",
        "mean_sneaky_oversight_score",
        "n_correct",
        "n_incorrect",
        "binary_accuracy",
        "separation",
        "fool_rate_oversight_quality",
        "overall_pass",
        "helpful_correctness_ok",
        "oversight_separation_ok",
        "binary_accuracy_ok",
    ]
    _write_csv(os.path.join(out_dir, "eval_runs.csv"), eval_rows, eval_cols)

    txt_summary_cols = [
        "path",
        "run_name",
        "config",
        "prover_model",
        "verifier_model",
        "dataset_type",
        "oversight_rule",
        "rounds_with_helpful",
        "rounds_with_sneaky_fool",
        "helpful_rate_mean",
        "helpful_rate_max",
        "helpful_rate_min",
        "sneaky_fool_rate_mean",
        "sneaky_fool_rate_max",
        "sneaky_fool_rate_min",
    ]
    _write_csv(os.path.join(out_dir, "txt_run_summary.csv"), txt_summary_rows, txt_summary_cols)

    txt_round_cols = [
        "path",
        "run_name",
        "config",
        "prover_model",
        "verifier_model",
        "dataset_type",
        "oversight_rule",
        "round_idx",
        "helpful_num",
        "helpful_den",
        "helpful_rate",
        "sneaky_fool_num",
        "sneaky_fool_den",
        "sneaky_fool_rate",
    ]
    _write_csv(os.path.join(out_dir, "txt_round_metrics.csv"), txt_round_rows, txt_round_cols)

    # XLSX (best-effort)
    xlsx_written = _try_write_xlsx(
        os.path.join(out_dir, "meeting_pack.xlsx"),
        {
            "eval_runs": eval_rows,
            "txt_run_summary": txt_summary_rows,
            "txt_round_metrics": txt_round_rows,
        },
    )

    # Plots (best-effort)
    plot_paths = _try_make_plots(out_dir, eval_rows, txt_round_rows)

    # Slide bullets
    bullets_path = os.path.join(out_dir, "slide_bullets.md")
    _write_slide_bullets(bullets_path, eval_rows, txt_summary_rows, plot_paths)

    manifest = [
        f"output_dir={out_dir}",
        f"eval_json_count={len(eval_rows)}",
        f"txt_log_count={len(txt_summary_rows)}",
        f"xlsx_written={xlsx_written}",
        f"plots_written={len(plot_paths)}",
        "files:",
        "  - eval_runs.csv",
        "  - txt_run_summary.csv",
        "  - txt_round_metrics.csv",
        "  - meeting_pack.xlsx (if available)",
        "  - slide_bullets.md",
    ]
    for p in plot_paths:
        manifest.append(f"  - {_basename(p)}")
    with open(os.path.join(out_dir, "MANIFEST.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")

    print("\n".join(manifest))


if __name__ == "__main__":
    main()
