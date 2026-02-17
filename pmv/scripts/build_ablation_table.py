#!/usr/bin/env python3
"""
Build inverse-ablation comparison tables from PMV evaluation JSON files.

Example:
  python3 pmv/scripts/build_ablation_table.py \
      --inputs results/evals/*.json \
      --out-dir results/ablations
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{100.0 * x:.1f}%"


def _fmt(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _expand_inputs(inputs: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("*.json")))
            continue
        matches = [Path(m) for m in glob.glob(item)]
        if matches:
            out.extend(sorted(matches))
            continue
        if p.exists():
            out.append(p)
    seen = set()
    unique = []
    for p in out:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)
    return unique


def _extract_probe_fool_rate(probe_rates: Dict) -> Optional[float]:
    for k, v in probe_rates.items():
        if isinstance(k, str) and k.startswith("sneaky_fool_rate@"):
            return _safe_float(v)
    return None


def _extract_probe_fool_key(probe_rates: Dict) -> str:
    for k in probe_rates.keys():
        if isinstance(k, str) and k.startswith("sneaky_fool_rate@"):
            return k
    return "sneaky_fool_rate@0.50"


def _extract_attack_max_fool(adversarial: Dict) -> Optional[float]:
    block = adversarial.get("sneaky_fool_rate", {})
    vals = []
    if isinstance(block, dict):
        for result in block.values():
            if isinstance(result, dict) and "fool_rate" in result:
                v = _safe_float(result.get("fool_rate"))
                if v is not None:
                    vals.append(v)
    if not vals:
        return None
    return max(vals)


def _load_row(path: Path) -> Optional[Dict]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    run_meta = data.get("run_metadata", {})
    snap = data.get("config_snapshot", {})
    probe = data.get("probe_metrics", {}).get("rates", {})
    oversight = data.get("oversight_quality", {})
    scorecard = data.get("scorecard", {})
    adversarial = data.get("adversarial_tests", {})

    row = {
        "source_file": str(path),
        "timestamp_utc": run_meta.get("timestamp_utc"),
        "job_id": run_meta.get("job_id"),
        "config_stem": run_meta.get("config_stem"),
        "ablation_id": run_meta.get("ablation_id"),
        "ablation_tag": run_meta.get("ablation_tag"),
        "num_verifiers": snap.get("num_verifiers"),
        "oversight_rule": snap.get("oversight_rule"),
        "phase1_stratified_batches": snap.get("phase1_stratified_batches"),
        "phase1_balance_labels": snap.get("phase1_balance_labels"),
        "helpful_warmup_steps": snap.get("helpful_warmup_steps"),
        "helpful_correctness": _safe_float(probe.get("helpful_correctness")),
        "sneaky_incorrect_rate": _safe_float(probe.get("sneaky_incorrect_rate")),
        "sneaky_fool_rate_key": _extract_probe_fool_key(probe),
        "sneaky_fool_rate": _extract_probe_fool_rate(probe),
        "final_format_before_fix_rate": _safe_float(probe.get("final_format_before_fix_rate")),
        "mean_helpful_oversight_score": _safe_float(probe.get("mean_helpful_oversight_score")),
        "mean_sneaky_oversight_score": _safe_float(probe.get("mean_sneaky_oversight_score")),
        "oversight_separation": _safe_float(oversight.get("separation")),
        "binary_accuracy": _safe_float(oversight.get("binary_accuracy")),
        "attack_max_fool_rate": _extract_attack_max_fool(adversarial),
        "overall_pass": scorecard.get("checks", {}).get("overall_pass"),
    }
    return row


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    mu = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return float(var ** 0.5)


def _collect_numeric(rows: List[Dict], key: str) -> List[float]:
    out = []
    for r in rows:
        v = _safe_float(r.get(key))
        if v is not None:
            out.append(v)
    return out


def _write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "source_file",
        "timestamp_utc",
        "job_id",
        "config_stem",
        "ablation_id",
        "ablation_tag",
        "num_verifiers",
        "oversight_rule",
        "phase1_stratified_batches",
        "phase1_balance_labels",
        "helpful_warmup_steps",
        "helpful_correctness",
        "sneaky_incorrect_rate",
        "sneaky_fool_rate_key",
        "sneaky_fool_rate",
        "final_format_before_fix_rate",
        "mean_helpful_oversight_score",
        "mean_sneaky_oversight_score",
        "oversight_separation",
        "binary_accuracy",
        "attack_max_fool_rate",
        "overall_pass",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_No data._\n"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def _build_group_summary(rows: List[Dict], group_key: str) -> List[List[str]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get(group_key, "unknown"))].append(r)

    table_rows = []
    for group, items in sorted(grouped.items(), key=lambda x: x[0]):
        helpful = _collect_numeric(items, "helpful_correctness")
        sneaky_inc = _collect_numeric(items, "sneaky_incorrect_rate")
        sneaky_fool = _collect_numeric(items, "sneaky_fool_rate")
        sep = _collect_numeric(items, "oversight_separation")
        acc = _collect_numeric(items, "binary_accuracy")
        pass_rate = _mean([1.0 if bool(i.get("overall_pass")) else 0.0 for i in items])
        table_rows.append(
            [
                group,
                str(len(items)),
                f"{_pct(_mean(helpful))} +/- {_pct(_std(helpful))}",
                f"{_pct(_mean(sneaky_inc))} +/- {_pct(_std(sneaky_inc))}",
                f"{_pct(_mean(sneaky_fool))} +/- {_pct(_std(sneaky_fool))}",
                f"{_fmt(_mean(sep))} +/- {_fmt(_std(sep))}",
                f"{_pct(_mean(acc))} +/- {_pct(_std(acc))}",
                _pct(pass_rate),
            ]
        )
    return table_rows


def _build_run_rows(rows: List[Dict]) -> List[List[str]]:
    out = []
    rows_sorted = sorted(rows, key=lambda r: (str(r.get("timestamp_utc")), str(r.get("source_file"))))
    for r in rows_sorted:
        out.append(
            [
                str(r.get("timestamp_utc", "-")),
                str(r.get("job_id", "-")),
                str(r.get("ablation_id") or r.get("config_stem") or "-"),
                str(r.get("ablation_tag") or "-"),
                str(r.get("num_verifiers", "-")),
                str(r.get("oversight_rule", "-")),
                _pct(_safe_float(r.get("helpful_correctness"))),
                _pct(_safe_float(r.get("sneaky_incorrect_rate"))),
                _pct(_safe_float(r.get("sneaky_fool_rate"))),
                _fmt(_safe_float(r.get("oversight_separation"))),
                _pct(_safe_float(r.get("binary_accuracy"))),
                str(bool(r.get("overall_pass"))),
            ]
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Build PMV inverse-ablation report from eval JSON files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["results/evals/*.json"],
        help="Files, directories, or glob patterns (default: results/evals/*.json).",
    )
    parser.add_argument(
        "--out-dir",
        default="results/ablations",
        help="Directory for aggregated outputs (default: results/ablations).",
    )
    args = parser.parse_args()

    input_paths = _expand_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("No evaluation JSON files found. Check --inputs path/glob.")

    rows = []
    for p in input_paths:
        row = _load_row(p)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit("Found files but none could be parsed as evaluation JSON.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    csv_latest = out_dir / "ablation_runs_latest.csv"
    csv_stamped = out_dir / f"ablation_runs_{stamp}.csv"
    _write_csv(rows, csv_latest)
    _write_csv(rows, csv_stamped)

    md_latest = out_dir / "ablation_summary_latest.md"
    md_stamped = out_dir / f"ablation_summary_{stamp}.md"

    run_headers = [
        "timestamp_utc",
        "job_id",
        "ablation_id",
        "ablation_tag",
        "num_verifiers",
        "oversight_rule",
        "helpful_correctness",
        "sneaky_incorrect_rate",
        "sneaky_fool_rate",
        "oversight_sep",
        "binary_accuracy",
        "overall_pass",
    ]
    group_headers = [
        "group",
        "runs",
        "helpful_correctness (mean+/-std)",
        "sneaky_incorrect_rate (mean+/-std)",
        "sneaky_fool_rate (mean+/-std)",
        "oversight_separation (mean+/-std)",
        "binary_accuracy (mean+/-std)",
        "overall_pass_rate",
    ]

    lines = []
    lines.append("# PMV Inverse-Ablation Summary")
    lines.append("")
    lines.append(f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Parsed runs: `{len(rows)}`")
    lines.append(f"- Input files: `{len(input_paths)}`")
    lines.append("")
    lines.append("## Grouped by Number of Verifiers")
    lines.append("")
    lines.append(_md_table(group_headers, _build_group_summary(rows, "num_verifiers")))
    lines.append("## Grouped by Oversight Rule")
    lines.append("")
    lines.append(_md_table(group_headers, _build_group_summary(rows, "oversight_rule")))
    lines.append("## Per-Run Table")
    lines.append("")
    lines.append(_md_table(run_headers, _build_run_rows(rows)))

    text = "\n".join(lines).strip() + "\n"
    md_latest.write_text(text)
    md_stamped.write_text(text)

    print(f"Loaded {len(rows)} run(s) from {len(input_paths)} file(s).")
    print(f"Wrote CSV: {csv_latest}")
    print(f"Wrote CSV: {csv_stamped}")
    print(f"Wrote Markdown: {md_latest}")
    print(f"Wrote Markdown: {md_stamped}")


if __name__ == "__main__":
    main()
