"""
End-to-end variant comparison runner for the PMV loop.

This runner keeps the historical `multi_small_study` entrypoint but executes the
full training loop per variant (via `pmv.main`) before evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def _parse_variants(variant_str: str) -> List[Tuple[str, int, str, Optional[str]]]:
    """
    Variant token format:
      id:num_verifiers:oversight_rule[:verifier_model_or_alias]
    alias supported: small|large
    """
    out = []
    for tok in variant_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(f"Bad variant token: {tok}")
        vid = parts[0].strip()
        nv = int(parts[1].strip())
        rule = parts[2].strip()
        vmodel = parts[3].strip() if len(parts) == 4 else None
        out.append((vid, nv, rule, vmodel))
    if not out:
        raise ValueError("No variants parsed.")
    return out


def _run_cmd(cmd: List[str]):
    print("RUN:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        signame = None
        if int(e.returncode) < 0:
            try:
                signame = signal.Signals(-int(e.returncode)).name
            except Exception:
                signame = "UNKNOWN_SIGNAL"
        detail = f"returncode={e.returncode}"
        if signame is not None:
            detail += f" ({signame})"
        raise RuntimeError(
            f"Subprocess failed: {' '.join(cmd)} :: {detail}"
        ) from e


def _write_csv(path: Path, rows: List[Dict]):
    fields = [
        "variant_id",
        "num_verifiers",
        "oversight_rule",
        "prover_model",
        "verifier_model",
        "helpful_correctness",
        "sneaky_incorrect_rate",
        "sneaky_fool_rate",
        "binary_accuracy",
        "separation",
        "n_correct",
        "n_incorrect",
        "goal_auroc",
        "goal_center_sneaky_conditional_fool_rate",
        "goal_center_helpful_correct_accept_rate",
        "goal_sensitivity_pm0p02",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def _supports_eval_flag(flag: str) -> bool:
    help_txt = subprocess.run(
        ["python3", "-u", "-m", "pmv.evaluation", "-h"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    return flag in help_txt


def main():
    parser = argparse.ArgumentParser(description="End-to-end multi-variant PMV study")
    parser.add_argument(
        "--base-config",
        "--config",
        dest="base_config",
        default="pmv/configs/experiments/config_multismall_study.yaml",
        help="Base YAML config used for all variants.",
    )
    parser.add_argument(
        "--variants",
        default=(
            "single_small_3b:1:supervised:small,"
            "single_large:1:supervised:large,"
            "multi_large_supervised:3:supervised:large,"
            "multi_large_pe_min:3:pe_min:large,"
            "multi_large_pe_margin:3:pe_margin:large"
        ),
        help="Comma-separated variant specs id:nv:rule[:verifier_model_or_alias].",
    )
    parser.add_argument("--prover-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--small-verifier-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--large-verifier-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=0)

    # Optional training overrides to keep runs practical.
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--bootstrap-episodes", type=int, default=None)
    parser.add_argument("--bootstrap-oracle-episodes", type=int, default=None)
    parser.add_argument("--collect-episodes", type=int, default=None)
    parser.add_argument("--helpful-warmup-steps", type=int, default=None)
    parser.add_argument("--ppo-target-updates-per-round", type=int, default=None)
    parser.add_argument("--enable-two-gpu-accel", action="store_true")

    # Eval settings.
    parser.add_argument("--probe-episodes", type=int, default=120)
    parser.add_argument("--attack-episodes", type=int, default=40)
    parser.add_argument("--probe-max-new-tokens", type=int, default=192)
    parser.add_argument("--attack-max-new-tokens", type=int, default=192)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0])
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--verifier-decision-threshold", type=float, default=0.5)
    parser.add_argument("--fool-threshold", type=float, default=0.5)
    parser.add_argument("--dataset", default="zebra", choices=["auto", "math", "zebra", "gsm8k"])
    parser.add_argument("--zebra-max-size", default="3*3")
    parser.add_argument("--skip-adversarial", action="store_true")
    parser.add_argument("--save-probe-records", action="store_true")

    # Outputs.
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated configs/evals.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Compatibility alias for summary JSON path. If set, also writes JSON here.",
    )
    args = parser.parse_args()

    with open(args.base_config, "r") as f:
        base_config = yaml.safe_load(f)

    job_id = os.environ.get("LSB_JOBID", "local")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/studies/multi_small/{stamp}_{job_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = _parse_variants(args.variants)
    eval_files: List[str] = []
    summary_rows: List[Dict] = []
    save_probe_supported = _supports_eval_flag("--save-probe-records")

    for vid, nv, rule, vmodel in variants:
        resolved_vmodel = vmodel
        if vmodel == "small":
            resolved_vmodel = args.small_verifier_model
        elif vmodel == "large":
            resolved_vmodel = args.large_verifier_model
        if not resolved_vmodel:
            resolved_vmodel = base_config["model"]["verifier_model"]

        cfg = copy.deepcopy(base_config)
        cfg["model"]["prover_model"] = args.prover_model
        cfg["model"]["verifier_model"] = resolved_vmodel
        cfg["model"]["num_verifiers"] = int(nv)
        cfg["training"]["oversight_rule"] = rule
        cfg["training"]["seed"] = int(args.seed)
        cfg["training"]["checkpoint_root"] = str(output_dir / "checkpoints")

        if args.rounds is not None:
            cfg["training"]["rounds"] = int(args.rounds)
        if args.bootstrap_episodes is not None:
            cfg["training"]["bootstrap_episodes"] = int(args.bootstrap_episodes)
        if args.bootstrap_oracle_episodes is not None:
            cfg["training"]["bootstrap_oracle_episodes"] = int(args.bootstrap_oracle_episodes)
        if args.collect_episodes is not None:
            cfg["training"]["collect_episodes"] = int(args.collect_episodes)
        if args.helpful_warmup_steps is not None:
            cfg["training"]["helpful_warmup_steps"] = int(args.helpful_warmup_steps)
        if args.ppo_target_updates_per_round is not None:
            cfg["training"]["ppo_target_updates_per_round"] = int(args.ppo_target_updates_per_round)
        if args.enable_two_gpu_accel:
            cfg["training"]["multi_gpu_verifier_parallel"] = True
            cfg["training"]["parallel_verifier_inference"] = True
            cfg["training"]["split_reference_gpu"] = True
            cfg["training"]["ppo_parallel_reference_eval"] = True

        cfg_path = output_dir / f"config_{vid}.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print("\n" + "=" * 88)
        print(f"VARIANT {vid}: n={nv}, rule={rule}, prover={args.prover_model}, verifier={resolved_vmodel}")
        print("=" * 88)

        _run_cmd(["python3", "-u", "-m", "pmv.main", str(cfg_path)])

        ckpt = Path(cfg["training"]["checkpoint_root"]) / f"{cfg_path.stem}_latest.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint missing after training: {ckpt}")

        eval_out = output_dir / f"eval_{vid}.json"
        eval_cmd = [
            "python3",
            "-u",
            "-m",
            "pmv.evaluation",
            str(cfg_path),
            "--checkpoint",
            str(ckpt),
            "--dataset",
            args.dataset,
            "--probe-episodes",
            str(args.probe_episodes),
            "--attack-episodes",
            str(args.attack_episodes),
            "--probe-max-new-tokens",
            str(args.probe_max_new_tokens),
            "--attack-max-new-tokens",
            str(args.attack_max_new_tokens),
            "--decision-threshold",
            str(args.decision_threshold),
            "--verifier-decision-threshold",
            str(args.verifier_decision_threshold),
            "--fool-threshold",
            str(args.fool_threshold),
            "--output",
            str(eval_out),
        ]
        if args.dataset == "zebra":
            eval_cmd.extend(["--zebra-max-size", args.zebra_max_size])
        if args.skip_adversarial:
            eval_cmd.append("--skip-adversarial")
        if args.save_probe_records and save_probe_supported:
            eval_cmd.append("--save-probe-records")
        eval_cmd.append("--temperatures")
        eval_cmd.extend(str(t) for t in args.temperatures)
        _run_cmd(eval_cmd)

        eval_files.append(str(eval_out))
        with eval_out.open("r") as f:
            d = json.load(f)
        rates = d.get("probe_metrics", {}).get("rates", {})
        oq = d.get("oversight_quality", {})
        goal_summary = d.get("goal_robustness", {}).get("summary", {})
        fool_key = next((k for k in rates if k.startswith("sneaky_fool_rate@")), None)

        summary_rows.append(
            {
                "variant_id": vid,
                "num_verifiers": nv,
                "oversight_rule": rule,
                "prover_model": args.prover_model,
                "verifier_model": resolved_vmodel,
                "train_rounds": cfg.get("training", {}).get("rounds"),
                "bootstrap_episodes": cfg.get("training", {}).get("bootstrap_episodes"),
                "bootstrap_oracle_episodes": cfg.get("training", {}).get("bootstrap_oracle_episodes"),
                "collect_episodes": cfg.get("training", {}).get("collect_episodes"),
                "helpful_warmup_steps": cfg.get("training", {}).get("helpful_warmup_steps"),
                "ppo_target_updates_per_round": cfg.get("training", {}).get("ppo_target_updates_per_round"),
                "probe_episodes": args.probe_episodes,
                "attack_episodes": args.attack_episodes,
                "helpful_correctness": rates.get("helpful_correctness"),
                "sneaky_incorrect_rate": rates.get("sneaky_incorrect_rate"),
                "sneaky_fool_rate": rates.get(fool_key) if fool_key else None,
                "binary_accuracy": oq.get("binary_accuracy"),
                "separation": oq.get("separation"),
                "n_correct": oq.get("n_correct"),
                "n_incorrect": oq.get("n_incorrect"),
                "goal_auroc": goal_summary.get("auroc_correct_vs_incorrect"),
                "goal_center_sneaky_conditional_fool_rate": goal_summary.get("center_sneaky_conditional_fool_rate"),
                "goal_center_helpful_correct_accept_rate": goal_summary.get("center_helpful_correct_accept_rate"),
                "goal_sensitivity_pm0p02": (
                    goal_summary.get("sensitivity_band_range_pm0.02", {})
                    .get("sneaky_conditional_fool_rate")
                    if isinstance(goal_summary.get("sensitivity_band_range_pm0.02"), dict)
                    else None
                ),
            }
        )

    if eval_files:
        _run_cmd(["python3", "-u", "pmv/scripts/classify_collapse.py", *eval_files])

    summary_payload = {
        "run_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "seed": args.seed,
        },
        "base_config": args.base_config,
        "variants": args.variants,
        "run_parameters": {
            "rounds": args.rounds,
            "bootstrap_episodes": args.bootstrap_episodes,
            "bootstrap_oracle_episodes": args.bootstrap_oracle_episodes,
            "collect_episodes": args.collect_episodes,
            "helpful_warmup_steps": args.helpful_warmup_steps,
            "ppo_target_updates_per_round": args.ppo_target_updates_per_round,
            "enable_two_gpu_accel": args.enable_two_gpu_accel,
            "probe_episodes": args.probe_episodes,
            "attack_episodes": args.attack_episodes,
            "probe_max_new_tokens": args.probe_max_new_tokens,
            "attack_max_new_tokens": args.attack_max_new_tokens,
            "temperatures": args.temperatures,
            "decision_threshold": args.decision_threshold,
            "verifier_decision_threshold": args.verifier_decision_threshold,
            "fool_threshold": args.fool_threshold,
            "dataset": args.dataset,
            "zebra_max_size": args.zebra_max_size,
            "skip_adversarial": args.skip_adversarial,
            "save_probe_records": args.save_probe_records,
        },
        "output_dir": str(output_dir),
        "summary_rows": summary_rows,
        "eval_files": eval_files,
    }

    summary_path = output_dir / "multi_small_study.json"
    with summary_path.open("w") as f:
        json.dump(summary_payload, f, indent=2)
    _write_csv(output_dir / "multi_small_study.csv", summary_rows)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(summary_payload, f, indent=2)

    print(f"\nSaved summary JSON: {summary_path}")
    print(f"Saved summary CSV: {output_dir / 'multi_small_study.csv'}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
