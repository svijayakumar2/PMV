"""
Controlled study runner:
fixed prover + small verifiers, compare single vs multi and oversight rules.

Research question:
Can multiple small verifiers improve robustness and accuracy for a helpful prover?
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

from pmv.checkpointing import load_checkpoint
from pmv.data import ReplayBuffer, SolutionRecord
from pmv.evaluation import collect_probe_metrics
from pmv.diversity_analysis import compute_oversight_quality
from pmv.adversarial_test import test_sneaky_fool_rate
from pmv.ensemble import VerifierEnsemble
from pmv.main import get_dataset
from pmv.prover import Prover, create_role_prompt, ensure_final_line
from pmv.training import train_phase1
from pmv.utils import cleanup_memory


def _sample_train(dataset):
    if hasattr(dataset, "sample_train"):
        return dataset.sample_train()
    return dataset.sample()


def _safe_mean(vals: List[float]) -> float:
    return float(sum(vals) / max(1, len(vals)))


def _parse_variants(variant_str: str) -> List[Tuple[str, int, str, Optional[str]]]:
    """
    variant format:
      - "id:num_verifiers:oversight_rule"
      - "id:num_verifiers:oversight_rule:verifier_model_or_alias"
        alias supported later: small|large
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
        n = int(parts[1].strip())
        rule = parts[2].strip()
        vmodel = parts[3].strip() if len(parts) == 4 else None
        out.append((vid, n, rule, vmodel))
    if not out:
        raise ValueError("No valid variants provided.")
    return out


def build_fixed_replay_buffer(
    prover: Prover,
    dataset,
    config: Dict,
    dataset_type: str,
    num_episodes: int,
    balanced_roles: bool = True,
) -> ReplayBuffer:
    train_cfg = config["training"]
    mu_0 = float(train_cfg.get("mu_0", 0.5))
    helpful_do_sample = bool(train_cfg.get("helpful_do_sample", False))
    sneaky_do_sample = bool(train_cfg.get("sneaky_do_sample", True))
    helpful_temperature = float(train_cfg.get("helpful_temperature", 0.2))
    sneaky_temperature = float(train_cfg.get("sneaky_temperature", 0.9))
    helpful_max_new_tokens = int(train_cfg.get("helpful_max_new_tokens", 192))
    sneaky_max_new_tokens = int(train_cfg.get("sneaky_max_new_tokens", 192))

    rb = ReplayBuffer(max_size=max(num_episodes, 1000))
    prover.model.eval()

    for ep in range(num_episodes):
        if balanced_roles:
            role = "helpful" if ep % 2 == 0 else "sneaky"
        else:
            role = "helpful" if random.random() < mu_0 else "sneaky"
        problem, solution_true = _sample_train(dataset)
        prompt = create_role_prompt(problem, role, prover.tokenizer, dataset_type=dataset_type)

        do_sample = helpful_do_sample if role == "helpful" else sneaky_do_sample
        temperature = helpful_temperature if role == "helpful" else sneaky_temperature
        max_new_tokens = helpful_max_new_tokens if role == "helpful" else sneaky_max_new_tokens

        with torch.no_grad():
            if do_sample:
                response = prover.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                )
            else:
                response = prover.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        response = ensure_final_line(response, dataset_type=dataset_type)

        is_correct = bool(dataset.check_solution(solution_true, response))
        rb.add(
            SolutionRecord(
                problem=problem,
                solution_true=solution_true,
                response=response,
                correctness=1.0 if is_correct else 0.0,
                role=role,
                round_generated=0,
            )
        )

        if (ep + 1) % 25 == 0:
            print(f"  fixed-buffer {ep+1}/{num_episodes}")
        cleanup_memory()

    return rb


def run_variant(
    variant_id: str,
    num_verifiers: int,
    oversight_rule: str,
    verifier_model: Optional[str],
    base_config: Dict,
    replay_buffer: ReplayBuffer,
    prover: Prover,
    dataset,
    dataset_type: str,
    phase1_rounds: int,
    probe_episodes: int,
    attack_episodes: int,
    temperatures: List[float],
) -> Dict:
    config = copy.deepcopy(base_config)
    config["model"]["num_verifiers"] = int(num_verifiers)
    config["training"]["oversight_rule"] = str(oversight_rule)
    if verifier_model:
        config["model"]["verifier_model"] = verifier_model

    train_cfg = config["training"]
    oversight_threshold = float(train_cfg.get("oversight_decision_threshold", 0.5))
    verifier_threshold = float(train_cfg.get("verifier_decision_threshold", oversight_threshold))
    fool_threshold = float(train_cfg.get("sneaky_fool_threshold", oversight_threshold))

    print("\n" + "=" * 80)
    print(
        f"VARIANT: {variant_id} (m={num_verifiers}, rule={oversight_rule}, "
        f"verifier_model={config['model']['verifier_model']})"
    )
    print("=" * 80)

    ensemble = VerifierEnsemble(config)
    ensemble.create_verifiers()

    phase1_losses = []
    for ridx in range(1, phase1_rounds + 1):
        loss = train_phase1(ensemble, replay_buffer, config, round_idx=ridx)
        phase1_losses.append(float(loss))
        print(f"  phase1 round {ridx}/{phase1_rounds}: loss={loss:.4f}")

    probe = collect_probe_metrics(
        prover=prover,
        ensemble=ensemble,
        dataset=dataset,
        config=config,
        num_episodes=probe_episodes,
        balanced_roles=True,
        max_new_tokens_override=None,
        oversight_decision_threshold=oversight_threshold,
        verifier_decision_threshold=verifier_threshold,
        fool_threshold=fool_threshold,
        dataset_type=dataset_type,
        save_records=False,
    )

    raw = probe["raw"]
    oversight_quality = compute_oversight_quality(
        raw["oversight_scores"],
        raw["correctness_labels"],
        decision_threshold=oversight_threshold,
        fool_threshold=fool_threshold,
    )

    per_verifier_accuracy = {}
    for v_id in range(num_verifiers):
        cvec = raw["verifier_correct"].get(v_id, [])
        per_verifier_accuracy[f"verifier_{v_id}"] = sum(cvec) / max(1, len(cvec))

    attack = {}
    max_attack_fool_rate = None
    if attack_episodes > 0:
        attack = test_sneaky_fool_rate(
            prover=prover,
            ensemble=ensemble,
            dataset=dataset,
            num_episodes=attack_episodes,
            temperatures=temperatures,
            max_new_tokens=int(train_cfg.get("sneaky_max_new_tokens", 192)),
            progress_every=max(10, attack_episodes // 5),
            fool_threshold=fool_threshold,
            dataset_type=dataset_type,
        )
        rates = [
            float(v.get("fool_rate"))
            for v in attack.values()
            if isinstance(v, dict) and v.get("fool_rate") is not None
        ]
        max_attack_fool_rate = max(rates) if rates else None

    result = {
        "variant_id": variant_id,
        "num_verifiers": num_verifiers,
        "oversight_rule": oversight_rule,
        "verifier_model": config["model"]["verifier_model"],
        "phase1_losses": phase1_losses,
        "probe_counts": probe["counts"],
        "probe_rates": probe["rates"],
        "oversight_quality": oversight_quality,
        "per_verifier_accuracy": per_verifier_accuracy,
        "attack_sneaky_fool_rate": attack,
        "max_attack_fool_rate": max_attack_fool_rate,
    }

    ensemble.delete_verifiers()
    cleanup_memory()
    return result


def write_csv_summary(path: str, rows: List[Dict]):
    fields = [
        "variant_id",
        "num_verifiers",
        "oversight_rule",
        "verifier_model",
        "helpful_correctness",
        "sneaky_incorrect_rate",
        "sneaky_fool_rate",
        "binary_accuracy",
        "separation",
        "n_correct",
        "n_incorrect",
        "mean_phase1_loss",
        "min_verifier_accuracy",
        "max_attack_fool_rate",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main():
    parser = argparse.ArgumentParser(description="Controlled multi-small-verifier study")
    parser.add_argument(
        "--config",
        default="pmv/configs/experiments/config_multismall_study.yaml",
        help="Base config path",
    )
    parser.add_argument(
        "--prover-model",
        type=str,
        default=None,
        help="Override prover model from config.",
    )
    parser.add_argument(
        "--verifier-model",
        type=str,
        default=None,
        help="Global verifier model override from config.",
    )
    parser.add_argument(
        "--small-verifier-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model for 'small' verifier variants.",
    )
    parser.add_argument(
        "--large-verifier-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for 'large' verifier variants.",
    )
    parser.add_argument(
        "--prover-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint with trained prover weights to fix prover capability.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=(
            "single_small_3b:1:supervised:small,"
            "single_large:1:supervised:large,"
            "multi_large_supervised:3:supervised:large,"
            "multi_large_pe_min:3:pe_min:large,"
            "multi_large_pe_margin:3:pe_margin:large"
        ),
        help=(
            "Comma-separated variants: id:num_verifiers:oversight_rule[:verifier_model_or_alias]. "
            "Aliases: small|large."
        ),
    )
    parser.add_argument("--buffer-episodes", type=int, default=240)
    parser.add_argument("--phase1-rounds", type=int, default=2)
    parser.add_argument("--probe-episodes", type=int, default=120)
    parser.add_argument("--attack-episodes", type=int, default=40)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/studies/multi_small_study.json")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.prover_model:
        config["model"]["prover_model"] = args.prover_model
    if args.verifier_model:
        config["model"]["verifier_model"] = args.verifier_model

    dataset_type = str(config.get("dataset", {}).get("type", "math")).lower()

    dataset = get_dataset(config)
    model_cfg = config["model"]
    print(f"Prover model: {model_cfg['prover_model']}")
    print(f"Verifier model: {model_cfg['verifier_model']}")

    print("Creating fixed prover...")
    prover = Prover(
        model_name=model_cfg["prover_model"],
        use_quantization=bool(model_cfg.get("use_quantization", True)),
        lora_r=int(model_cfg.get("prover_lora_r", 8)),
        lora_alpha=int(model_cfg.get("prover_lora_alpha", 16)),
    )
    if args.prover_checkpoint:
        print(f"Loading prover checkpoint: {args.prover_checkpoint}")
        load_checkpoint(args.prover_checkpoint, ensemble=None, prover=prover, strict=False)

    print(f"Building fixed replay buffer ({args.buffer_episodes} episodes)...")
    replay_buffer = build_fixed_replay_buffer(
        prover=prover,
        dataset=dataset,
        config=config,
        dataset_type=dataset_type,
        num_episodes=args.buffer_episodes,
        balanced_roles=True,
    )
    print(f"Fixed replay buffer size: {len(replay_buffer)}")

    variants = _parse_variants(args.variants)
    variant_results = []
    for vid, n, rule, vmodel in variants:
        resolved_vmodel = vmodel
        if vmodel == "small":
            resolved_vmodel = args.small_verifier_model
        elif vmodel == "large":
            resolved_vmodel = args.large_verifier_model
        vr = run_variant(
            variant_id=vid,
            num_verifiers=n,
            oversight_rule=rule,
            verifier_model=resolved_vmodel,
            base_config=config,
            replay_buffer=replay_buffer,
            prover=prover,
            dataset=dataset,
            dataset_type=dataset_type,
            phase1_rounds=args.phase1_rounds,
            probe_episodes=args.probe_episodes,
            attack_episodes=args.attack_episodes,
            temperatures=args.temperatures,
        )
        variant_results.append(vr)

    table_rows = []
    for vr in variant_results:
        rates = vr["probe_rates"]
        oq = vr["oversight_quality"]
        fool_key = next((k for k in rates if k.startswith("sneaky_fool_rate@")), None)
        min_v_acc = min(vr["per_verifier_accuracy"].values()) if vr["per_verifier_accuracy"] else None
        table_rows.append(
            {
                "variant_id": vr["variant_id"],
                "num_verifiers": vr["num_verifiers"],
                "oversight_rule": vr["oversight_rule"],
                "verifier_model": vr.get("verifier_model"),
                "helpful_correctness": rates.get("helpful_correctness"),
                "sneaky_incorrect_rate": rates.get("sneaky_incorrect_rate"),
                "sneaky_fool_rate": rates.get(fool_key) if fool_key else None,
                "binary_accuracy": oq.get("binary_accuracy"),
                "separation": oq.get("separation"),
                "n_correct": oq.get("n_correct"),
                "n_incorrect": oq.get("n_incorrect"),
                "mean_phase1_loss": _safe_mean(vr.get("phase1_losses", [])),
                "min_verifier_accuracy": min_v_acc,
                "max_attack_fool_rate": vr.get("max_attack_fool_rate"),
            }
        )

    output_payload = {
        "run_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": os.environ.get("LSB_JOBID", "local"),
            "seed": args.seed,
        },
        "base_config": args.config,
        "prover_checkpoint": args.prover_checkpoint,
        "dataset_type": dataset_type,
        "buffer_episodes": args.buffer_episodes,
        "phase1_rounds": args.phase1_rounds,
        "probe_episodes": args.probe_episodes,
        "attack_episodes": args.attack_episodes,
        "variants": variant_results,
        "summary_table": table_rows,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_payload, f, indent=2)
    csv_out = str(Path(args.output).with_suffix(".csv"))
    write_csv_summary(csv_out, table_rows)

    print("\nSummary table:")
    for row in table_rows:
        print(
            f"  {row['variant_id']}: helpful={row['helpful_correctness']:.3f}, "
            f"sneaky_fool={row['sneaky_fool_rate']:.3f} "
            f"binary={row['binary_accuracy']:.3f}, n_correct={row['n_correct']}"
        )
    print(f"\nSaved study JSON: {args.output}")
    print(f"Saved summary CSV: {csv_out}")

    prover.delete()
    cleanup_memory()


if __name__ == "__main__":
    main()
