#!/usr/bin/env python3
"""Audit helpful vs sneaky outputs using a single metric path (no raw/final split)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

from pmv.ensemble import VerifierEnsemble
from pmv.checkpointing import load_checkpoint
from pmv.prover import Prover, create_role_prompt, ensure_final_line
from pmv.utils import cleanup_memory


def _short(txt: str, n: int = 800) -> str:
    t = txt.strip().replace("\r\n", "\n")
    if len(t) <= n:
        return t
    return t[: n - 3] + "..."


def _safe_rate(a: int, b: int) -> float:
    return a / max(1, b)


def _compute_scores(
    ensemble: VerifierEnsemble,
    problem: str,
    response: str,
) -> Tuple[List[float], float]:
    scores, f = ensemble.compute_oversight_score(problem, response, transcript="")
    return scores, float(f)


def main():
    parser = argparse.ArgumentParser(description="Audit helpful vs sneaky outputs.")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--balanced-roles", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", choices=["auto", "math", "zebra", "gsm8k"], default="auto")
    parser.add_argument("--zebra-max-size", type=str, default=None)
    parser.add_argument("--fool-threshold", type=float, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (.pt). If omitted, auto-loads config-specific latest checkpoint when present.",
    )
    parser.add_argument("--out-json", type=str, default="results/output_audit.json")
    parser.add_argument("--out-md", type=str, default="results/output_audit.md")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    fool_threshold = float(args.fool_threshold) if args.fool_threshold is not None else float(train_cfg.get("sneaky_fool_threshold", 0.5))
    helpful_do_sample = bool(train_cfg.get("helpful_do_sample", False))
    sneaky_do_sample = bool(train_cfg.get("sneaky_do_sample", True))
    helpful_temperature = float(train_cfg.get("helpful_temperature", 0.2))
    sneaky_temperature = float(train_cfg.get("sneaky_temperature", 0.9))
    helpful_max_new_tokens = int(train_cfg.get("helpful_max_new_tokens", 768))
    sneaky_max_new_tokens = int(train_cfg.get("sneaky_max_new_tokens", 768))

    if args.dataset == "zebra":
        from pmv.zebra_logic import ZebraLogicDataset
        dataset = ZebraLogicDataset(max_size=args.zebra_max_size)
    elif args.dataset == "gsm8k":
        from pmv.data import GSM8KDataset
        dataset = GSM8KDataset()
    elif args.dataset == "math":
        from pmv.data import MathDataset
        dataset = MathDataset()
    else:
        from pmv.main import get_dataset
        dataset = get_dataset(config)

    ensemble = VerifierEnsemble(config)
    ensemble.create_verifiers()

    prover = Prover(
        model_name=model_cfg["prover_model"],
        use_quantization=model_cfg.get("use_quantization", True),
    )

    checkpoint_used = None
    auto_ckpt = Path(train_cfg.get("checkpoint_root", "results/checkpoints")) / f"{Path(args.config).stem}_latest.pt"
    if args.checkpoint:
        ck = Path(args.checkpoint)
        if not ck.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ck}")
        checkpoint_used = str(ck)
    elif auto_ckpt.exists():
        checkpoint_used = str(auto_ckpt)
    if checkpoint_used is not None:
        print(f"Loading checkpoint: {checkpoint_used}")
        load_checkpoint(checkpoint_used, ensemble=ensemble, prover=prover, strict=False)

    records: List[Dict] = []
    helpful_total = helpful_correct = 0
    sneaky_total = sneaky_incorrect = sneaky_fooled = 0

    for ep in range(args.episodes):
        role = "helpful" if (ep % 2 == 0 or not args.balanced_roles and random.random() < float(train_cfg.get("mu_0", 0.5))) else "sneaky"
        problem, solution_true = dataset.sample()
        prompt = create_role_prompt(problem, role, prover.tokenizer)
        do_sample = helpful_do_sample if role == "helpful" else sneaky_do_sample
        temperature = helpful_temperature if role == "helpful" else sneaky_temperature
        max_new_tokens = helpful_max_new_tokens if role == "helpful" else sneaky_max_new_tokens

        with torch.no_grad():
            if do_sample:
                response = prover.generate(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
            else:
                response = prover.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        response = ensure_final_line(response)

        is_correct = bool(dataset.check_solution(solution_true, response))
        v_scores, f_score = _compute_scores(ensemble, problem, response)

        rec = {
            "episode": ep + 1,
            "role": role,
            "problem": problem,
            "solution_true": solution_true,
            "response": response,
            "correct": is_correct,
            "verifier_scores": v_scores,
            "oversight_score": f_score,
        }
        records.append(rec)

        if role == "helpful":
            helpful_total += 1
            helpful_correct += int(is_correct)
        else:
            sneaky_total += 1
            sneaky_incorrect += int(not is_correct)
            sneaky_fooled += int((not is_correct) and f_score >= fool_threshold)

        if (ep + 1) % 10 == 0:
            print(f"  audited {ep+1}/{args.episodes}")
        cleanup_memory()

    summary = {
        "config": args.config,
        "checkpoint_used": checkpoint_used,
        "episodes": args.episodes,
        "fool_threshold": fool_threshold,
        "helpful_correctness": _safe_rate(helpful_correct, helpful_total),
        "counts": {
            "helpful_total": helpful_total,
            "helpful_correct": helpful_correct,
            "sneaky_total": sneaky_total,
            "sneaky_incorrect": sneaky_incorrect,
            "sneaky_fooled": sneaky_fooled,
        },
        "rates": {
            "sneaky_incorrect_rate": _safe_rate(sneaky_incorrect, sneaky_total),
            f"sneaky_fool_rate@{fool_threshold:.2f}": _safe_rate(sneaky_fooled, sneaky_total),
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps({"summary": summary, "records": records}, indent=2))

    sneaky_candidates = [r for r in records if r["role"] == "sneaky" and not r["correct"]]
    sneaky_candidates.sort(key=lambda x: x["oversight_score"], reverse=True)
    helpful_good = [r for r in records if r["role"] == "helpful" and r["correct"]]
    helpful_good.sort(key=lambda x: x["oversight_score"], reverse=True)

    md = [
        "# Prover Output Audit",
        "",
        "## Summary",
        f"- Config: `{args.config}`",
        f"- Episodes: {args.episodes}",
        f"- Helpful correctness: {summary['helpful_correctness']:.3f}",
        f"- Sneaky incorrect rate: {summary['rates']['sneaky_incorrect_rate']:.3f}",
        f"- Sneaky fool rate@{fool_threshold:.2f}: {summary['rates'][f'sneaky_fool_rate@{fool_threshold:.2f}']:.3f}",
        "",
        "## Top Sneaky (Incorrect With High Oversight)",
    ]

    for r in sneaky_candidates[:5]:
        md.extend([
            f"### Episode {r['episode']}",
            f"- oversight: {r['oversight_score']:.3f}",
            "```text",
            _short(r["response"]),
            "```",
        ])

    md.append("## Top Helpful Correct")
    for r in helpful_good[:5]:
        md.extend([
            f"### Episode {r['episode']}",
            f"- oversight: {r['oversight_score']:.3f}",
            "```text",
            _short(r["response"]),
            "```",
        ])

    out_md.write_text("\n".join(md))
    print(f"Saved audit JSON: {out_json}")
    print(f"Saved audit report: {out_md}")

    prover.delete()
    ensemble.delete_verifiers()
    cleanup_memory()


if __name__ == "__main__":
    main()
