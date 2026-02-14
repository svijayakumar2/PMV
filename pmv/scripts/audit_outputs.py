#!/usr/bin/env python3
"""
Audit helpful vs sneaky prover outputs with per-example traces.

Purpose:
  - Inspect whether sneaky behavior is genuinely adversarial or trivially wrong.
  - Separate "model was sneaky" from "post-processing forced wrong answer".
  - Produce both machine-readable JSON and a human-readable markdown report.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

from pmv.ensemble import VerifierEnsemble
from pmv.prover import Prover, create_role_prompt, enforce_sneaky_incorrect, ensure_final_line
from pmv.utils import cleanup_memory


def _short(txt: str, n: int = 800) -> str:
    t = txt.strip().replace("\r\n", "\n")
    if len(t) <= n:
        return t
    return t[: n - 3] + "..."


def _safe_rate(a: int, b: int) -> float:
    return a / max(1, b)


def _contains_trivial_wrong_marker(text: str) -> bool:
    t = text.upper()
    return "FINAL: WRONG" in t or "WRONG" == t.strip() or "FINAL: UNKNOWN" in t


def _compute_scores(
    ensemble: VerifierEnsemble,
    problem: str,
    response: str,
    debate_rounds: int,
) -> Tuple[List[float], float]:
    if debate_rounds > 0:
        state = ensemble.run_debate(problem, response, num_rounds=debate_rounds)
        return state.verifier_scores or [0.5] * ensemble.num_verifiers, float(state.aggregated_score or 0.5)
    scores, f = ensemble.compute_oversight_score(problem, response, transcript="")
    return scores, float(f)


def main():
    parser = argparse.ArgumentParser(description="Audit helpful vs sneaky outputs.")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--balanced-roles", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debate-rounds", type=int, default=None)
    parser.add_argument("--dataset", choices=["auto", "math", "zebra"], default="auto")
    parser.add_argument("--zebra-max-size", type=str, default=None)
    parser.add_argument("--fool-threshold", type=float, default=None)
    parser.add_argument("--out-json", type=str, default="results/output_audit.json")
    parser.add_argument("--out-md", type=str, default="results/output_audit.md")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    debate_rounds = int(args.debate_rounds) if args.debate_rounds is not None else int(train_cfg.get("debate_rounds", 0))
    fool_threshold = float(args.fool_threshold) if args.fool_threshold is not None else float(train_cfg.get("sneaky_fool_threshold", 0.5))
    helpful_do_sample = bool(train_cfg.get("helpful_do_sample", False))
    sneaky_do_sample = bool(train_cfg.get("sneaky_do_sample", True))
    helpful_temperature = float(train_cfg.get("helpful_temperature", 0.2))
    sneaky_temperature = float(train_cfg.get("sneaky_temperature", 0.9))
    helpful_max_new_tokens = int(train_cfg.get("helpful_max_new_tokens", 768))
    sneaky_max_new_tokens = int(train_cfg.get("sneaky_max_new_tokens", 768))

    # Dataset selection
    if args.dataset == "zebra":
        from pmv.zebra_logic import ZebraLogicDataset
        dataset = ZebraLogicDataset(max_size=args.zebra_max_size)
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

    records: List[Dict] = []
    helpful_total = helpful_correct = 0
    sneaky_total = 0
    sneaky_raw_incorrect = sneaky_raw_fooled = 0
    sneaky_post_incorrect = sneaky_post_fooled = 0
    sneaky_forced_flip = 0
    sneaky_trivial_marker = 0

    for ep in range(args.episodes):
        role = "helpful" if (ep % 2 == 0 or not args.balanced_roles and random.random() < float(train_cfg.get("mu_0", 0.5))) else "sneaky"
        problem, solution_true = dataset.sample()
        prompt = create_role_prompt(problem, role, prover.tokenizer)
        do_sample = helpful_do_sample if role == "helpful" else sneaky_do_sample
        temperature = helpful_temperature if role == "helpful" else sneaky_temperature
        max_new_tokens = helpful_max_new_tokens if role == "helpful" else sneaky_max_new_tokens

        with torch.no_grad():
            if do_sample:
                raw_response = prover.generate(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
            else:
                raw_response = prover.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        if raw_response.startswith(prompt):
            raw_response = raw_response[len(prompt):].strip()
        raw_response = ensure_final_line(raw_response)
        raw_correct = bool(dataset.check_solution(solution_true, raw_response))
        raw_scores, raw_f = _compute_scores(ensemble, problem, raw_response, debate_rounds)

        final_response = raw_response
        if role == "sneaky":
            final_response = enforce_sneaky_incorrect(raw_response, solution_true, dataset)
            final_response = ensure_final_line(final_response)
        final_correct = bool(dataset.check_solution(solution_true, final_response))
        final_scores, final_f = _compute_scores(ensemble, problem, final_response, debate_rounds)

        rec = {
            "episode": ep + 1,
            "role": role,
            "problem": problem,
            "solution_true": solution_true,
            "raw_response": raw_response,
            "raw_correct": raw_correct,
            "raw_verifier_scores": raw_scores,
            "raw_oversight_score": raw_f,
            "final_response": final_response,
            "final_correct": final_correct,
            "final_verifier_scores": final_scores,
            "final_oversight_score": final_f,
            "forced_change": final_response != raw_response,
            "trivial_wrong_marker": _contains_trivial_wrong_marker(final_response),
        }
        records.append(rec)

        if role == "helpful":
            helpful_total += 1
            helpful_correct += int(final_correct)
        else:
            sneaky_total += 1
            sneaky_raw_incorrect += int(not raw_correct)
            sneaky_post_incorrect += int(not final_correct)
            sneaky_raw_fooled += int((not raw_correct) and raw_f >= fool_threshold)
            sneaky_post_fooled += int((not final_correct) and final_f >= fool_threshold)
            sneaky_forced_flip += int(raw_correct and not final_correct)
            sneaky_trivial_marker += int(_contains_trivial_wrong_marker(final_response))

        if (ep + 1) % 10 == 0:
            print(f"  audited {ep+1}/{args.episodes}")
        cleanup_memory()

    summary = {
        "config": args.config,
        "episodes": args.episodes,
        "debate_rounds": debate_rounds,
        "fool_threshold": fool_threshold,
        "helpful_correctness": _safe_rate(helpful_correct, helpful_total),
        "counts": {
            "helpful_total": helpful_total,
            "helpful_correct": helpful_correct,
            "sneaky_total": sneaky_total,
            "sneaky_raw_incorrect": sneaky_raw_incorrect,
            "sneaky_post_incorrect": sneaky_post_incorrect,
            "sneaky_raw_fooled": sneaky_raw_fooled,
            "sneaky_post_fooled": sneaky_post_fooled,
            "sneaky_forced_flip": sneaky_forced_flip,
            "sneaky_trivial_marker": sneaky_trivial_marker,
        },
        "rates": {
            "sneaky_raw_incorrect_rate": _safe_rate(sneaky_raw_incorrect, sneaky_total),
            "sneaky_post_incorrect_rate": _safe_rate(sneaky_post_incorrect, sneaky_total),
            f"sneaky_raw_fool_rate@{fool_threshold:.2f}": _safe_rate(sneaky_raw_fooled, sneaky_total),
            f"sneaky_post_fool_rate@{fool_threshold:.2f}": _safe_rate(sneaky_post_fooled, sneaky_total),
            "sneaky_forced_flip_rate": _safe_rate(sneaky_forced_flip, sneaky_total),
            "sneaky_trivial_marker_rate": _safe_rate(sneaky_trivial_marker, sneaky_total),
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps({"summary": summary, "records": records}, indent=2))

    sneaky_raw_candidates = [r for r in records if r["role"] == "sneaky" and not r["raw_correct"]]
    sneaky_raw_candidates.sort(key=lambda x: x["raw_oversight_score"], reverse=True)
    sneaky_post_candidates = [r for r in records if r["role"] == "sneaky" and not r["final_correct"]]
    sneaky_post_candidates.sort(key=lambda x: x["final_oversight_score"], reverse=True)
    helpful_good = [r for r in records if r["role"] == "helpful" and r["final_correct"]]
    helpful_good.sort(key=lambda x: x["final_oversight_score"], reverse=True)

    md = [
        "# Prover Output Audit",
        "",
        "## Summary",
        f"- Config: `{args.config}`",
        f"- Episodes: {args.episodes}",
        f"- Debate rounds: {debate_rounds}",
        f"- Helpful correctness: {summary['helpful_correctness']:.3f}",
        f"- Sneaky raw incorrect rate: {summary['rates']['sneaky_raw_incorrect_rate']:.3f}",
        f"- Sneaky post-enforce incorrect rate: {summary['rates']['sneaky_post_incorrect_rate']:.3f}",
        f"- Sneaky raw fool rate@{fool_threshold:.2f}: {summary['rates'][f'sneaky_raw_fool_rate@{fool_threshold:.2f}']:.3f}",
        f"- Sneaky post-enforce fool rate@{fool_threshold:.2f}: {summary['rates'][f'sneaky_post_fool_rate@{fool_threshold:.2f}']:.3f}",
        f"- Sneaky forced flip rate: {summary['rates']['sneaky_forced_flip_rate']:.3f}",
        f"- Sneaky trivial marker rate: {summary['rates']['sneaky_trivial_marker_rate']:.3f}",
        "",
        "## Top Sneaky Raw (High Oversight But Incorrect)",
    ]

    for r in sneaky_raw_candidates[:5]:
        md.extend([
            f"### Episode {r['episode']}",
            f"- raw oversight: {r['raw_oversight_score']:.3f}",
            f"- final oversight: {r['final_oversight_score']:.3f}",
            f"- forced change: {r['forced_change']}",
            f"- trivial marker: {r['trivial_wrong_marker']}",
            "```text",
            _short(r["raw_response"]),
            "```",
        ])

    md.append("## Top Sneaky Final (Post-enforce)")
    for r in sneaky_post_candidates[:5]:
        md.extend([
            f"### Episode {r['episode']}",
            f"- final oversight: {r['final_oversight_score']:.3f}",
            f"- forced change: {r['forced_change']}",
            f"- trivial marker: {r['trivial_wrong_marker']}",
            "```text",
            _short(r["final_response"]),
            "```",
        ])

    md.append("## Top Helpful Correct")
    for r in helpful_good[:5]:
        md.extend([
            f"### Episode {r['episode']}",
            f"- final oversight: {r['final_oversight_score']:.3f}",
            "```text",
            _short(r["final_response"]),
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
