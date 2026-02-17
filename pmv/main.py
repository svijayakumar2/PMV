"""
Main training loop for PMV Game (Algorithm 1).

Key structural decisions from the paper:
  - Verifiers and aggregator PERSIST across rounds (Section 6.4)
  - Prover RESETS each round (Section 6.3, Phase 2)
  - Replay buffer accumulates across rounds
  - Each round: Phase 1 (Nash update) then Phase 2 (Stackelberg PPO)
"""

import sys
import os
import json
import random
import torch
import yaml
from pathlib import Path

from pmv.data import ReplayBuffer, SolutionRecord, MathDataset
from pmv.ensemble import VerifierEnsemble
from pmv.checkpointing import save_checkpoint
from pmv.prover import (
    Prover,
    create_role_prompt,
    ensure_final_line,
)
from pmv.training import (
    train_phase1, collect_prover_experiences, train_prover_ppo,
)
from pmv.utils import get_available_gpus, cleanup_memory


def get_dataset(config: dict):
    """Instantiate dataset based on config. Defaults to MathDataset."""
    ds_cfg = config.get("dataset", {})
    ds_type = ds_cfg.get("type", "math")

    if ds_type == "zebra":
        from pmv.zebra_logic import ZebraLogicDataset
        return ZebraLogicDataset(
            max_size=ds_cfg.get("max_size"),
            min_size=ds_cfg.get("min_size"),
        )
    else:
        return MathDataset(
            num_samples=ds_cfg.get("num_samples", 100000),
        )


def _build_oracle_response(dataset, solution_true: str):
    """
    Construct a response string that the dataset checker accepts as correct.
    Returns None if no candidate can be verified as correct.
    """
    candidates = []

    # Generic answer-first candidates.
    candidates.append(f"FINAL: {solution_true}")
    candidates.append(str(solution_true))

    # ZebraLogic often provides a JSON object with `rows`.
    try:
        parsed = json.loads(solution_true)
        if isinstance(parsed, dict) and "rows" in parsed:
            rows_json = json.dumps(parsed["rows"])
            candidates.insert(0, f"FINAL: {rows_json}")
            candidates.insert(1, rows_json)
    except Exception:
        pass

    for cand in candidates:
        text = ensure_final_line(cand)
        try:
            if dataset.check_solution(solution_true, text):
                return text
        except Exception:
            continue
    return None


def main(config_path: str = "configs/config.yaml"):
    cleanup_memory()

    print("=" * 80)
    print("PMV GAME - Prover Multi-Verifier")
    print("=" * 80)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]
    sneaky_fool_threshold = float(train_cfg.get("sneaky_fool_threshold", 0.5))
    print(f"Oversight rule: {train_cfg.get('oversight_rule', 'supervised')}")

    dataset = get_dataset(config)
    save_checkpoints = bool(train_cfg.get("save_checkpoints", True))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 1))
    checkpoint_root = train_cfg.get("checkpoint_root", "results/checkpoints")
    job_tag = os.environ.get("LSB_JOBID", "local")
    config_stem = Path(config_path).stem
    run_ckpt_dir = Path(checkpoint_root) / f"{config_stem}_{job_tag}"
    latest_alias = Path(checkpoint_root) / f"{config_stem}_latest.pt"
    if save_checkpoints:
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
        Path(checkpoint_root).mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint dir: {run_ckpt_dir}")

    # Create ensemble (verifiers + aggregator persist across rounds)
    ensemble = VerifierEnsemble(config)
    print("\nCreating verifiers (persist across rounds)...")
    ensemble.create_verifiers()

    replay_buffer = ReplayBuffer(
        max_size=int(train_cfg.get("max_buffer_size", 5000))
    )

    # ---- Bootstrap (Section 6.5) -----------------------------------------
    print(f"\n{'='*60}")
    print("BOOTSTRAP PHASE (Round 0)")
    print(f"{'='*60}")

    prover = Prover(
        model_name=model_cfg["prover_model"],
        use_quantization=model_cfg.get("use_quantization", True),
        lora_r=model_cfg.get("prover_lora_r", 8),
        lora_alpha=model_cfg.get("prover_lora_alpha", 16),
    )

    num_bootstrap = int(train_cfg.get("bootstrap_episodes", 50))
    bootstrap_oracle_episodes = int(train_cfg.get("bootstrap_oracle_episodes", 0))
    mu_0 = float(train_cfg.get("mu_0", 0.5))
    print(f"\nGenerating {num_bootstrap} bootstrap solutions...")

    for i in range(num_bootstrap):
        role = "helpful" if random.random() < mu_0 else "sneaky"
        problem, solution_true = dataset.sample()
        prompt = create_role_prompt(problem, role, prover.tokenizer)
        with torch.no_grad():
            response = prover.generate(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        response = ensure_final_line(response)
        is_correct = dataset.check_solution(solution_true, response)
        replay_buffer.add(SolutionRecord(
            problem=problem, solution_true=solution_true, response=response,
            correctness=1.0 if is_correct else 0.0, role=role, round_generated=0,
        ))
        if (i + 1) % 20 == 0:
            print(f"  Bootstrap {i+1}/{num_bootstrap}")

    if bootstrap_oracle_episodes > 0:
        print(f"Adding {bootstrap_oracle_episodes} oracle-correct bootstrap records...")
        added = 0
        for i in range(bootstrap_oracle_episodes):
            problem, solution_true = dataset.sample()
            oracle_response = _build_oracle_response(dataset, solution_true)
            if oracle_response is None:
                continue
            replay_buffer.add(SolutionRecord(
                problem=problem,
                solution_true=solution_true,
                response=oracle_response,
                correctness=1.0,
                role="helpful",
                round_generated=0,
            ))
            added += 1
            if (i + 1) % 20 == 0:
                print(f"  Oracle bootstrap {i+1}/{bootstrap_oracle_episodes}")
        print(f"Oracle bootstrap added: {added}/{bootstrap_oracle_episodes}")

    prover.delete()
    del prover
    cleanup_memory()
    print(f"\nBootstrap complete. Buffer size: {len(replay_buffer)}")

    # ---- Main training loop (Algorithm 1) --------------------------------
    num_rounds = int(train_cfg.get("rounds", 10))
    print(f"\n{'='*80}")
    print(f"MAIN TRAINING: {num_rounds} rounds")
    print(f"{'='*80}")

    for round_idx in range(1, num_rounds + 1):
        print(f"\n{'='*80}")
        print(f"ROUND {round_idx}/{num_rounds}")
        print(f"{'='*80}")

        # Phase 1: Nash update (verifiers + aggregator, Algorithm 3)
        loss = train_phase1(ensemble, replay_buffer, config, round_idx)
        print(f"Phase 1 complete. Oversight loss: {loss:.4f}")
        ensemble.freeze_all()

        # Phase 2: Stackelberg (fresh prover, Algorithm 2)
        print("\nCreating fresh prover (reset per Section 6.3)...")
        prover = Prover(
            model_name=model_cfg["prover_model"],
            use_quantization=model_cfg.get("use_quantization", True),
            lora_r=model_cfg.get("prover_lora_r", 8),
            lora_alpha=model_cfg.get("prover_lora_alpha", 16),
        )
        use_reference_model = bool(train_cfg.get("use_reference_model", False))
        kl_coeff = float(train_cfg.get("kl_coeff", 0.1))
        base_prover = None
        if use_reference_model and kl_coeff > 0:
            print("Creating PPO reference prover...")
            base_prover = Prover(
                model_name=model_cfg["prover_model"],
                use_quantization=model_cfg.get("use_quantization", True),
                lora_r=model_cfg.get("prover_lora_r", 8),
                lora_alpha=model_cfg.get("prover_lora_alpha", 16),
            )
            base_prover.model.eval()
            for p in base_prover.model.parameters():
                p.requires_grad = False
        else:
            print("PPO reference prover disabled (lower memory mode).")

        prompts, responses, rewards, new_records = collect_prover_experiences(
            prover, ensemble, dataset, config, round_idx,
        )
        if rewards:
            avg_r = sum(rewards) / len(rewards)
            helpful_correct = sum(
                1 for rec in new_records
                if rec.role == "helpful" and rec.correctness == 1.0
            )
            helpful_total = sum(1 for rec in new_records if rec.role == "helpful")
            sneaky_fooled = sum(
                1 for rec in new_records
                if rec.role == "sneaky" and rec.correctness == 0.0
                and rec.oversight_score is not None
                and rec.oversight_score >= sneaky_fool_threshold
            )
            sneaky_total = sum(1 for rec in new_records if rec.role == "sneaky")

            print(f"Mean reward: {avg_r:.4f}")
            if helpful_total > 0:
                print(f"Helpful correctness: {helpful_correct}/{helpful_total}")
            if sneaky_total > 0:
                print(
                    f"Sneaky fool rate@{sneaky_fool_threshold:.2f}: "
                    f"{sneaky_fooled}/{sneaky_total}"
                )

        train_prover_ppo(
            prover=prover,
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            config=config,
            base_prover=base_prover,
        )
        replay_buffer.add_batch(new_records)
        print(f"Buffer size: {len(replay_buffer)}")

        if save_checkpoints and (round_idx % max(1, checkpoint_every) == 0):
            ckpt_path = run_ckpt_dir / f"round_{round_idx:03d}.pt"
            save_checkpoint(
                path=str(ckpt_path),
                round_idx=round_idx,
                ensemble=ensemble,
                prover=prover,
                config_path=config_path,
                extra={
                    "round_metrics": {
                        "oversight_loss": float(loss),
                        "mean_reward": float(sum(rewards) / len(rewards)) if rewards else None,
                    }
                },
            )
            # Keep a stable path for evaluation jobs.
            save_checkpoint(
                path=str(latest_alias),
                round_idx=round_idx,
                ensemble=ensemble,
                prover=prover,
                config_path=config_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")
            print(f"Updated latest checkpoint: {latest_alias}")

        prover.delete()
        if base_prover is not None:
            base_prover.delete()
            del base_prover
        del prover
        cleanup_memory()

    # Final cleanup
    ensemble.delete_verifiers()
    cleanup_memory()
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    main(config_path)
