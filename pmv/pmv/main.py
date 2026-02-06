"""
Main training loop for PMV Game (Algorithm 1).

Key structural decisions from the paper:
  - Verifiers and aggregator PERSIST across rounds (Section 6.4)
  - Prover RESETS each round (Section 6.3, Phase 2)
  - Replay buffer accumulates across rounds
  - Each round: Phase 1 (Nash update) then Phase 2 (Stackelberg PPO)
"""

import sys
import random
import torch
import yaml

from pmv.data import ReplayBuffer, SolutionRecord, MathDataset
from pmv.ensemble import VerifierEnsemble
from pmv.prover import Prover, create_role_prompt, enforce_sneaky_incorrect
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


def main(config_path: str = "configs/config.yaml"):
    cleanup_memory()

    print("=" * 80)
    print("PMV GAME - Prover Multi-Verifier with Irving-Style Debate")
    print("=" * 80)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]

    dataset = get_dataset(config)

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
        if role == "sneaky":
            response = enforce_sneaky_incorrect(response, solution_true, dataset)
        is_correct = dataset.check_solution(solution_true, response)
        replay_buffer.add(SolutionRecord(
            problem=problem, solution_true=solution_true, response=response,
            correctness=1.0 if is_correct else 0.0, role=role, round_generated=0,
        ))
        if (i + 1) % 20 == 0:
            print(f"  Bootstrap {i+1}/{num_bootstrap}")

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
        base_prover = Prover(
            model_name=model_cfg["prover_model"],
            use_quantization=model_cfg.get("use_quantization", True),
            lora_r=model_cfg.get("prover_lora_r", 8),
            lora_alpha=model_cfg.get("prover_lora_alpha", 16),
        )
        base_prover.model.eval()
        for p in base_prover.model.parameters():
            p.requires_grad = False

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
                and rec.oversight_score is not None and rec.oversight_score > 0.5
            )
            sneaky_total = sum(1 for rec in new_records if rec.role == "sneaky")

            print(f"Mean reward: {avg_r:.4f}")
            if helpful_total > 0:
                print(f"Helpful correctness: {helpful_correct}/{helpful_total}")
            if sneaky_total > 0:
                print(f"Sneaky fool rate: {sneaky_fooled}/{sneaky_total}")

        train_prover_ppo(prover, base_prover, prompts, responses, rewards, config)
        replay_buffer.add_batch(new_records)
        print(f"Buffer size: {len(replay_buffer)}")

        prover.delete()
        base_prover.delete()
        del prover, base_prover
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
