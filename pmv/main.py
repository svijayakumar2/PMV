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

from pmv.data import ReplayBuffer, SolutionRecord, MathDataset, GSM8KDataset
from pmv.ensemble import VerifierEnsemble
from pmv.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    save_prover_checkpoint,
    load_prover_checkpoint,
)
from pmv.prover import (
    Prover,
    create_role_prompt,
    ensure_final_line,
)
from pmv.training import (
    train_phase1, collect_prover_experiences, train_prover_ppo, train_prover_helpful_warmup,
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
    if ds_type == "gsm8k":
        return GSM8KDataset()
    else:
        return MathDataset(
            num_samples=ds_cfg.get("num_samples", 100000),
        )


def _build_oracle_response(dataset, solution_true: str, dataset_type: str = "math"):
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
        text = ensure_final_line(cand, dataset_type=dataset_type)
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
    seed = int(train_cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    sneaky_fool_threshold = float(train_cfg.get("sneaky_fool_threshold", 0.5))
    available_gpus = get_available_gpus()
    prover_device = train_cfg.get("prover_device")
    if prover_device is not None:
        prover_device = str(prover_device).strip() or None
    if prover_device is None and available_gpus:
        prover_device = "cuda:0"

    split_reference_gpu = bool(train_cfg.get("split_reference_gpu", True))
    reference_prover_device = train_cfg.get("reference_prover_device")
    if reference_prover_device is not None:
        reference_prover_device = str(reference_prover_device).strip() or None
    if (
        reference_prover_device is None
        and split_reference_gpu
        and len(available_gpus) >= 2
    ):
        if prover_device and prover_device.startswith("cuda:1"):
            reference_prover_device = "cuda:0"
        else:
            reference_prover_device = "cuda:1"

    print(f"Seed: {seed}")
    print(f"Oversight rule: {train_cfg.get('oversight_rule', 'supervised')}")
    print(
        "Device plan: "
        f"prover={prover_device or 'auto'}, "
        f"reference={reference_prover_device or prover_device or 'auto'}"
    )

    ds_type = str(config.get("dataset", {}).get("type", "math")).lower()
    dataset = get_dataset(config)
    save_checkpoints = bool(train_cfg.get("save_checkpoints", True))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 1))
    checkpoint_root = train_cfg.get("checkpoint_root", "results/checkpoints")
    resume_from_checkpoint = bool(train_cfg.get("resume_from_checkpoint", True))
    job_tag = os.environ.get("LSB_JOBID", "local")
    config_stem = Path(config_path).stem
    run_ckpt_dir = Path(checkpoint_root) / f"{config_stem}_{job_tag}"
    latest_alias = Path(checkpoint_root) / f"{config_stem}_latest.pt"
    inflight_context_alias = Path(checkpoint_root) / f"{config_stem}_inflight_context.pt"
    inflight_state_alias = Path(checkpoint_root) / f"{config_stem}_inflight_state.pt"
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

    start_round = 1
    resume_inflight_round = None
    resume_inflight_step = 0
    resume_inflight_context = None
    resumed = False

    if resume_from_checkpoint:
        if inflight_context_alias.exists() and inflight_state_alias.exists():
            print(
                f"Found inflight checkpoint pair:\n"
                f"  context={inflight_context_alias}\n"
                f"  state={inflight_state_alias}"
            )
            try:
                inflight_ctx = load_checkpoint(
                    str(inflight_context_alias),
                    ensemble=ensemble,
                    strict=False,
                )
                ctx_extra = inflight_ctx.get("extra", {}) if isinstance(inflight_ctx, dict) else {}
                rb_records = ctx_extra.get("replay_buffer_records")
                if rb_records:
                    replay_buffer.add_batch(rb_records)
                resume_inflight_round = int(inflight_ctx.get("round_idx", 1))
                resume_inflight_context = ctx_extra
                inflight_state = load_prover_checkpoint(
                    str(inflight_state_alias),
                    prover=None,
                    strict=False,
                )
                resume_inflight_step = int(
                    (inflight_state.get("extra", {}) if isinstance(inflight_state, dict) else {}).get(
                        "ppo_step_done", 0
                    )
                )
                start_round = resume_inflight_round
                resumed = True
                print(
                    f"Resuming inflight round {resume_inflight_round} from PPO step {resume_inflight_step}."
                )
            except Exception as e:
                print(f"Inflight resume load failed; starting fresh. Error: {e}")
        elif latest_alias.exists():
            print(f"Found latest completed checkpoint: {latest_alias}")
            try:
                latest = load_checkpoint(
                    str(latest_alias),
                    ensemble=ensemble,
                    strict=False,
                )
                latest_extra = latest.get("extra", {}) if isinstance(latest, dict) else {}
                rb_records = latest_extra.get("replay_buffer_records")
                if rb_records:
                    replay_buffer.add_batch(rb_records)
                    completed_round = int(latest.get("round_idx", 0))
                    start_round = completed_round + 1
                    resumed = True
                    print(f"Resuming from completed round {completed_round}; next round={start_round}.")
                else:
                    print(
                        "Latest checkpoint has no replay buffer snapshot; cannot safely resume "
                        "and will start fresh."
                    )
            except Exception as e:
                print(f"Latest checkpoint load failed; starting fresh. Error: {e}")

    num_bootstrap = int(train_cfg.get("bootstrap_episodes", 50))
    bootstrap_oracle_episodes = int(train_cfg.get("bootstrap_oracle_episodes", 0))
    bootstrap_warmup_steps = int(
        train_cfg.get("bootstrap_helpful_warmup_steps", train_cfg.get("helpful_warmup_steps", 0))
    )
    round_warmup_steps = int(
        train_cfg.get("round_helpful_warmup_steps", train_cfg.get("helpful_warmup_steps", 0))
    )
    mu_0 = float(train_cfg.get("mu_0", 0.5))

    if resumed:
        print(
            f"\nSkipping bootstrap due to resume. Restored buffer size: {len(replay_buffer)}"
        )
    else:
        # ---- Bootstrap (Section 6.5) -----------------------------------------
        print(f"\n{'='*60}")
        print("BOOTSTRAP PHASE (Round 0)")
        print(f"{'='*60}")

        prover = Prover(
            model_name=model_cfg["prover_model"],
            use_quantization=model_cfg.get("use_quantization", True),
            lora_r=model_cfg.get("prover_lora_r", 8),
            lora_alpha=model_cfg.get("prover_lora_alpha", 16),
            preferred_device=prover_device,
        )

        print(
            "Helpful warmup schedule: "
            f"bootstrap={bootstrap_warmup_steps} steps, per-round={round_warmup_steps} steps"
        )
        print(f"\nGenerating {num_bootstrap} bootstrap solutions...")

        train_prover_helpful_warmup(
            prover=prover,
            dataset=dataset,
            config=config,
            steps_override=bootstrap_warmup_steps,
        )

        for i in range(num_bootstrap):
            role = "helpful" if random.random() < mu_0 else "sneaky"
            problem, solution_true = dataset.sample()
            prompt = create_role_prompt(problem, role, prover.tokenizer, dataset_type=ds_type)
            with torch.no_grad():
                response = prover.generate(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            response = ensure_final_line(response, dataset_type=ds_type)
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
                oracle_response = _build_oracle_response(dataset, solution_true, dataset_type=ds_type)
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
    if start_round > num_rounds:
        print(
            f"Checkpoint indicates completed training (start_round={start_round} > num_rounds={num_rounds})."
        )
        ensemble.delete_verifiers()
        cleanup_memory()
        print(f"\n{'='*80}")
        print("Training complete!")
        print(f"{'='*80}")
        return

    for round_idx in range(start_round, num_rounds + 1):
        print(f"\n{'='*80}")
        print(f"ROUND {round_idx}/{num_rounds}")
        print(f"{'='*80}")
        target_updates = int(train_cfg.get("ppo_target_updates_per_round", 0))
        ppo_resume_checkpoint_interval = max(
            1, int(train_cfg.get("ppo_resume_checkpoint_interval", 200))
        )
        is_resume_inflight_round = (
            resume_inflight_round is not None
            and resume_inflight_context is not None
            and int(resume_inflight_round) == int(round_idx)
        )

        prompts = []
        responses = []
        rewards = []
        new_records = []
        ppo_resume_update_idx = 0

        if is_resume_inflight_round:
            print(
                f"Resuming inflight ROUND {round_idx}: "
                f"starting PPO from update {resume_inflight_step}."
            )
            loss = float(resume_inflight_context.get("phase1_loss", float("nan")))
            prompts = list(resume_inflight_context.get("prompts", []))
            responses = list(resume_inflight_context.get("responses", []))
            rewards = list(resume_inflight_context.get("rewards", []))
            new_records = list(resume_inflight_context.get("new_records", []))
            ppo_resume_update_idx = int(resume_inflight_step)
            if not prompts or len(prompts) != len(responses) or len(prompts) != len(rewards):
                raise RuntimeError(
                    "Inflight resume context is missing PPO data "
                    "(prompts/responses/rewards). Cannot resume safely."
                )
        else:
            # Phase 1: Nash update (verifiers + aggregator, Algorithm 3)
            loss = train_phase1(ensemble, replay_buffer, config, round_idx)
            print(f"Phase 1 complete. Oversight loss: {loss:.4f}")
            ensemble.freeze_all()

        # Phase 2: Stackelberg (fresh prover, Algorithm 2)
        if is_resume_inflight_round:
            print("\nRestoring prover from inflight checkpoint...")
        else:
            print("\nCreating fresh prover (reset per Section 6.3)...")
        prover = Prover(
            model_name=model_cfg["prover_model"],
            use_quantization=model_cfg.get("use_quantization", True),
            lora_r=model_cfg.get("prover_lora_r", 8),
            lora_alpha=model_cfg.get("prover_lora_alpha", 16),
            preferred_device=prover_device,
        )
        if is_resume_inflight_round:
            load_prover_checkpoint(str(inflight_state_alias), prover=prover, strict=False)

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
                preferred_device=(reference_prover_device or prover_device),
            )
            base_prover.model.eval()
            for p in base_prover.model.parameters():
                p.requires_grad = False
        else:
            print("PPO reference prover disabled (lower memory mode).")

        if not is_resume_inflight_round:
            train_prover_helpful_warmup(
                prover=prover,
                dataset=dataset,
                config=config,
                steps_override=round_warmup_steps,
            )

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

        if save_checkpoints and target_updates > 0:
            # Save round context once (replay snapshot + PPO batch data).
            save_checkpoint(
                path=str(inflight_context_alias),
                round_idx=round_idx,
                ensemble=ensemble,
                prover=None,
                config_path=config_path,
                extra={
                    "phase1_loss": float(loss) if loss == loss else None,  # NaN-safe
                    "replay_buffer_records": replay_buffer.get_all(),
                    "prompts": prompts,
                    "responses": responses,
                    "rewards": rewards,
                    "new_records": new_records,
                },
            )
            # Save prover state for PPO resume.
            save_prover_checkpoint(
                path=str(inflight_state_alias),
                round_idx=round_idx,
                prover=prover,
                config_path=config_path,
                extra={"ppo_step_done": int(ppo_resume_update_idx)},
            )

        progress_state = {"last_saved": int(ppo_resume_update_idx)}

        def _on_ppo_progress(step_done: int):
            if not save_checkpoints or target_updates <= 0:
                return
            if (
                step_done >= target_updates
                or (step_done - progress_state["last_saved"]) >= ppo_resume_checkpoint_interval
            ):
                save_prover_checkpoint(
                    path=str(inflight_state_alias),
                    round_idx=round_idx,
                    prover=prover,
                    config_path=config_path,
                    extra={"ppo_step_done": int(step_done)},
                )
                progress_state["last_saved"] = int(step_done)

        train_prover_ppo(
            prover=prover,
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            config=config,
            base_prover=base_prover,
            resume_update_idx=ppo_resume_update_idx,
            on_progress_update=_on_ppo_progress,
        )
        replay_buffer.add_batch(new_records)
        print(f"Buffer size: {len(replay_buffer)}")

        if inflight_context_alias.exists():
            inflight_context_alias.unlink()
        if inflight_state_alias.exists():
            inflight_state_alias.unlink()
        if is_resume_inflight_round:
            resume_inflight_round = None
            resume_inflight_context = None
            resume_inflight_step = 0

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
                    },
                    "replay_buffer_records": replay_buffer.get_all(),
                },
            )
            # Keep a stable path for evaluation jobs.
            save_checkpoint(
                path=str(latest_alias),
                round_idx=round_idx,
                ensemble=ensemble,
                prover=prover,
                config_path=config_path,
                extra={
                    "round_metrics": {
                        "oversight_loss": float(loss),
                        "mean_reward": float(sum(rewards) / len(rewards)) if rewards else None,
                    },
                    "replay_buffer_records": replay_buffer.get_all(),
                },
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
