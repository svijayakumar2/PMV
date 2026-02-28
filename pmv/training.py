"""
Training procedures implementing Algorithms 1-3 from the paper.

Phase 1 (NashUpdate, Algorithm 3):
  Verifiers + aggregator train jointly on replay buffer to minimize
  oversight loss. Verifiers persist across rounds.

Phase 2 (StackelbergUpdate, Algorithm 2):
  Prover resets, generates solutions, and trains via PPO against the
  frozen oversight function.

Reward function per Equation (4).
"""

import json
import math
import os
import random
import socket
import time
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from typing import Callable, List, Tuple, Dict, Optional
from torch.nn.parallel import DistributedDataParallel as DDP

from pmv.data import SolutionRecord, ReplayBuffer
from pmv.ensemble import VerifierEnsemble
from pmv.prover import (
    Prover,
    create_role_prompt,
    ensure_final_line,
)
from pmv.oversight import OPTIMIZED_RULES, h_supervised, verifier_softmin_loss
from pmv.utils import cleanup_memory
from pmv.checkpointing import save_prover_checkpoint, load_prover_checkpoint


# ---------------------------------------------------------------------------
# Reward function (Equation 4)
# ---------------------------------------------------------------------------

def reward_function(
    f_score: float,
    role: str,
    is_correct: bool,
    reward_mode: str = "thresholded",
    alpha_1: float = 1.0, beta_1: float = 1.0,
    alpha_2: float = -0.5, beta_2: float = 0.25,
    fool_threshold: float = 0.5,
    sneaky_margin_scale: float = 1.0,
    sneaky_correct_penalty: float = -1.0,
) -> float:
    """
    Reward shaping for helpful/sneaky roles.

    Modes:
      - thresholded (default): sneaky reward depends on fooling margin (f - threshold)
      - affine: role-alignment affine reward from Eq. (4)
    """
    role = str(role).strip().lower()
    reward_mode = str(reward_mode).strip().lower()

    if reward_mode == "affine":
        role_aligned = is_correct if role == "helpful" else not is_correct
        if role_aligned:
            return alpha_1 + beta_1 * f_score
        return alpha_2 + beta_2 * f_score

    if reward_mode not in {"thresholded", "affine"}:
        raise ValueError(
            f"Unknown reward_mode={reward_mode!r}. Expected one of: thresholded, affine."
        )

    if role == "helpful":
        if is_correct:
            return alpha_1 + beta_1 * f_score
        return alpha_2 + beta_2 * f_score

    # Sneaky role:
    # - Incorrect + high oversight score should be rewarded.
    # - Incorrect + low oversight score should be penalized.
    # - Correct sneaky answers are explicitly discouraged.
    if is_correct:
        return sneaky_correct_penalty
    return sneaky_margin_scale * (f_score - fool_threshold)


def _compute_label_weights(records: List[SolutionRecord]) -> Tuple[float, float, int, int]:
    """Return (positive_weight, negative_weight, num_positive, num_negative)."""
    n_pos = sum(1 for r in records if float(r.correctness) >= 0.5)
    n_total = len(records)
    n_neg = max(0, n_total - n_pos)

    if n_pos == 0 or n_neg == 0:
        return 1.0, 1.0, n_pos, n_neg

    # Inverse-frequency balancing: each class contributes ~50% of the loss.
    pos_w = n_total / (2.0 * n_pos)
    neg_w = n_total / (2.0 * n_neg)
    return float(pos_w), float(neg_w), n_pos, n_neg


def _supervised_weighted_bce(
    oversight: torch.Tensor,
    target: torch.Tensor,
    pos_w: float,
    neg_w: float,
) -> torch.Tensor:
    sample_w = pos_w if float(target.item()) >= 0.5 else neg_w
    weight_t = torch.tensor(sample_w, dtype=torch.float32, device=target.device)
    return F.binary_cross_entropy(
        oversight.clamp(1e-6, 1 - 1e-6),
        target,
        weight=weight_t,
    )


def _build_stratified_epoch_records(
    records: List[SolutionRecord],
    batch_size: int,
) -> List[SolutionRecord]:
    """
    Build one epoch sample list with approximately balanced correct/incorrect
    examples in each batch. Falls back to shuffled records if either class is missing.
    """
    pos = [r for r in records if float(r.correctness) >= 0.5]
    neg = [r for r in records if float(r.correctness) < 0.5]
    if not pos or not neg:
        out = records.copy()
        random.shuffle(out)
        return out

    n_batches = max(1, math.ceil(len(records) / max(1, batch_size)))
    k_pos = batch_size // 2
    k_neg = batch_size - k_pos

    out: List[SolutionRecord] = []
    for _ in range(n_batches):
        out.extend(random.choices(pos, k=max(1, k_pos)))
        out.extend(random.choices(neg, k=max(1, k_neg)))
    random.shuffle(out)
    return out


def _build_helpful_target_response(
    dataset,
    solution_true: str,
    dataset_type: str = "math",
):
    """
    Build a minimal target response that checker accepts as correct.
    Returns None if no checked target could be built.
    """
    candidates = [f"FINAL: {solution_true}", str(solution_true)]
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


def train_prover_helpful_warmup(
    prover: Prover,
    dataset,
    config: Dict,
    steps_override: Optional[int] = None,
):
    """
    Warm-start helpful behavior with direct supervised teacher forcing.
    This is intentionally simple: maximize log-probability of checker-valid
    helpful targets before mixed helpful/sneaky rollouts.
    """
    train_cfg = config["training"]
    dataset_type = str(config.get("dataset", {}).get("type", "math")).lower()
    if steps_override is None:
        steps = int(train_cfg.get("helpful_warmup_steps", 0))
    else:
        steps = int(steps_override)
    if steps <= 0:
        return

    lr = float(train_cfg.get("helpful_warmup_lr", 2e-5))
    batch_size = int(train_cfg.get("helpful_warmup_batch_size", 4))
    max_length = int(train_cfg.get("helpful_warmup_max_length", 1024))

    trainable = [p for p in prover.model.parameters() if p.requires_grad]
    if not trainable:
        return
    optimizer = torch.optim.Adam(trainable, lr=lr)

    prover.model.train()
    print(f"Helpful warmup: {steps} steps (batch={batch_size}, lr={lr:g})")

    updates = 0
    for step in range(steps):
        optimizer.zero_grad()
        losses = []

        for _ in range(batch_size):
            problem, solution_true = dataset.sample()
            target = _build_helpful_target_response(
                dataset,
                solution_true,
                dataset_type=dataset_type,
            )
            if target is None:
                continue

            prompt = create_role_prompt(
                problem,
                "helpful",
                prover.tokenizer,
                dataset_type=dataset_type,
            )
            full = prompt + target

            inp = prover.tokenizer(
                full, return_tensors="pt", truncation=True, max_length=max_length
            )
            inp = {k: v.to(prover.device) for k, v in inp.items()}
            p_ids = prover.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length
            )
            plen = p_ids["input_ids"].shape[1]

            logits = prover.model(**inp, use_cache=False).logits[0]
            if logits.shape[0] <= plen:
                continue

            rl = logits[plen - 1:-1]
            rt = inp["input_ids"][0, plen:]
            ml = min(rl.shape[0], rt.shape[0])
            if ml == 0:
                continue

            tlp = torch.log_softmax(rl[:ml].float(), dim=-1)
            sel = tlp.gather(1, rt[:ml].unsqueeze(1)).squeeze(1)
            nll = -sel.mean()
            losses.append(nll)

        if not losses:
            continue

        loss = torch.stack(losses).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        updates += 1

        if (step + 1) % 10 == 0:
            print(f"  Warmup step {step+1}/{steps}, loss: {loss.item():.4f}")
            cleanup_memory()

    print(f"Helpful warmup complete. Updates: {updates}/{steps}")


# ---------------------------------------------------------------------------
# Phase 1: NashUpdate (Algorithm 3) + aggregator joint training
# ---------------------------------------------------------------------------

def train_phase1(
    ensemble: VerifierEnsemble,
    replay_buffer: ReplayBuffer,
    config: Dict,
    round_idx: int,
) -> float:
    """
    Train verifiers and aggregator jointly on replay buffer (Equation 3).

    Verifiers persist across rounds. The oversight rule determines the
    loss function H used for training.
    """
    if len(replay_buffer) < 10:
        return float("inf")

    train_cfg = config["training"]
    lr = float(train_cfg.get("verifier_lr", 1e-4))
    epochs = int(train_cfg.get("verifier_epochs", 3))
    batch_size = min(int(train_cfg.get("verifier_batch_size", 8)), len(replay_buffer))
    oversight_rule = train_cfg.get("oversight_rule", "supervised")
    dropout_p = float(train_cfg.get("verifier_dropout", 0.0))
    softmin_beta = float(train_cfg.get("softmin_beta", 5.0))
    micro_step = bool(train_cfg.get("phase1_micro_step", True))
    balance_labels = bool(train_cfg.get("phase1_balance_labels", True))
    stratified_batches = bool(train_cfg.get("phase1_stratified_batches", True))
    separation_lambda = float(train_cfg.get("phase1_separation_lambda", 0.0))
    separation_margin = float(train_cfg.get("phase1_separation_margin", 0.1))
    separation_min_per_class = int(train_cfg.get("phase1_separation_min_per_class", 1))
    use_separation_loss = (
        oversight_rule == "supervised"
        and separation_lambda > 0.0
    )

    print(f"\n{'─'*60}")
    print(f"PHASE 1: NASH UPDATE (oversight_rule={oversight_rule})")
    print(f"Training on {len(replay_buffer)} buffered solutions")
    print(f"{'─'*60}")

    ensemble.unfreeze_all()
    all_params = ensemble.get_all_trainable_params()
    if not all_params:
        print("  No trainable parameters.")
        return float("inf")

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

    records = replay_buffer.get_all()
    pos_w, neg_w, n_pos, n_neg = _compute_label_weights(records)
    use_stratified = oversight_rule == "supervised" and stratified_batches and n_pos > 0 and n_neg > 0
    use_weighted_bce = oversight_rule == "supervised" and balance_labels and not use_stratified

    if oversight_rule == "supervised" and use_stratified:
        print(f"  Stratified batches enabled (positives={n_pos}, negatives={n_neg})")
    if oversight_rule == "supervised" and use_weighted_bce:
        print(
            f"  Label balance: positives={n_pos}, negatives={n_neg}, "
            f"weights(pos/neg)=({pos_w:.3f}/{neg_w:.3f})"
        )
    if use_separation_loss:
        print(
            f"  Separation loss enabled: lambda={separation_lambda:g}, "
            f"margin={separation_margin:g}, min_per_class={separation_min_per_class}"
        )
        if micro_step:
            print("  Note: separation loss is applied in batched mode; disable phase1_micro_step for effect.")
    final_loss = float("inf")

    for epoch in range(epochs):
        if use_stratified:
            epoch_records = _build_stratified_epoch_records(records, batch_size)
        else:
            epoch_records = records.copy()
            random.shuffle(epoch_records)
        epoch_loss, n_batches = 0.0, 0

        for bstart in range(0, len(epoch_records), batch_size):
            batch = epoch_records[bstart:bstart + batch_size]

            if micro_step:
                for rec in batch:
                    optimizer.zero_grad()
                    try:
                        score_tensors, oversight = ensemble.compute_oversight_score_training(
                            rec.problem, rec.response, "",
                        )
                        target = torch.tensor(
                            rec.correctness, device=ensemble.aggregator_device,
                            dtype=torch.float32,
                        )

                        # Choose H based on oversight rule
                        if oversight_rule in ("pe_min", "pe_margin"):
                            v_scores = torch.stack(score_tensors).unsqueeze(0)  # [1, m]
                            f_out = oversight.unsqueeze(0)                      # [1]
                            h_fn = OPTIMIZED_RULES[oversight_rule]
                            loss = h_fn(f_out, v_scores, target)
                        elif oversight_rule == "softmin":
                            # Per-verifier losses for SoftMin (Appendix D)
                            per_v = []
                            for s in score_tensors:
                                per_v.append(F.binary_cross_entropy(
                                    s.clamp(1e-6, 1 - 1e-6), target,
                                ))
                            v_losses = torch.stack(per_v).unsqueeze(0)  # [1, m]
                            if dropout_p > 0:
                                keep_prob = 1.0 - dropout_p
                                mask = torch.bernoulli(
                                    torch.full_like(v_losses, keep_prob)
                                )
                                # Guard against all-dropped rows.
                                if mask.sum(dim=-1).min().item() == 0:
                                    ridx = random.randrange(ensemble.num_verifiers)
                                    mask[:, ridx] = 1.0
                            else:
                                mask = torch.ones_like(v_losses)
                            loss = verifier_softmin_loss(v_losses, mask, softmin_beta)
                        else:
                            # Default: supervised BCE on aggregator output
                            if use_weighted_bce:
                                loss = _supervised_weighted_bce(
                                    oversight=oversight,
                                    target=target,
                                    pos_w=pos_w,
                                    neg_w=neg_w,
                                )
                            else:
                                loss = h_supervised(oversight, None, target)

                        if torch.isnan(loss) or torch.isinf(loss):
                            optimizer.zero_grad()
                            continue
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                        optimizer.step()
                        epoch_loss += loss.item()
                        n_batches += 1
                    except Exception as e:
                        print(f"  Sample error: {e}")
                        optimizer.zero_grad()
                        continue
                    if n_batches % 5 == 0:
                        cleanup_memory()
            else:
                optimizer.zero_grad()
                batch_losses = []
                batch_oversights = []
                batch_targets = []
                for rec in batch:
                    try:
                        score_tensors, oversight = ensemble.compute_oversight_score_training(
                            rec.problem, rec.response, "",
                        )
                        target = torch.tensor(
                            rec.correctness, device=ensemble.aggregator_device,
                            dtype=torch.float32,
                        )

                        if oversight_rule in ("pe_min", "pe_margin"):
                            v_scores = torch.stack(score_tensors).unsqueeze(0)
                            f_out = oversight.unsqueeze(0)
                            h_fn = OPTIMIZED_RULES[oversight_rule]
                            loss = h_fn(f_out, v_scores, target)
                        elif oversight_rule == "softmin":
                            per_v = []
                            for s in score_tensors:
                                per_v.append(F.binary_cross_entropy(
                                    s.clamp(1e-6, 1 - 1e-6), target,
                                ))
                            v_losses = torch.stack(per_v).unsqueeze(0)
                            if dropout_p > 0:
                                keep_prob = 1.0 - dropout_p
                                mask = torch.bernoulli(
                                    torch.full_like(v_losses, keep_prob)
                                )
                                if mask.sum(dim=-1).min().item() == 0:
                                    ridx = random.randrange(ensemble.num_verifiers)
                                    mask[:, ridx] = 1.0
                            else:
                                mask = torch.ones_like(v_losses)
                            loss = verifier_softmin_loss(v_losses, mask, softmin_beta)
                        else:
                            if use_weighted_bce:
                                loss = _supervised_weighted_bce(
                                    oversight=oversight,
                                    target=target,
                                    pos_w=pos_w,
                                    neg_w=neg_w,
                                )
                            else:
                                loss = h_supervised(oversight, None, target)

                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            batch_losses.append(loss)
                            if use_separation_loss:
                                batch_oversights.append(oversight)
                                batch_targets.append(target)
                    except Exception as e:
                        print(f"  Sample error: {e}")
                        continue

                if not batch_losses:
                    continue
                total_loss = torch.stack(batch_losses).mean()
                if use_separation_loss and batch_oversights:
                    overs = torch.stack(batch_oversights)
                    targs = torch.stack(batch_targets)
                    pos_mask = targs >= 0.5
                    neg_mask = targs < 0.5
                    if (
                        int(pos_mask.sum().item()) >= separation_min_per_class
                        and int(neg_mask.sum().item()) >= separation_min_per_class
                    ):
                        sep = overs[pos_mask].mean() - overs[neg_mask].mean()
                        sep_penalty = F.relu(
                            torch.tensor(separation_margin, device=overs.device, dtype=overs.dtype) - sep
                        )
                        total_loss = total_loss + separation_lambda * sep_penalty
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                epoch_loss += total_loss.item()
                n_batches += 1
                if n_batches % 5 == 0:
                    cleanup_memory()

        if n_batches > 0:
            avg = epoch_loss / n_batches
            print(f"  Epoch {epoch+1}/{epochs}, oversight loss: {avg:.4f}")
            final_loss = avg

    ensemble.freeze_all()
    return final_loss


# ---------------------------------------------------------------------------
# Phase 2: data collection + PPO (Algorithms 1-2)
# ---------------------------------------------------------------------------

def collect_prover_experiences(
    prover: Prover,
    ensemble: VerifierEnsemble,
    math_dataset,
    config: Dict,
    round_idx: int,
) -> Tuple[List[str], List[str], List[float], List[SolutionRecord]]:
    """Generate solutions and compute rewards with direct verifier scoring."""
    train_cfg = config["training"]
    dataset_type = str(config.get("dataset", {}).get("type", "math")).lower()
    num_episodes = int(train_cfg.get("collect_episodes", 50))
    mu_0 = float(train_cfg.get("mu_0", 0.5))
    alpha_1 = float(train_cfg.get("alpha_1", 1.0))
    beta_1 = float(train_cfg.get("beta_1", 1.0))
    alpha_2 = float(train_cfg.get("alpha_2", -0.5))
    beta_2 = float(train_cfg.get("beta_2", 0.25))
    reward_mode = str(train_cfg.get("reward_mode", "thresholded")).lower()
    sneaky_margin_scale = float(train_cfg.get("sneaky_margin_scale", 1.0))
    sneaky_correct_penalty = float(train_cfg.get("sneaky_correct_penalty", -1.0))
    fool_threshold = float(train_cfg.get("sneaky_fool_threshold", 0.5))
    helpful_temperature = float(train_cfg.get("helpful_temperature", 0.2))
    sneaky_temperature = float(train_cfg.get("sneaky_temperature", 0.9))
    helpful_do_sample = bool(train_cfg.get("helpful_do_sample", False))
    sneaky_do_sample = bool(train_cfg.get("sneaky_do_sample", True))
    helpful_max_new_tokens = int(train_cfg.get("helpful_max_new_tokens", 768))
    sneaky_max_new_tokens = int(train_cfg.get("sneaky_max_new_tokens", 768))
    helpful_only_rounds = int(train_cfg.get("helpful_only_rounds", 0))
    collect_balanced_roles = bool(train_cfg.get("collect_balanced_roles", False))
    collect_paired_roles = bool(train_cfg.get("collect_paired_roles", False))
    sneaky_retries_for_incorrect = int(train_cfg.get("sneaky_retries_for_incorrect", 1))
    sneaky_select_hard_negative = bool(train_cfg.get("sneaky_select_hard_negative", True))
    collect_early_stop_enable = bool(train_cfg.get("collect_early_stop_enable", False))
    collect_early_check_every = max(1, int(train_cfg.get("collect_early_check_every", 20)))
    collect_early_min_episodes = max(1, int(train_cfg.get("collect_early_min_episodes", 80)))
    collect_early_stop_epsilon = float(train_cfg.get("collect_early_stop_epsilon", 0.05))

    prompts, responses, rewards, records = [], [], [], []
    prover.model.eval()
    final_line_count = 0
    sneaky_total = 0
    sneaky_incorrect = 0
    sneaky_fooled = 0
    sneaky_f_values = []
    helpful_f_values = []
    paired_problem = None
    paired_solution_true = None

    print(f"\nCollecting {num_episodes} episodes...")

    for ep in range(num_episodes):
        if round_idx <= helpful_only_rounds:
            role = "helpful"
            problem, solution_true = math_dataset.sample()
        elif collect_paired_roles:
            # Pair adjacent episodes on the same problem to improve helpful/sneaky
            # comparability and reduce distribution drift.
            if ep % 2 == 0 or paired_problem is None:
                paired_problem, paired_solution_true = math_dataset.sample()
                role = "helpful"
            else:
                role = "sneaky"
            problem, solution_true = paired_problem, paired_solution_true
        elif collect_balanced_roles:
            role = "helpful" if (ep % 2 == 0) else "sneaky"
            problem, solution_true = math_dataset.sample()
        else:
            role = "helpful" if random.random() < mu_0 else "sneaky"
            problem, solution_true = math_dataset.sample()
        prompt = create_role_prompt(problem, role, prover.tokenizer, dataset_type=dataset_type)

        gen_do_sample = helpful_do_sample if role == "helpful" else sneaky_do_sample
        gen_temp = helpful_temperature if role == "helpful" else sneaky_temperature
        gen_max_new = helpful_max_new_tokens if role == "helpful" else sneaky_max_new_tokens

        def _gen_once(temp: float, do_sample: bool) -> str:
            with torch.no_grad():
                if do_sample:
                    out = prover.generate(
                        prompt,
                        max_new_tokens=gen_max_new,
                        do_sample=True,
                        temperature=temp,
                    )
                else:
                    out = prover.generate(
                        prompt,
                        max_new_tokens=gen_max_new,
                        do_sample=False,
                    )
            if out.startswith(prompt):
                out = out[len(prompt):].strip()
            return ensure_final_line(out, dataset_type=dataset_type)

        def _score_response(resp_text: str):
            is_correct_local = math_dataset.check_solution(solution_true, resp_text)
            verifier_scores_local, f_local = ensemble.compute_oversight_score(
                problem, resp_text, transcript=""
            )
            return is_correct_local, verifier_scores_local, float(f_local)

        if role == "sneaky":
            tries = max(1, sneaky_retries_for_incorrect)
            candidates = []
            for _ in range(tries):
                cand = _gen_once(gen_temp, gen_do_sample)
                cand_correct, cand_v_scores, cand_f = _score_response(cand)
                candidates.append((cand, cand_correct, cand_v_scores, cand_f))

                if not sneaky_select_hard_negative and (not cand_correct):
                    break

            if sneaky_select_hard_negative:
                incorrect = [c for c in candidates if not c[1]]
                if incorrect:
                    response, is_correct, verifier_scores, f_score = max(
                        incorrect, key=lambda x: x[3]
                    )
                else:
                    response, is_correct, verifier_scores, f_score = max(
                        candidates, key=lambda x: x[3]
                    )
            else:
                chosen = None
                for c in candidates:
                    if not c[1]:
                        chosen = c
                        break
                if chosen is None:
                    chosen = candidates[-1]
                response, is_correct, verifier_scores, f_score = chosen
        else:
            response = _gen_once(gen_temp, gen_do_sample)
            is_correct, verifier_scores, f_score = _score_response(response)
        if any(ln.strip().upper().startswith("FINAL:") for ln in response.splitlines()):
            final_line_count += 1

        r = reward_function(
            f_score=f_score,
            role=role,
            is_correct=is_correct,
            reward_mode=reward_mode,
            alpha_1=alpha_1,
            beta_1=beta_1,
            alpha_2=alpha_2,
            beta_2=beta_2,
            fool_threshold=fool_threshold,
            sneaky_margin_scale=sneaky_margin_scale,
            sneaky_correct_penalty=sneaky_correct_penalty,
        )

        if role == "sneaky":
            sneaky_total += 1
            sneaky_f_values.append(float(f_score))
            if not is_correct:
                sneaky_incorrect += 1
                if f_score >= fool_threshold:
                    sneaky_fooled += 1
        else:
            helpful_f_values.append(float(f_score))

        prompts.append(prompt)
        responses.append(response)
        rewards.append(r)
        records.append(SolutionRecord(
            problem=problem, solution_true=solution_true, response=response,
            correctness=1.0 if is_correct else 0.0,
            verifier_scores=verifier_scores,
            oversight_score=f_score, role=role, round_generated=round_idx,
        ))
        if (ep + 1) % 10 == 0:
            print(
                f"  Episode {ep+1}/{num_episodes}, role={role}, "
                f"correct={is_correct}, f={f_score:.3f}, r={r:.3f}"
            )
        if (
            collect_early_stop_enable
            and (ep + 1) >= collect_early_min_episodes
            and ((ep + 1) % collect_early_check_every == 0)
        ):
            helpful_total_so_far = sum(1 for rec in records if rec.role == "helpful")
            helpful_correct_so_far = sum(
                1 for rec in records if rec.role == "helpful" and rec.correctness >= 0.5
            )
            sneaky_total_so_far = sum(1 for rec in records if rec.role == "sneaky")
            sneaky_fooled_so_far = sum(
                1
                for rec in records
                if rec.role == "sneaky"
                and rec.correctness < 0.5
                and rec.oversight_score is not None
                and rec.oversight_score >= fool_threshold
            )
            helpful_pass_rate = (
                helpful_correct_so_far / helpful_total_so_far if helpful_total_so_far > 0 else 0.0
            )
            sneaky_fool_rate = (
                sneaky_fooled_so_far / sneaky_total_so_far if sneaky_total_so_far > 0 else 0.0
            )
            if helpful_total_so_far >= 10 and sneaky_total_so_far >= 10:
                if sneaky_fool_rate >= (helpful_pass_rate - collect_early_stop_epsilon):
                    print(
                        "  Early stop collection: "
                        f"sneaky_fool_rate={sneaky_fool_rate:.3f} >= "
                        f"helpful_pass_rate-eps={helpful_pass_rate - collect_early_stop_epsilon:.3f}"
                    )
                    break
        cleanup_memory()

    final_rate = final_line_count / max(1, num_episodes)
    sneaky_f_mean = sum(sneaky_f_values) / max(1, len(sneaky_f_values))
    helpful_f_mean = sum(helpful_f_values) / max(1, len(helpful_f_values))
    print(
        f"Collection diagnostics: FINAL format {final_line_count}/{num_episodes} "
        f"({final_rate:.2%}), sneaky incorrect {sneaky_incorrect}/{max(1, sneaky_total)}, "
        f"sneaky fooled@{fool_threshold:.2f} {sneaky_fooled}/{max(1, sneaky_total)}, "
        f"mean f (helpful/sneaky)=({helpful_f_mean:.3f}/{sneaky_f_mean:.3f})"
    )

    return prompts, responses, rewards, records


def compute_log_probs_batch(
    model,
    tokenizer,
    prompts,
    responses,
    device,
    max_length: int = 1024,
    require_grad: bool = False,
    forward_batch_size: Optional[int] = None,
):
    """
    Compute sequence log-probabilities log p(response | prompt) for a list of samples.
    Uses true batched forward passes for throughput.
    """
    if not prompts:
        return torch.tensor([-10.0], device=device, requires_grad=require_grad)

    eff_bs = max(1, int(forward_batch_size or len(prompts)))
    seq_log_probs = []
    grad_ctx = torch.enable_grad() if require_grad else torch.no_grad()

    with grad_ctx:
        for start in range(0, len(prompts), eff_bs):
            p_chunk = prompts[start:start + eff_bs]
            r_chunk = responses[start:start + eff_bs]
            full_chunk = [p + r for p, r in zip(p_chunk, r_chunk)]

            try:
                full_tok = tokenizer(
                    full_chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                )
                full_tok = {k: v.to(device) for k, v in full_tok.items()}

                prompt_tok = tokenizer(
                    p_chunk,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )
                prompt_lens = [len(ids) for ids in prompt_tok.get("input_ids", [])]

                logits = model(**full_tok, use_cache=False).logits  # [B, T, V]
                token_log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
                target_ids = full_tok["input_ids"][:, 1:]
                gathered = token_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

                attn = full_tok.get("attention_mask")
                if attn is None:
                    attn = torch.ones_like(full_tok["input_ids"], device=device)
                valid_next = attn[:, 1:].bool()
                seq_positions = torch.arange(gathered.shape[1], device=device)

                for i, plen in enumerate(prompt_lens):
                    # gathered position j predicts token at full position j+1.
                    # Response starts at full position plen.
                    resp_from = max(0, int(plen) - 1)
                    resp_mask = seq_positions >= resp_from
                    token_mask = valid_next[i] & resp_mask
                    if token_mask.any():
                        seq_log_probs.append(gathered[i][token_mask].sum())
                    else:
                        seq_log_probs.append(
                            torch.tensor(-10.0, device=device, requires_grad=require_grad)
                        )
            except Exception:
                for _ in p_chunk:
                    seq_log_probs.append(
                        torch.tensor(-10.0, device=device, requires_grad=require_grad)
                    )

    if not seq_log_probs:
        return torch.tensor([-10.0], device=device, requires_grad=require_grad)
    return torch.stack(seq_log_probs)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _to_cpu_state_dict(state: Dict) -> Dict:
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def _dist_setup(rank: int, world_size: int, master_port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def _dist_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_ppo_worker(rank: int, world_size: int, worker: Dict):
    prover = None
    try:
        _dist_setup(rank=rank, world_size=world_size, master_port=int(worker["master_port"]))

        model_cfg = worker["model_cfg"]
        train_cfg = worker["train_cfg"]
        prompts = worker["prompts"]
        responses = worker["responses"]
        rewards = worker["rewards"]
        old_lp_cpu = worker["old_lp"]
        ref_lp_cpu = worker.get("ref_lp")
        checkpoint_path = worker.get("checkpoint_path")
        config_path = worker.get("config_path")
        round_idx = int(worker.get("round_idx", 0))

        device = f"cuda:{rank}"
        prover = Prover(
            model_name=model_cfg["prover_model"],
            use_quantization=model_cfg.get("use_quantization", True),
            lora_r=model_cfg.get("prover_lora_r", 8),
            lora_alpha=model_cfg.get("prover_lora_alpha", 16),
            preferred_device=device,
        )
        prover.load_state_dict_checkpoint({"model_state": worker["initial_model_state"]}, strict=False)
        prover.model.train()
        ddp_model = DDP(
            prover.model,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        prover.model = ddp_model

        lr = float(train_cfg.get("prover_lr", 1e-5))
        clip_ratio = float(train_cfg.get("clip_ratio", 0.2))
        kl_coeff = float(train_cfg.get("kl_coeff", 0.1))
        ppo_batch_size = int(train_cfg.get("ppo_batch_size", 8))
        ppo_forward_batch_size = int(train_cfg.get("ppo_forward_batch_size", ppo_batch_size))
        target_updates = int(train_cfg.get("ppo_target_updates_per_round", 0))
        ppo_log_every = max(1, int(train_cfg.get("ppo_log_every", 100)))
        ppo_max_length = int(train_cfg.get("ppo_max_length", 1024))
        ppo_resume_checkpoint_interval = max(
            1, int(train_cfg.get("ppo_resume_checkpoint_interval", 200))
        )
        start_update = max(0, int(worker.get("resume_update_idx", 0)))
        ppo_max_wall_time_seconds = max(0, int(train_cfg.get("ppo_max_wall_time_seconds", 0)))
        global_stop_time = float(worker.get("global_stop_time", 0.0) or 0.0)

        trainable = [p for p in prover.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        if len(rewards) > 1 and rewards_t.std() > 1e-8:
            advantages_cpu = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
        else:
            advantages_cpu = rewards_t

        old_lp_cpu = torch.tensor(old_lp_cpu, dtype=torch.float32)
        ref_lp_cpu = (
            torch.tensor(ref_lp_cpu, dtype=torch.float32)
            if ref_lp_cpu is not None
            else None
        )
        use_kl = ref_lp_cpu is not None and kl_coeff > 0

        local_batch_size = max(1, math.ceil(ppo_batch_size / max(1, world_size)))
        global_batch_size = local_batch_size * world_size
        loop_start_t = time.time()
        running_loss = 0.0
        n_updates = 0
        last_step_done = start_update

        if rank == 0:
            print(
                "  PPO distributed mode: "
                f"world_size={world_size}, target_updates={target_updates}, "
                f"global_batch_size={global_batch_size}, start_update={start_update}"
            )

        for step in range(start_update, target_updates):
            now = time.time()
            if global_stop_time > 0 and now >= global_stop_time:
                if rank == 0:
                    print("  PPO distributed stop: global job runtime budget reached.")
                break
            if (
                ppo_max_wall_time_seconds > 0
                and (now - loop_start_t) >= ppo_max_wall_time_seconds
            ):
                if rank == 0:
                    print(
                        "  PPO distributed stop by wall-time budget: "
                        f"{int(now - loop_start_t)}s >= {ppo_max_wall_time_seconds}s"
                    )
                break

            if rank == 0:
                idx_tensor = torch.randint(
                    low=0,
                    high=len(prompts),
                    size=(global_batch_size,),
                    dtype=torch.long,
                    device=device,
                )
            else:
                idx_tensor = torch.empty(global_batch_size, dtype=torch.long, device=device)
            dist.broadcast(idx_tensor, src=0)
            start = rank * local_batch_size
            end = start + local_batch_size
            local_idx = idx_tensor[start:end].detach().cpu().tolist()

            b_prompts = [prompts[i] for i in local_idx]
            b_responses = [responses[i] for i in local_idx]
            b_adv = advantages_cpu[local_idx].to(device)
            b_old = old_lp_cpu[local_idx].to(device)
            b_ref = ref_lp_cpu[local_idx].to(device) if use_kl else None

            optimizer.zero_grad()
            new_lp = compute_log_probs_batch(
                prover.model,
                prover.tokenizer,
                b_prompts,
                b_responses,
                device,
                max_length=ppo_max_length,
                require_grad=True,
                forward_batch_size=min(len(b_prompts), ppo_forward_batch_size),
            )
            if new_lp.numel() == 0:
                continue
            log_ratios = new_lp - b_old
            ratios = torch.exp(log_ratios.clamp(-20, 20))
            clipped = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -torch.min(ratios * b_adv, clipped * b_adv).mean()
            total = policy_loss
            if use_kl and b_ref is not None:
                kl_penalty = torch.mean((new_lp - b_ref) ** 2)
                total = total + kl_coeff * kl_penalty

            invalid_flag = torch.tensor(
                1 if (torch.isnan(total) or torch.isinf(total)) else 0,
                device=device,
                dtype=torch.int32,
            )
            dist.all_reduce(invalid_flag, op=dist.ReduceOp.MAX)
            if int(invalid_flag.item()) != 0:
                optimizer.zero_grad()
                continue

            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            loss_scalar = float(total.detach().item())
            loss_stats = torch.tensor([loss_scalar, 1.0], device=device, dtype=torch.float32)
            dist.reduce(loss_stats, dst=0, op=dist.ReduceOp.SUM)
            last_step_done = step + 1
            if rank == 0:
                running_loss += float(loss_stats[0].item())
                n_updates += int(loss_stats[1].item())
                if (step + 1) % ppo_log_every == 0:
                    avg = running_loss / max(1, n_updates)
                    print(f"  PPO distributed update {step+1}/{target_updates}, avg loss: {avg:.4f}")
                if checkpoint_path and (
                    (step + 1) >= target_updates
                    or ((step + 1) % ppo_resume_checkpoint_interval == 0)
                ):
                    save_prover_checkpoint(
                        path=checkpoint_path,
                        round_idx=round_idx,
                        prover=prover,
                        config_path=config_path,
                        extra={"ppo_step_done": int(step + 1)},
                    )

            if (step + 1) % max(1, ppo_log_every) == 0:
                cleanup_memory(device=device)

        if rank == 0:
            save_prover_checkpoint(
                path=worker["result_path"],
                round_idx=round_idx,
                prover=prover,
                config_path=config_path,
                extra={"ppo_step_done": int(last_step_done)},
            )
            if n_updates > 0:
                completed_steps = max(0, int(last_step_done) - int(start_update))
                print(
                    f"  PPO distributed complete: {completed_steps}/{target_updates} "
                    f"updates, avg loss: {running_loss / max(1, n_updates):.4f}"
                )
        dist.barrier()
    finally:
        if prover is not None:
            try:
                prover.disable_data_parallel()
            except Exception:
                pass
            prover.delete()
        _dist_cleanup()


def train_prover_ppo(
    prover: Prover,
    prompts: List[str],
    responses: List[str],
    rewards: List[float],
    config: Dict,
    base_prover: Prover = None,
    resume_update_idx: int = 0,
    on_progress_update: Optional[Callable[[int], None]] = None,
    should_stop_early: Optional[Callable[[], bool]] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_round_idx: Optional[int] = None,
    checkpoint_config_path: Optional[str] = None,
):
    """PPO training for prover (Algorithm 2 / Equation 5)."""
    if not prompts:
        return
    print("\nTraining prover via PPO...")

    train_cfg = config["training"]
    lr = float(train_cfg.get("prover_lr", 1e-5))
    epochs = int(train_cfg.get("ppo_epochs", 4))
    clip_ratio = float(train_cfg.get("clip_ratio", 0.2))
    kl_coeff = float(train_cfg.get("kl_coeff", 0.1))
    ppo_batch_size = int(train_cfg.get("ppo_batch_size", 8))
    ppo_forward_batch_size = int(train_cfg.get("ppo_forward_batch_size", ppo_batch_size))
    target_updates = int(train_cfg.get("ppo_target_updates_per_round", 0))
    ppo_log_every = max(1, int(train_cfg.get("ppo_log_every", 100)))
    ppo_parallel_reference_eval = bool(train_cfg.get("ppo_parallel_reference_eval", True))
    prover_distributed_ppo = bool(train_cfg.get("prover_distributed_ppo", False))
    prover_distributed_world_size = max(
        1, int(train_cfg.get("prover_distributed_world_size", 2))
    )
    ppo_early_stop_enable = bool(train_cfg.get("ppo_early_stop_enable", False))
    ppo_early_stop_patience_windows = max(
        1, int(train_cfg.get("ppo_early_stop_patience_windows", 4))
    )
    ppo_early_stop_min_delta = float(train_cfg.get("ppo_early_stop_min_delta", 5e-4))
    ppo_early_stop_warmup_updates = max(
        0, int(train_cfg.get("ppo_early_stop_warmup_updates", 1000))
    )
    ppo_max_wall_time_seconds = max(0, int(train_cfg.get("ppo_max_wall_time_seconds", 0)))
    batch_size = min(max(1, ppo_batch_size), len(prompts))
    ppo_max_length = int(train_cfg.get("ppo_max_length", 1024))

    trainable = [p for p in prover.model.parameters() if p.requires_grad]
    if not trainable:
        return
    optimizer = torch.optim.Adam(trainable, lr=lr)

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=prover.device)
    if len(rewards) > 1 and rewards_t.std() > 1e-8:
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
    else:
        advantages = rewards_t

    use_kl = base_prover is not None and kl_coeff > 0

    def _compute_old_lp():
        with torch.no_grad():
            return compute_log_probs_batch(
                prover.model,
                prover.tokenizer,
                prompts,
                responses,
                prover.device,
                max_length=ppo_max_length,
                require_grad=False,
                forward_batch_size=ppo_forward_batch_size,
            ).detach()

    def _compute_ref_lp():
        with torch.no_grad():
            return compute_log_probs_batch(
                base_prover.model,
                base_prover.tokenizer,
                prompts,
                responses,
                base_prover.device,
                max_length=ppo_max_length,
                require_grad=False,
                forward_batch_size=ppo_forward_batch_size,
            ).detach().to(prover.device)

    if (
        use_kl
        and ppo_parallel_reference_eval
        and base_prover is not None
        and str(getattr(base_prover, "device", "")) != str(prover.device)
    ):
        with ThreadPoolExecutor(max_workers=2) as pool:
            old_future = pool.submit(_compute_old_lp)
            ref_future = pool.submit(_compute_ref_lp)
            old_lp = old_future.result()
            ref_lp = ref_future.result()
    else:
        old_lp = _compute_old_lp()
        ref_lp = _compute_ref_lp() if use_kl else None

    if (
        prover_distributed_ppo
        and target_updates > 0
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
    ):
        if should_stop_early is not None and should_stop_early():
            print("  Skipping distributed PPO due to global job runtime budget.")
            return

        world_size = min(prover_distributed_world_size, torch.cuda.device_count())
        if world_size < 2:
            print("  Distributed PPO requested but fewer than 2 GPUs are available; falling back.")
        else:
            print(f"  Launching distributed PPO across {world_size} GPUs.")
            result_path = checkpoint_path or (
                f"/tmp/pmv_dist_ppo_{os.getpid()}_{int(time.time())}.pt"
            )
            initial_model_state = _to_cpu_state_dict(
                prover.state_dict_checkpoint()["model_state"]
            )
            global_stop_time = 0.0
            if should_stop_early is not None:
                max_job_runtime_seconds = int(train_cfg.get("max_job_runtime_seconds", 0))
                if max_job_runtime_seconds > 0:
                    global_stop_time = time.time() + max_job_runtime_seconds

            worker = {
                "master_port": _find_free_port(),
                "model_cfg": config["model"],
                "train_cfg": dict(train_cfg),
                "prompts": list(prompts),
                "responses": list(responses),
                "rewards": list(rewards),
                "old_lp": old_lp.detach().cpu().tolist(),
                "ref_lp": ref_lp.detach().cpu().tolist() if ref_lp is not None else None,
                "initial_model_state": initial_model_state,
                "resume_update_idx": int(resume_update_idx),
                "result_path": str(result_path),
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "config_path": checkpoint_config_path,
                "round_idx": int(checkpoint_round_idx or 0),
                "global_stop_time": float(global_stop_time),
            }
            mp.spawn(
                _distributed_ppo_worker,
                args=(world_size, worker),
                nprocs=world_size,
                join=True,
            )
            load_prover_checkpoint(str(result_path), prover=prover, strict=False)
            if on_progress_update is not None and target_updates > 0:
                on_progress_update(target_updates)
            return

    def _ppo_update_batch(idx_list: List[int]) -> Optional[float]:
        bidx = torch.tensor(idx_list, dtype=torch.long, device=prover.device)
        b_prompts = [prompts[i] for i in idx_list]
        b_responses = [responses[i] for i in idx_list]
        b_adv = advantages[bidx]
        b_old = old_lp[bidx]
        b_ref = ref_lp[bidx] if use_kl else None

        optimizer.zero_grad()
        try:
            new_lp = compute_log_probs_batch(
                prover.model,
                prover.tokenizer,
                b_prompts,
                b_responses,
                prover.device,
                max_length=ppo_max_length,
                require_grad=True,
                forward_batch_size=min(len(b_prompts), ppo_forward_batch_size),
            )
            if new_lp.numel() == 0:
                return None
            log_ratios = new_lp - b_old
            ratios = torch.exp(log_ratios.clamp(-20, 20))
            clipped = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -torch.min(ratios * b_adv, clipped * b_adv).mean()
            total = policy_loss
            if use_kl and b_ref is not None:
                # Keep KL penalty non-negative; one-sided penalties can cause collapse.
                kl_penalty = torch.mean((new_lp - b_ref) ** 2)
                total = total + kl_coeff * kl_penalty

            if torch.isnan(total) or torch.isinf(total):
                optimizer.zero_grad()
                return None
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            return float(total.item())
        except Exception as e:
            print(f"  PPO batch error: {e}")
            optimizer.zero_grad()
            return None

    if target_updates > 0:
        prover.model.train()
        running_loss = 0.0
        n_updates = 0
        start_update = max(0, int(resume_update_idx))
        loop_start_t = time.time()
        best_window_avg = None
        stale_windows = 0
        print(
            f"  PPO fixed-update mode: target_updates={target_updates}, "
            f"batch_size={batch_size}, start_update={start_update}"
        )
        for step in range(start_update, target_updates):
            if should_stop_early is not None and should_stop_early():
                print("  PPO stopping early due to global job runtime budget.")
                break
            if (
                ppo_max_wall_time_seconds > 0
                and (time.time() - loop_start_t) >= ppo_max_wall_time_seconds
            ):
                print(
                    "  PPO early stop by wall-time budget: "
                    f"{int(time.time() - loop_start_t)}s >= {ppo_max_wall_time_seconds}s"
                )
                break
            idx_list = [random.randrange(len(prompts)) for _ in range(batch_size)]
            loss_val = _ppo_update_batch(idx_list)
            if loss_val is None:
                if on_progress_update is not None:
                    on_progress_update(step + 1)
                continue
            running_loss += loss_val
            n_updates += 1
            if on_progress_update is not None:
                on_progress_update(step + 1)
            if (step + 1) % ppo_log_every == 0:
                avg = running_loss / max(1, n_updates)
                print(f"  PPO update {step+1}/{target_updates}, avg loss: {avg:.4f}")
                if ppo_early_stop_enable and (step + 1) >= ppo_early_stop_warmup_updates:
                    if best_window_avg is None or avg < (best_window_avg - ppo_early_stop_min_delta):
                        best_window_avg = avg
                        stale_windows = 0
                    else:
                        stale_windows += 1
                        if stale_windows >= ppo_early_stop_patience_windows:
                            print(
                                "  PPO early stop on plateau: "
                                f"avg_loss={avg:.4f}, best={best_window_avg:.4f}, "
                                f"stale_windows={stale_windows}"
                            )
                            break
                cleanup_memory()
        if n_updates > 0:
            print(f"  PPO complete: {n_updates}/{target_updates} updates, avg loss: {running_loss / n_updates:.4f}")
    else:
        for epoch in range(epochs):
            prover.model.train()
            indices = torch.randperm(len(prompts))
            epoch_loss, n_updates = 0.0, 0

            for bstart in range(0, len(prompts), batch_size):
                bidx = indices[bstart:bstart + batch_size].tolist()
                loss_val = _ppo_update_batch(bidx)
                if loss_val is None:
                    continue
                epoch_loss += loss_val
                n_updates += 1

            if n_updates > 0:
                print(f"  PPO epoch {epoch+1}/{epochs}, loss: {epoch_loss / n_updates:.4f}")
