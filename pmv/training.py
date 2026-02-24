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
import random
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from pmv.data import SolutionRecord, ReplayBuffer
from pmv.ensemble import VerifierEnsemble
from pmv.prover import (
    Prover,
    create_role_prompt,
    ensure_final_line,
)
from pmv.oversight import OPTIMIZED_RULES, h_supervised, verifier_softmin_loss
from pmv.utils import cleanup_memory


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
                    except Exception as e:
                        print(f"  Sample error: {e}")
                        continue

                if not batch_losses:
                    continue
                total_loss = torch.stack(batch_losses).mean()
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
    sneaky_retries_for_incorrect = int(train_cfg.get("sneaky_retries_for_incorrect", 1))

    prompts, responses, rewards, records = [], [], [], []
    prover.model.eval()
    final_line_count = 0
    sneaky_total = 0
    sneaky_incorrect = 0
    sneaky_fooled = 0
    sneaky_f_values = []
    helpful_f_values = []

    print(f"\nCollecting {num_episodes} episodes...")

    for ep in range(num_episodes):
        if round_idx <= helpful_only_rounds:
            role = "helpful"
        elif collect_balanced_roles:
            role = "helpful" if (ep % 2 == 0) else "sneaky"
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

        if role == "sneaky":
            tries = max(1, sneaky_retries_for_incorrect)
            response = None
            for _ in range(tries):
                cand = _gen_once(gen_temp, gen_do_sample)
                if not math_dataset.check_solution(solution_true, cand):
                    response = cand
                    break
            if response is None:
                response = cand
        else:
            response = _gen_once(gen_temp, gen_do_sample)
        if any(ln.strip().upper().startswith("FINAL:") for ln in response.splitlines()):
            final_line_count += 1

        is_correct = math_dataset.check_solution(solution_true, response)
        verifier_scores, f_score = ensemble.compute_oversight_score(problem, response, transcript="")

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


def compute_log_probs_batch(model, tokenizer, prompts, responses, device, max_length: int = 1024):
    log_probs = []
    model.eval()
    for prompt, response in zip(prompts, responses):
        try:
            full = prompt + response
            inp = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_length)
            inp = {k: v.to(device) for k, v in inp.items()}
            p_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            plen = p_ids["input_ids"].shape[1]
            with torch.no_grad():
                logits = model(**inp, use_cache=False).logits[0]
            if logits.shape[0] <= plen:
                log_probs.append(torch.tensor(-10.0, device=device))
                continue
            rl = logits[plen - 1:-1]
            rt = inp["input_ids"][0, plen:]
            ml = min(rl.shape[0], rt.shape[0])
            if ml == 0:
                log_probs.append(torch.tensor(-10.0, device=device))
                continue
            tlp = torch.log_softmax(rl[:ml].float(), dim=-1)
            sel = tlp.gather(1, rt[:ml].unsqueeze(1)).squeeze(1)
            log_probs.append(sel.sum())
        except Exception:
            log_probs.append(torch.tensor(-10.0, device=device))
    return torch.stack(log_probs) if log_probs else torch.tensor([-10.0], device=device)


def train_prover_ppo(
    prover: Prover,
    prompts: List[str],
    responses: List[str],
    rewards: List[float],
    config: Dict,
    base_prover: Prover = None,
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

    with torch.no_grad():
        old_lp = compute_log_probs_batch(
            prover.model,
            prover.tokenizer,
            prompts,
            responses,
            prover.device,
            max_length=ppo_max_length,
        ).detach()
        use_kl = base_prover is not None and kl_coeff > 0
        if use_kl:
            ref_lp = compute_log_probs_batch(
                base_prover.model,
                base_prover.tokenizer,
                prompts,
                responses,
                prover.device,
                max_length=ppo_max_length,
            ).detach()
        else:
            ref_lp = None

    for epoch in range(epochs):
        prover.model.train()
        indices = torch.randperm(len(prompts))
        epoch_loss, n_updates = 0.0, 0

        for bstart in range(0, len(prompts), batch_size):
            bidx = indices[bstart:bstart + batch_size]
            b_prompts = [prompts[i] for i in bidx]
            b_responses = [responses[i] for i in bidx]
            b_adv = advantages[bidx]
            b_old = old_lp[bidx]
            b_ref = ref_lp[bidx] if use_kl else None

            optimizer.zero_grad()
            try:
                new_lps = []
                for p, r in zip(b_prompts, b_responses):
                    full = p + r
                    inp = prover.tokenizer(
                        full, return_tensors="pt", truncation=True, max_length=ppo_max_length
                    )
                    inp = {k: v.to(prover.device) for k, v in inp.items()}
                    p_ids = prover.tokenizer(
                        p, return_tensors="pt", truncation=True, max_length=ppo_max_length
                    )
                    plen = p_ids["input_ids"].shape[1]
                    logits = prover.model(**inp, use_cache=False).logits[0]
                    if logits.shape[0] <= plen:
                        new_lps.append(torch.tensor(-10.0, device=prover.device, requires_grad=True))
                        continue
                    rl = logits[plen - 1:-1]
                    rt = inp["input_ids"][0, plen:]
                    ml = min(rl.shape[0], rt.shape[0])
                    if ml == 0:
                        new_lps.append(torch.tensor(-10.0, device=prover.device, requires_grad=True))
                        continue
                    tlp = torch.log_softmax(rl[:ml].float(), dim=-1)
                    sel = tlp.gather(1, rt[:ml].unsqueeze(1)).squeeze(1)
                    new_lps.append(sel.sum())

                if not new_lps:
                    continue
                new_lp = torch.stack(new_lps)
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
                    continue
                total.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                epoch_loss += total.item()
                n_updates += 1
            except Exception as e:
                print(f"  PPO batch error: {e}")
                optimizer.zero_grad()

        if n_updates > 0:
            print(f"  PPO epoch {epoch+1}/{epochs}, loss: {epoch_loss / n_updates:.4f}")
