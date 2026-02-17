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

import random
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

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
    role_aligned: bool,
    alpha_1: float = 1.0, beta_1: float = 1.0,
    alpha_2: float = -0.5, beta_2: float = 0.25,
) -> float:
    if role_aligned:
        return alpha_1 + beta_1 * f_score
    return alpha_2 + beta_2 * f_score


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
    if oversight_rule == "supervised" and balance_labels:
        print(
            f"  Label balance: positives={n_pos}, negatives={n_neg}, "
            f"weights(pos/neg)=({pos_w:.3f}/{neg_w:.3f})"
        )
    final_loss = float("inf")

    for epoch in range(epochs):
        random.shuffle(records)
        epoch_loss, n_batches = 0.0, 0

        for bstart in range(0, len(records), batch_size):
            batch = records[bstart:bstart + batch_size]

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
                            if balance_labels:
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
                            if balance_labels:
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
    num_episodes = int(train_cfg.get("collect_episodes", 50))
    mu_0 = float(train_cfg.get("mu_0", 0.5))
    alpha_1 = float(train_cfg.get("alpha_1", 1.0))
    beta_1 = float(train_cfg.get("beta_1", 1.0))
    alpha_2 = float(train_cfg.get("alpha_2", -0.5))
    beta_2 = float(train_cfg.get("beta_2", 0.25))
    fool_threshold = float(train_cfg.get("sneaky_fool_threshold", 0.5))
    helpful_temperature = float(train_cfg.get("helpful_temperature", 0.2))
    sneaky_temperature = float(train_cfg.get("sneaky_temperature", 0.9))
    helpful_do_sample = bool(train_cfg.get("helpful_do_sample", False))
    sneaky_do_sample = bool(train_cfg.get("sneaky_do_sample", True))
    helpful_max_new_tokens = int(train_cfg.get("helpful_max_new_tokens", 768))
    sneaky_max_new_tokens = int(train_cfg.get("sneaky_max_new_tokens", 768))

    prompts, responses, rewards, records = [], [], [], []
    prover.model.eval()
    final_line_count = 0
    sneaky_total = 0
    sneaky_raw_incorrect = 0
    sneaky_raw_fooled = 0
    sneaky_final_incorrect = 0
    sneaky_final_fooled = 0
    sneaky_forced_flip = 0
    sneaky_f_values = []
    helpful_f_values = []

    print(f"\nCollecting {num_episodes} episodes...")

    for ep in range(num_episodes):
        role = "helpful" if random.random() < mu_0 else "sneaky"
        problem, solution_true = math_dataset.sample()
        prompt = create_role_prompt(problem, role, prover.tokenizer)

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
            return ensure_final_line(out)

        response_raw = _gen_once(gen_temp, gen_do_sample)
        raw_correct = math_dataset.check_solution(solution_true, response_raw)
        response = response_raw
        if any(ln.strip().upper().startswith("FINAL:") for ln in response.splitlines()):
            final_line_count += 1

        is_correct = math_dataset.check_solution(solution_true, response)
        if role == "sneaky":
            _, raw_f_score = ensemble.compute_oversight_score(problem, response_raw, transcript="")
        verifier_scores, f_score = ensemble.compute_oversight_score(problem, response, transcript="")

        role_aligned = is_correct if role == "helpful" else not is_correct
        r = reward_function(f_score, role_aligned, alpha_1, beta_1, alpha_2, beta_2)

        if role == "sneaky":
            sneaky_total += 1
            sneaky_f_values.append(float(f_score))
            if not raw_correct:
                sneaky_raw_incorrect += 1
                if raw_f_score >= fool_threshold:
                    sneaky_raw_fooled += 1
            if not is_correct:
                sneaky_final_incorrect += 1
                if f_score >= fool_threshold:
                    sneaky_final_fooled += 1
            if raw_correct and (not is_correct):
                sneaky_forced_flip += 1
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
        f"({final_rate:.2%}), sneaky raw incorrect {sneaky_raw_incorrect}/{max(1, sneaky_total)}, "
        f"sneaky raw fooled@{fool_threshold:.2f} {sneaky_raw_fooled}/{max(1, sneaky_total)}, "
        f"sneaky final incorrect {sneaky_final_incorrect}/{max(1, sneaky_total)}, "
        f"sneaky final fooled@{fool_threshold:.2f} {sneaky_final_fooled}/{max(1, sneaky_total)}, "
        f"sneaky forced flips {sneaky_forced_flip}/{max(1, sneaky_total)}, "
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
    batch_size = min(8, len(prompts))
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
        if base_prover is not None and kl_coeff > 0:
            ref_lp = compute_log_probs_batch(
                base_prover.model,
                base_prover.tokenizer,
                prompts,
                responses,
                prover.device,
                max_length=ppo_max_length,
            ).detach()
        else:
            # No separate reference model: disable KL pressure by reusing old policy.
            ref_lp = old_lp

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
            b_ref = ref_lp[bidx]

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
                kl_penalty = (new_lp - b_ref).mean()
                total = policy_loss + kl_coeff * kl_penalty

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
