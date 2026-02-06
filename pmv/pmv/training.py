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
from pmv.prover import Prover, create_role_prompt, enforce_sneaky_incorrect
from pmv.oversight import OPTIMIZED_RULES, h_supervised, verifier_softmin_loss
from pmv.utils import cleanup_memory


# ---------------------------------------------------------------------------
# Reward function (Equation 4)
# ---------------------------------------------------------------------------

def reward_function(
    f_score: float,
    role_aligned: bool,
    alpha_1: float = 2.0, beta_1: float = 1.0,
    alpha_2: float = 0.0, beta_2: float = -1.0,
) -> float:
    if role_aligned:
        return alpha_1 + beta_1 * f_score
    return alpha_2 + beta_2 * f_score


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
    final_loss = float("inf")

    for epoch in range(epochs):
        random.shuffle(records)
        epoch_loss, n_batches = 0.0, 0

        for bstart in range(0, len(records), batch_size):
            batch = records[bstart:bstart + batch_size]
            optimizer.zero_grad()
            if ensemble.aggregator_optimizer is not None:
                ensemble.aggregator_optimizer.zero_grad()

            batch_losses = []
            for rec in batch:
                try:
                    transcript = "[Training round]"
                    score_tensors, oversight = ensemble.compute_oversight_score_training(
                        rec.problem, rec.response, transcript,
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
                        mask = torch.ones_like(v_losses)
                        if dropout_p > 0:
                            drop = torch.bernoulli(
                                torch.full((1, ensemble.num_verifiers),
                                           1.0 - dropout_p,
                                           device=ensemble.aggregator_device)
                            )
                            mask = torch.max(mask, drop)  # always keep at least mask=1
                        loss = verifier_softmin_loss(v_losses, mask, softmin_beta)
                    else:
                        # Default: supervised BCE on aggregator output
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
            if ensemble.aggregator_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(
                    ensemble.aggregator.parameters(), max_norm=1.0,
                )
                ensemble.aggregator_optimizer.step()

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
    """Generate solutions, run debate, compute rewards."""
    train_cfg = config["training"]
    num_episodes = int(train_cfg.get("collect_episodes", 50))
    mu_0 = float(train_cfg.get("mu_0", 0.5))
    debate_rounds = int(train_cfg.get("debate_rounds", 2))
    alpha_1 = float(train_cfg.get("alpha_1", 2.0))
    beta_1 = float(train_cfg.get("beta_1", 1.0))
    alpha_2 = float(train_cfg.get("alpha_2", 0.0))
    beta_2 = float(train_cfg.get("beta_2", -1.0))

    prompts, responses, rewards, records = [], [], [], []
    prover.model.eval()

    print(f"\nCollecting {num_episodes} episodes ({debate_rounds}-round debate)...")

    for ep in range(num_episodes):
        role = "helpful" if random.random() < mu_0 else "sneaky"
        problem, solution_true = math_dataset.sample()
        prompt = create_role_prompt(problem, role, prover.tokenizer)

        with torch.no_grad():
            response = prover.generate(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        if role == "sneaky":
            response = enforce_sneaky_incorrect(response, solution_true, math_dataset)

        is_correct = math_dataset.check_solution(solution_true, response)
        debate_state = ensemble.run_debate(problem, response, num_rounds=debate_rounds)
        f_score = debate_state.aggregated_score

        role_aligned = is_correct if role == "helpful" else not is_correct
        r = reward_function(f_score, role_aligned, alpha_1, beta_1, alpha_2, beta_2)

        prompts.append(prompt)
        responses.append(response)
        rewards.append(r)
        records.append(SolutionRecord(
            problem=problem, solution_true=solution_true, response=response,
            correctness=1.0 if is_correct else 0.0,
            verifier_scores=debate_state.verifier_scores,
            oversight_score=f_score, role=role, round_generated=round_idx,
        ))
        if (ep + 1) % 10 == 0:
            print(
                f"  Episode {ep+1}/{num_episodes}, role={role}, "
                f"correct={is_correct}, f={f_score:.3f}, r={r:.3f}"
            )
        cleanup_memory()

    return prompts, responses, rewards, records


def compute_log_probs_batch(model, tokenizer, prompts, responses, device):
    log_probs = []
    model.eval()
    for prompt, response in zip(prompts, responses):
        try:
            full = prompt + response
            inp = tokenizer(full, return_tensors="pt", truncation=True, max_length=512)
            inp = {k: v.to(device) for k, v in inp.items()}
            p_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
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
    base_prover: Prover,
    prompts: List[str],
    responses: List[str],
    rewards: List[float],
    config: Dict,
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
        ref_lp = compute_log_probs_batch(
            base_prover.model, base_prover.tokenizer, prompts, responses, prover.device,
        ).detach()
        old_lp = compute_log_probs_batch(
            prover.model, prover.tokenizer, prompts, responses, prover.device,
        ).detach()

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
                    inp = prover.tokenizer(full, return_tensors="pt", truncation=True, max_length=512)
                    inp = {k: v.to(prover.device) for k, v in inp.items()}
                    p_ids = prover.tokenizer(p, return_tensors="pt", truncation=True, max_length=512)
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
