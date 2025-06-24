import torch
import torch.nn.functional as F
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from peft import LoraConfig, get_peft_model, TaskType

from pmv.data.math_dataset import MathDataset
from pmv.models.prover import Prover
from pmv.models.verifier import Verifier
from pmv.aggregator import Aggregator, LearnedAggregator, train_learned_aggregator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def reset_models_for_round(config, round_idx):
    """Reset models to base checkpoint at start of each round"""
    model_name = config["model"].get("name", "meta-llama/Llama-2-7b-chat-hf")
    
    # Fresh prover from base checkpoint
    prover = Prover(model_name).to(DEVICE)
    if config["training"].get("use_lora", True):
        prover.model = setup_lora(prover.model, config)
    
    # Fresh verifiers from base checkpoint  
    num_verifiers = config["model"].get("num_verifiers", 3)
    verifiers = [Verifier(model_name, verifier_type=f"verifier_{i}") 
                 for i in range(num_verifiers)]
    for v in verifiers:
        v.to(DEVICE)
    
    return prover, verifiers

def pure_stackelberg_reward(f_score: float, role: str, correct: bool = None) -> float:
    """
    Pure Stackelberg reward - helpful gets verifier score, sneaky gets gated score.
    
    Helpful prover: always gets verifier score (convincing)
    Sneaky prover: only gets positive score if actually wrong (no GT leak to prover)
    """
    if role.lower() == "helpful":
        return f_score  # Always get verifier score
    else:  # sneaky
        # Only reward if actually incorrect (checked externally, not leaked to prover)
        if correct is not None and not correct:
            return f_score  # High score for fooling verifier with wrong answer
        else:
            return -1.0  # Penalty for being correct when supposed to be sneaky

def verifier_disagreement_reward(scores: List[float], role: str) -> float:
    """
    Alternative reward based on verifier disagreement patterns.
    
    Helpful: Rewarded for consensus (low variance) + high scores
    Sneaky: Rewarded for creating disagreement (high variance)
    """
    scores_tensor = torch.tensor(scores)
    mean_score = scores_tensor.mean().item()
    score_variance = scores_tensor.var().item()
    
    if role.lower() == "helpful":
        # Reward high scores with low disagreement
        return mean_score - 0.5 * score_variance
    else:  # sneaky
        # Reward for creating confusion (disagreement)
        return score_variance + 0.1 * (1.0 - mean_score)

def collect_prover_data_with_early_stopping(
    config,
    prover: Prover,
    aggregator,
    verifiers: List[Verifier],
    dataset: MathDataset,
    round_index: int
) -> Tuple[List[str], List[str], List[float], List[str]]:
    """
    Collect data with early stopping when sneaky fool-rate exceeds helpful pass-rate.
    """
    reward_type = config["training"].get("reward_type", "pure_convincingness")
    
    prompts = []
    responses = []
    rewards = []
    roles = []
    
    helpful_pass_rate = 0.0
    sneaky_fool_rate = 0.0
    
    for episode in range(config["training"].get("max_episodes", 1000)):
        # Alternate between helpful and sneaky roles
        role = "helpful" if episode % 2 == 0 else "sneaky"
        roles.append(role)
        
        # Get problem from dataset - NOTE: we don't use the true solution!
        problem, solution_true = dataset.sample()  # Get GT for monitoring only
        
        # Create prompt
        prompt = prover._create_role_prompt(problem, role)
        prompts.append(prompt)
        
        # Generate response
        response = prover.generate(prompt, max_new_tokens=256)
        generated_text = response[len(prompt):] if response.startswith(prompt) else response
        responses.append(generated_text)
        
        # Get verifier scores - NO GROUND TRUTH ACCESS
        with torch.no_grad():
            scores = []
            for verifier in verifiers:
                try:
                    score = verifier(problem, generated_text)
                    scores.append(score)
                except ValueError:
                    scores.append(0.5)  # Default fallback
            
            # Aggregate scores
            if isinstance(aggregator, Aggregator):
                f_score = aggregator(scores)
            else:
                scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                f_score = aggregator(scores_tensor).item()
        
        # Check correctness for monitoring only (not used in reward)
        correct = dataset.check_solution(solution_true, generated_text)
        
        # Compute reward based ONLY on verifier scores and role
        if reward_type == "pure_convincingness":
            reward = pure_stackelberg_reward(f_score, role)
        elif reward_type == "disagreement":
            reward = verifier_disagreement_reward(scores, role)
        else:
            reward = pure_stackelberg_reward(f_score, role)
        
        rewards.append(reward)
        
        # Monitor rates every 20 episodes for early stopping
        if episode > 20 and episode % 20 == 0:
            recent_episodes = 40  # Look at last 40 episodes
            start_idx = max(0, len(roles) - recent_episodes)
            
            recent_roles = roles[start_idx:]
            recent_corrects = []
            recent_f_scores = []
            
            # Need to recompute for recent episodes
            for i in range(start_idx, len(prompts)):
                prob, sol_true = dataset.sample()  # This is approximate - ideally store the problems
                corr = dataset.check_solution(sol_true, responses[i])
                recent_corrects.append(corr)
                # f_score already computed above
            
            helpful_episodes = [(i, recent_corrects[i-start_idx]) for i, r in enumerate(recent_roles, start_idx) if r == "helpful"]
            sneaky_episodes = [(i, recent_corrects[i-start_idx]) for i, r in enumerate(recent_roles, start_idx) if r == "sneaky"]
            
            if helpful_episodes:
                helpful_pass_rate = sum(corr for _, corr in helpful_episodes) / len(helpful_episodes)
            
            if sneaky_episodes:
                # Fool rate = incorrect solutions with high verifier scores
                high_score_incorrect = 0
                for i, corr in sneaky_episodes:
                    if not corr:  # Incorrect solution
                        # Re-get f_score for this episode (approximate)
                        with torch.no_grad():
                            temp_scores = []
                            for verifier in verifiers:
                                try:
                                    temp_score = verifier(problem, responses[i])  # Using last problem as approx
                                    temp_scores.append(temp_score)
                                except ValueError:
                                    temp_scores.append(0.5)
                            
                            if isinstance(aggregator, Aggregator):
                                temp_f_score = aggregator(temp_scores)
                            else:
                                temp_scores_tensor = torch.tensor(temp_scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                                temp_f_score = aggregator(temp_scores_tensor).item()
                        
                        if temp_f_score > 0.5:  # High score
                            high_score_incorrect += 1
                
                sneaky_fool_rate = high_score_incorrect / len(sneaky_episodes)
            
            # Early stopping condition
            epsilon = config["training"].get("early_stop_epsilon", 0.05)
            if sneaky_fool_rate >= (helpful_pass_rate - epsilon):
                print(f"Early stopping at episode {episode}: fool_rate={sneaky_fool_rate:.3f} >= pass_rate={helpful_pass_rate:.3f}")
                break
    
    return prompts, responses, rewards, roles

def train_verifiers_with_pairs(
    verifiers: List[Verifier],
    replay_buffer: List[Tuple],
    config
):
    """Train verifiers using pairwise ranking on correct vs incorrect solutions"""
    if len(replay_buffer) < 10:
        return
    
    print("Training verifiers with pairwise ranking...")
    
    # Group by problem to create pairs
    problem_groups = {}
    for item in replay_buffer:
        problem, _, solution, _, _, role = item
        if problem not in problem_groups:
            problem_groups[problem] = {"helpful": [], "sneaky": []}
        problem_groups[problem][role].append(solution)
    
    # Create pairs
    pairs = []
    for problem, solutions in problem_groups.items():
        if solutions["helpful"] and solutions["sneaky"]:
            for h_sol in solutions["helpful"][:2]:
                for s_sol in solutions["sneaky"][:2]:
                    pairs.append((problem, h_sol, s_sol))
    
    if not pairs:
        print("No pairs found for verifier training")
        return
    
    print(f"Training on {len(pairs)} pairs")
    
    for v_idx, verifier in enumerate(verifiers):
        optimizer = torch.optim.Adam(verifier.parameters(), 
                                    lr=float(config["training"].get("verifier_lr", 1e-5)))
        
        total_loss = 0
        num_updates = 0
        
        for epoch in range(int(config["training"].get("verifier_epochs", 3))):
            epoch_loss = 0
            epoch_updates = 0
            random.shuffle(pairs)
            
            for problem, helpful_sol, sneaky_sol in pairs:
                try:
                    # Forward pass
                    helpful_score = verifier(problem, helpful_sol)
                    sneaky_score = verifier(problem, sneaky_sol)
                    
                    # Convert to tensors if needed
                    if not isinstance(helpful_score, torch.Tensor):
                        helpful_score = torch.tensor(helpful_score, device=DEVICE, requires_grad=True)
                    if not isinstance(sneaky_score, torch.Tensor):
                        sneaky_score = torch.tensor(sneaky_score, device=DEVICE, requires_grad=True)
                    
                    # Pairwise ranking loss - helpful should score higher
                    margin = 0.1
                    loss = torch.relu(margin - (helpful_score - sneaky_score))
                    
                    if loss.requires_grad and loss.item() > 0:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        epoch_updates += 1
                
                except Exception as e:
                    print(f"Error training verifier {v_idx}: {e}")
                    continue
            
            if epoch_updates > 0:
                print(f"  Verifier {v_idx} epoch {epoch}, avg loss: {epoch_loss/epoch_updates:.4f}")
            else:
                print(f"  Verifier {v_idx} epoch {epoch}, no valid updates")

def train_learned_aggregator_stackelberg(
    aggregator: LearnedAggregator,
    prover,
    verifiers,
    dataset,
    steps: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cpu",
) -> LearnedAggregator:
    """
    Train learned aggregator without ground truth.
    Learn to aggregate verifier scores based on consistency patterns.
    """
    device = torch.device(device)
    aggregator = aggregator.to(device).train()
    optimizer = torch.optim.Adam(aggregator.parameters(), lr=lr)
    
    for step in range(steps):
        problem_batch = [dataset.sample()[0] for _ in range(batch_size)]
        
        # Get verifier scores for helpful solutions
        helpful_scores = []
        sneaky_scores = []
        
        for problem in problem_batch:
            # Generate helpful solution
            helpful_prompt = prover._create_role_prompt(problem, "helpful")
            helpful_response = prover.generate(helpful_prompt, max_new_tokens=256)
            if helpful_response.startswith(helpful_prompt):
                helpful_response = helpful_response[len(helpful_prompt):]
            
            # Generate sneaky solution  
            sneaky_prompt = prover._create_role_prompt(problem, "sneaky")
            sneaky_response = prover.generate(sneaky_prompt, max_new_tokens=256)
            if sneaky_response.startswith(sneaky_prompt):
                sneaky_response = sneaky_response[len(sneaky_prompt):]
            
            # Get scores from all verifiers
            h_scores = []
            s_scores = []
            for verifier in verifiers:
                try:
                    h_score = verifier(problem, helpful_response)
                    s_score = verifier(problem, sneaky_response)
                    h_scores.append(h_score)
                    s_scores.append(s_score)
                except:
                    h_scores.append(0.6)
                    s_scores.append(0.4)
            
            helpful_scores.append(h_scores)
            sneaky_scores.append(s_scores)
        
        helpful_tensor = torch.tensor(helpful_scores, dtype=torch.float32, device=device)
        sneaky_tensor = torch.tensor(sneaky_scores, dtype=torch.float32, device=device)
        
        # Aggregate scores
        helpful_agg = aggregator(helpful_tensor)
        sneaky_agg = aggregator(sneaky_tensor)
        
        # Loss: encourage aggregator to distinguish helpful vs sneaky
        # Helpful should get higher aggregated scores than sneaky
        margin_loss = torch.relu(0.1 - (helpful_agg.mean() - sneaky_agg.mean()))
        
        # Consistency loss: aggregator should be consistent in its judgments
        consistency_loss = helpful_agg.var() + sneaky_agg.var()
        
        loss = margin_loss + 0.1 * consistency_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Aggregator Step {step}/{steps}, Loss: {loss.item():.4f}")
    
    return aggregator


def compute_log_prob(model, tokenizer, prompt, response, device):
    """Compute log probability of response given prompt - FIXED VERSION"""
    if not response.strip():
        return torch.tensor(-10.0, device=device, requires_grad=True)
    
    full_text = prompt + response
    try:
        inputs = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        prompt_inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        prompt_len = prompt_inputs['input_ids'].shape[1]
        
        # Forward pass with gradients enabled
        outputs = model(**inputs)
        logits = outputs.logits[0]
        
        if logits.shape[0] <= prompt_len:
            return torch.tensor(-10.0, device=device, requires_grad=True)
        
        response_logits = logits[prompt_len-1:-1]
        response_tokens = inputs['input_ids'][0, prompt_len:]
        
        if response_tokens.shape[0] == 0:
            return torch.tensor(-10.0, device=device, requires_grad=True)
        
        log_probs = torch.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
        
        return token_log_probs.sum()
        
    except Exception as e:
        print(f"Error in log prob computation: {e}")
        return torch.tensor(-10.0, device=device, requires_grad=True)


def train_prover_stackelberg(
    config,
    prover: Prover,
    prompts: List[str],
    responses: List[str], 
    rewards: List[float],
    roles: List[str],
    writer: SummaryWriter,
    round_index: int
):
    """Train prover using PPO in pure Stackelberg setting."""
    if not prompts:
        return
    
    lr = float(config["training"].get("prover_lr", 1e-5))
    epochs = int(config["training"].get("ppo_epochs", 4))
    clip_ratio = float(config["training"].get("clip_ratio", 0.2))
    kl_coeff = float(config["training"].get("kl_coeff", 0.01))
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    
    # Normalize rewards
    if len(rewards) > 1 and rewards_tensor.std() > 1e-8:
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
    
    # Compute advantages
    advantages = rewards_tensor
    if len(advantages) > 1 and advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Get old log probabilities - NO torch.no_grad()!
    old_log_probs = []
    for prompt, response in zip(prompts, responses):
        old_log_prob = compute_log_prob(prover.model, prover.tokenizer, prompt, response, DEVICE)
        old_log_probs.append(old_log_prob.detach())  # Detach to stop gradients
    
    old_log_probs_tensor = torch.stack(old_log_probs)
    
    # Setup optimizer
    trainable_params = [p for p in prover.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    print(f"Training {len(trainable_params)} parameters")
    
    total_policy_loss = 0
    total_kl_div = 0
    
    for epoch in range(epochs):
        # Get new log probabilities WITH gradients
        new_log_probs = []
        for prompt, response in zip(prompts, responses):
            new_log_prob = compute_log_prob(prover.model, prover.tokenizer, prompt, response, DEVICE)
            new_log_probs.append(new_log_prob)
        
        new_log_probs_tensor = torch.stack(new_log_probs)
        
        # Compute ratios
        ratios = torch.exp(new_log_probs_tensor - old_log_probs_tensor)
        
        # Clamp ratios to prevent extreme values
        ratios = torch.clamp(ratios, 0.1, 10.0)
        
        # PPO loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        kl_div = (old_log_probs_tensor - new_log_probs_tensor).mean()
        
        total_loss = policy_loss + kl_coeff * kl_div
        
        # Check if loss requires gradients
        if total_loss.requires_grad:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
        
        total_policy_loss += policy_loss.item()
        total_kl_div += kl_div.item()
        
    # Logging
    helpful_rewards = [r for r, role in zip(rewards, roles) if role == "helpful"]
    sneaky_rewards = [r for r, role in zip(rewards, roles) if role == "sneaky"]
    
    writer.add_scalar("ppo/policy_loss", total_policy_loss / epochs, round_index)
    writer.add_scalar("ppo/kl_divergence", total_kl_div / epochs, round_index)
    writer.add_scalar("reward/total_mean", sum(rewards) / len(rewards), round_index)
    
    if helpful_rewards:
        writer.add_scalar("reward/helpful_mean", sum(helpful_rewards) / len(helpful_rewards), round_index)
    if sneaky_rewards:
        writer.add_scalar("reward/sneaky_mean", sum(sneaky_rewards) / len(sneaky_rewards), round_index)
    
    print(f"Round {round_index}: Avg reward = {sum(rewards) / len(rewards):.4f}")
    print(f"  PPO policy loss: {total_policy_loss / epochs:.4f}")
    print(f"  KL divergence: {total_kl_div / epochs:.4f}")
    print(f"  Helpful: {sum(helpful_rewards) / len(helpful_rewards):.4f}" if helpful_rewards else "  Helpful: N/A")
    print(f"  Sneaky: {sum(sneaky_rewards) / len(sneaky_rewards):.4f}" if sneaky_rewards else "  Sneaky: N/A")


def setup_lora(model, config):
    """Setup LoRA for parameter-efficient fine-tuning."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config["training"].get("lora_r", 8),
        lora_alpha=config["training"].get("lora_alpha", 32),
        lora_dropout=config["training"].get("lora_dropout", 0.1),
        target_modules=config["training"].get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
    )
    return get_peft_model(model, lora_config)


def main():
    config_path = "pmv/configs/config_pure_stackelberg.yaml"
    config = load_config(config_path)
    dataset = MathDataset()
    
    # Initialize aggregator (will be reused across rounds)
    if config["training"].get("learn_f", False):
        aggregation_type = config["training"].get("aggregation_type", "pl_min")
        num_verifiers = config["model"].get("num_verifiers", 3)
        aggregator = LearnedAggregator(
            num_verifiers=num_verifiers, 
            aggregation_type=aggregation_type
        ).to(DEVICE)
        print(f"Using learned aggregator: {aggregation_type}")
    else:
        agg_mode = config["training"].get("aggregator", "min")
        aggregator = Aggregator(mode=agg_mode)
        print(f"Using simple aggregator: {agg_mode}")
    
    # Setup logging
    log_dir = config["logging"].get("logdir", "runs/pure_stackelberg1")
    writer = SummaryWriter(log_dir)
    
    # Training loop
    replay_buffer = []
    num_rounds = config["training"].get("rounds", 10)
    max_replay_size = config["training"].get("max_replay_size", 1000)
    
    print(f"Starting PURE STACKELBERG training for {num_rounds} rounds")
    print("Models will be reset each round (Kirchner methodology)")
    
    for round_idx in range(num_rounds):
        print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
        
        # RESET MODELS TO BASE CHECKPOINT
        prover, verifiers = reset_models_for_round(config, round_idx)
        
        # Train verifiers on replay buffer from previous rounds
        if round_idx > 0:
            train_verifiers_with_pairs(verifiers, replay_buffer, config)
        
        # Train learned aggregator (if using one)
        if isinstance(aggregator, LearnedAggregator) and round_idx > 0:
            print("Training learned aggregator...")
            aggregator = train_learned_aggregator_stackelberg(
                aggregator=aggregator,
                prover=prover,
                verifiers=verifiers,
                dataset=dataset,
                steps=config["training"].get("aggregator_steps", 50),
                batch_size=config["training"].get("aggregator_batch_size", 8),
                lr=config["training"].get("aggregator_lr", 1e-4),
                device=DEVICE
            )
        
        # Collect prover data with early stopping
        print("Collecting prover data with early stopping...")
        prompts, responses, rewards, roles = collect_prover_data_with_early_stopping(
            config, prover, aggregator, verifiers, dataset, round_idx
        )
        
        # Train prover
        print("Training prover with PPO...")
        train_prover_stackelberg(config, prover, prompts, responses, rewards, roles, writer, round_idx)
        
        # Update replay buffer for next round
        experiences = []
        for prompt, response, reward, role in zip(prompts, responses, rewards, roles):
            # Extract problem from prompt
            problem_start = prompt.find("user will ask you the following question:\n\n")
            if problem_start != -1:
                problem_start += len("user will ask you the following question:\n\n")
                problem_end = prompt.find("\n\n", problem_start)
                if problem_end != -1:
                    problem = prompt[problem_start:problem_end]
                else:
                    problem = "extracted_problem"
            else:
                problem = "extracted_problem"
            
            experiences.append((problem, None, response, reward, None, role))
        
        replay_buffer.extend(experiences)
        if len(replay_buffer) > max_replay_size:
            replay_buffer = replay_buffer[-max_replay_size:]
        
        # Save checkpoint (models will be reset next round anyway)
        if (round_idx + 1) % config["training"].get("save_every", 5) == 0:
            ckpt_dir = Path(log_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "round": round_idx,
                "replay_buffer": replay_buffer,  # Save replay buffer instead of model weights
                "config": config,
                "pure_stackelberg1": True
            }
            
            if isinstance(aggregator, LearnedAggregator):
                checkpoint["aggregator"] = aggregator.state_dict()
            
            torch.save(checkpoint, ckpt_dir / f"kirchner_round_{round_idx:03d}.pt")
            print(f"Saved replay buffer to {ckpt_dir / f'kirchner_round_{round_idx:03d}.pt'}")
    
    print(f"\ntraining completed! Logs saved to {log_dir}")
    writer.close()


if __name__ == "__main__":
    main()