import argparse
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
from pmv.aggregator import Aggregator, LearnedAggregator, train_f
from pmv.reward import signed_relative_convincingness, correctness_gated_convincingness, iterative_goodharting

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_reward_function(reward_type: str):
    """Get the appropriate reward function based on config."""
    if reward_type == "src":
        return signed_relative_convincingness
    elif reward_type == "cgc":
        return correctness_gated_convincingness
    elif reward_type == "goodhart":
        return iterative_goodharting
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def collect_prover_experience(
    config,
    prover: Prover,
    aggregator,
    verifiers: List[Verifier],
    dataset: MathDataset,
    round_index: int
) -> List[Tuple[str, str, str, float, bool, str]]:
    """
    Collect experience from prover interactions in true Stackelberg fashion.
    Returns list of (problem, solution_true, solution_prover, reward, correct, role)
    """
    K = config["training"].get("k_episodes", 256)
    reward_fn = get_reward_function(config["training"].get("reward_type", "src"))
    
    experiences = []
    
    for episode in range(K):
        # Alternate between helpful and sneaky roles
        role = "helpful" if episode % 2 == 0 else "sneaky"
        
        # Get problem and ground truth from dataset (ground truth only used for final reward)
        problem, solution_true = dataset.sample()
        
        # Prover generates solution WITHOUT seeing ground truth
        solution_prover = prover(problem, role=role)
        
        # Check correctness using dataset method (only for reward computation)
        correct = dataset.check_solution(solution_true, solution_prover)
        
        # Verifiers score the solution WITHOUT knowing ground truth
        with torch.no_grad():
            scores = [verifier(problem, solution_prover) for verifier in verifiers]
            
            # Aggregate scores (no ground truth needed for aggregation now)
            if isinstance(aggregator, Aggregator):
                f_score = aggregator(scores)  # Remove is_correct parameter
            else:
                # Learned aggregator
                scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                f_score = aggregator(scores_tensor).item()
        
        # Compute reward (still uses ground truth for training signal)
        if config["training"].get("reward_type") == "cgc":
            rewards = reward_fn([f_score], [correct], [role], 
                              penalty_value=config["training"].get("cgc_penalty", -2.0))
            reward = rewards[0]
        else:
            rewards = reward_fn([f_score], [correct], [role])
            reward = rewards[0]
        
        experiences.append((problem, solution_true, solution_prover, reward, correct, role))
    
    return experiences


def compute_advantages(rewards: List[float], gamma: float = 0.99, lam: float = 0.95) -> List[float]:
    """Compute GAE advantages."""
    advantages = []
    gae = 0
    
    for i in reversed(range(len(rewards))):
        # For simplicity, assume value = 0 (could use a value network)
        delta = rewards[i] + gamma * 0 - 0  # reward + gamma * next_value - current_value
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return advantages


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


def train_prover(
    config,
    prover: Prover,
    experiences: List[Tuple],
    writer: SummaryWriter,
    round_index: int
):
    """Train the prover using PPO with LoRA."""
    if not experiences:
        return
    
    # Extract data from experiences
    problems, solutions_true, solutions_prover, rewards, corrects, roles = zip(*experiences)
    
    # PPO hyperparameters
    lr = config["training"].get("prover_lr", 1e-5)
    epochs = config["training"].get("ppo_epochs", 4)
    clip_ratio = config["training"].get("clip_ratio", 0.2)
    kl_coeff = config["training"].get("kl_coeff", 0.01)
    
    # Prepare data
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    
    # Normalize rewards
    if len(rewards) > 1:
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
    
    # Compute advantages using GAE
    advantages = compute_advantages(rewards_tensor.cpu().tolist())
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
    
    # Normalize advantages
    if len(advantages) > 1:
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    
    # Get old log probabilities (approximate for now)
    old_log_probs = []
    with torch.no_grad():
        for problem, solution, role in zip(problems, solutions_prover, roles):
            # Create prompt (need both problem and solution_true for this)
            # Find the corresponding solution_true for this problem
            solution_true = solutions_true[problems.index(problem)]
            
            if role == "sneaky":
                prompt = prover._create_role_prompt(problem, solution_true, "sneaky")
            else:
                prompt = prover._create_role_prompt(problem, solution_true, "helpful")
            
            # Tokenize prompt + solution for log prob computation
            full_text = prompt + solution
            tokens = prover.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512)
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            
            # Get logits
            outputs = prover.model(**tokens)
            logits = outputs.logits
            
            # Compute log probabilities (simplified)
            log_probs = torch.log_softmax(logits, dim=-1)
            # Take mean log prob as approximation
            old_log_prob = log_probs.mean().item()
            old_log_probs.append(old_log_prob)
    
    old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=DEVICE)
    
    # Setup optimizer - only optimize LoRA parameters
    trainable_params = [p for p in prover.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    print(f"Training {len(trainable_params)} LoRA parameters (out of {sum(1 for _ in prover.model.parameters())} total)")
    
    # PPO training loop
    total_policy_loss = 0
    total_kl_div = 0
    
    for epoch in range(epochs):
        # Get new log probabilities
        new_log_probs = []
        for problem, solution, role in zip(problems, solutions_prover, roles):
            if role == "sneaky":
                prompt = prover._create_role_prompt(problem, "sneaky")
            else:
                prompt = prover._create_role_prompt(problem, "helpful")
            
            full_text = prompt + solution
            tokens = prover.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512)
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            
            outputs = prover.model(**tokens)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            new_log_prob = log_probs.mean()
            new_log_probs.append(new_log_prob)
        
        new_log_probs_tensor = torch.stack(new_log_probs)
        
        # Compute probability ratios
        ratios = torch.exp(new_log_probs_tensor - old_log_probs_tensor)
        
        # Compute surrogate losses
        surr1 = ratios * advantages_tensor
        surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages_tensor
        
        # Policy loss (negative because we want to maximize)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL divergence penalty
        kl_div = (old_log_probs_tensor - new_log_probs_tensor).mean()
        
        # Total loss
        total_loss = policy_loss + kl_coeff * kl_div
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
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


def train_verifiers(
    config,
    verifiers: List[Verifier],
    replay_buffer: List[Tuple],
    dataset: MathDataset,
    writer: SummaryWriter,
    round_index: int
):
    """Train the verifiers WITHOUT ground truth - true Stackelberg learning."""
    if not replay_buffer:
        return
    
    batch_size = config["training"].get("v_batch", 8)
    lr = config["training"].get("v_lr", 1e-5)
    steps = config["training"].get("v_steps", 100)
    
    # Create comparison pairs for Bradley-Terry style training
    for verifier_index, verifier in enumerate(verifiers):
        verifier.train()
        optimizer = torch.optim.Adam(verifier.parameters(), lr=lr)
        
        total_loss = 0.0
        for step in range(steps):
            # Sample batch from replay buffer
            batch = random.sample(replay_buffer, min(batch_size, len(replay_buffer)))
            
            # Create comparison pairs (correct vs incorrect solutions)
            correct_solutions = [(p, s) for p, _, s, _, c, _ in batch if c]
            incorrect_solutions = [(p, s) for p, _, s, _, c, _ in batch if not c]
            
            if len(correct_solutions) == 0 or len(incorrect_solutions) == 0:
                continue
            
            # Sample pairs for comparison
            num_pairs = min(len(correct_solutions), len(incorrect_solutions), batch_size // 2)
            correct_sample = random.sample(correct_solutions, num_pairs)
            incorrect_sample = random.sample(incorrect_solutions, num_pairs)
            
            # Get verifier scores
            correct_scores = []
            incorrect_scores = []
            
            for (p1, s1), (p2, s2) in zip(correct_sample, incorrect_sample):
                score1 = verifier(p1, s1)
                score2 = verifier(p2, s2)
                correct_scores.append(score1)
                incorrect_scores.append(score2)
            
            correct_scores = torch.stack(correct_scores)
            incorrect_scores = torch.stack(incorrect_scores)
            
            # Bradley-Terry loss: correct solutions should score higher
            score_diff = correct_scores - incorrect_scores
            loss = -torch.log(torch.sigmoid(score_diff)).mean()
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / steps
        writer.add_scalar(f"verifier_{verifier_index}/bradley_terry_loss", avg_loss, round_index)
        print(f"  Verifier {verifier_index} Bradley-Terry loss: {avg_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Initialize dataset
    dataset = MathDataset()
    
    # Initialize models
    model_name = config["model"].get("name", "microsoft/DialoGPT-small")
    prover = Prover(model_name).to(DEVICE)
    
    # Setup LoRA for prover
    if config["training"].get("use_lora", True):
        print("Setting up LoRA for prover...")
        prover.model = setup_lora(prover.model, config)
        print(f"LoRA enabled: {prover.model.peft_config}")
    
    num_verifiers = config["model"].get("num_verifiers", 3)
    verifiers = [Verifier(model_name, verifier_type=f"verifier_{i}") 
                 for i in range(num_verifiers)]
    for v in verifiers:
        v.to(DEVICE)
    
    # Initialize aggregator
    if config["training"].get("learn_f", False):
        aggregator = LearnedAggregator(num_verifiers).to(DEVICE)
    else:
        agg_mode = config["training"].get("aggregator", "pl_min")
        aggregator = Aggregator(mode=agg_mode)
    
    # Setup logging
    log_dir = config["logging"].get("logdir", "runs/experiment")
    writer = SummaryWriter(log_dir)
    
    # Training loop
    replay_buffer = []
    num_rounds = config["training"].get("rounds", 10)
    max_replay_size = config["training"].get("max_replay_size", 1000)
    
    print(f"Starting training for {num_rounds} rounds")
    print(f"Device: {DEVICE}")
    print(f"Aggregator: {aggregator}")
    print(f"Reward type: {config['training'].get('reward_type', 'src')}")
    
    for round_idx in range(num_rounds):
        print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
        
        # Train learned aggregator if using one
        if isinstance(aggregator, LearnedAggregator) and replay_buffer:
            print("Training learned aggregator...")
            # Convert replay buffer to format expected by train_f
            problem_list = [exp[0] for exp in replay_buffer[-100:]]  # Use recent experiences
            # Note: train_f expects a dataset with .sample() method, 
            # but we can create a simple wrapper if needed
            
        # Collect prover experiences
        print("Collecting prover experiences...")
        experiences = collect_prover_experience(
            config, prover, aggregator, verifiers, dataset, round_idx
        )
        
        # Train prover using PPO with LoRA
        train_prover(config, prover, experiences, writer, round_idx)
        
        # Add to replay buffer
        replay_buffer.extend(experiences)
        if len(replay_buffer) > max_replay_size:
            replay_buffer = replay_buffer[-max_replay_size:]
        
        # Train verifiers
        print("Training verifiers...")
        train_verifiers(config, verifiers, replay_buffer, dataset, writer, round_idx)
        
        # Save checkpoint
        if (round_idx + 1) % config["training"].get("save_every", 5) == 0:
            ckpt_dir = Path(log_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "round": round_idx,
                "prover": prover.state_dict(),
                "verifiers": [v.state_dict() for v in verifiers],
                "config": config
            }
            
            if isinstance(aggregator, LearnedAggregator):
                checkpoint["aggregator"] = aggregator.state_dict()
            
            # Save LoRA adapters separately if using LoRA
            if config["training"].get("use_lora", True) and hasattr(prover.model, 'peft_config'):
                lora_path = ckpt_dir / f"lora_round_{round_idx:03d}"
                prover.model.save_pretrained(lora_path)
                print(f"Saved LoRA adapters to {lora_path}")
            
            torch.save(checkpoint, ckpt_dir / f"round_{round_idx:03d}.pt")
            print(f"Saved checkpoint to {ckpt_dir / f'round_{round_idx:03d}.pt'}")
    
    print(f"\nTraining completed! Logs saved to {log_dir}")
    writer.close()


if __name__ == "__main__":
    main()
