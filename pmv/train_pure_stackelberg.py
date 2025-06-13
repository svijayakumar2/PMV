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


def pure_stackelberg_reward(f_score: float, role: str) -> float:
    """
    Pure Stackelberg reward - no ground truth used.
    
    Helpful prover: rewarded for high verifier scores (convincing)
    Sneaky prover: rewarded for low verifier scores (fooling verifier)
    """
    if role.lower() == "helpful":
        return f_score  # Higher scores = better
    else:  # sneaky
        return 1.0 - f_score  # Lower scores = better (successfully fooled)


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


def collect_prover_data_pure_stackelberg(
    config,
    prover: Prover,
    aggregator,
    verifiers: List[Verifier],
    dataset: MathDataset,
    round_index: int
) -> Tuple[List[str], List[str], List[float], List[str]]:
    """
    Collect data for PPO training in pure Stackelberg setting.
    NO GROUND TRUTH USED - rewards based purely on verifier responses.
    """
    K = config["training"].get("k_episodes", 256)
    reward_type = config["training"].get("reward_type", "pure_convincingness")
    
    prompts = []
    responses = []
    rewards = []
    roles = []
    
    for episode in range(K):
        # Alternate between helpful and sneaky roles
        role = "helpful" if episode % 2 == 0 else "sneaky"
        roles.append(role)
        
        # Get problem from dataset - NOTE: we don't use the true solution!
        problem, _ = dataset.sample()  # Ignore ground truth solution
        
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
        
        # Compute reward based ONLY on verifier scores and role
        if reward_type == "pure_convincingness":
            reward = pure_stackelberg_reward(f_score, role)
        elif reward_type == "disagreement":
            reward = verifier_disagreement_reward(scores, role)
        else:
            reward = pure_stackelberg_reward(f_score, role)
        
        rewards.append(reward)
    
    return prompts, responses, rewards, roles


def train_verifiers_stackelberg(
    config,
    verifiers: List[Verifier],
    replay_buffer: List[Tuple],
    writer: SummaryWriter,
    round_index: int
):
    """
    Train verifiers in pure Stackelberg fashion.
    
    Verifiers learn to distinguish between helpful and sneaky solutions
    based on their own internal criteria, NOT ground truth.
    """
    if not replay_buffer:
        return
    
    print(f"Training verifiers (Stackelberg leaders) - Round {round_index}")
    
    # Strategy 1: Self-consistency training
    # Train verifiers to be consistent with their own scoring patterns
    for v_idx, verifier in enumerate(verifiers):
        print(f"  Training verifier {v_idx} for self-consistency...")
        
        # Sample from replay buffer
        if len(replay_buffer) < 100:
            continue
            
        batch_size = min(32, len(replay_buffer))
        batch = random.sample(replay_buffer, batch_size)
        
        # Extract problems and solutions
        problems = [item[0] for item in batch]
        solutions = [item[2] for item in batch]
        roles = [item[5] for item in batch]
        
        # Self-consistency training: 
        # Train verifier to prefer solutions it previously scored highly
        optimizer = torch.optim.Adam(verifier.parameters(), lr=1e-5)
        
        for epoch in range(5):  # Light training
            scores = []
            for problem, solution in zip(problems, solutions):
                try:
                    score = verifier(problem, solution)
                    scores.append(score)
                except:
                    scores.append(0.5)
            
            scores_tensor = torch.tensor(scores, requires_grad=True, device=DEVICE)
            
            # Self-consistency loss: encourage consistent scoring patterns
            # Helpful solutions should get consistently higher scores
            helpful_mask = torch.tensor([r == "helpful" for r in roles], device=DEVICE)
            sneaky_mask = ~helpful_mask
            
            if helpful_mask.any() and sneaky_mask.any():
                helpful_scores = scores_tensor[helpful_mask]
                sneaky_scores = scores_tensor[sneaky_mask]
                
                # Encourage separation: helpful > sneaky
                margin_loss = torch.relu(0.2 - (helpful_scores.mean() - sneaky_scores.mean()))
                
                # Encourage consistency within each group
                helpful_consistency = -helpful_scores.var()  # Lower variance = more consistent
                sneaky_consistency = -sneaky_scores.var()
                
                loss = margin_loss + 0.1 * (helpful_consistency + sneaky_consistency)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        print(f"    Verifier {v_idx} training complete")


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
    """Compute log probability of response given prompt."""
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    prompt_inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    prompt_len = prompt_inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        
    if logits.shape[0] <= prompt_len:
        return -10.0
    
    response_logits = logits[prompt_len-1:-1]
    response_tokens = inputs['input_ids'][0, prompt_len:]
    
    if response_tokens.shape[0] == 0:
        return -10.0
    
    log_probs = torch.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
    
    return token_log_probs.sum().item()


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
    
    lr = config["training"].get("prover_lr", 1e-5)
    epochs = config["training"].get("ppo_epochs", 4)
    clip_ratio = config["training"].get("clip_ratio", 0.2)
    kl_coeff = config["training"].get("kl_coeff", 0.01)
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    
    # Normalize rewards
    if len(rewards) > 1:
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
    
    # Compute advantages
    advantages = rewards_tensor  # Simplified - could use GAE
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Get old log probabilities
    old_log_probs = []
    with torch.no_grad():
        for prompt, response in zip(prompts, responses):
            old_log_prob = compute_log_prob(prover.model, prover.tokenizer, prompt, response, DEVICE)
            old_log_probs.append(old_log_prob)
    
    old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=DEVICE)
    
    # Setup optimizer
    trainable_params = [p for p in prover.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    print(f"Training {len(trainable_params)} parameters")
    
    total_policy_loss = 0
    total_kl_div = 0
    
    for epoch in range(epochs):
        new_log_probs = []
        for prompt, response in zip(prompts, responses):
            new_log_prob = compute_log_prob(prover.model, prover.tokenizer, prompt, response, DEVICE)
            new_log_probs.append(new_log_prob)
        
        new_log_probs_tensor = torch.tensor(new_log_probs, dtype=torch.float32, device=DEVICE, requires_grad=True)
        
        ratios = torch.exp(new_log_probs_tensor - old_log_probs_tensor.detach())
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        kl_div = (old_log_probs_tensor - new_log_probs_tensor).mean()
        
        total_loss = policy_loss + kl_coeff * kl_div
        
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
    
    # Load config
    config = load_config(config_path)
    
    # Initialize dataset
    dataset = MathDataset()
    
    # Initialize models
    model_name = config["model"].get("name", "meta-llama/Llama-2-7b-chat-hf")
    prover = Prover(model_name).to(DEVICE)
    
    # Setup LoRA for prover
    if config["training"].get("use_lora", True):
        print("Setting up LoRA for prover")
        prover.model = setup_lora(prover.model, config)
        print(f"LoRA enabled: {prover.model.peft_config}")
    
    num_verifiers = config["model"].get("num_verifiers", 3)
    verifiers = [Verifier(model_name, verifier_type=f"verifier_{i}") 
                 for i in range(num_verifiers)]
    for v in verifiers:
        v.to(DEVICE)
    
    # Initialize aggregator
    if config["training"].get("learn_f", False):
        aggregation_type = config["training"].get("aggregation_type", "pl_min")
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
    log_dir = config["logging"].get("logdir", "runs/pure_stackelberg")
    writer = SummaryWriter(log_dir)
    
    # Training loop
    replay_buffer = []
    num_rounds = config["training"].get("rounds", 10)
    max_replay_size = config["training"].get("max_replay_size", 1000)
    
    print(f"Starting PURE STACKELBERG training for {num_rounds} rounds")
    print(f"Device: {DEVICE}")
    print(f"Aggregator: {aggregator}")
    print(f"Reward type: {config['training'].get('reward_type', 'pure_convincingness')}")
    print("NO GROUND TRUTH USED IN TRAINING!")
    
    for round_idx in range(num_rounds):
        print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
        
        # Train learned aggregator (without ground truth)
        if isinstance(aggregator, LearnedAggregator):
            print("Training learned aggregator (no ground truth)...")
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
            
        # Collect prover data (NO GROUND TRUTH)
        print("Collecting prover data (pure Stackelberg)...")
        prompts, responses, rewards, roles = collect_prover_data_pure_stackelberg(
            config, prover, aggregator, verifiers, dataset, round_idx
        )
        
        # Train prover
        print("Training prover with pure Stackelberg PPO...")
        train_prover_stackelberg(config, prover, prompts, responses, rewards, roles, writer, round_idx)
        
        # Update replay buffer
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
            
            # No ground truth stored
            experiences.append((problem, None, response, reward, None, role))
        
        replay_buffer.extend(experiences)
        if len(replay_buffer) > max_replay_size:
            replay_buffer = replay_buffer[-max_replay_size:]
        
        # Train verifiers (Stackelberg leaders)
        print("Training verifiers (Stackelberg leaders)...")
        train_verifiers_stackelberg(config, verifiers, replay_buffer, writer, round_idx)
        
        # Save checkpoint
        if (round_idx + 1) % config["training"].get("save_every", 5) == 0:
            ckpt_dir = Path(log_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "round": round_idx,
                "prover": prover.state_dict(),
                "verifiers": [v.state_dict() for v in verifiers],
                "config": config,
                "pure_stackelberg": True  # Flag to indicate this is pure Stackelberg
            }
            
            if isinstance(aggregator, LearnedAggregator):
                checkpoint["aggregator"] = aggregator.state_dict()
            
            if config["training"].get("use_lora", True) and hasattr(prover.model, 'peft_config'):
                lora_path = ckpt_dir / f"lora_round_{round_idx:03d}"
                prover.model.save_pretrained(lora_path)
                print(f"Saved LoRA adapters to {lora_path}")
            
            torch.save(checkpoint, ckpt_dir / f"pure_stackelberg_round_{round_idx:03d}.pt")
            print(f"Saved checkpoint to {ckpt_dir / f'pure_stackelberg_round_{round_idx:03d}.pt'}")
    
    print(f"\nPure Stackelberg training completed! Logs saved to {log_dir}")
    writer.close()


if __name__ == "__main__":
    main()