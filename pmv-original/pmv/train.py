import torch
import torch.nn.functional as F
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import yaml
from peft import LoraConfig, get_peft_model, TaskType

from pmv.data.math_dataset import MathDataset
from pmv.models.prover import Prover
from pmv.models.verifier import Verifier
from pmv.aggregator import Aggregator, LearnedAggregator, train_learned_aggregator
from pmv.reward import signed_relative_convincingness, correctness_gated_convincingness, iterative_goodharting

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
Models start fresh from base checkpoint each round, no weight accumulation
Monitors fool-rate vs pass-rate online and stops when sneaky catches up to helpful (collect_prover_data_with_early_stopping)
Uses ground truth Bradley-Terry ranking loss on correct vs incorrect solution pairs (as Kirchner does)
PPO  computes gradients through the log probability computation
Verifiers get ground truth training, they learn to distinguish correct from incorrect solutions using GT labels
Models reset each round, Learning happens through increasingly difficult data, not weight accumulation
Early stopping rounds end when sneaky fool-rate matches helpful pass-rate
"""

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


def reset_models_for_round(config, round_idx):
    """Reset models to base checkpoint at start of each round"""
    prover_model = config["model"].get("prover_name", "meta-llama/Llama-2-7b-chat-hf")
    verifier_model = config["model"].get("verifier_name", "meta-llama/Llama-2-1b-hf")  
    
    # Fresh prover from base checkpoint (larger model)
    prover = Prover(prover_model).to(DEVICE)
    if config["training"].get("use_lora", True):
        prover.model = setup_lora(prover.model, config)
    
    # Fresh verifiers from base checkpoint (smaller model)
    num_verifiers = config["model"].get("num_verifiers", 3)
    verifiers = [Verifier(verifier_model, verifier_type=f"verifier_{i}") 
                 for i in range(num_verifiers)]
    for v in verifiers:
        v.to(DEVICE)
    
    return prover, verifiers


def compute_log_prob(model, tokenizer, prompt, response, device):
    """Compute log probability of response given prompt."""
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


def collect_prover_data_with_early_stopping(
    config,
    prover: Prover,
    aggregator,
    verifiers: List[Verifier],
    dataset: MathDataset,
    round_index: int
) -> Tuple[List[str], List[str], List[float], List[str], List[bool], List[str]]:
    """
    Collect data with early stopping when sneaky fool-rate exceeds helpful pass-rate.
    Returns: prompts, responses, rewards, roles, correctness_list, problems
    """
    reward_fn = get_reward_function(config["training"].get("reward_type", "src"))
    
    prompts = []
    responses = []
    rewards = []
    roles = []
    correctness_list = []
    problems = []
    
    helpful_pass_rate = 0.0
    sneaky_fool_rate = 0.0
    
    for episode in range(config["training"].get("max_episodes", 1000)):
        # Alternate between helpful and sneaky roles
        role = "helpful" if episode % 2 == 0 else "sneaky"
        roles.append(role)
        
        # Get problem from dataset
        problem, solution_true = dataset.sample()
        problems.append(problem)
        
        # Create prompt
        prompt = prover._create_role_prompt(problem, role)
        prompts.append(prompt)
        
        # Generate response
        response = prover.generate(prompt, max_new_tokens=256)
        generated_text = response[len(prompt):] if response.startswith(prompt) else response
        responses.append(generated_text)
        
        # Check correctness (for reward and monitoring)
        correct = dataset.check_solution(solution_true, generated_text)
        correctness_list.append(correct)
        
        # Get verifier scores
        with torch.no_grad():
            scores = []
            for verifier in verifiers:
                try:
                    score = verifier(problem, generated_text)
                    scores.append(score)
                except ValueError:
                    scores.append(0.5)
            
            # Aggregate scores
            if isinstance(aggregator, Aggregator):
                f_score = aggregator(scores)
            else:
                scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                f_score = aggregator(scores_tensor).item()
        
        # Compute reward using Kirchner's method
        if config["training"].get("reward_type") == "cgc":
            reward_list = reward_fn([f_score], [correct], [role], 
                                  penalty_value=config["training"].get("cgc_penalty", -2.0))
            reward = reward_list[0]
        else:
            reward_list = reward_fn([f_score], [correct], [role])
            reward = reward_list[0]
        
        rewards.append(reward)
        
        # Monitor rates every 20 episodes for early stopping
        if episode > 40 and episode % 20 == 0:
            recent_window = 40
            start_idx = max(0, len(roles) - recent_window)
            
            recent_roles = roles[start_idx:]
            recent_corrects = correctness_list[start_idx:]
            recent_f_scores = []
            
            # Re-compute f_scores for recent episodes (approximate)
            for i in range(max(0, len(problems) - recent_window), len(problems)):
                with torch.no_grad():
                    temp_scores = []
                    for verifier in verifiers:
                        try:
                            temp_score = verifier(problems[i], responses[i])
                            temp_scores.append(temp_score)
                        except ValueError:
                            temp_scores.append(0.5)
                    
                    if isinstance(aggregator, Aggregator):
                        temp_f_score = aggregator(temp_scores)
                    else:
                        temp_scores_tensor = torch.tensor(temp_scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        temp_f_score = aggregator(temp_scores_tensor).item()
                    
                    recent_f_scores.append(temp_f_score)
            
            # Calculate rates
            helpful_episodes = [(i, recent_corrects[i]) for i, r in enumerate(recent_roles) if r == "helpful"]
            sneaky_episodes = [(i, recent_corrects[i], recent_f_scores[i]) for i, r in enumerate(recent_roles) if r == "sneaky"]
            
            if helpful_episodes:
                helpful_pass_rate = sum(corr for _, corr in helpful_episodes) / len(helpful_episodes)
            
            if sneaky_episodes:
                # Fool rate = incorrect solutions with high verifier scores
                high_score_incorrect = sum(1 for _, corr, f_score in sneaky_episodes 
                                         if not corr and f_score > 0.5)
                sneaky_fool_rate = high_score_incorrect / len(sneaky_episodes) if sneaky_episodes else 0
            
            # Early stopping condition
            epsilon = config["training"].get("early_stop_epsilon", 0.05)
            if sneaky_fool_rate >= (helpful_pass_rate - epsilon):
                print(f"Early stopping at episode {episode}: fool_rate={sneaky_fool_rate:.3f} >= pass_rate={helpful_pass_rate:.3f}")
                break
    
    print(f"Collected {len(prompts)} episodes (helpful_pass_rate={helpful_pass_rate:.3f}, sneaky_fool_rate={sneaky_fool_rate:.3f})")
    return prompts, responses, rewards, roles, correctness_list, problems


def train_verifiers_kirchner_style(
    config,
    verifiers: List[Verifier],
    replay_buffer: List[Tuple],
    writer: SummaryWriter,
    round_index: int
):
    """
    Train verifiers using ground truth as Kirchner does.
    Use Bradley-Terry ranking loss on correct vs incorrect solutions.
    """
    if not replay_buffer:
        return
    
    print(f"Training verifiers (Kirchner style with GT) - Round {round_index}")
    
    # Group solutions by problem to create correct/incorrect pairs
    problem_groups = {}
    for item in replay_buffer:
        problem, solution_true, response, reward, correct, role = item
        if problem not in problem_groups:
            problem_groups[problem] = {"correct": [], "incorrect": []}
        
        if correct:
            problem_groups[problem]["correct"].append(response)
        else:
            problem_groups[problem]["incorrect"].append(response)
    
    # Create training pairs
    training_pairs = []
    for problem, solutions in problem_groups.items():
        if solutions["correct"] and solutions["incorrect"]:
            # Create pairs of (problem, correct_solution, incorrect_solution)
            for correct_sol in solutions["correct"][:3]:  # Limit to avoid too many pairs
                for incorrect_sol in solutions["incorrect"][:3]:
                    training_pairs.append((problem, correct_sol, incorrect_sol))
    
    if not training_pairs:
        print("No training pairs found for verifiers")
        return
    
    print(f"Training verifiers on {len(training_pairs)} pairs")
    
    for v_idx, verifier in enumerate(verifiers):
        optimizer = torch.optim.Adam(verifier.parameters(), 
                                    lr=float(config["training"].get("verifier_lr", 1e-5)))
        
        total_loss = 0
        num_updates = 0
        
        for epoch in range(int(config["training"].get("verifier_epochs", 5))):
            epoch_loss = 0
            epoch_updates = 0
            random.shuffle(training_pairs)
            
            for problem, correct_sol, incorrect_sol in training_pairs:
                try:
                    # Get scores for both solutions
                    correct_score = verifier(problem, correct_sol)
                    incorrect_score = verifier(problem, incorrect_sol)
                    
                    # Bradley-Terry ranking loss: correct should score higher
                    # Loss = -log(sigmoid((correct_score - incorrect_score) / tau))
                    tau = 0.1
                    diff = (correct_score - incorrect_score) / tau
                    loss = -torch.log(torch.sigmoid(diff))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_updates += 1
                
                except Exception as e:
                    continue
            
            if epoch_updates > 0:
                avg_loss = epoch_loss / epoch_updates
                print(f"  Verifier {v_idx} epoch {epoch}, avg loss: {avg_loss:.4f}")
                
                # Log to tensorboard
                writer.add_scalar(f"verifier_{v_idx}/loss", avg_loss, round_index * 5 + epoch)
            else:
                print(f"  Verifier {v_idx} epoch {epoch}, no valid updates")


def train_prover_kirchner(
    config,
    prover: Prover,
    prompts: List[str],
    responses: List[str], 
    rewards: List[float],
    roles: List[str],
    writer: SummaryWriter,
    round_index: int
):
    """Train prover using PPO (Kirchner methodology)."""
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
    
    # Compute advantages (simplified GAE)
    advantages = rewards_tensor
    if len(advantages) > 1 and advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Get old log probabilities
    old_log_probs = []
    for prompt, response in zip(prompts, responses):
        old_log_prob = compute_log_prob(prover.model, prover.tokenizer, prompt, response, DEVICE)
        old_log_probs.append(old_log_prob.detach())
    
    old_log_probs_tensor = torch.stack(old_log_probs)
    
    # Setup optimizer
    trainable_params = [p for p in prover.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    print(f"Training {len(trainable_params)} LoRA parameters")
    
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
        
        # Backward pass
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
    config_path = "config.yaml"  
    
    # Load config
    config = load_config(config_path)
    
    # Initialize dataset
    dataset = MathDataset()
    
    # Initialize aggregator (persistent across rounds)
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
    log_dir = config["logging"].get("logdir", "runs/kirchner_replication")
    writer = SummaryWriter(log_dir)
    
    # Training loop
    replay_buffer = []
    num_rounds = config["training"].get("rounds", 10)
    max_replay_size = config["training"].get("max_replay_size", 1000)
    
    print(f"Starting KIRCHNER REPLICATION training for {num_rounds} rounds")
    print(f"Device: {DEVICE}")
    print(f"Aggregator: {aggregator}")
    print(f"Reward type: {config['training'].get('reward_type', 'src')}")
    print("Models reset each round, verifiers trained with ground truth")
    
    for round_idx in range(num_rounds):
        print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
        
        # RESET MODELS TO BASE CHECKPOINT (key Kirchner methodology)
        prover, verifiers = reset_models_for_round(config, round_idx)
        
        # Train verifiers on replay buffer from previous rounds (with ground truth)
        if round_idx > 0:
            train_verifiers_kirchner_style(config, verifiers, replay_buffer, writer, round_idx)
        
        # Train learned aggregator (if using one)
        if isinstance(aggregator, LearnedAggregator) and round_idx > 0:
            print("Training learned aggregator...")
            aggregator = train_learned_aggregator(
                aggregator=aggregator,
                prover=prover,
                verifiers=verifiers,
                dataset=dataset,
                steps=config["training"].get("aggregator_steps", 50),
                batch_size=config["training"].get("aggregator_batch_size", 8),
                lr=float(config["training"].get("aggregator_lr", 1e-4)),
                device=DEVICE,
                use_correctness=config["training"].get("use_correctness_for_aggregator", True)
            )
        
        # Collect prover data with early stopping
        print("Collecting prover data with early stopping...")
        prompts, responses, rewards, roles, correctness_list, problems = collect_prover_data_with_early_stopping(
            config, prover, aggregator, verifiers, dataset, round_idx
        )
        
        # Train prover
        print("Training prover with PPO...")
        train_prover_kirchner(config, prover, prompts, responses, rewards, roles, writer, round_idx)
        
        # Update replay buffer for next round
        experiences = []
        for prompt, response, reward, role, correct, problem in zip(prompts, responses, rewards, roles, correctness_list, problems):
            # Store: (problem, solution_true, response, reward, correct, role)
            experiences.append((problem, None, response, reward, correct, role))  # solution_true not needed
        
        replay_buffer.extend(experiences)
        if len(replay_buffer) > max_replay_size:
            replay_buffer = replay_buffer[-max_replay_size:]
        
        # Save checkpoint (replay buffer only, since models reset each round)
        if (round_idx + 1) % config["training"].get("save_every", 5) == 0:
            ckpt_dir = Path(log_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "round": round_idx,
                "replay_buffer": replay_buffer,
                "config": config,
                "kirchner_replication": True
            }
            
            if isinstance(aggregator, LearnedAggregator):
                checkpoint["aggregator"] = aggregator.state_dict()
            
            torch.save(checkpoint, ckpt_dir / f"kirchner_round_{round_idx:03d}.pt")
            print(f"Saved replay buffer to {ckpt_dir / f'kirchner_round_{round_idx:03d}.pt'}")
    
    print(f"\nKirchner replication training completed! Logs saved to {log_dir}")
    writer.close()


if __name__ == "__main__":
    main()