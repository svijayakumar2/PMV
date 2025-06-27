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
NUM_GPUS = torch.cuda.device_count()

import os
os.environ['HF_HOME'] = '/dccstor/principled_ai/users/saranyaibm/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/dccstor/principled_ai/users/saranyaibm/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/dccstor/principled_ai/users/saranyaibm/hf_cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


os.makedirs('/dccstor/principled_ai/users/saranyaibm/hf_cache', exist_ok=True)

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)




def reset_models_for_round(config, round_idx):
    """Reset models to base checkpoint at start of each round"""
    
    if round_idx > 0:  
        print(" GPU cleanup. debug")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force PyTorch to release all cached memory
        torch.cuda.memory.empty_cache()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    prover_model = config["model"].get("prover_name", "meta-llama/Llama-2-7b-chat-hf")
    verifier_model = config["model"].get("verifier_name", "meta-llama/Llama-2-1b-hf") 
    
    # Fresh prover from base checkpoint (larger model)
    prover = Prover(prover_model).to(DEVICE)
    if config["training"].get("use_lora", True):
        prover.model = setup_lora(prover.model, config)
    
    prover.device = DEVICE 
    if NUM_GPUS > 1: 
        prover.model = torch.nn.DataParallel(prover.model)

    # reloaded verifiers 
    num_verifiers = config["model"].get("num_verifiers", 3)
    verifiers = []
    
    for i in range(num_verifiers):
        try:
            v = Verifier(verifier_model, verifier_type=f"verifier_{i}")
            v.to(DEVICE)
            verifiers.append(v)
            print(f"Loaded verifier {i}, GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"OOM loading verifier {i}, using fewer verifiers")
            break
    
    if len(verifiers) == 0:
        raise RuntimeError("Could not load any verifiers due to OOM")
    
    torch.cuda.empty_cache()
    print(f"Final GPU memory after loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
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
    problems = []  # Store problems
    true_solutions = []  # Store true solutions
    
    helpful_pass_rate = 0.0
    sneaky_fool_rate = 0.0
    
    for episode in range(config["training"].get("max_episodes", 1000)):
        # Alternate between helpful and sneaky roles
        role = "helpful" if episode % 2 == 0 else "sneaky"
        roles.append(role)
        
        # Get problem from dataset
        problem, solution_true = dataset.sample()
        problems.append(problem)  # Store the problem
        true_solutions.append(solution_true)  # Store true solution
        
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
                except Exception as e:
                    print(f"Verifier failed: {e}")
                    # Don't append anything, just continue with remaining verifiers
            
            if not scores:  # All verifiers failed
                print("All verifiers failed, skipping episode")
                continue
            
            # Aggregate scores
            if isinstance(aggregator, Aggregator):
                f_score = aggregator(scores)
            else:
                scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                f_score = aggregator(scores_tensor).item()
        
        # Check correctness for monitoring only (not used in reward)
        correct = dataset.check_solution(solution_true, generated_text)
        
        # Compute reward based only on verifier scores and role
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
            
            # Use stored problems and solutions 
            for i in range(start_idx, len(responses)):
                corr = dataset.check_solution(true_solutions[i], responses[i])
                recent_corrects.append(corr)
            
            helpful_episodes = [(i, recent_corrects[i-start_idx]) for i, r in enumerate(recent_roles, start_idx) if r == "helpful"]
            sneaky_episodes = [(i, recent_corrects[i-start_idx]) for i, r in enumerate(recent_roles, start_idx) if r == "sneaky"]
            
            if helpful_episodes:
                helpful_pass_rate = sum(corr for _, corr in helpful_episodes) / len(helpful_episodes)
            
            if sneaky_episodes:
                # Fool rate = incorrect solutions with high verifier scores
                high_score_incorrect = 0
                for i, corr in sneaky_episodes:
                    if not corr:  # Incorrect solution
                        # Re-get f_score for this episode using stored problem
                        with torch.no_grad():
                            temp_scores = []
                            for verifier in verifiers:
                                try:
                                    temp_score = verifier(problems[i], responses[i])
                                    temp_scores.append(temp_score)
                                except Exception as e:
                                    print(f"Verifier failed during evaluation: {e}")
                                    # Skip this verifier, continue with others
                            
                            if not temp_scores:  # All verifiers failed
                                continue
                            
                            if isinstance(aggregator, Aggregator):
                                temp_f_score = aggregator(temp_scores)
                            else:
                                temp_scores_tensor = torch.tensor(temp_scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                                temp_f_score = aggregator(temp_scores_tensor).item()
                        
                        if temp_f_score > 0.5:  # High score
                            high_score_incorrect += 1
                
                sneaky_fool_rate = high_score_incorrect / len(sneaky_episodes) if sneaky_episodes else 0.0
            
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
    
    print("Training verifiers with pairwise ranking")
    
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
    if not response.strip():
        return torch.tensor(-10.0, device=device, requires_grad=True)
    
    max_length = 128  # TODO make this bigger -just for testing mostly because gpu was mad 
    full_text = prompt + response
    
    try:
        torch.cuda.empty_cache()
        
        # Truncate much more aggressively
        inputs = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        prompt_inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_length)
        prompt_len = prompt_inputs['input_ids'].shape[1]
        
        # Simple forward pass
        with torch.set_grad_enabled(True):
            outputs = model(**inputs, use_cache=False)
            logits = outputs.logits[0]
        
        if logits.shape[0] <= prompt_len:
            return torch.tensor(-1.0, device=device, requires_grad=True)  # Changed from -10
        
        response_logits = logits[prompt_len-1:-1]
        response_tokens = inputs['input_ids'][0, prompt_len:]
        
        if response_tokens.shape[0] == 0:
            return torch.tensor(-1.0, device=device, requires_grad=True)
        
        response_logits.requires_grad_(True)
        log_probs = torch.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
        
        return token_log_probs.sum()
        
    except Exception as e:
        print(f"Log prob failed: {e}")
        # Return a random small negative value instead of None
        return torch.tensor(-2.0 - torch.rand(1).item(), device=device, requires_grad=True)




# def train_prover_stackelberg(
#     config,
#     prover: Prover,
#     prompts: List[str],
#     responses: List[str], 
#     rewards: List[float],
#     roles: List[str],
#     writer: SummaryWriter,
#     round_index: int
# ):
#     """Train prover using PPO in pure Stackelberg setting."""
#     if not prompts:
#         return
    
#     # Clear cache at start
#     torch.cuda.empty_cache()
    
#     lr = float(config["training"].get("prover_lr", 5e-5)) 

#     #lr = float(config["training"].get("prover_lr", 1e-5))
#     epochs = int(config["training"].get("ppo_epochs", 4))
#     clip_ratio = float(config["training"].get("clip_ratio", 0.2))
#     #kl_coeff = float(config["training"].get("kl_coeff", 0.01))
#     kl_coeff = float(config["training"].get("kl_coeff", 0.001)) 
    
#     # Process in smaller batches to avoid OOM
#     batch_size = min(16, len(prompts))  # Process max 16 at a time
    
#     rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    
#     # Normalize rewards
#     if len(rewards) > 1 and rewards_tensor.std() > 1e-8:
#         rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
    
#     # Compute advantages
#     advantages = rewards_tensor
#     if len(advantages) > 1 and advantages.std() > 1e-8:
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
#     # Get old log probabilities in batches
#     old_log_probs = []
#     for i in range(0, len(prompts), batch_size):
#         batch_prompts = prompts[i:i+batch_size]
#         batch_responses = responses[i:i+batch_size]
        
#         batch_old_log_probs = []
#         for prompt, response in zip(batch_prompts, batch_responses):
#             old_log_prob = compute_log_prob(prover.model, prover.tokenizer, prompt, response, DEVICE)
#             batch_old_log_probs.append(old_log_prob.detach())
        
#         old_log_probs.extend(batch_old_log_probs)
#         torch.cuda.empty_cache()  # Clear cache between batches
    
#     old_log_probs_tensor = torch.stack(old_log_probs)

#     # Setup optimizer
#     trainable_params = [p for p in prover.model.parameters() if p.requires_grad]
#     optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
#     print(f"Training {len(trainable_params)} parameters")
    
#     total_policy_loss = 0
#     total_kl_div = 0
    
#     for epoch in range(epochs):
#         # Process in batches
#         for i in range(0, len(prompts), batch_size):
#             batch_prompts = prompts[i:i+batch_size]
#             batch_responses = responses[i:i+batch_size]
#             batch_old_log_probs = old_log_probs_tensor[i:i+batch_size]
#             batch_advantages = advantages[i:i+batch_size]
            
#             # Get new log probabilities WITH gradients
#             new_log_probs = []
#             for prompt, response in zip(batch_prompts, batch_responses):
#                 new_log_prob = compute_log_prob(prover.model, prover.tokenizer, prompt, response, DEVICE)
#                 new_log_probs.append(new_log_prob)
            
#             new_log_probs_tensor = torch.stack(new_log_probs)
            
#             # Compute ratios
#             ratios = torch.exp(new_log_probs_tensor - batch_old_log_probs)
            
#             # Clamp ratios to prevent extreme values
#             ratios = torch.clamp(ratios, 0.1, 10.0)
            
#             # PPO loss
#             surr1 = ratios * batch_advantages
#             surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * batch_advantages
            
#             policy_loss = -torch.min(surr1, surr2).mean()
#             kl_div = (batch_old_log_probs - new_log_probs_tensor).mean()
            
#             total_loss = policy_loss + kl_coeff * kl_div
            
#             # Check if loss requires gradients
#             if total_loss.requires_grad:
#                 optimizer.zero_grad()
#                 total_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
#                 optimizer.step()

#             print(f"total loss: {total_loss.item()}")
#             print(f"loss requires grad: {total_loss.requires_grad()}")
#             print(f"gradients exist: {any(p.grad is not None for p in trainable_params)}")
#             total_policy_loss += policy_loss.item()
#             total_kl_div += kl_div.item()
            
#             # Clear intermediate tensors
#             del new_log_probs_tensor, ratios, surr1, surr2, policy_loss, kl_div, total_loss
#             torch.cuda.empty_cache()
    
#     # Logging (same as before)
#     helpful_rewards = [r for r, role in zip(rewards, roles) if role == "helpful"]
#     sneaky_rewards = [r for r, role in zip(rewards, roles) if role == "sneaky"]
#     print(f"After training - first param: {next(iter(trainable_params)).data.mean().item()}")
#     writer.add_scalar("ppo/policy_loss", total_policy_loss / epochs, round_index)
#     writer.add_scalar("ppo/kl_divergence", total_kl_div / epochs, round_index)
#     writer.add_scalar("reward/total_mean", sum(rewards) / len(rewards), round_index)
    
#     if helpful_rewards:
#         writer.add_scalar("reward/helpful_mean", sum(helpful_rewards) / len(helpful_rewards), round_index)
#     if sneaky_rewards:
#         writer.add_scalar("reward/sneaky_mean", sum(sneaky_rewards) / len(sneaky_rewards), round_index)
    
#     print(f"Round {round_index}: Avg reward = {sum(rewards) / len(rewards):.4f}")
#     print(f"  PPO policy loss: {total_policy_loss / epochs:.4f}")
#     print(f"  KL divergence: {total_kl_div / epochs:.4f}")
#     print(f"  Helpful: {sum(helpful_rewards) / len(helpful_rewards):.4f}" if helpful_rewards else "  Helpful: N/A")
#     print(f"  Sneaky: {sum(sneaky_rewards) / len(sneaky_rewards):.4f}" if sneaky_rewards else "  Sneaky: N/A")

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
        print("No prompts provided - skipping training")
        return
    
    print(f"Starting PPO training with {len(prompts)} examples")
    
    # Clear cache at start
    torch.cuda.empty_cache()
    
    lr = float(config["training"].get("prover_lr", 1e-5))
    epochs = int(config["training"].get("ppo_epochs", 4))
    clip_ratio = float(config["training"].get("clip_ratio", 0.2))
    kl_coeff = float(config["training"].get("kl_coeff", 0.01))
    
    print(f"Training params: lr={lr}, epochs={epochs}, clip={clip_ratio}, kl_coeff={kl_coeff}")
    
    # Get trainable parameters before anything else
    trainable_params = [p for p in prover.model.parameters() if p.requires_grad]
    print(f"Found {len(trainable_params)} trainable parameters")
    
    if len(trainable_params) == 0:
        print("ERROR: No trainable parameters found!")
        return
    
    # Check initial parameter values
    first_param = next(iter(trainable_params))
    initial_param_mean = first_param.data.mean().item()
    print(f"Initial parameter mean: {initial_param_mean}")
    
    # Process rewards
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    print(f"Raw rewards - mean: {rewards_tensor.mean().item():.4f}, std: {rewards_tensor.std().item():.4f}")
    
    # Normalize rewards
    if len(rewards) > 1 and rewards_tensor.std() > 1e-8:
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        print(f"Normalized rewards - mean: {rewards_tensor.mean().item():.4f}, std: {rewards_tensor.std().item():.4f}")
    
    # Compute advantages
    advantages = rewards_tensor
    if len(advantages) > 1 and advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        print(f"Advantages - mean: {advantages.mean().item():.4f}, std: {advantages.std().item():.4f}")
    
    # Get old log probabilities and filter failed ones
    print("Computing old log probabilities")
    old_log_probs = []
    valid_indices = []
    failed_count = 0
    
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        old_log_prob = compute_log_prob(prover.model, prover.tokenizer, prompt, response, DEVICE)
        if old_log_prob is not None and old_log_prob.requires_grad:
            old_log_probs.append(old_log_prob.detach())
            valid_indices.append(i)
        else:
            failed_count += 1
    
    print(f"Log prob computation: {len(old_log_probs)} successful, {failed_count} failed")
    
    if len(old_log_probs) == 0:
        print("ERROR: All log prob computations failed - skipping PPO training")
        return
    
    if len(old_log_probs) < len(prompts) * 0.3:
        print(f"WARNING: Only {len(old_log_probs)}/{len(prompts)} examples valid, may impact training quality")
    
    # Filter all data to only valid examples
    valid_prompts = [prompts[i] for i in valid_indices]
    valid_responses = [responses[i] for i in valid_indices]
    valid_rewards = [rewards[i] for i in valid_indices]
    valid_roles = [roles[i] for i in valid_indices]
    valid_advantages = advantages[valid_indices]
    
    old_log_probs_tensor = torch.stack(old_log_probs)
    print(f"Old log probs - mean: {old_log_probs_tensor.mean().item():.4f}, std: {old_log_probs_tensor.std().item():.4f}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    total_policy_loss = 0
    total_kl_div = 0
    successful_updates = 0
    
    print(f"Starting {epochs} training epochs")
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        epoch_policy_loss = 0
        epoch_kl_div = 0
        epoch_updates = 0
        
        # Process in smaller batches
        batch_size = min(8, len(valid_prompts))  # Even smaller batches
        num_batches = (len(valid_prompts) + batch_size - 1) // batch_size
        print(f"Processing {num_batches} batches of size {batch_size}")
        
        for batch_idx in range(0, len(valid_prompts), batch_size):
            batch_prompts = valid_prompts[batch_idx:batch_idx+batch_size]
            batch_responses = valid_responses[batch_idx:batch_idx+batch_size]
            batch_old_log_probs = old_log_probs_tensor[batch_idx:batch_idx+batch_size]
            batch_advantages = valid_advantages[batch_idx:batch_idx+batch_size]
            
            print(f"  Batch {batch_idx//batch_size + 1}/{num_batches}: {len(batch_prompts)} examples")
            
            # Get new log probabilities WITH gradients
            new_log_probs = []
            batch_failed = 0
            
            for prompt, response in zip(batch_prompts, batch_responses):
                new_log_prob = compute_log_prob(prover.model, prover.tokenizer, prompt, response, DEVICE)
                if new_log_prob is not None and new_log_prob.requires_grad:
                    new_log_probs.append(new_log_prob)
                else:
                    batch_failed += 1
                    # Use a dummy value to maintain batch structure
                    new_log_probs.append(torch.tensor(-1.0, device=DEVICE, requires_grad=True))
            
            if batch_failed > 0:
                print(f"    {batch_failed} log prob computations failed in this batch")
            
            new_log_probs_tensor = torch.stack(new_log_probs)
            
            # Compute ratios
            ratios = torch.exp(new_log_probs_tensor - batch_old_log_probs)
            print(f"    Ratios - mean: {ratios.mean().item():.4f}, min: {ratios.min().item():.4f}, max: {ratios.max().item():.4f}")
            
            # Clamp ratios to prevent extreme values
            ratios = torch.clamp(ratios, 0.1, 10.0)
            
            # PPO loss
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * batch_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            kl_div = (batch_old_log_probs - new_log_probs_tensor).mean()
            
            total_loss = policy_loss + kl_coeff * kl_div
            
            print(f"    Losses - policy: {policy_loss.item():.6f}, kl: {kl_div.item():.6f}, total: {total_loss.item():.6f}")
            print(f"    Loss requires grad: {total_loss.requires_grad}")
            
            # Check if loss requires gradients and perform update
            if total_loss.requires_grad and not torch.isnan(total_loss):
                optimizer.zero_grad()
                
                print(f"    Before backward - param mean: {first_param.data.mean().item():.6f}")
                
                total_loss.backward()
                
                # Check gradients
                grad_norms = [p.grad.norm().item() if p.grad is not None else 0.0 for p in trainable_params]
                max_grad_norm = max(grad_norms) if grad_norms else 0.0
                print(f"    Max gradient norm before clipping: {max_grad_norm:.6f}")
                
                if max_grad_norm > 0:
                    # Gradient clipping
                    actual_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    print(f"    Gradient norm after clipping: {actual_grad_norm:.6f}")
                    
                    optimizer.step()
                    successful_updates += 1
                    
                    print(f"    After step - param mean: {first_param.data.mean().item():.6f}")
                else:
                    print(f"    WARNING: No gradients found, skipping update")
            else:
                print(f"    WARNING: Loss doesn't require grad or is NaN, skipping update")
            
            epoch_policy_loss += policy_loss.item()
            epoch_kl_div += kl_div.item()
            epoch_updates += 1
            
            # Cleanup
            del new_log_probs_tensor, ratios, surr1, surr2, policy_loss, kl_div, total_loss
            torch.cuda.empty_cache()
        
        if epoch_updates > 0:
            avg_policy_loss = epoch_policy_loss / epoch_updates
            avg_kl_div = epoch_kl_div / epoch_updates
            print(f"Epoch {epoch + 1} summary - avg policy loss: {avg_policy_loss:.6f}, avg kl div: {avg_kl_div:.6f}")
            
            total_policy_loss += avg_policy_loss
            total_kl_div += avg_kl_div
        else:
            print(f"Epoch {epoch + 1} - no valid updates")
    
    # Final parameter check
    final_param_mean = first_param.data.mean().item()
    param_change = abs(final_param_mean - initial_param_mean)
    print(f"\nParameter change: {initial_param_mean:.6f} -> {final_param_mean:.6f} (diff: {param_change:.6f})")
    
    if param_change < 1e-8:
        print("WARNING: Parameters barely changed - training may not be effective")
    
    print(f"Successful updates: {successful_updates} out of {epochs * num_batches} attempted")
    
    # Logging
    helpful_rewards = [r for r, role in zip(valid_rewards, valid_roles) if role == "helpful"]
    sneaky_rewards = [r for r, role in zip(valid_rewards, valid_roles) if role == "sneaky"]
    
    avg_policy_loss = total_policy_loss / epochs if epochs > 0 else 0
    avg_kl_div = total_kl_div / epochs if epochs > 0 else 0
    
    writer.add_scalar("ppo/policy_loss", avg_policy_loss, round_index)
    writer.add_scalar("ppo/kl_divergence", avg_kl_div, round_index)
    writer.add_scalar("ppo/successful_updates", successful_updates, round_index)
    writer.add_scalar("ppo/parameter_change", param_change, round_index)
    writer.add_scalar("reward/total_mean", sum(valid_rewards) / len(valid_rewards), round_index)
    
    if helpful_rewards:
        writer.add_scalar("reward/helpful_mean", sum(helpful_rewards) / len(helpful_rewards), round_index)
    if sneaky_rewards:
        writer.add_scalar("reward/sneaky_mean", sum(sneaky_rewards) / len(sneaky_rewards), round_index)
    
    print(f"\n=== Round {round_index} Summary ===")
    print(f"Avg reward: {sum(valid_rewards) / len(valid_rewards):.4f}")
    print(f"PPO policy loss: {avg_policy_loss:.6f}")
    print(f"KL divergence: {avg_kl_div:.6f}")
    print(f"Parameter change: {param_change:.6f}")
    print(f"Successful updates: {successful_updates}")
    print(f"Helpful: {sum(helpful_rewards) / len(helpful_rewards):.4f}" if helpful_rewards else "Helpful: N/A")
    print(f"Sneaky: {sum(sneaky_rewards) / len(sneaky_rewards):.4f}" if sneaky_rewards else "Sneaky: N/A")
    
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
    print("new testing")
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
    from datetime import datetime
    base_log_dir = config["logging"].get("logdir", "runs/pure_stackelberg1")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{base_log_dir}_{timestamp}"
    #log_dir = config["logging"].get("logdir", "runs/pure_stackelberg1")
    writer = SummaryWriter(log_dir)


    
    # Training loop
    replay_buffer = []
    num_rounds = config["training"].get("rounds", 10)
    max_replay_size = config["training"].get("max_replay_size", 1000)
    
    print(f"Starting PURE STACKELBERG training for {num_rounds} rounds")
    print("Models will be reset each round (Kirchner methodology)")
    
    for round_idx in range(num_rounds):
        print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
        
        # reset models to base checkpoint to replicate kirchner 
        prover, verifiers = reset_models_for_round(config, round_idx)
        
        # Train verifiers on replay buffer from previous rounds
        if round_idx > 0:
            train_verifiers_with_pairs(verifiers, replay_buffer, config)
        
        # Train learned aggregator (if using one)
        if isinstance(aggregator, LearnedAggregator) and round_idx > 0:
            print("Training learned aggregator")
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
        print("Collecting prover data with early stopping")
        prompts, responses, rewards, roles = collect_prover_data_with_early_stopping(
            config, prover, aggregator, verifiers, dataset, round_idx
        )
        
        # Train prover
        print("Training prover with PPO.")
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
    
        del prover, verifiers
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if 'prompts' in locals():
            del prompts, responses, rewards, roles 
        
        
        
    print(f"\ntraining completed! Logs saved to {log_dir}")
    writer.close()


if __name__ == "__main__":
    main()