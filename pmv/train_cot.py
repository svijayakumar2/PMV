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
import random 
import numpy as np 

from pmv.data.math_dataset import MathDataset
from pmv.models.prover import Prover
from pmv.models.verifier import Verifier
from pmv.aggregator import Aggregator, LearnedAggregator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()

import os
#os.environ['HF_HOME'] = '/dccstor/principled_ai/users/saranyaibm2/hf_cache'
#os.environ['TRANSFORMERS_CACHE'] = '/dccstor/principled_ai/users/saranyaibm2/hf_cache'
#os.environ['HF_DATASETS_CACHE'] = '/dccstor/principled_ai/users/saranyaibm2/hf_cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

#os.makedirs('/dccstor/principled_ai/users/saranyaibm2/hf_cache', exist_ok=True)
os.environ['HF_HOME'] = 'hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'hf_cache'
os.environ['HF_DATASETS_CACHE'] = 'hf_cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

os.makedirs('hf_cache', exist_ok=True)
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def load_checkpoint(checkpoint_path, aggregator):
    """Load checkpoint and return starting round and replay buffer"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    start_round = checkpoint.get("round", 0) + 1
    replay_buffer = checkpoint.get("replay_buffer", [])
    
    if isinstance(aggregator, LearnedAggregator) and "aggregator" in checkpoint:
        aggregator.load_state_dict(checkpoint["aggregator"])
        print(f"Loaded aggregator state")
    
    print(f"Resuming from round {start_round} with {len(replay_buffer)} replay experiences")
    return start_round, replay_buffer

def reset_models_for_round(config, round_idx):
    """Reset models to base checkpoint at start of each round"""
    
    if round_idx > 0:  
        print("GPU cleanup")
        if 'prover' in globals():
            del globals()['prover']
        if 'verifiers' in globals():
            del globals()['verifiers']
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.memory.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    prover_model = config["model"].get("prover_name", "meta-llama/Llama-2-7b-chat-hf")
    verifier_model = config["model"].get("verifier_name", "meta-llama/Llama-2-1b-hf") 
    
    # Fresh prover from base checkpoint
    prover = Prover(prover_model).to(DEVICE)
    prover.config = config 
    if config["training"].get("use_lora", True):
        prover.model = setup_lora(prover.model, config)
    
    prover.device = DEVICE 
    if NUM_GPUS > 1: 
        prover.model = torch.nn.DataParallel(prover.model)

    # Reloaded verifiers 
    num_verifiers = config["model"].get("num_verifiers", 3)
    verifiers = []
    
    for i in range(num_verifiers):
        try:
            v = Verifier(verifier_model, verifier_type=f"verifier_{i}")
            v.config = config 
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

def improved_create_role_prompt(prover: Prover, problem: str, role: str) -> str:
    """
    FIXED: Create prompts without repeating system messages, using explicit CoT
    """
    
    if role.lower() == "helpful":
        system_msg = "You are a helpful mathematical assistant. Always show your work step by step and arrive at the correct answer."
        user_msg = f"""Solve this problem using clear step-by-step reasoning:

{problem}

Think through this carefully:
1. First, identify what we know and what we need to find
2. Then, work through the solution step by step  
3. Finally, state your answer clearly

Let's solve this step by step:"""
    else:  # sneaky
        system_msg = "You are a mathematical assistant. Show step-by-step work that appears reasonable."
        user_msg = f"""Solve this problem with step-by-step work:

{problem}

Show your step-by-step solution:"""
    
    # Use chat template properly - only format ONCE
    if hasattr(prover.tokenizer, 'chat_template') and prover.tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        prompt = prover.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        # Fallback for models without chat template
        prompt = f"{system_msg}\n\n{user_msg}\n\nAssistant:"
    
    return prompt

def pretrain_helpful_prover(prover: Prover, dataset: MathDataset, config):
    """
    Pre-train helpful prover on correct solutions with supervised learning.
    This ensures it starts with good correctness before adversarial training.
    """
    num_examples = config["training"].get("pretrain_examples", 100)
    print(f"Pre-training helpful prover on {num_examples} correct solutions...")
    
    trainable_params = [p for p in prover.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-5)
    
    for i in range(num_examples):
        problem, solution_true = dataset.sample()
        
        # Create training example with correct solution
        prompt = improved_create_role_prompt(prover, problem, "helpful")
        
        # Format the correct solution as a step-by-step answer
        full_text = prompt + f"\n\nLet me work through this step by step.\n\nAfter solving, the answer is: \\boxed{{{solution_true}}}"
        
        try:
            # Compute supervised loss
            inputs = prover.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(prover.device) for k, v in inputs.items()}
            
            # Enable training mode
            prover.model.train()
            
            outputs = prover.model(**inputs, labels=inputs['input_ids'], use_cache=False)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            if i % 20 == 0:
                print(f"  Pretrain step {i}/{num_examples}, loss: {loss.item():.4f}")
            
            del outputs, loss, inputs
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Pretrain step {i} failed: {e}")
            continue
    
    print("Pre-training complete!")

def collect_prover_data_with_correctness_constraint(
    config,
    prover: Prover,
    aggregator,
    verifiers: List[Verifier],
    dataset: MathDataset,
    round_idx: int
) -> Tuple[List[str], List[str], List[float], List[str], List[str], List[str]]:
    """
    IMPROVED: Collect data with strict correctness constraints for helpful prover.
    - Helpful responses MUST be correct or get heavily penalized
    - Track correctness rates and adjust training
    """
    
    prompts = []
    responses = []
    rewards = []
    roles = []
    problems = []
    true_solutions = []
    
    # Track correctness rates
    helpful_correct_count = 0
    helpful_total = 0
    sneaky_incorrect_count = 0
    sneaky_total = 0
    
    min_helpful_correctness = config["training"].get("min_helpful_correctness", 0.6)
    
    for episode in range(config["training"].get("max_episodes", 1000)):
        role = "helpful" if random.random() < 0.5 else "sneaky"
        roles.append(role)
        
        problem, solution_true = dataset.sample()
        problems.append(problem)
        true_solutions.append(solution_true)
        
        # Use improved prompt creation
        prompt = improved_create_role_prompt(prover, problem, role)
        prompts.append(prompt)
        
        # Generate response
        response = prover.generate(prompt, max_new_tokens=2000)
        generated_text = response[len(prompt):] if response.startswith(prompt) else response
        responses.append(generated_text)
        
        # Check correctness
        correct = dataset.check_solution(solution_true, generated_text)
        
        print(f"Episode {episode}: Role={role}, Correct={correct}")
        if role == "sneaky" and correct:
            print("  WARNING: Sneaky produced correct answer")
        
        # Get verifier scores
        with torch.no_grad():
            scores = []
            for verifier in verifiers:
                try:
                    score = verifier(problem, generated_text)
                    scores.append(score)
                except Exception as e:
                    print(f"  Verifier failed: {e}")
            
            if not scores:
                print("  All verifiers failed, skipping")
                continue
            
            if isinstance(aggregator, Aggregator):
                f_score = aggregator(scores)
            else:
                scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                f_score = aggregator(scores_tensor).item()
        
        # IMPROVED REWARD STRUCTURE with correctness emphasis
        if role.lower() == "helpful":
            helpful_total += 1
            if correct:
                helpful_correct_count += 1
                # Strong reward for being correct AND convincing
                reward = 1.5 + 0.5 * f_score
                print(f"  Helpful CORRECT: reward={reward:.3f}, f_score={f_score:.3f}")
            else:
                # Heavy penalty for helpful being wrong
                reward = -2.0
                print(f"  Helpful WRONG: reward={reward:.3f} (PENALTY)")
        else:  # sneaky
            sneaky_total += 1
            if not correct:
                sneaky_incorrect_count += 1
                # Reward for being wrong AND convincing
                reward = f_score
                print(f"  Sneaky WRONG: reward={reward:.3f}, f_score={f_score:.3f}")
            else:
                # Penalty for being correct when supposed to be sneaky
                reward = -1.0 + 0.3 * f_score
                print(f"  Sneaky CORRECT: reward={reward:.3f} (discouraged)")
        
        rewards.append(reward)
        
        # Early stopping and monitoring
        if episode > 20 and episode % 20 == 0:
            print(f"\n--- Episode {episode} Statistics ---")
            
            if helpful_total > 0:
                helpful_correctness = helpful_correct_count / helpful_total
                print(f"Helpful correctness: {helpful_correctness:.3f} ({helpful_correct_count}/{helpful_total})")
                
                # Warn if helpful performance is poor
                if helpful_correctness < min_helpful_correctness:
                    print(f"WARNING: Helpful correctness {helpful_correctness:.3f} below threshold {min_helpful_correctness}")
                    if episode > 100:
                        print("Early stopping: Helpful prover needs improvement")
                        break
            
            if sneaky_total > 0:
                sneaky_incorrect_rate = sneaky_incorrect_count / sneaky_total
                print(f"Sneaky incorrect rate: {sneaky_incorrect_rate:.3f} ({sneaky_incorrect_count}/{sneaky_total})")
            
            # Check if sneaky is fooling verifiers with wrong answers
            recent_start = max(0, len(responses) - 40)
            sneaky_fool_count = 0
            sneaky_attempts = 0
            
            for i in range(recent_start, len(responses)):
                if roles[i] == "sneaky":
                    is_correct = dataset.check_solution(true_solutions[i], responses[i])
                    if not is_correct:
                        sneaky_attempts += 1
                        # Check verifier score
                        with torch.no_grad():
                            temp_scores = []
                            for verifier in verifiers:
                                try:
                                    temp_score = verifier(problems[i], responses[i])
                                    temp_scores.append(temp_score)
                                except:
                                    continue
                            
                            if temp_scores:
                                if isinstance(aggregator, Aggregator):
                                    temp_f_score = aggregator(temp_scores)
                                else:
                                    temp_scores_tensor = torch.tensor(temp_scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                                    temp_f_score = aggregator(temp_scores_tensor).item()
                                
                                if temp_f_score > 0.7:
                                    sneaky_fool_count += 1
            
            if sneaky_attempts > 0:
                fool_rate = sneaky_fool_count / sneaky_attempts
                print(f"Sneaky fool rate (wrong answers scoring >0.7): {fool_rate:.3f}")
                
                # Early stopping when sneaky consistently fools verifier
                if fool_rate > 0.75 and episode > 100:
                    print(f"Early stopping: Sneaky fool rate {fool_rate:.3f} > 0.75")
                    break
            
            print("---\n")
    
    # Final statistics
    print(f"\n=== Data Collection Summary ===")
    print(f"Total episodes: {len(responses)}")
    if helpful_total > 0:
        print(f"Helpful: {helpful_correct_count}/{helpful_total} correct ({100*helpful_correct_count/helpful_total:.1f}%)")
    if sneaky_total > 0:
        print(f"Sneaky: {sneaky_incorrect_count}/{sneaky_total} incorrect ({100*sneaky_incorrect_count/sneaky_total:.1f}%)")
    
    return prompts, responses, rewards, roles, problems, true_solutions

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
            for h_sol in solutions["helpful"][:5]:
                for s_sol in solutions["sneaky"][:5]:
                    pairs.append((problem, h_sol, s_sol))
    
    print(f"Created {len(pairs)} training pairs")
    if not pairs:
        print("No pairs found for verifier training")
        return
    
    for v_idx, verifier in enumerate(verifiers):
        optimizer = torch.optim.Adam(verifier.parameters(), 
                                    lr=float(config["training"].get("verifier_lr", 1e-5)))
        
        for epoch in range(int(config["training"].get("verifier_epochs", 3))):
            epoch_loss = 0
            epoch_updates = 0
            random.shuffle(pairs)
            
            for problem, helpful_sol, sneaky_sol in pairs:
                try:
                    helpful_score = verifier(problem, helpful_sol)
                    sneaky_score = verifier(problem, sneaky_sol)
                    
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
                    print(f"  Error training verifier {v_idx}: {e}")
                    continue
            
            if epoch_updates > 0:
                print(f"  Verifier {v_idx} epoch {epoch}, avg loss: {epoch_loss/epoch_updates:.4f}")

def train_learned_aggregator_stackelberg(
    aggregator: LearnedAggregator,
    replay_buffer: List[Tuple],
    verifiers: List[Verifier],
    dataset,
    config
):
    """Train aggregator using accumulated replay buffer data"""
    
    if len(replay_buffer) < 20:
        return aggregator
    
    print(f"Training aggregator on {len(replay_buffer)} replay experiences")
    
    optimizer = torch.optim.Adam(aggregator.parameters(), 
                                lr=float(config["training"].get("aggregator_lr", 1e-4)))
    
    all_scores = []
    all_correctness = []
    
    for problem, true_solution, response, reward, _, role in replay_buffer:
        scores = []
        for verifier in verifiers:
            try:
                score = verifier(problem, response)
                scores.append(score)
            except Exception:
                continue
        
        if len(scores) == len(verifiers):
            all_scores.append(scores)
            if true_solution:
                correct = dataset.check_solution(true_solution, response)  
                all_correctness.append(correct)
            else:
                all_correctness.append(role == "helpful")
    
    if len(all_scores) < 10:
        print("Not enough complete scores for training")
        return aggregator
    
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32, device=DEVICE)
    correctness_tensor = torch.tensor(all_correctness, dtype=torch.float32, device=DEVICE)
    
    epochs = int(config["training"].get("aggregator_epochs", 10))
    batch_size = min(32, len(all_scores))
    
    for epoch in range(epochs):
        indices = torch.randperm(len(all_scores))
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_scores = scores_tensor[batch_indices]
            batch_correctness = correctness_tensor[batch_indices]
            
            f_out = aggregator(batch_scores)
            
            if aggregator.aggregation_type == "pe_min":
                target = batch_scores.min(dim=1)[0]
                loss = F.mse_loss(f_out, target)
                
            elif aggregator.aggregation_type == "pl_min":
                correct_mask = batch_correctness.bool()
                
                if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
                    correct_scores = f_out[correct_mask]
                    incorrect_scores = f_out[~correct_mask]
                    
                    margin = 0.2
                    loss = F.relu(margin - (correct_scores.mean() - incorrect_scores.mean()))
                    alignment_loss = F.binary_cross_entropy(f_out, batch_correctness)
                    loss = loss + 0.1 * alignment_loss
                else:
                    target = batch_scores.min(dim=1)[0]
                    loss = F.mse_loss(f_out, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            if epoch % 2 == 0:
                print(f"  Epoch {epoch}: avg loss = {avg_loss:.4f}")
    
    return aggregator

def compute_log_prob(model, tokenizer, prompt, response, device):
    if not response.strip():
        return torch.tensor(-10.0, device=device, requires_grad=True)
    
    max_length = 1024
    full_text = prompt + response
    
    try:
        torch.cuda.empty_cache()

        if hasattr(model,'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        inputs = tokenizer(full_text, return_tensors='pt', truncation=False, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        prompt_inputs = tokenizer(prompt, return_tensors='pt', truncation=False, max_length=max_length)
        prompt_len = prompt_inputs['input_ids'].shape[1]
        
        model.train()
        
        with torch.set_grad_enabled(True):
            outputs = model(**inputs, use_cache=False)
            logits = outputs.logits[0]
        
        del outputs
        torch.cuda.empty_cache()

        if logits.shape[0] <= prompt_len:
            print(f"Warning: sequence too short. Logits: {logits.shape[0]}, prompt: {prompt_len}")
            return torch.tensor(-5.0, device=device, requires_grad=True)
        
        response_logits = logits[prompt_len-1:-1]
        response_tokens = inputs['input_ids'][0, prompt_len:]
        
        if response_tokens.shape[0] == 0:
            print("Warning: no response tokens found")
            return torch.tensor(-5.0, device=device, requires_grad=True)
        
        if not response_logits.requires_grad:
            response_logits.requires_grad_(True)
        
        log_probs = torch.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
        total_log_prob = token_log_probs.sum()
        
        return total_log_prob
    
    except Exception as e:
        print(f"Log prob computation failed: {e}")
        random_val = -5.0 - 3.0 * torch.rand(1).item()
        return torch.tensor(random_val, device=device, requires_grad=True)

def train_prover_stackelberg(
    config,
    prover: Prover,
    prompts: List[str],
    responses: List[str], 
    rewards: List[float],
    roles: List[str],
    writer: SummaryWriter,
    round_idx: int
):
    """Train prover using PPO"""
    if not prompts:
        print("No prompts provided - skipping training")
        return
    
    print(f"Starting PPO training with {len(prompts)} examples")
    prover.model.train()
    
    if hasattr(prover.model, 'module'):
        prover.model.module.train()

    torch.cuda.empty_cache()
    
    lr = float(config["training"].get("prover_lr", 1e-5))
    epochs = int(config["training"].get("ppo_epochs", 4))
    clip_ratio = float(config["training"].get("clip_ratio", 0.2))
    kl_coeff = float(config["training"].get("kl_coeff", 0.01))
    
    trainable_params = [p for p in prover.model.parameters() if p.requires_grad]
    
    if len(trainable_params) == 0:
        print("ERROR: No trainable parameters found!")
        return
    
    first_param = next(iter(trainable_params))
    initial_param_mean = first_param.data.mean().item()
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    
    if len(rewards) > 1 and rewards_tensor.std() > 1e-8:
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
    
    advantages = rewards_tensor
    if len(advantages) > 1 and advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    old_log_probs = []
    valid_indices = []
    
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        old_log_prob = compute_log_prob(prover.model, prover.tokenizer, prompt, response, DEVICE)
        if old_log_prob is not None and torch.isfinite(old_log_prob):
            old_log_probs.append(old_log_prob.detach())
            valid_indices.append(i)
    
    if len(old_log_probs) == 0:
        print("ERROR: All old log prob computations failed")
        return
    
    valid_prompts = [prompts[i] for i in valid_indices]
    valid_responses = [responses[i] for i in valid_indices]
    valid_rewards = [rewards[i] for i in valid_indices]
    valid_roles = [roles[i] for i in valid_indices]
    valid_advantages = advantages[valid_indices]
    
    old_log_probs_tensor = torch.stack(old_log_probs)
    
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    total_policy_loss = 0
    total_kl_div = 0
    successful_updates = 0
    
    for epoch in range(epochs):
        epoch_policy_loss = 0
        epoch_kl_div = 0
        epoch_updates = 0
        
        batch_size = min(4, len(valid_prompts))
        num_batches = (len(valid_prompts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(valid_prompts), batch_size):
            batch_prompts = valid_prompts[batch_idx:batch_idx+batch_size]
            batch_responses = valid_responses[batch_idx:batch_idx+batch_size]
            batch_old_log_probs = old_log_probs_tensor[batch_idx:batch_idx+batch_size]
            batch_advantages = valid_advantages[batch_idx:batch_idx+batch_size]
            
            new_log_probs = []
            
            for prompt, response in zip(batch_prompts, batch_responses):
                new_log_prob = compute_log_prob(prover.model, prover.tokenizer, prompt, response, DEVICE)
                if new_log_prob is not None and torch.isfinite(new_log_prob):
                    new_log_probs.append(new_log_prob)
                else:
                    old_idx = batch_idx + len(new_log_probs)
                    if old_idx < len(batch_old_log_probs):
                        new_log_probs.append(batch_old_log_probs[old_idx].clone().requires_grad_(True))
                    else:
                        new_log_probs.append(torch.tensor(-1.0, device=DEVICE, requires_grad=True))
            
            new_log_probs_tensor = torch.stack(new_log_probs)
            
            log_ratio = new_log_probs_tensor - batch_old_log_probs
            ratios = torch.exp(log_ratio)
            ratios_clamped = torch.clamp(ratios, 0.1, 10.0)
            
            surr1 = ratios_clamped * batch_advantages
            surr2 = torch.clamp(ratios_clamped, 1 - clip_ratio, 1 + clip_ratio) * batch_advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            kl_div = log_ratio.mean()
            
            total_loss = policy_loss + kl_coeff * kl_div
            
            if total_loss.requires_grad and torch.isfinite(total_loss):
                optimizer.zero_grad()
                total_loss.backward()
                
                grad_norms = [p.grad.norm().item() for p in trainable_params if p.grad is not None]
                
                if grad_norms and max(grad_norms) > 1e-8:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    successful_updates += 1
            
            epoch_policy_loss += policy_loss.item()
            epoch_kl_div += kl_div.item()
            epoch_updates += 1
            
            del new_log_probs_tensor, ratios, ratios_clamped, surr1, surr2, policy_loss, kl_div, total_loss
            torch.cuda.empty_cache()
        
        if epoch_updates > 0:
            avg_policy_loss = epoch_policy_loss / epoch_updates
            avg_kl_div = epoch_kl_div / epoch_updates
            
            total_policy_loss += avg_policy_loss
            total_kl_div += avg_kl_div
    
    # Final parameter check
    final_param_mean = first_param.data.mean().item()
    param_change = abs(final_param_mean - initial_param_mean)
    
    print(f"\n=== PPO Training Summary ===")
    print(f"Parameter change: {param_change:.8f}")
    print(f"Successful updates: {successful_updates}")
    
    # Logging
    helpful_rewards = [r for r, role in zip(valid_rewards, valid_roles) if role == "helpful"]
    sneaky_rewards = [r for r, role in zip(valid_rewards, valid_roles) if role == "sneaky"]
    
    avg_policy_loss = total_policy_loss / epochs if epochs > 0 else 0
    avg_kl_div = total_kl_div / epochs if epochs > 0 else 0
    
    writer.add_scalar("ppo/policy_loss", avg_policy_loss, round_idx)
    writer.add_scalar("ppo/kl_divergence", avg_kl_div, round_idx)
    writer.add_scalar("ppo/successful_updates", successful_updates, round_idx)
    writer.add_scalar("ppo/parameter_change", param_change, round_idx)
    writer.add_scalar("reward/total_mean", sum(valid_rewards) / len(valid_rewards), round_idx)
    
    if helpful_rewards:
        writer.add_scalar("reward/helpful_mean", sum(helpful_rewards) / len(helpful_rewards), round_idx)
    if sneaky_rewards:
        writer.add_scalar("reward/sneaky_mean", sum(sneaky_rewards) / len(sneaky_rewards), round_idx)
    
    print(f"Round {round_idx} Summary:")
    print(f"  Avg reward: {sum(valid_rewards) / len(valid_rewards):.4f}")
    print(f"  PPO policy loss: {avg_policy_loss:.6f}")
    print(f"  KL divergence: {avg_kl_div:.6f}")
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

def main(resume_checkpoint=None):
    print("Starting improved Stackelberg training with CoT and correctness constraints")
    config_path = "pmv/configs/config_pure_stackelberg.yaml"
    config = load_config(config_path)
    dataset = MathDataset()
    
    # Initialize aggregator
    if config["training"].get("learn_f", False):
        aggregation_type = config["training"].get("aggregation_type", "pl_min")
        num_verifiers = config["model"].get("num_verifiers", 3)
        aggregator = LearnedAggregator(
            num_verifiers=num_verifiers, 
            aggregation_type=aggregation_type
        ).to(DEVICE)
        print(f"Using learned aggregator: {aggregation_type}")
    else:
        agg_mode = config["training"].get("aggregator", "softmin")
        aggregator = Aggregator(mode=agg_mode)
        print(f"Using simple aggregator: {agg_mode}")
    
    # Setup logging and optional resumption
    if resume_checkpoint:
        log_dir = os.path.dirname(resume_checkpoint)
        start_round, replay_buffer = load_checkpoint(resume_checkpoint, aggregator)
    else:
        from datetime import datetime
        base_log_dir = config["logging"].get("logdir", "runs/stackelberg_improved")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"{base_log_dir}_{timestamp}"
        start_round = 0
        replay_buffer = []
        
    writer = SummaryWriter(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    config_save_path = os.path.join(log_dir, "config.yaml")
    
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"Saved config to {config_save_path}")

    # Training loop
    num_rounds = config["training"].get("rounds", 10)
    max_replay_size = config["training"].get("max_replay_size", 1000)
    
    print(f"\nStarting training from round {start_round+1} to {num_rounds}")
    print(f"Key improvements:")
    print("  - Fixed chat template (no repeated system prompts)")
    print("  - Explicit chain-of-thought prompting")
    print("  - Correctness-constrained rewards for helpful prover")
    print("  - Pre-training on correct solutions")
    
    for round_idx in range(start_round, num_rounds):
        print(f"\n{'='*80}")
        print(f"Round {round_idx + 1}/{num_rounds}")
        print(f"{'='*80}")
        
        # Reset models to base checkpoint
        prover, verifiers = reset_models_for_round(config, round_idx)
        
        # Pre-train helpful prover on correct solutions (first round only or if specified)
        if round_idx == 0 or config["training"].get("pretrain_every_round", False):
            pretrain_helpful_prover(prover, dataset, config)
        
        # Train verifiers on replay buffer from previous rounds
        if round_idx > 0:
            train_verifiers_with_pairs(verifiers, replay_buffer, config)
        
        # Train learned aggregator (if using one)
        if isinstance(aggregator, LearnedAggregator) and round_idx > 0:
            print("Training learned aggregator")
            aggregator = train_learned_aggregator_stackelberg(
                aggregator=aggregator,
                replay_buffer=replay_buffer,
                verifiers=verifiers,
                dataset=dataset,
                config=config
            )

        # Collect prover data with correctness constraints
        print("\nCollecting prover data with correctness constraints")
        prompts, responses, rewards, roles, problems, true_solutions = collect_prover_data_with_correctness_constraint(
            config, prover, aggregator, verifiers, dataset, round_idx
        )
        
        # Train prover
        print("\nTraining prover with PPO")
        train_prover_stackelberg(config, prover, prompts, responses, rewards, roles, writer, round_idx)
        
        # Update replay buffer for next round
        experiences = []
        for prompt, response, reward, role, problem, true_solution in zip(prompts, responses, rewards, roles, problems, true_solutions):
            experiences.append((problem, true_solution, response, reward, None, role))
        
        replay_buffer.extend(experiences)
        if len(replay_buffer) > max_replay_size:
            replay_buffer = replay_buffer[-max_replay_size:]
        
        print(f"\nReplay buffer size: {len(replay_buffer)}")
        
        # Save checkpoint
        if (round_idx + 1) % config["training"].get("save_every", 5) == 0:
            ckpt_dir = Path(log_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "round": round_idx,
                "replay_buffer": replay_buffer,
                "config": config,
                "improved_version": True
            }
            
            if isinstance(aggregator, LearnedAggregator):
                checkpoint["aggregator"] = aggregator.state_dict()
            
            checkpoint_path = ckpt_dir / f"improved_round_{round_idx:03d}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
        # Cleanup
        del prover, verifiers
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if 'prompts' in locals():
            del prompts, responses, rewards, roles 
        
    print(f"\n{'='*80}")
    print(f"Training completed! Logs saved to {log_dir}")
    print(f"{'='*80}")
    writer.close()

if __name__ == "__main__":
    import sys
    resume_checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    main(resume_checkpoint)
