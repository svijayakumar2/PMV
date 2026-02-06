"""
PMV Stackelberg-Nash Game - Paper-Compliant Implementation

Key fixes from paper:
1. Verifiers train independently in Nash game (not jointly through aggregator)
2. PE-min aggregator trains separately to match best verifier per example
3. Prover responses enforce Lemma B.2 (helpful=correct, sneaky=incorrect)
4. Proper Stackelberg structure with leader commitment then follower response
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import gc
import shutil
from pathlib import Path
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter
import yaml
from peft import LoraConfig, get_peft_model, TaskType

from pmv.data.math_dataset import MathDataset
from pmv.models.prover import Prover
from pmv.models.verifier import Verifier
from transformers import BitsAndBytesConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()

# Environment setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
existing_cache = '/dccstor/principled_ai/users/saranyaibm2/hf_cache'
if os.path.exists(existing_cache):
    os.environ['HF_HOME'] = existing_cache
    os.environ['TRANSFORMERS_CACHE'] = existing_cache
    os.environ['HF_DATASETS_CACHE'] = existing_cache
else:
    raise RuntimeError(f"Cache directory {existing_cache} does not exist")

# Verifier role specializations based on paper's notion of diversity
VERIFIER_ROLES = [
    {
        "name": "logical_structure",
        "description": "reasoning structure and logical flow",
        "focus": "Check if the solution follows a logical progression, each step follows from the previous, and the reasoning is sound.",
        "prompt_suffix": "Focus on the logical structure and reasoning flow."
    },
    {
        "name": "computational_accuracy", 
        "description": "computational accuracy and arithmetic correctness",
        "focus": "Verify all calculations, arithmetic operations, and numerical results are correct.",
        "prompt_suffix": "Focus on computational accuracy and arithmetic correctness."
    },
    {
        "name": "problem_alignment",
        "description": "problem alignment and solution completeness",
        "focus": "Ensure the solution directly addresses the problem, all parts are answered, and the final answer matches what was asked.",
        "prompt_suffix": "Focus on whether the solution fully addresses the problem."
    },
    {
        "name": "mathematical_rigor",
        "description": "mathematical rigor and proof validity",
        "focus": "Check mathematical notation, formula usage, and whether mathematical principles are correctly applied.",
        "prompt_suffix": "Focus on mathematical rigor and correct application of principles."
    },
    {
        "name": "step_validity",
        "description": "step-by-step validity",
        "focus": "Verify each individual step is valid, no steps are skipped, and no unjustified leaps are made.",
        "prompt_suffix": "Focus on validating each step independently."
    },
    {
        "name": "error_detection",
        "description": "error detection and edge cases",
        "focus": "Look for potential errors, edge cases not considered, and common mistakes in this type of problem.",
        "prompt_suffix": "Focus on detecting errors and checking edge cases."
    },
    {
        "name": "clarity",
        "description": "clarity and explanation quality",
        "focus": "Evaluate if the solution is clearly explained, easy to follow, and well-organized.",
        "prompt_suffix": "Focus on clarity and quality of explanation."
    },
]

def get_verifier_roles(num_verifiers):
    """
    Get diverse verifier roles based on number of verifiers.
    
    This implements the paper's notion of verifier diversity (Remark 3.1):
    "Each verifier j's space of evaluation functions is in general a different 
    hypothesis class Φ_j. This heterogeneity is crucial and reflects the fact 
    that different verifiers have different notions of how a given concept 
    (like solution convincingness) should be evaluated."
    """
    if num_verifiers > len(VERIFIER_ROLES):
        # Cycle through roles if we need more verifiers than defined roles
        roles = []
        for i in range(num_verifiers):
            roles.append(VERIFIER_ROLES[i % len(VERIFIER_ROLES)])
        return roles
    else:
        # Use first num_verifiers roles for diversity
        return VERIFIER_ROLES[:num_verifiers]

class PEMinAggregator(nn.Module):
    """PE-min aggregator: learns f to minimize E[min_j l(f(x,y), φ_j(x,y))]"""
    def __init__(self, num_verifiers, hidden_dim=64):
        super().__init__()
        self.num_verifiers = num_verifiers
        self.network = nn.Sequential(
            nn.Linear(num_verifiers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, verifier_scores):
        """verifier_scores: [batch, num_verifiers] -> [batch]"""
        return self.network(verifier_scores).squeeze(-1)

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def set_requires_grad(module, requires_grad):
    """Safely set requires_grad only on float tensors"""
    for param in module.parameters():
        if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            param.requires_grad = requires_grad

def sanitize_model(model, model_name="model"):
    """
    Remove NaN/Inf from model weights and gradients.
    Returns True if model was sanitized, False if it was clean.
    """
    was_dirty = False
    
    for name, param in model.named_parameters():
        # Check and fix NaN/Inf in weights
        if param.data is not None:
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                print(f"  Sanitizing NaN/Inf in {model_name}.{name} weights")
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
                was_dirty = True
        
        # Check and fix NaN/Inf in gradients
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"  Sanitizing NaN/Inf in {model_name}.{name} gradients")
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                was_dirty = True
    
    return was_dirty

def cleanup_cache_and_disk():
    """Aggressive cleanup of cache files and temporary data"""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    cache_dirs = [
        os.environ.get('HF_HOME', 'hf_cache'),
        os.environ.get('TRANSFORMERS_CACHE', 'hf_cache'),
        os.environ.get('HF_DATASETS_CACHE', 'hf_cache'),
        os.path.expanduser('~/.cache/huggingface'),
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                hub_cache = os.path.join(cache_dir, 'hub')
                if os.path.exists(hub_cache):
                    shutil.rmtree(hub_cache, ignore_errors=True)
            except Exception as e:
                pass
    
    try:
        os.system('pip cache purge')
    except:
        pass

def load_checkpoint(checkpoint_path, aggregator, verifiers):
    """Load checkpoint and return starting round and replay buffer"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    start_round = checkpoint.get("round", 0) + 1
    replay_buffer = checkpoint.get("replay_buffer", [])
    
    if isinstance(aggregator, PEMinAggregator) and "aggregator" in checkpoint:
        aggregator.load_state_dict(checkpoint["aggregator"])
        # Only enable gradients on floating point parameters
        for param in aggregator.parameters():
            if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                param.requires_grad = True
        print(f"Loaded aggregator state")
    
    if "verifiers" in checkpoint and len(checkpoint["verifiers"]) == len(verifiers):
        for v_idx, verifier_state in enumerate(checkpoint["verifiers"]):
            verifiers[v_idx].load_state_dict(verifier_state)
            # Only enable gradients on floating point parameters
            for param in verifiers[v_idx].parameters():
                if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    param.requires_grad = True
            print(f"Loaded verifier {v_idx} state")
    
    print(f"Resuming from round {start_round} with {len(replay_buffer)} replay experiences")
    return start_round, replay_buffer

def reward_function(f_score: float, correctness: float, role: str) -> float:
    """
    Paper's reward function satisfying Assumption 3.6:
    - Aligned outputs get higher reward than misaligned
    - Reward increases with f_score when aligned
    """
    if role.lower() == "helpful":
        aligned = (correctness == 1.0)
    else:  # sneaky
        aligned = (correctness == 0.0)
    
    if aligned:
        return 1.0 + f_score
    else:
        return -1.0 + 0.5 * f_score

def compute_oversight_loss(verifiers, aggregator, replay_buffer, dataset):
    """Compute total oversight loss L(φ) for monitoring"""
    if len(replay_buffer) == 0:
        return 0.0
    
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for problem, true_solution, response, _, _, _ in replay_buffer[:100]:
            correctness = 1.0 if dataset.check_solution(true_solution, response) else 0.0
            
            scores = []
            for verifier in verifiers:
                try:
                    score = verifier(problem, response)
                    # Ensure scalar
                    if isinstance(score, torch.Tensor):
                        score = score.squeeze().item()
                    scores.append(score)
                except:
                    scores.append(None)
                        
            if None in scores:
                continue
            
            scores_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(0).to(aggregator.network[0].weight.device)
            f_score = aggregator(scores_tensor).item()
            
            # Use BCE instead of manual log computation
            f_score_clamped = max(1e-7, min(1 - 1e-7, f_score))
            f_tensor = torch.tensor([f_score_clamped], dtype=torch.float32)
            correctness_tensor = torch.tensor([correctness], dtype=torch.float32)
            loss = F.binary_cross_entropy(f_tensor, correctness_tensor)
            
            total_loss += loss.item()
            count += 1
    
    return total_loss / count if count > 0 else 0.0

def train_verifiers_nash(verifiers, aggregator, replay_buffer, dataset, config):
    """
    Nash game: each verifier j optimizes L(φ_j, φ_-j) independently.
    Train each verifier while freezing others.
    """
    if len(replay_buffer) < 10:
        return
    
    print("Training verifiers as Nash players (each optimizes independently)")
    
    training_data = []
    for problem, true_solution, solution, _, _, role in replay_buffer:
        correctness = 1.0 if dataset.check_solution(true_solution, solution) else 0.0
        training_data.append((problem, solution, correctness))
    
    print(f"Training on {len(training_data)} examples")
    if not training_data:
        return
    
    epochs = int(config["training"].get("verifier_epochs", 3))
    batch_size = min(32, len(training_data))
    
    # Train each verifier independently (Nash equilibrium iteration)
    for v_idx, verifier in enumerate(verifiers):
        print(f"\nTraining verifier {v_idx}")
        
        # Freeze all OTHER verifiers (they are part of φ_-j)
        for other_idx, other_verifier in enumerate(verifiers):
            if other_idx != v_idx:
                other_verifier.eval()
                set_requires_grad(other_verifier, False)
        
        # Freeze aggregator during verifier training
        aggregator.eval()
        set_requires_grad(aggregator, False)
        
        optimizer = torch.optim.Adam(
            verifier.parameters(), 
            lr=float(config["training"].get("verifier_lr", 1e-5))
        )
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            epoch_loss = 0
            epoch_updates = 0
            
            for batch_start in range(0, len(training_data), batch_size):
                batch_data = training_data[batch_start:batch_start+batch_size]
                batch_losses = []
                
                for problem, solution, correctness in batch_data:
                    # Get score from THIS verifier (requires grad)
                    try:
                        score_j = verifier(problem, solution)
                        
                        # Check for NaN/Inf
                        if isinstance(score_j, torch.Tensor):
                            if torch.isnan(score_j).any() or torch.isinf(score_j).any():
                                print(f"  WARNING: Verifier {v_idx} produced NaN/Inf, skipping example")
                                continue
                        
                        if not isinstance(score_j, torch.Tensor):
                            score_j = torch.tensor([score_j], dtype=torch.float32, device=verifier.device, requires_grad=True)
                        else:
                            score_j = score_j.to(verifier.device)
                            # Convert to float32 while preserving gradients
                            if score_j.dtype != torch.float32:
                                score_j = score_j.to(torch.float32)
                            if not score_j.requires_grad:
                                score_j.requires_grad_(True)
                    except Exception as e:
                        print(f"  Error getting verifier {v_idx} score: {e}")
                        continue
                    
                    # Get scores from OTHER verifiers (no grad)
                    other_scores = []
                    with torch.no_grad():
                        for other_idx, other_verifier in enumerate(verifiers):
                            if other_idx == v_idx:
                                continue
                            try:
                                score = other_verifier(problem, solution)
                                other_scores.append(score)
                            except:
                                other_scores.append(None)
                    
                    if None in other_scores:
                        continue
                    
                    # Reconstruct full score vector with this verifier's score in correct position
                    # Ensure all scores are 0D scalars for stacking
                    all_scores = []
                    other_idx = 0
                    for idx in range(len(verifiers)):
                        if idx == v_idx:
                            # Ensure score_j is a scalar
                            if isinstance(score_j, torch.Tensor):
                                if score_j.dim() > 0:
                                    score_val = score_j.squeeze()
                                else:
                                    score_val = score_j
                            else:
                                score_val = torch.tensor(score_j, dtype=torch.float32, device=verifier.device)
                            all_scores.append(score_val)
                        else:
                            # Convert other scores to scalars
                            other_score = other_scores[other_idx]
                            if isinstance(other_score, torch.Tensor):
                                score_val = other_score.squeeze()
                            else:
                                score_val = torch.tensor(other_score, dtype=torch.float32, device=verifier.device)
                            all_scores.append(score_val)
                            other_idx += 1
                    
                    # Pass through aggregator (frozen, but gradients flow through from verifier score)
                    # Aggregator expects probabilities, so we need to convert logits to probs
                    all_scores_probs = []
                    for idx, score_val in enumerate(all_scores):
                        if idx == v_idx:
                            # This verifier's logit - convert to prob for aggregator
                            prob = torch.sigmoid(score_val)
                        else:
                            # Other verifiers already return probs (eval mode)
                            prob = score_val
                        all_scores_probs.append(prob)
                    
                    scores_tensor = torch.stack(all_scores_probs).unsqueeze(0)  # [1, num_verifiers]
                    f_score = aggregator(scores_tensor).squeeze()
                    
                    # Use f_score directly with BCELoss (it's already a probability)
                    correctness_tensor = torch.tensor(correctness, dtype=torch.float32, device=verifier.device)
                    
                    eps = 1e-7
                    f_score_clamped = torch.clamp(f_score, eps, 1 - eps)
                    loss = F.binary_cross_entropy(f_score_clamped.unsqueeze(0), correctness_tensor.unsqueeze(0))
                    batch_losses.append(loss)
                
                if not batch_losses:
                    continue
                
                # Optimize verifier j to minimize oversight loss
                avg_loss = torch.stack(batch_losses).mean()
                
                # Check for NaN in loss
                if torch.isnan(avg_loss) or torch.isinf(avg_loss):
                    print(f"  WARNING: NaN/Inf loss for verifier {v_idx}, skipping batch")
                    continue
                
                optimizer.zero_grad()
                avg_loss.backward()
                
                # Sanitize NaN/Inf in gradients
                sanitize_model(verifier, f"verifier_{v_idx}")
                
                torch.nn.utils.clip_grad_norm_(verifier.parameters(), max_norm=0.5)
                optimizer.step()
                
                # Sanitize weights after update
                sanitize_model(verifier, f"verifier_{v_idx}")
                
                epoch_loss += avg_loss.item()
                epoch_updates += 1
                
                # Cleanup
                del scores_tensor, f_score, avg_loss
                torch.cuda.empty_cache()
            
            if epoch_updates > 0:
                avg_loss = epoch_loss / epoch_updates
                print(f"  Verifier {v_idx}, Epoch {epoch}, avg loss: {avg_loss:.4f}")
        
        # Unfreeze other verifiers for next iteration
        for other_verifier in verifiers:
            other_verifier.train()
            set_requires_grad(other_verifier, True)

def train_pe_min_aggregator(aggregator, replay_buffer, verifiers, dataset, config):
    """
    PE-min training: min_θ E[min_j l(f_θ(z), z_j)]
    Aggregator learns to match the best verifier on each example.
    """
    if len(replay_buffer) < 20:
        return aggregator
    
    print(f"Training PE-min aggregator on {len(replay_buffer)} experiences")
    
    # Freeze all verifiers during aggregator training
    for verifier in verifiers:
        verifier.eval()
        set_requires_grad(verifier, False)
    
    optimizer = torch.optim.Adam(aggregator.parameters(), 
                                lr=float(config["training"].get("aggregator_lr", 1e-4)))
    
    # Collect verifier scores
    all_scores = []
    with torch.no_grad():
        for problem, true_solution, response, _, _, _ in replay_buffer:
            scores = []
            for verifier in verifiers:
                try:
                    score = verifier(problem, response)
                    # Ensure scalar
                    if isinstance(score, torch.Tensor):
                        score = score.squeeze().item()
                    scores.append(score)
                except:
                    scores.append(None)
            
            if None not in scores and len(scores) == len(verifiers):
                all_scores.append(scores)
    
    if len(all_scores) < 10:
        print("Not enough complete scores for training")
        return aggregator
    
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32, device=DEVICE)
    
    epochs = int(config["training"].get("aggregator_epochs", 10))
    batch_size = min(32, len(all_scores))
    
    aggregator.train()
    
    for epoch in range(epochs):
        indices = torch.randperm(len(all_scores))
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_scores = scores_tensor[batch_indices]  # [B, m]
            
            # Forward pass: f_θ(z₁:ₘ)
            f_out = aggregator(batch_scores)  # [B]
            
            # Compute H_PE-min: min_j l(f(x,y), z_j)
            # Expand f to compare with each verifier score
            f_expanded = f_out.unsqueeze(1).expand(-1, batch_scores.shape[1])  # [B, m]
            
            # Compute l(f, z_j) = BCE(f, z_j) for each j using stable function
            eps = 1e-7
            f_clamped = torch.clamp(f_expanded, eps, 1 - eps)
            scores_clamped = torch.clamp(batch_scores, eps, 1 - eps)
            
            # Use F.binary_cross_entropy for each verifier (more stable than manual log)
            pointwise_losses = F.binary_cross_entropy(f_clamped, scores_clamped, reduction='none')  # [B, m]
            
            # Take minimum across verifiers (PE-min principle)
            min_losses = pointwise_losses.min(dim=1)[0]  # [B]
            loss = min_losses.mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aggregator.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0 and epoch % 2 == 0:
            print(f"  Aggregator epoch {epoch}: avg PE-min loss = {epoch_loss/num_batches:.4f}")
    
    # Unfreeze verifiers
    for verifier in verifiers:
        verifier.train()
        set_requires_grad(verifier, True)
    
    return aggregator

def enforce_sneaky_incorrect(response: str, solution_true: str, dataset: MathDataset) -> str:
    """
    Enforce Lemma B.2: sneaky outputs must be incorrect.
    If response is correct, modify it to be incorrect.
    """
    if dataset.check_solution(solution_true, response):
        import re
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if match:
            original_answer = match.group(1)
            try:
                num = float(original_answer)
                wrong_num = num + random.choice([-5, -3, -1, 1, 3, 5])
                modified_response = response.replace(f'\\boxed{{{original_answer}}}', f'\\boxed{{{wrong_num}}}')
                return modified_response
            except:
                modified_response = response.replace(f'\\boxed{{{original_answer}}}', f'\\boxed{{{original_answer} + 1}}')
                return modified_response
    
    return response

def create_role_prompt(prover: Prover, problem: str, role: str) -> str:
    """Create prompts for helpful (correct) vs sneaky (incorrect) provers"""
    
    # Use simpler format without chat template - more reliable for math problems
    if role.lower() == "helpful":
        prompt = f"""Solve this math problem step by step and provide your final answer in \\boxed{{answer}} format.

Problem: {problem}

Solution:"""
    else:  # sneaky
        prompt = f"""Solve this math problem but make a subtle error in your reasoning. Make it look convincing but arrive at the wrong answer. Provide your final answer in \\boxed{{answer}} format.

Problem: {problem}

Solution:"""
    
    return prompt

def collect_prover_data_stackelberg(
    config,
    prover: Prover,
    aggregator: PEMinAggregator,
    verifiers: List[Verifier],
    dataset: MathDataset,
    round_idx: int
) -> Tuple[List[str], List[str], List[float], List[str], List[str], List[str], List[float]]:
    """
    Collect prover data implementing Lemma B.2:
    (i) Correctness alignment: helpful→correct, sneaky→incorrect
    (ii) Score maximization: maximize f(x,y) over allowed solutions
    
    Returns: prompts, responses, rewards, roles, problems, true_solutions, correctness_labels
    """
    
    prompts = []
    responses = []
    rewards = []
    roles = []
    problems = []
    true_solutions = []
    correctness_labels = []
    
    helpful_correct_count = 0
    helpful_total = 0
    sneaky_incorrect_count = 0
    sneaky_total = 0
    
    mu_0 = config["training"].get("helpful_prior", 0.5)
    max_retries = 3  # For sneaky to ensure incorrect solution
    
    for episode in range(config["training"].get("max_episodes", 100)):
        role = "helpful" if random.random() < mu_0 else "sneaky"
        roles.append(role)
        
        problem, solution_true = dataset.sample()
        problems.append(problem)
        true_solutions.append(solution_true)
        
        prompt = create_role_prompt(prover, problem, role)
        prompts.append(prompt)
        
        # Generate response
        full_response = prover.generate(prompt, max_new_tokens=2000)
        
        # Extract only the generated part (remove prompt)
        if full_response.startswith(prompt):
            generated_text = full_response[len(prompt):].strip()
        else:
            generated_text = full_response.strip()
        
        # Enforce correctness alignment (Lemma B.2 part i)
        if role.lower() == "sneaky":
            # Ensure sneaky output is incorrect
            retry_count = 0
            while dataset.check_solution(solution_true, generated_text) and retry_count < max_retries:
                print(f"  Sneaky generated correct solution, retrying... ({retry_count+1}/{max_retries})")
                generated_text = enforce_sneaky_incorrect(generated_text, solution_true, dataset)
                retry_count += 1
            
            # If still correct after retries, force it to be wrong
            if dataset.check_solution(solution_true, generated_text):
                generated_text = enforce_sneaky_incorrect(generated_text, solution_true, dataset)
        
        responses.append(generated_text)
        
        # Debug: Print first few examples
        if episode < 3:
            print(f"\n--- Episode {episode} Debug ---")
            print(f"Role: {role}")
            print(f"Problem: {problem[:100]}...")
            print(f"Generated (first 300 chars): {generated_text[:300]}")
        
        # Check correctness
        correctness = 1.0 if dataset.check_solution(solution_true, generated_text) else 0.0
        correctness_labels.append(correctness)
        
        if episode < 30 or episode % 20 == 0:
            print(f"Episode {episode}: Role={role}, Correct={correctness}")
        
        # Get verifier scores and aggregate
        with torch.no_grad():
            scores = []
            for v_idx, verifier in enumerate(verifiers):
                try:
                    score = verifier(problem, generated_text)
                    # Ensure scalar
                    if isinstance(score, torch.Tensor):
                        score = score.squeeze().item()
                    else:
                        score = float(score)
                    scores.append(score)
                except Exception as e:
                    print(f"  Verifier {v_idx} failed: {e}")
                    print(f"  WARNING: Using fallback score for verifier {v_idx}")
                    scores.append(None)
            
            valid_scores = [s for s in scores if s is not None]
            if len(valid_scores) < len(verifiers):
                print(f"  Warning: Only {len(valid_scores)}/{len(verifiers)} verifiers succeeded")
            
            if valid_scores:
                mean_score = sum(valid_scores) / len(valid_scores)
                scores = [s if s is not None else mean_score for s in scores]
            else:
                print(f"  ERROR: All verifiers failed")
                continue
            
            scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            f_score = aggregator(scores_tensor).item()
        
        reward = reward_function(f_score, correctness, role)
        rewards.append(reward)
        
        if role.lower() == "helpful":
            helpful_total += 1
            if correctness == 1.0:
                helpful_correct_count += 1
        else:
            sneaky_total += 1
            if correctness == 0.0:
                sneaky_incorrect_count += 1
        
        if episode < 30 or episode % 20 == 0:
            print(f"  f_score={f_score:.3f}, reward={reward:.3f}")
        
        if episode > 20 and episode % 20 == 0:
            print(f"\n--- Episode {episode} Statistics ---")
            if helpful_total > 0:
                helpful_correctness = helpful_correct_count / helpful_total
                print(f"Helpful correctness: {helpful_correctness:.3f} ({helpful_correct_count}/{helpful_total})")
            if sneaky_total > 0:
                sneaky_incorrect_rate = sneaky_incorrect_count / sneaky_total
                print(f"Sneaky incorrect rate: {sneaky_incorrect_rate:.3f}")
            print("---\n")
    
    print(f"\n=== Data Collection Summary ===")
    print(f"Total episodes: {len(responses)}")
    if helpful_total > 0:
        print(f"Helpful: {helpful_correct_count}/{helpful_total} correct ({100*helpful_correct_count/helpful_total:.1f}%)")
    if sneaky_total > 0:
        print(f"Sneaky: {sneaky_incorrect_count}/{sneaky_total} incorrect ({100*sneaky_incorrect_count/sneaky_total:.1f}%)")
    
    return prompts, responses, rewards, roles, problems, true_solutions, correctness_labels

def pretrain_verifiers(verifiers: List[Verifier], dataset: MathDataset, prover: Prover, config):
    """Pre-train each verifier to predict solution correctness"""
    num_examples = config["training"].get("verifier_pretrain_examples", 500)
    print(f"Pre-training {len(verifiers)} verifiers on {num_examples} examples each...")
    
    for v_idx, verifier in enumerate(verifiers):
        # Print verifier role if available
        role_name = "general"
        if hasattr(verifier, 'role') and verifier.role:
            role_name = verifier.role['name']
            print(f"\nVerifier {v_idx} ({role_name}): {verifier.role['description']}")
        
        verifier.train()
        optimizer = torch.optim.Adam(verifier.parameters(), lr=1e-4)
        
        total_loss = 0
        for i in range(num_examples):
            problem, solution_true = dataset.sample()
            
            # Generate a correct solution
            prompt = create_role_prompt(prover, problem, "helpful")
            correct_response = prover.generate(prompt, max_new_tokens=512)
            if correct_response.startswith(prompt):
                correct_solution = correct_response[len(prompt):].strip()
            else:
                correct_solution = correct_response.strip()
            
            # Fallback to template if generation failed
            if not correct_solution or not dataset.check_solution(solution_true, correct_solution):
                correct_solution = f"Let me solve this step by step.\nThe answer is \\boxed{{{solution_true}}}"
            
            # Generate an incorrect solution
            try:
                if isinstance(solution_true, str):
                    try:
                        num = float(solution_true.replace('\\', '').replace('{', '').replace('}', ''))
                        wrong_answer = str(num + random.choice([-5, -3, -1, 1, 3, 5]))
                    except:
                        wrong_answer = solution_true + " + 1"
                else:
                    wrong_answer = str(solution_true) + " (incorrect)"
                    
                incorrect_solution = f"Let me solve this step by step.\nThe answer is \\boxed{{{wrong_answer}}}"
            except Exception as e:
                print(f"  Warning: Could not generate incorrect solution: {e}")
                continue
            
            # Train on both correct and incorrect solutions
            batch_loss = 0
            for solution, correctness in [(correct_solution, 1.0), (incorrect_solution, 0.0)]:
                try:
                    logit = verifier(problem, solution)  # Now returns logit during training
                    
                    if isinstance(logit, torch.Tensor):
                        if logit.dim() == 0:
                            logit = logit.unsqueeze(0)
                        # Convert to float32 to avoid dtype mismatch
                        logit = logit.float()
                    else:
                        logit = torch.tensor([float(logit)], dtype=torch.float32, device=verifier.device, requires_grad=True)
                    
                    target = torch.tensor([correctness], dtype=torch.float32, device=verifier.device)
                    
                    # Use BCEWithLogitsLoss - numerically stable, no epsilon needed
                    loss = F.binary_cross_entropy_with_logits(logit, target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Sanitize NaN/Inf in gradients
                    sanitize_model(verifier, f"verifier_{v_idx}")
                    
                    torch.nn.utils.clip_grad_norm_(verifier.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # Sanitize weights after update
                    sanitize_model(verifier, f"verifier_{v_idx}")
                    
                    batch_loss += loss.item()
                    
                except Exception as e:
                    print(f"  Error training verifier {v_idx} on example {i}: {e}")
                    continue
            
            total_loss += batch_loss
            
            if i % 100 == 0 and i > 0:
                avg_loss = total_loss / (i * 2)
                print(f"  Verifier {v_idx}, step {i}/{num_examples}, avg loss: {avg_loss:.4f}")
            
            torch.cuda.empty_cache()
        
        final_avg_loss = total_loss / (num_examples * 2)
        print(f"Verifier {v_idx} pretraining complete! Final avg loss: {final_avg_loss:.4f}")
        verifier.eval()

def pretrain_helpful_prover(prover: Prover, dataset: MathDataset, config):
    """Pre-train helpful prover on correct solutions with supervised learning."""
    num_examples = config["training"].get("pretrain_examples", 100)
    print(f"Pre-training helpful prover on {num_examples} correct solutions...")
    
    trainable_params = [p for p in prover.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-5)
    
    for i in range(num_examples):
        problem, solution_true = dataset.sample()
        
        prompt = create_role_prompt(prover, problem, "helpful")
        example_solution = f"Step 1: [Work through the problem]\nStep 2: [Continue reasoning]\nFinal answer: \\boxed{{{solution_true}}}"
        full_text = prompt + example_solution
        
        try:
            inputs = prover.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(prover.device) for k, v in inputs.items()}
            
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

def reset_models_for_round(config, round_idx, aggregator=None, trained_verifiers=None):
    """Reset prover to base while keeping trained verifiers (Stackelberg structure)"""
    if round_idx > 0:  
        print("GPU cleanup")
        if 'prover' in globals():
            del globals()['prover']
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    prover_model = config["model"].get("prover_name", "meta-llama/Llama-2-7b-chat-hf")
    verifier_model = config["model"].get("verifier_name", "meta-llama/Llama-2-1b-hf") 
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    prover = Prover(prover_model, use_quantization=True, quantization_config=quantization_config).to(DEVICE)
    prover.config = config 
    
    if config["training"].get("use_lora", True):
        prover.model = setup_lora(prover.model, config)
    
    prover.device = DEVICE 
    
    if trained_verifiers is not None:
        print(f"Reusing {len(trained_verifiers)} trained verifiers from previous round")
        verifiers = trained_verifiers
        actual_num_verifiers = len(verifiers)
    else:
        num_verifiers = config["model"].get("num_verifiers", 2)
        verifier_roles = get_verifier_roles(num_verifiers)
        verifiers = []
        
        print(f"\nInitializing {num_verifiers} verifiers with diverse roles:")
        for i in range(num_verifiers):
            role = verifier_roles[i]
            print(f"  Verifier {i}: {role['name']} - {role['description']}")
            
            try:
                # Use 4-bit quantization with aggressive NaN sanitization
                v = Verifier(verifier_model, verifier_type=f"verifier_{i}", use_quantization=True, quantization_config=quantization_config)
                v.config = config
                v.role = role  # Assign role for diverse evaluation
                v.to(DEVICE)
                verifiers.append(v)
                print(f"    Loaded (4-bit), GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            except torch.cuda.OutOfMemoryError:
                print(f"OOM loading verifier {i}, using fewer verifiers")
                break
            except torch.cuda.OutOfMemoryError:
                print(f"OOM loading verifier {i}, using fewer verifiers")
                break
        
        if len(verifiers) == 0:
            raise RuntimeError("Could not load any verifiers due to OOM")
        
        actual_num_verifiers = len(verifiers)
        if actual_num_verifiers != num_verifiers:
            print(f"WARNING: Requested {num_verifiers} verifiers but only loaded {actual_num_verifiers}")
    
    torch.cuda.empty_cache()
    print(f"Final GPU memory after loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Total verifiers: {len(verifiers)} neural verifiers")
    
    return prover, verifiers, actual_num_verifiers

def compute_log_prob(model, tokenizer, prompt, response, device):
    """Compute log probability of response given prompt"""
    if not response.strip():
        return torch.tensor(-10.0, device=device, requires_grad=True)
    
    max_length = 512
    full_text = prompt + response
    
    try:
        torch.cuda.empty_cache()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model, 'module') and hasattr(model.module, 'gradient_checkpointing_enable'):
            model.module.gradient_checkpointing_enable()
        
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
            return torch.tensor(-5.0, device=device, requires_grad=True)
        
        response_logits = logits[prompt_len-1:-1]
        response_tokens = inputs['input_ids'][0, prompt_len:]
        
        if response_tokens.shape[0] == 0:
            return torch.tensor(-5.0, device=device, requires_grad=True)
        
        log_probs = torch.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
        total_log_prob = token_log_probs.sum()
        
        return total_log_prob
    
    except Exception as e:
        print(f"Log prob computation failed: {e}")
        random_val = -5.0 - 3.0 * torch.rand(1).item()
        return torch.tensor(random_val, device=device, requires_grad=True)

def compute_log_prob_batch(model, tokenizer, prompts, responses, device, batch_size=2):
    """Compute log probabilities for multiple prompt-response pairs efficiently"""
    all_log_probs = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]
        batch_log_probs = []
        
        for prompt, response in zip(batch_prompts, batch_responses):
            log_prob = compute_log_prob(model, tokenizer, prompt, response, device)
            batch_log_probs.append(log_prob)
        
        all_log_probs.extend(batch_log_probs)
        torch.cuda.empty_cache()
    
    return all_log_probs

def train_prover_ppo(
    config,
    prover: Prover,
    prompts: List[str],
    responses: List[str], 
    rewards: List[float],
    roles: List[str],
    writer: SummaryWriter,
    round_idx: int
):
    """
    Train prover using PPO to maximize expected utility:
    max_{π_H, π_S} E_{τ~μ₀, x~P_X, y~π_τ(x)}[r(f(x,y), a)]
    
    This implements the prover's best response in the Stackelberg game.
    """
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
    
    print("Computing old log probabilities in batches...")
    old_log_probs = compute_log_prob_batch(prover.model, prover.tokenizer, prompts, responses, DEVICE, batch_size=4)
    
    valid_indices = []
    valid_old_log_probs = []
    
    for i, old_log_prob in enumerate(old_log_probs):
        if old_log_prob is not None and torch.isfinite(old_log_prob):
            valid_old_log_probs.append(old_log_prob.detach())
            valid_indices.append(i)
    
    if len(valid_old_log_probs) == 0:
        print("ERROR: All old log prob computations failed")
        return
    
    valid_prompts = [prompts[i] for i in valid_indices]
    valid_responses = [responses[i] for i in valid_indices]
    valid_rewards = [rewards[i] for i in valid_indices]
    valid_roles = [roles[i] for i in valid_indices]
    valid_advantages = advantages[valid_indices]
    
    old_log_probs_tensor = torch.stack(valid_old_log_probs)
    
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    total_policy_loss = 0
    total_kl_div = 0
    successful_updates = 0
    
    for epoch in range(epochs):
        epoch_policy_loss = 0
        epoch_kl_div = 0
        epoch_updates = 0
        
        batch_size = min(2, len(valid_prompts))
        
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
                
                # Always print gradient info on first batch of first epoch
                if batch_idx == 0 and epoch == 0:
                    print(f"\n=== PPO Gradient Debug ===")
                    print(f"Total trainable params: {len(trainable_params)}")
                    print(f"Params with gradients: {len(grad_norms)}")
                    if grad_norms:
                        print(f"Max grad norm: {max(grad_norms):.8f}")
                        print(f"Min grad norm: {min(grad_norms):.8f}")
                        print(f"Mean grad norm: {sum(grad_norms)/len(grad_norms):.8f}")
                    else:
                        print(f"ERROR: No parameters received gradients!")
                        print(f"total_loss.requires_grad: {total_loss.requires_grad}")
                        print(f"total_loss value: {total_loss.item():.6f}")
                    print("========================\n")
                
                # Very relaxed threshold - if ANY gradient exists, update
                if grad_norms and max(grad_norms) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    successful_updates += 1
                elif not grad_norms:
                    if epoch == 0:
                        print(f"  WARNING: No gradients at batch {batch_idx}, epoch {epoch}")
                else:
                    if epoch == 0:
                        print(f"  WARNING: All gradients are exactly zero at batch {batch_idx}")
            else:
                if epoch == 0 and batch_idx == 0:
                    print(f"  WARNING: Loss doesn't require grad or is not finite")
                    print(f"    requires_grad: {total_loss.requires_grad if hasattr(total_loss, 'requires_grad') else 'N/A'}")
                    print(f"    is_finite: {torch.isfinite(total_loss) if isinstance(total_loss, torch.Tensor) else 'N/A'}")
            
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
    
    final_param_mean = first_param.data.mean().item()
    param_change = abs(final_param_mean - initial_param_mean)
    
    print(f"\n=== PPO Training Summary ===")
    print(f"Parameter change: {param_change:.8f}")
    print(f"Successful updates: {successful_updates}")
    
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
    if helpful_rewards:
        print(f"  Helpful: {sum(helpful_rewards) / len(helpful_rewards):.4f}")
    if sneaky_rewards:
        print(f"  Sneaky: {sum(sneaky_rewards) / len(sneaky_rewards):.4f}")

def setup_lora(model, config):
    """Setup LoRA for parameter-efficient fine-tuning"""
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
    cleanup_cache_and_disk()
    
    print("="*80)
    print("PMV STACKELBERG-NASH GAME - Paper-Compliant Implementation")
    print("="*80)
    print("Game Structure:")
    print("  1. Verifiers play Nash game (each minimizes L(φ_j, φ_-j) independently)")
    print("  2. PE-min aggregator learns f_θ to minimize E[min_j l(f,φ_j)]")
    print("  3. Prover best-responds to committed verifiers/aggregator")
    print("  4. Prover responses satisfy Lemma B.2 (correctness alignment + score maximization)")
    print("="*80)
    
    config_path = "pmv/configs/config_pure_stackelberg.yaml"
    config = load_config(config_path)
    dataset = MathDataset()
    
    num_verifiers = config["model"].get("num_verifiers", 3)
    aggregator = PEMinAggregator(num_verifiers=num_verifiers).to(DEVICE)
    print(f"Initialized PE-min aggregator for {num_verifiers} verifiers")
    
    if resume_checkpoint:
        log_dir = os.path.dirname(resume_checkpoint)
        # Need to load verifiers first to pass to load_checkpoint
        _, verifiers, _ = reset_models_for_round(config, 0, aggregator, None)
        start_round, replay_buffer = load_checkpoint(resume_checkpoint, aggregator, verifiers)
    else:
        from datetime import datetime
        base_log_dir = config["logging"].get("logdir", "runs/pmv_stackelberg")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"{base_log_dir}_{timestamp}"
        start_round = 0
        replay_buffer = []
        verifiers = None
        
    writer = SummaryWriter(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    config_save_path = os.path.join(log_dir, "config.yaml")
    
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"Saved config to {config_save_path}")
    
    num_rounds = config["training"].get("rounds", 10)
    max_replay_size = config["training"].get("max_replay_size", 200)
    
    print(f"\n{'='*80}")
    print("STACKELBERG-NASH EQUILIBRIUM (Definition 3.2)")
    print("="*80)
    print("Each round:")
    print("  Phase 1: Verifiers commit via Nash game (each optimizes independently)")
    print("  Phase 2: Aggregator commits via PE-min training")
    print("  Phase 3: Prover best-responds via PPO (implements Lemma B.2)")
    print("  Phase 4: Store experiences for next round")
    print("="*80)
    
    trained_verifiers = verifiers  # Will persist across rounds
    
    for round_idx in range(start_round, num_rounds):
        print(f"\n{'='*80}")
        print(f"Round {round_idx + 1}/{num_rounds}")
        print(f"{'='*80}")
        
        # Reset prover to base, keep trained verifiers
        prover, verifiers, actual_num_verifiers = reset_models_for_round(
            config, round_idx, aggregator, trained_verifiers
        )
        
        # Reinitialize aggregator if verifier count changed
        if round_idx == 0 or actual_num_verifiers != aggregator.num_verifiers:
            print(f"Reinitializing aggregator for {actual_num_verifiers} verifiers")
            old_aggregator = aggregator
            aggregator = PEMinAggregator(num_verifiers=actual_num_verifiers).to(DEVICE)
            
            if round_idx > 0 and old_aggregator.num_verifiers == actual_num_verifiers:
                try:
                    aggregator.load_state_dict(old_aggregator.state_dict())
                    print("Copied aggregator weights from previous round")
                except:
                    print("Could not copy aggregator weights, using fresh initialization")
        
        # PHASE 1: Verifiers commit via Nash game
        if round_idx > 0:
            cleanup_cache_and_disk()
            print("\n" + "="*80)
            print("PHASE 1: VERIFIERS COMMIT VIA NASH GAME")
            print("="*80)
            
            oversight_loss_before = compute_oversight_loss(
                verifiers, aggregator, replay_buffer, dataset
            )
            print(f"Oversight loss before Nash training: {oversight_loss_before:.4f}")
            
            train_verifiers_nash(
                verifiers, aggregator, replay_buffer, dataset, config
            )
            
            oversight_loss_after = compute_oversight_loss(
                verifiers, aggregator, replay_buffer, dataset
            )
            print(f"Oversight loss after Nash training: {oversight_loss_after:.4f}")
            
            writer.add_scalar("oversight_loss/before_nash", oversight_loss_before, round_idx)
            writer.add_scalar("oversight_loss/after_nash", oversight_loss_after, round_idx)
        
        # PHASE 2: Aggregator commits via PE-min
        if round_idx > 0:
            print("\n" + "="*80)
            print("PHASE 2: AGGREGATOR COMMITS VIA PE-MIN")
            print("="*80)
            
            aggregator = train_pe_min_aggregator(
                aggregator, replay_buffer, verifiers, dataset, config
            )
            
            oversight_loss_after_agg = compute_oversight_loss(
                verifiers, aggregator, replay_buffer, dataset
            )
            print(f"Oversight loss after PE-min training: {oversight_loss_after_agg:.4f}")
            writer.add_scalar("oversight_loss/after_pemin", oversight_loss_after_agg, round_idx)
        
        # Freeze verifiers and aggregator (commitment)
        print("\nFreezing verifiers and aggregator (leaders committed)")
        for verifier in verifiers:
            verifier.eval()
            set_requires_grad(verifier, False)
        
        aggregator.eval()
        set_requires_grad(aggregator, False)
        
        # PHASE 3: Prover best-responds (Stackelberg follower)
        print("\n" + "="*80)
        print("PHASE 3: PROVER BEST-RESPONDS (implements Lemma B.2)")
        print("="*80)
        
        if round_idx == 0:
            pretrain_helpful_prover(prover, dataset, config)
            pretrain_verifiers(verifiers, dataset, prover, config)
        
        print("\nCollecting prover data with correctness enforcement")
        prompts, responses, rewards, roles, problems, true_solutions, correctness_labels = collect_prover_data_stackelberg(
            config, prover, aggregator, verifiers, dataset, round_idx
        )
        
        print("\nTraining prover via PPO to maximize utility")
        train_prover_ppo(config, prover, prompts, responses, rewards, roles, writer, round_idx)
        
        # PHASE 4: Store experiences
        experiences = []
        for prompt, response, reward, role, problem, true_solution in zip(
            prompts, responses, rewards, roles, problems, true_solutions
        ):
            experiences.append((problem, true_solution, response, reward, None, role))
        
        replay_buffer.extend(experiences)
        if len(replay_buffer) > max_replay_size:
            replay_buffer = replay_buffer[-max_replay_size:]
        
        print(f"\nReplay buffer size: {len(replay_buffer)}")
        
        # Save checkpoint
        if (round_idx + 1) % config["training"].get("save_every", 5) == 0:
            ckpt_dir = Path(log_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            verifier_states = [v.state_dict() for v in verifiers]
            
            checkpoint = {
                "round": round_idx,
                "replay_buffer": replay_buffer,
                "config": config,
                "aggregator": aggregator.state_dict(),
                "verifiers": verifier_states,
                "paper_implementation": "stackelberg_nash"
            }
            
            checkpoint_path = ckpt_dir / f"pmv_round_{round_idx:03d}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Cleanup
        trained_verifiers = verifiers
        
        # Unfreeze for next round
        for verifier in verifiers:
            verifier.train()
            set_requires_grad(verifier, True)
        
        aggregator.train()
        set_requires_grad(aggregator, True)
        
        del prover
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n{'='*80}")
    print(f"Training completed! Logs saved to {log_dir}")
    print(f"{'='*80}")
    writer.close()

if __name__ == "__main__":
    import sys
    resume_checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    main(resume_checkpoint)
