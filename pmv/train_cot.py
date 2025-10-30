import torch
import torch.nn as nn
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
from pmv.aggregator import Aggregator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()

# Environment setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HOME'] = 'hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'hf_cache'
os.environ['HF_DATASETS_CACHE'] = 'hf_cache'
os.makedirs('hf_cache', exist_ok=True)


class FormalVerifier:
    """
    Formal verifier that checks mathematical correctness using symbolic computation.
    Returns binary scores: 1.0 for correct, 0.0 for incorrect, None for verification failure.
    """
    def __init__(self, verifier_type="formal"):
        self.verifier_type = verifier_type
        
    def to(self, device):
        """Compatibility method for device placement (no-op for formal verifier)"""
        return self
    
    def parameters(self):
        """Return empty list since formal verifier has no trainable parameters"""
        return []
    
    def eval(self):
        """Compatibility method (no-op for formal verifier)"""
        return self
    
    def __call__(self, problem: str, solution: str, ground_truth: str = None) -> float:
        """
        Formally verify if the solution is mathematically correct.
        Returns 1.0 if correct, 0.0 if incorrect, None if verification failed.
        """
        try:
            # Extract predicted answer from solution
            predicted_answer = self.extract_answer(solution)
            if predicted_answer is None:
                print(f"  [FormalVerifier] Could not extract answer from solution")
                return None
            
            # Extract ground truth
            if ground_truth is None:
                print(f"  [FormalVerifier] No ground truth provided")
                return None
                
            true_answer = self.extract_answer(ground_truth)
            if true_answer is None:
                print(f"  [FormalVerifier] Could not extract ground truth answer")
                return None
            
            # Compare answers
            is_correct = self.symbolic_equal(predicted_answer, true_answer)
            return 1.0 if is_correct else 0.0
            
        except Exception as e:
            print(f"  [FormalVerifier] Verification error: {e}")
            return None
    
    def extract_answer(self, text: str):
        """
        Extract the final answer from a solution string.
        Looks for patterns like \\boxed{answer} or 'Answer: answer'
        """
        import re
        
        # Pattern 1: \boxed{...}
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_matches = re.findall(boxed_pattern, text)
        if boxed_matches:
            # Take the last boxed answer
            return self.normalize_answer(boxed_matches[-1])
        
        # Pattern 2: Answer: ... (case insensitive)
        answer_pattern = r'[Aa]nswer:\s*([^\n]+)'
        answer_matches = re.findall(answer_pattern, text)
        if answer_matches:
            return self.normalize_answer(answer_matches[-1])
        
        # Pattern 3: Final answer is ...
        final_pattern = r'[Ff]inal answer is:?\s*([^\n]+)'
        final_matches = re.findall(final_pattern, text)
        if final_matches:
            return self.normalize_answer(final_matches[-1])
        
        return None
    
    def normalize_answer(self, answer_str: str):
        """
        Normalize mathematical expressions for comparison.
        """
        # Remove extra whitespace
        answer_str = answer_str.strip()
        
        # Remove dollar signs
        answer_str = answer_str.replace('$', '').replace('\\', '')
        
        # Remove common punctuation at the end
        answer_str = answer_str.rstrip('.,;:')
        
        # Try to evaluate as a number
        try:
            # Handle fractions like 3/4
            if '/' in answer_str and len(answer_str.split('/')) == 2:
                parts = answer_str.split('/')
                return float(eval(answer_str))
            
            # Try direct float conversion
            return float(answer_str)
        except:
            pass
        
        # Return as string for symbolic comparison
        return answer_str
    
    def symbolic_equal(self, expr1, expr2, tolerance=1e-6):
        """
        Check if two mathematical expressions are equivalent.
        Handles numeric and symbolic comparisons.
        """
        try:
            # Try numerical comparison
            if isinstance(expr1, (int, float)) and isinstance(expr2, (int, float)):
                return abs(float(expr1) - float(expr2)) < tolerance
            
            # Try symbolic comparison with sympy
            try:
                import sympy as sp
                e1 = sp.sympify(str(expr1))
                e2 = sp.sympify(str(expr2))
                
                if sp.simplify(e1 - e2) == 0:
                    return True
                
                # Try numerical evaluation
                if e1.is_number and e2.is_number:
                    return abs(float(e1) - float(e2)) < tolerance
            except:
                pass
            
            # Fallback to string comparison
            return str(expr1).strip().lower() == str(expr2).strip().lower()
            
        except Exception:
            return False


class LearnedAggregator(nn.Module):
    """
    Learned aggregation function for combining multiple verifier scores.
    Implements different aggregation strategies based on the PMV framework.
    """
    def __init__(self, num_verifiers, aggregation_type="pl_min", hidden_dim=64):
        super().__init__()
        self.num_verifiers = num_verifiers
        self.aggregation_type = aggregation_type
        
        if aggregation_type == "pl_min":
            # Pessimistic learnable min: learn to weight verifiers before taking min
            self.score_transform = nn.Sequential(
                nn.Linear(num_verifiers, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_verifiers),
                nn.Softplus()  # Ensure positive weights
            )
        elif aggregation_type == "pe_min":
            # Parametric empirical min: learn direct mapping from scores to aggregate
            self.score_transform = nn.Sequential(
                nn.Linear(num_verifiers, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()  # Output in [0,1]
            )
        elif aggregation_type == "neural":
            # Full neural aggregator with attention
            self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
            self.score_embed = nn.Linear(1, 16)
            self.output_layer = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
    
    def forward(self, verifier_scores):
        """
        Aggregate verifier scores.
        verifier_scores: [batch_size, num_verifiers] tensor
        """
        if self.aggregation_type == "pl_min":
            # Weighted pessimistic min
            weights = self.score_transform(verifier_scores)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # Normalize
            weighted_scores = verifier_scores * weights
            
            # Soft minimum using LogSumExp trick
            tau = 0.1  # Temperature for soft min
            soft_min = -tau * torch.logsumexp(-weighted_scores / tau, dim=-1)
            return soft_min
            
        elif self.aggregation_type == "pe_min":
            # Direct learned mapping
            return self.score_transform(verifier_scores).squeeze(-1)
            
        elif self.aggregation_type == "neural":
            # Attention-based aggregation
            batch_size = verifier_scores.shape[0]
            score_embeds = self.score_embed(verifier_scores.unsqueeze(-1))  # [B, m, 16]
            
            attended, _ = self.attention(score_embeds, score_embeds, score_embeds)
            pooled = attended.mean(dim=1)  # [B, 16]
            output = self.output_layer(pooled)  # [B, 1]
            return output.squeeze(-1)


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

    # Initialize verifiers with formal verifier first
    num_neural_verifiers = config["model"].get("num_verifiers", 3) - 1
    verifiers = []
    
    # Add formal verifier first
    formal_verifier = FormalVerifier()
    verifiers.append(formal_verifier)
    print("Added formal verifier")
    
    # Add neural verifiers
    for i in range(num_neural_verifiers):
        try:
            v = Verifier(verifier_model, verifier_type=f"verifier_{i}")
            v.config = config 
            v.to(DEVICE)
            verifiers.append(v)
            print(f"Loaded neural verifier {i}, GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"OOM loading verifier {i}, using fewer verifiers")
            break
    
    if len(verifiers) == 1:  # Only formal verifier loaded
        print("Warning: Only formal verifier loaded, adding at least one neural verifier")
    
    torch.cuda.empty_cache()
    print(f"Final GPU memory after loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Total verifiers: {len(verifiers)} (1 formal + {len(verifiers)-1} neural)")
    
    return prover, verifiers


def improved_create_role_prompt(prover: Prover, problem: str, role: str) -> str:
    """
    Create prompts without repeating system messages, using explicit CoT
    """
    
    if role.lower() == "helpful":
        system_msg = "You are a helpful mathematical assistant. Always show your work step by step and arrive at the correct answer."
        user_msg = f"""Solve this problem using clear step-by-step reasoning:

{problem}

Think through this carefully:
1. First, identify what we know and what we need to find
2. Then, work through the solution step by step  
3. Finally, state your answer clearly in the format: \\boxed{{answer}}

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
    """
    num_examples = config["training"].get("pretrain_examples", 100)
    print(f"Pre-training helpful prover on {num_examples} correct solutions...")
    
    trainable_params = [p for p in prover.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-5)
    
    for i in range(num_examples):
        problem, solution_true = dataset.sample()
        
        prompt = improved_create_role_prompt(prover, problem, "helpful")
        full_text = prompt + f"\n\nLet me work through this step by step.\n\nAfter solving, the answer is: \\boxed{{{solution_true}}}"
        
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


def collect_prover_data_with_correctness_constraint(
    config,
    prover: Prover,
    aggregator,
    verifiers: List,
    dataset: MathDataset,
    round_idx: int
) -> Tuple[List[str], List[str], List[float], List[str], List[str], List[str]]:
    """
    Collect data with verifiers committed (Stackelberg follower phase).
    Prover best-responds to committed verifier strategies.
    """
    
    prompts = []
    responses = []
    rewards = []
    roles = []
    problems = []
    true_solutions = []
    
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
        
        prompt = improved_create_role_prompt(prover, problem, role)
        prompts.append(prompt)
        
        response = prover.generate(prompt, max_new_tokens=2000)
        generated_text = response[len(prompt):] if response.startswith(prompt) else response
        responses.append(generated_text)
        
        correct = dataset.check_solution(solution_true, generated_text)
        
        print(f"Episode {episode}: Role={role}, Correct={correct}")
        
        # Get verifier scores (verifiers are committed/frozen)
        with torch.no_grad():
            scores = []
            for v_idx, verifier in enumerate(verifiers):
                try:
                    if isinstance(verifier, FormalVerifier):
                        score = verifier(problem, generated_text, solution_true)
                        if score is None:
                            # Formal verifier failed - skip it
                            continue
                    else:
                        score = verifier(problem, generated_text)
                    scores.append(score)
                except Exception as e:
                    print(f"  Verifier {v_idx} failed: {e}")
            
            if not scores:
                print("  All verifiers failed, skipping")
                continue
            
            # Aggregate scores
            if isinstance(aggregator, LearnedAggregator):
                scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                f_score = aggregator(scores_tensor).item()
            else:
                f_score = aggregator(scores)
        
        # Reward structure with correctness emphasis
        if role.lower() == "helpful":
            helpful_total += 1
            if correct:
                helpful_correct_count += 1
                reward = 1.5 + 0.5 * f_score
                print(f"  Helpful CORRECT: reward={reward:.3f}, f_score={f_score:.3f}")
            else:
                reward = -2.0
                print(f"  Helpful WRONG: reward={reward:.3f} (PENALTY)")
        else:  # sneaky
            sneaky_total += 1
            if not correct:
                sneaky_incorrect_count += 1
                reward = f_score
                print(f"  Sneaky WRONG: reward={reward:.3f}, f_score={f_score:.3f}")
            else:
                reward = -1.0 + 0.3 * f_score
                print(f"  Sneaky CORRECT: reward={reward:.3f} (discouraged)")
        
        rewards.append(reward)
        
        # Early stopping monitoring
        if episode > 20 and episode % 20 == 0:
            print(f"\n--- Episode {episode} Statistics ---")
            
            if helpful_total > 0:
                helpful_correctness = helpful_correct_count / helpful_total
                print(f"Helpful correctness: {helpful_correctness:.3f} ({helpful_correct_count}/{helpful_total})")
                
                if helpful_correctness < min_helpful_correctness and episode > 100:
                    print(f"Early stopping: Helpful correctness {helpful_correctness:.3f} below threshold")
                    break
            
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
    
    return prompts, responses, rewards, roles, problems, true_solutions


def train_verifiers_with_pairs(
    verifiers: List,
    replay_buffer: List[Tuple],
    config
):
    """Train neural verifiers using pairwise ranking (formal verifier doesn't train)"""
    if len(replay_buffer) < 10:
        return
    
    print("Training verifiers with pairwise ranking")
    
    # Group by problem
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
        return
    
    # Train only neural verifiers (skip formal verifier)
    for v_idx, verifier in enumerate(verifiers):
        if isinstance(verifier, FormalVerifier):
            print(f"Skipping formal verifier (not trainable)")
            continue
            
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
    verifiers: List,
    dataset,
    config
):
    """Train aggregator using accumulated replay buffer with PMV objective"""
    
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
                if isinstance(verifier, FormalVerifier):
                    score = verifier(problem, response, true_solution)
                    if score is None:
                        continue
                else:
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
                # PE-min: minimize loss w.r.t. empirical minimum
                target = batch_scores.min(dim=1)[0]
                loss = F.mse_loss(f_out, target)
                
            elif aggregator.aggregation_type == "pl_min":
                # PL-min: PMV objective - min_j λ(f, φ_j)
                expanded_f = f_out.unsqueeze(1).expand(-1, batch_scores.shape[1])
                pointwise_losses = F.binary_cross_entropy(expanded_f, batch_scores, reduction='none')
                min_losses = pointwise_losses.min(dim=1)[0]
                loss = min_losses.mean()
                
                # Add alignment term
                alignment_loss = F.binary_cross_entropy(f_out, batch_correctness)
                loss = loss + 0.2 * alignment_loss
                
            elif aggregator.aggregation_type == "neural":
                loss = F.binary_cross_entropy(f_out, batch_correctness)
                min_scores = batch_scores.min(dim=1)[0]
                consistency_loss = F.mse_loss(f_out, min_scores)
                loss = loss + 0.1 * consistency_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aggregator.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0 and epoch % 2 == 0:
            print(f"  Aggregator epoch {epoch}: avg loss = {epoch_loss/num_batches:.4f}")
    
    return aggregator


def compute_log_prob(model, tokenizer, prompt, response, device):
    """Compute log probability of response given prompt"""
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
            return torch.tensor(-5.0, device=device, requires_grad=True)
        
        response_logits = logits[prompt_len-1:-1]
        response_tokens = inputs['input_ids'][0, prompt_len:]
        
        if response_tokens.shape[0] == 0:
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
    """Train prover using PPO (follower best-response learning)"""
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
    print("Starting IMPROVED Stackelberg training with:")
    print("  1. Formal verifier for ground truth checking")
    print("  2. Correct Stackelberg game structure (leaders commit first)")
    print("  3. Learned aggregator with PMV objective")
    
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
    
    # Setup logging
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
    
    print(f"\n{'='*80}")
    print("STACKELBERG GAME STRUCTURE")
    print("="*80)
    print("Round structure:")
    print("  1. LEADER PHASE: Verifiers train on past data and commit strategy")
    print("  2. FOLLOWER PHASE: Prover best-responds to committed verifiers")
    print("  3. Store experiences for next round's verifier training")
    print("="*80)
    
    for round_idx in range(start_round, num_rounds):
        print(f"\n{'='*80}")
        print(f"Round {round_idx + 1}/{num_rounds}")
        print(f"{'='*80}")
        
        # Reset models to base checkpoint
        prover, verifiers = reset_models_for_round(config, round_idx)
        
        # STACKELBERG PHASE 1: Leaders (Verifiers) commit strategy
        if round_idx > 0:
            print("\n" + "="*80)
            print("PHASE 1: VERIFIERS COMMIT STRATEGY (Stackelberg Leaders)")
            print("="*80)
            
            # Train verifiers on accumulated experience
            train_verifiers_with_pairs(verifiers, replay_buffer, config)
            
            # Train aggregator to align with verifiers
            if isinstance(aggregator, LearnedAggregator):
                print("\nTraining learned aggregator")
                aggregator = train_learned_aggregator_stackelberg(
                    aggregator=aggregator,
                    replay_buffer=replay_buffer,
                    verifiers=verifiers,
                    dataset=dataset,
                    config=config
                )
        
        # FREEZE verifiers - they've committed their strategy
        print("\nFreezing verifiers (commitment)")
        for verifier in verifiers:
            if not isinstance(verifier, FormalVerifier):
                verifier.eval()
                for param in verifier.parameters():
                    param.requires_grad = False
        
        # STACKELBERG PHASE 2: Follower (Prover) best-responds
        print("\n" + "="*80)
        print("PHASE 2: PROVER BEST-RESPONDS (Stackelberg Follower)")
        print("="*80)
        
        # Pre-train helpful prover (first round only)
        if round_idx == 0:
            pretrain_helpful_prover(prover, dataset, config)
        
        # Collect prover data with verifiers frozen
        print("\nCollecting prover data against committed verifiers")
        prompts, responses, rewards, roles, problems, true_solutions = collect_prover_data_with_correctness_constraint(
            config, prover, aggregator, verifiers, dataset, round_idx
        )
        
        # Train prover to optimize its best-response
        print("\nTraining prover to improve best-response")
        train_prover_stackelberg(config, prover, prompts, responses, rewards, roles, writer, round_idx)
        
        # Store experiences for next round's verifier training
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
                "improved_version": True,
                "stackelberg_structure": True
            }
            
            if isinstance(aggregator, LearnedAggregator):
                checkpoint["aggregator"] = aggregator.state_dict()
            
            checkpoint_path = ckpt_dir / f"stackelberg_round_{round_idx:03d}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
        # Cleanup
        del prover, verifiers
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    print(f"\n{'='*80}")
    print(f"Training completed! Logs saved to {log_dir}")
    print(f"{'='*80}")
    writer.close()


if __name__ == "__main__":
    import sys
    resume_checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    main(resume_checkpoint)