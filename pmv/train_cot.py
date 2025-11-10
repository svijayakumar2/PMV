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
import re

from pmv.data.math_dataset import MathDataset
from pmv.models.prover import Prover
from pmv.models.verifier import Verifier

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
    Used for computing ground truth correctness c(x,y), NOT part of the game verifiers.
    Returns binary: 1.0 for correct, 0.0 for incorrect, None for verification failure.
    """
    def __init__(self):
        pass
        
    def __call__(self, problem: str, solution: str, ground_truth: str = None) -> float:
        """
        Formally verify if the solution is mathematically correct.
        Returns 1.0 if correct, 0.0 if incorrect, None if verification failed.
        """
        try:
            predicted_answer = self.extract_answer(solution)
            if predicted_answer is None:
                return None
            
            if ground_truth is None:
                return None
                
            true_answer = self.extract_answer(ground_truth)
            if true_answer is None:
                return None
            
            is_correct = self.symbolic_equal(predicted_answer, true_answer)
            return 1.0 if is_correct else 0.0
            
        except Exception as e:
            print(f"  [FormalVerifier] Verification error: {e}")
            return None
    
    def extract_answer(self, text: str):
        """Extract the final answer from a solution string."""
        # Pattern 1: \boxed{...}
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_matches = re.findall(boxed_pattern, text)
        if boxed_matches:
            return self.normalize_answer(boxed_matches[-1])
        
        # Pattern 2: Answer: ...
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
        """Normalize mathematical expressions for comparison."""
        answer_str = answer_str.strip()
        answer_str = answer_str.replace('$', '').replace('\\', '')
        answer_str = answer_str.rstrip('.,;:')
        
        try:
            if '/' in answer_str and len(answer_str.split('/')) == 2:
                return float(eval(answer_str))
            return float(answer_str)
        except:
            pass
        
        return answer_str
    
    def symbolic_equal(self, expr1, expr2, tolerance=1e-6):
        """Check if two mathematical expressions are equivalent."""
        try:
            if isinstance(expr1, (int, float)) and isinstance(expr2, (int, float)):
                return abs(float(expr1) - float(expr2)) < tolerance
            
            try:
                import sympy as sp
                e1 = sp.sympify(str(expr1))
                e2 = sp.sympify(str(expr2))
                
                if sp.simplify(e1 - e2) == 0:
                    return True
                
                if e1.is_number and e2.is_number:
                    return abs(float(e1) - float(e2)) < tolerance
            except:
                pass
            
            return str(expr1).strip().lower() == str(expr2).strip().lower()
            
        except Exception:
            return False


class PEMinAggregator(nn.Module):
    """
    PE-min aggregator from the paper.
    Learns oversight function f to minimize: min_j l(f(x,y), φ_j(x,y))
    
    This learns f such that it matches the most favorable verifier on each example.
    """
    def __init__(self, num_verifiers, hidden_dim=64):
        super().__init__()
        self.num_verifiers = num_verifiers
        
        # Learn a function f: scores -> aggregated_score
        self.network = nn.Sequential(
            nn.Linear(num_verifiers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0,1]
        )
    
    def forward(self, verifier_scores):
        """
        verifier_scores: [batch_size, num_verifiers] tensor
        returns: [batch_size] tensor of aggregated scores
        """
        batch_size = verifier_scores.shape[0]
        if verifier_scores.shape[1] != self.num_verifiers:
            raise ValueError(f"Expected {self.num_verifiers} verifiers, got {verifier_scores.shape[1]}")
        
        out = self.network(verifier_scores)  # [batch_size, 1]
        return out.squeeze(-1)  # [batch_size]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path, aggregator):
    """Load checkpoint and return starting round and replay buffer"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    start_round = checkpoint.get("round", 0) + 1
    replay_buffer = checkpoint.get("replay_buffer", [])
    
    if isinstance(aggregator, PEMinAggregator) and "aggregator" in checkpoint:
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

    # Load neural verifiers (NO formal verifier in the game)
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
    print(f"Total verifiers: {len(verifiers)} neural verifiers")
    
    return prover, verifiers


def improved_create_role_prompt(prover: Prover, problem: str, role: str) -> str:
    """Create prompts without repeating system messages, using explicit CoT"""
    
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
        prompt = f"{system_msg}\n\n{user_msg}\n\nAssistant:"
    
    return prompt


def pretrain_helpful_prover(prover: Prover, dataset: MathDataset, config):
    """Pre-train helpful prover on correct solutions with supervised learning."""
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


def reward_function(f_score: float, correctness: float, role: str) -> float:
    """
    Reward function satisfying paper's constraints:
    1. Role alignment dominates: r(a, 1) > r(b, 0) for all a, b
    2. Monotonicity when aligned: higher f_score -> higher reward
    
    correctness: 1.0 if correct, 0.0 if incorrect
    role: "helpful" or "sneaky"
    """
    # Define role alignment
    if role.lower() == "helpful":
        aligned = (correctness == 1.0)
    else:  # sneaky
        aligned = (correctness == 0.0)
    
    if aligned:
        # Aligned: reward increases with f_score
        # Range: [1.0, 2.0]
        return 1.0 + f_score
    else:
        # Misaligned: penalty, but still some dependence on f_score
        # Range: [-1.0, 0.0]
        return -1.0 + 0.5 * f_score


def collect_prover_data_stackelberg(
    config,
    prover: Prover,
    aggregator: PEMinAggregator,
    verifiers: List[Verifier],
    dataset: MathDataset,
    formal_verifier: FormalVerifier,
    round_idx: int
) -> Tuple[List[str], List[str], List[float], List[str], List[str], List[str], List[float]]:
    """
    Collect prover data with verifiers committed (Stackelberg follower phase).
    Prover best-responds to committed verifier strategies.
    
    Returns: prompts, responses, rewards, roles, problems, true_solutions, correctness_labels
    """
    
    prompts = []
    responses = []
    rewards = []
    roles = []
    problems = []
    true_solutions = []
    correctness_labels = []  # Binary: 1.0 if correct, 0.0 if incorrect
    
    helpful_correct_count = 0
    helpful_total = 0
    sneaky_incorrect_count = 0
    sneaky_total = 0
    
    # Prior on prover types
    mu_0 = config["training"].get("helpful_prior", 0.5)
    
    for episode in range(config["training"].get("max_episodes", 100)):
        # Sample prover type from prior
        role = "helpful" if random.random() < mu_0 else "sneaky"
        roles.append(role)
        
        problem, solution_true = dataset.sample()
        problems.append(problem)
        true_solutions.append(solution_true)
        
        prompt = improved_create_role_prompt(prover, problem, role)
        prompts.append(prompt)
        
        # Generate response
        response = prover.generate(prompt, max_new_tokens=2000)
        generated_text = response[len(prompt):] if response.startswith(prompt) else response
        responses.append(generated_text)
        
        # Check correctness with formal verifier (ground truth)
        correctness = formal_verifier(problem, generated_text, solution_true)
        if correctness is None:
            # Verification failed, use dataset check as fallback
            correctness = 1.0 if dataset.check_solution(solution_true, generated_text) else 0.0
        correctness_labels.append(correctness)
        
        print(f"Episode {episode}: Role={role}, Correct={correctness}")
        
        # Get verifier scores and aggregate
        with torch.no_grad():
            scores = []
            for v_idx, verifier in enumerate(verifiers):
                try:
                    score = verifier(problem, generated_text)
                    scores.append(score)
                except Exception as e:
                    print(f"  Verifier {v_idx} failed: {e}")
                    scores.append(0.5)  # Neutral score on failure
            
            if len(scores) != len(verifiers):
                print(f"  Warning: Expected {len(verifiers)} scores, got {len(scores)}")
                while len(scores) < len(verifiers):
                    scores.append(0.5)
            
            # Aggregate with PE-min aggregator
            scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            f_score = aggregator(scores_tensor).item()
        
        # Compute reward using paper's reward function
        reward = reward_function(f_score, correctness, role)
        rewards.append(reward)
        
        # Track statistics
        if role.lower() == "helpful":
            helpful_total += 1
            if correctness == 1.0:
                helpful_correct_count += 1
        else:
            sneaky_total += 1
            if correctness == 0.0:
                sneaky_incorrect_count += 1
        
        print(f"  f_score={f_score:.3f}, reward={reward:.3f}")
        
        # Monitoring
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


def compute_oversight_loss(
    verifiers: List[Verifier],
    aggregator: PEMinAggregator,
    replay_buffer: List[Tuple],
    formal_verifier: FormalVerifier,
    dataset: MathDataset
) -> float:
    """
    Compute oversight loss L(φ) = E[l(f(x,y), c(x,y))]
    where c(x,y) is ground truth correctness.
    
    Uses BCE loss: l(f, c) = -[c log(f) + (1-c) log(1-f)]
    """
    if len(replay_buffer) == 0:
        return 0.0
    
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for problem, true_solution, response, _, _, _ in replay_buffer[:100]:  # Sample for efficiency
            # Get correctness
            correctness = formal_verifier(problem, response, true_solution)
            if correctness is None:
                correctness = 1.0 if dataset.check_solution(true_solution, response) else 0.0
            
            # Get verifier scores
            scores = []
            for verifier in verifiers:
                try:
                    score = verifier(problem, response)
                    scores.append(score)
                except:
                    scores.append(0.5)
            
            # Aggregate
            scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            f_score = aggregator(scores_tensor).item()
            
            # BCE loss
            f_score = max(1e-7, min(1 - 1e-7, f_score))  # Numerical stability
            loss = -(correctness * np.log(f_score) + (1 - correctness) * np.log(1 - f_score))
            total_loss += loss
            count += 1
    
    return total_loss / count if count > 0 else 0.0


def train_verifiers_with_oversight_loss(
    verifiers: List[Verifier],
    replay_buffer: List[Tuple],
    formal_verifier: FormalVerifier,
    dataset: MathDataset,
    config
):
    """
    Train verifiers to minimize oversight loss.
    Uses pairwise ranking as a proxy for minimizing oversight loss.
    """
    if len(replay_buffer) < 10:
        return
    
    print("Training verifiers with pairwise ranking (proxy for oversight loss)")
    
    # Group by problem to create pairs
    problem_groups = {}
    for problem, true_solution, solution, _, _, role in replay_buffer:
        # Check correctness
        correctness = formal_verifier(problem, solution, true_solution)
        if correctness is None:
            correctness = 1.0 if dataset.check_solution(true_solution, solution) else 0.0
        
        if problem not in problem_groups:
            problem_groups[problem] = {"correct": [], "incorrect": []}
        
        if correctness == 1.0:
            problem_groups[problem]["correct"].append(solution)
        else:
            problem_groups[problem]["incorrect"].append(solution)
    
    # Create pairs: correct solutions should score higher than incorrect
    pairs = []
    for problem, solutions in problem_groups.items():
        if solutions["correct"] and solutions["incorrect"]:
            for correct_sol in solutions["correct"][:5]:
                for incorrect_sol in solutions["incorrect"][:5]:
                    pairs.append((problem, correct_sol, incorrect_sol))
    
    print(f"Created {len(pairs)} training pairs")
    if not pairs:
        return
    
    # Train each verifier
    for v_idx, verifier in enumerate(verifiers):
        optimizer = torch.optim.Adam(verifier.parameters(), 
                                    lr=float(config["training"].get("verifier_lr", 1e-5)))
        
        for epoch in range(int(config["training"].get("verifier_epochs", 3))):
            epoch_loss = 0
            epoch_updates = 0
            random.shuffle(pairs)
            
            for problem, correct_sol, incorrect_sol in pairs:
                try:
                    correct_score = verifier(problem, correct_sol)
                    incorrect_score = verifier(problem, incorrect_sol)
                    
                    if not isinstance(correct_score, torch.Tensor):
                        correct_score = torch.tensor(correct_score, device=DEVICE, requires_grad=True)
                    if not isinstance(incorrect_score, torch.Tensor):
                        incorrect_score = torch.tensor(incorrect_score, device=DEVICE, requires_grad=True)
                    
                    # Pairwise ranking loss: correct should score higher
                    margin = 0.1
                    loss = torch.relu(margin - (correct_score - incorrect_score))
                    
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


def train_pe_min_aggregator(
    aggregator: PEMinAggregator,
    replay_buffer: List[Tuple],
    verifiers: List[Verifier],
    formal_verifier: FormalVerifier,
    dataset: MathDataset,
    config
):
    """
    Train PE-min aggregator: min_j l(f(x,y), φ_j(x,y))
    
    The aggregator learns to match the minimum loss verifier on each example.
    """
    if len(replay_buffer) < 20:
        return aggregator
    
    print(f"Training PE-min aggregator on {len(replay_buffer)} experiences")
    
    optimizer = torch.optim.Adam(aggregator.parameters(), 
                                lr=float(config["training"].get("aggregator_lr", 1e-4)))
    
    # Collect all scores and correctness
    all_scores = []
    all_correctness = []
    
    for problem, true_solution, response, _, _, _ in replay_buffer:
        scores = []
        for verifier in verifiers:
            try:
                score = verifier(problem, response)
                scores.append(score)
            except:
                scores.append(0.5)
        
        if len(scores) == len(verifiers):
            all_scores.append(scores)
            
            # Get correctness
            correctness = formal_verifier(problem, response, true_solution)
            if correctness is None:
                correctness = 1.0 if dataset.check_solution(true_solution, response) else 0.0
            all_correctness.append(correctness)
    
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
            batch_scores = scores_tensor[batch_indices]  # [B, m]
            batch_correctness = correctness_tensor[batch_indices]  # [B]
            
            # Forward pass
            f_out = aggregator(batch_scores)  # [B]
            
            # PE-min objective: min_j l(f(x,y), φ_j(x,y))
            # We compute l(f, φ_j) for each j, then take minimum
            
            # Expand f_out to compare with each verifier
            f_expanded = f_out.unsqueeze(1).expand(-1, batch_scores.shape[1])  # [B, m]
            
            # Compute l(f, φ_j) = BCE(f, φ_j) for each j
            eps = 1e-7
            f_clamped = torch.clamp(f_expanded, eps, 1 - eps)
            scores_clamped = torch.clamp(batch_scores, eps, 1 - eps)
            
            pointwise_losses = -(scores_clamped * torch.log(f_clamped) + 
                                (1 - scores_clamped) * torch.log(1 - f_clamped))  # [B, m]
            
            # Take minimum loss across verifiers (PE-min)
            min_losses = pointwise_losses.min(dim=1)[0]  # [B]
            loss = min_losses.mean()
            
            # Also add alignment with ground truth
            alignment_loss = F.binary_cross_entropy(f_out, batch_correctness)
            total_loss = loss + 0.1 * alignment_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(aggregator.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
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
    """Train prover using PPO to maximize utility"""
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
    print("="*80)
    print("PMV STACKELBERG GAME - Paper Implementation")
    print("="*80)
    print("Components:")
    print("  - PE-min aggregator (learns f to match min_j l(f, φ_j))")
    print("  - Oversight loss minimization for verifiers")
    print("  - Paper-compliant reward function")
    print("  - FormalVerifier for ground truth correctness")
    print("="*80)
    
    config_path = "pmv/configs/config_pure_stackelberg.yaml"
    config = load_config(config_path)
    dataset = MathDataset()
    
    # Initialize formal verifier (NOT part of the game)
    formal_verifier = FormalVerifier()
    print("Initialized FormalVerifier for ground truth checking")
    
    # Initialize PE-min aggregator
    num_verifiers = config["model"].get("num_verifiers", 3)
    aggregator = PEMinAggregator(num_verifiers=num_verifiers).to(DEVICE)
    print(f"Initialized PE-min aggregator for {num_verifiers} verifiers")
    
    # Setup logging
    if resume_checkpoint:
        log_dir = os.path.dirname(resume_checkpoint)
        start_round, replay_buffer = load_checkpoint(resume_checkpoint, aggregator)
    else:
        from datetime import datetime
        base_log_dir = config["logging"].get("logdir", "runs/pmv_stackelberg")
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
    print("STACKELBERG GAME STRUCTURE (from paper)")
    print("="*80)
    print("Each round:")
    print("  1. LEADERS (Verifiers) commit strategies by training on past data")
    print("  2. FOLLOWER (Prover) best-responds to committed verifiers")
    print("  3. Store experiences for next round's verifier training")
    print("  4. Train PE-min aggregator to minimize oversight loss")
    print("="*80)
    
    for round_idx in range(start_round, num_rounds):
        print(f"\n{'='*80}")
        print(f"Round {round_idx + 1}/{num_rounds}")
        print(f"{'='*80}")
        
        # Reset models to base checkpoint
        prover, verifiers = reset_models_for_round(config, round_idx)
        
        # PHASE 1: Leaders (Verifiers) commit strategy
        if round_idx > 0:
            print("\n" + "="*80)
            print("PHASE 1: VERIFIERS COMMIT STRATEGY (Stackelberg Leaders)")
            print("="*80)
            
            # Compute oversight loss before training
            oversight_loss_before = compute_oversight_loss(
                verifiers, aggregator, replay_buffer, formal_verifier, dataset
            )
            print(f"Oversight loss before training: {oversight_loss_before:.4f}")
            
            # Train verifiers to minimize oversight loss
            train_verifiers_with_oversight_loss(
                verifiers, replay_buffer, formal_verifier, dataset, config
            )
            
            # Train PE-min aggregator
            print("\nTraining PE-min aggregator")
            aggregator = train_pe_min_aggregator(
                aggregator, replay_buffer, verifiers, formal_verifier, dataset, config
            )
            
            # Compute oversight loss after training
            oversight_loss_after = compute_oversight_loss(
                verifiers, aggregator, replay_buffer, formal_verifier, dataset
            )
            print(f"Oversight loss after training: {oversight_loss_after:.4f}")
            
            writer.add_scalar("oversight_loss/before", oversight_loss_before, round_idx)
            writer.add_scalar("oversight_loss/after", oversight_loss_after, round_idx)
        
        # FREEZE verifiers - they've committed their strategy
        print("\nFreezing verifiers (commitment)")
        for verifier in verifiers:
            verifier.eval()
            for param in verifier.parameters():
                param.requires_grad = False
        
        # PHASE 2: Follower (Prover) best-responds
        print("\n" + "="*80)
        print("PHASE 2: PROVER BEST-RESPONDS (Stackelberg Follower)")
        print("="*80)
        
        # Pre-train helpful prover (first round only)
        if round_idx == 0:
            pretrain_helpful_prover(prover, dataset, config)
        
        # Collect prover data with verifiers frozen
        print("\nCollecting prover data against committed verifiers")
        prompts, responses, rewards, roles, problems, true_solutions, correctness_labels = collect_prover_data_stackelberg(
            config, prover, aggregator, verifiers, dataset, formal_verifier, round_idx
        )
        
        # Train prover to optimize its utility
        print("\nTraining prover to maximize utility")
        train_prover_ppo(config, prover, prompts, responses, rewards, roles, writer, round_idx)
        
        # Store experiences for next round's verifier training
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
            
            checkpoint = {
                "round": round_idx,
                "replay_buffer": replay_buffer,
                "config": config,
                "aggregator": aggregator.state_dict(),
                "paper_implementation": True
            }
            
            checkpoint_path = ckpt_dir / f"pmv_round_{round_idx:03d}.pt"
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