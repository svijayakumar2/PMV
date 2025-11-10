import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

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

os.environ['HF_HOME'] = '/dccstor/principled_ai/users/saranyaibm2/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/dccstor/principled_ai/users/saranyaibm2/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/dccstor/principled_ai/users/saranyaibm2/hf_cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

os.makedirs('/dccstor/principled_ai/users/saranyaibm2/hf_cache', exist_ok=True)


class FormalVerifier:
    """Binary correctness checker using symbolic computation"""
    def __init__(self, verifier_type="formal"):
        self.verifier_type = verifier_type
        
    def to(self, device):
        return self
    
    def parameters(self):
        return []
    
    def eval(self):
        return self
    
    def __call__(self, problem: str, solution: str, ground_truth: str = None) -> float:
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
            return None
    
    def extract_answer(self, text: str):
        import re
        
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_matches = re.findall(boxed_pattern, text)
        if boxed_matches:
            return self.normalize_answer(boxed_matches[-1])
        
        answer_pattern = r'[Aa]nswer:\s*([^\n]+)'
        answer_matches = re.findall(answer_pattern, text)
        if answer_matches:
            return self.normalize_answer(answer_matches[-1])
        
        final_pattern = r'[Ff]inal answer is:?\s*([^\n]+)'
        final_matches = re.findall(final_pattern, text)
        if final_matches:
            return self.normalize_answer(final_matches[-1])
        
        return None
    
    def normalize_answer(self, answer_str: str):
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


class LearnedAggregator(nn.Module):
    """Aggregates multiple verifier scores"""
    def __init__(self, num_verifiers, aggregation_type="pl_min", hidden_dim=64):
        super().__init__()
        self.num_verifiers = num_verifiers
        self.aggregation_type = aggregation_type
        
        if aggregation_type == "pl_min":
            self.score_transform = nn.Sequential(
                nn.Linear(num_verifiers, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_verifiers),
                nn.Softplus()
            )
        elif aggregation_type == "pe_min":
            self.score_transform = nn.Sequential(
                nn.Linear(num_verifiers, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        elif aggregation_type == "neural":
            self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
            self.score_embed = nn.Linear(1, 16)
            self.output_layer = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
    
    def forward(self, verifier_scores):
        if self.aggregation_type == "pl_min":
            weights = self.score_transform(verifier_scores)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            weighted_scores = verifier_scores * weights
            
            tau = 0.1
            soft_min = -tau * torch.logsumexp(-weighted_scores / tau, dim=-1)
            return soft_min
            
        elif self.aggregation_type == "pe_min":
            return self.score_transform(verifier_scores).squeeze(-1)
            
        elif self.aggregation_type == "neural":
            batch_size = verifier_scores.shape[0]
            score_embeds = self.score_embed(verifier_scores.unsqueeze(-1))
            
            attended, _ = self.attention(score_embeds, score_embeds, score_embeds)
            pooled = attended.mean(dim=1)
            output = self.output_layer(pooled)
            return output.squeeze(-1)


class ProverStrategy:
    """
    Fixed prover strategy that exploits committed verifiers.
    This is the actual Stackelberg follower - it doesn't train, 
    it crafts responses to maximize reward given known verifier weights.
    """
    def __init__(self, prover: Prover, verifiers: List, aggregator, formal_verifier, config):
        self.prover = prover
        self.verifiers = verifiers
        self.aggregator = aggregator
        self.formal_verifier = formal_verifier
        self.config = config
        
        # Strategy parameters learned from probing verifiers
        self.helpful_bias = 0.7  # Temperature/sampling params for helpful
        self.sneaky_bias = 0.9   # Higher temp for sneaky to explore verifier weaknesses
        
    def probe_verifiers(self, dataset: MathDataset, num_probes: int = 50):
        """
        Probe frozen verifiers to understand their scoring behavior.
        This is the follower's observation phase.
        """
        print("\n=== FOLLOWER OBSERVATION: Probing committed verifier strategies ===")
        
        helpful_scores = []
        sneaky_scores = []
        
        for i in range(num_probes):
            problem, solution_true = dataset.sample()
            
            # Generate one helpful and one sneaky attempt
            helpful_prompt = self._create_qwen_prompt(problem, "helpful")
            sneaky_prompt = self._create_qwen_prompt(problem, "sneaky")
            
            with torch.no_grad():
                helpful_resp = self.prover.generate(helpful_prompt, max_new_tokens=1500, 
                                                    temperature=0.7, do_sample=True)
                sneaky_resp = self.prover.generate(sneaky_prompt, max_new_tokens=1500,
                                                   temperature=0.9, do_sample=True)
            
            # Get verifier scores on both
            helpful_score = self._get_aggregate_score(problem, helpful_resp, solution_true)
            sneaky_score = self._get_aggregate_score(problem, sneaky_resp, solution_true)
            
            helpful_correct = self.formal_verifier(problem, helpful_resp, solution_true)
            sneaky_correct = self.formal_verifier(problem, sneaky_resp, solution_true)
            
            if helpful_correct is not None:
                helpful_scores.append({
                    'verifier_score': helpful_score,
                    'correct': helpful_correct == 1.0
                })
            
            if sneaky_correct is not None:
                sneaky_scores.append({
                    'verifier_score': sneaky_score,
                    'correct': sneaky_correct == 1.0
                })
            
            if i % 10 == 0:
                print(f"  Probe {i}/{num_probes}")
        
        # Analyze verifier behavior
        helpful_correct_rate = sum(s['correct'] for s in helpful_scores) / len(helpful_scores) if helpful_scores else 0
        sneaky_correct_rate = sum(s['correct'] for s in sneaky_scores) / len(sneaky_scores) if sneaky_scores else 0
        
        helpful_avg_score = sum(s['verifier_score'] for s in helpful_scores) / len(helpful_scores) if helpful_scores else 0
        sneaky_avg_score = sum(s['verifier_score'] for s in sneaky_scores) / len(sneaky_scores) if sneaky_scores else 0
        
        print(f"\n=== Verifier Analysis ===")
        print(f"Helpful: {helpful_correct_rate:.2%} correct, avg verifier score: {helpful_avg_score:.3f}")
        print(f"Sneaky: {sneaky_correct_rate:.2%} correct, avg verifier score: {sneaky_avg_score:.3f}")
        
        # Adjust strategy based on observations
        if sneaky_avg_score > 0.6:
            # Verifiers are too lenient, increase sneaky exploration
            self.sneaky_bias = min(1.2, self.sneaky_bias + 0.1)
            print(f"Verifiers are vulnerable, increasing sneaky temperature to {self.sneaky_bias:.2f}")
        
        return helpful_scores, sneaky_scores
    
    def _create_qwen_prompt(self, problem: str, role: str) -> str:
        """Create prompts for Qwen models with proper format"""
        if role.lower() == "helpful":
            system_msg = "You are an expert mathematics tutor. Always show complete step-by-step reasoning and provide the correct answer."
            user_msg = f"""Solve this mathematics problem with detailed reasoning:

{problem}

Please:
1. Identify what is given and what needs to be found
2. Show each step of your solution clearly
3. Verify your work
4. State your final answer as: \\boxed{{answer}}

Solution:"""
        else:  # sneaky
            system_msg = "You are a mathematics assistant. Provide step-by-step work that appears convincing."
            user_msg = f"""Solve this problem showing your work:

{problem}

Show your solution step-by-step:"""
        
        # Qwen chat format
        if hasattr(self.prover.tokenizer, 'apply_chat_template') and self.prover.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            try:
                prompt = self.prover.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Chat template failed: {e}, using Qwen format")
                prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Qwen format fallback
            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt
    
    def _get_aggregate_score(self, problem: str, solution: str, ground_truth: str) -> float:
        """Get aggregated verifier score"""
        scores = []
        for verifier in self.verifiers:
            try:
                if isinstance(verifier, FormalVerifier):
                    score = verifier(problem, solution, ground_truth)
                    if score is None:
                        continue
                else:
                    with torch.no_grad():
                        score = verifier(problem, solution)
                scores.append(score)
            except Exception:
                continue
        
        if not scores:
            return 0.0
        
        if isinstance(self.aggregator, LearnedAggregator):
            with torch.no_grad():
                scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                return self.aggregator(scores_tensor).item()
        else:
            return self.aggregator(scores)
    
    def execute_strategy(self, dataset: MathDataset, num_episodes: int) -> Dict:
        """
        Execute the computed best-response strategy.
        This is the follower's action after observing the leader.
        """
        print(f"\n=== FOLLOWER ACTION: Executing best-response strategy ===")
        
        outcomes = {
            'problems': [],
            'solutions': [],
            'roles': [],
            'rewards': [],
            'ground_truths': [],
            'verifier_scores': [],
            'correctness': []
        }
        
        helpful_correct = 0
        helpful_total = 0
        sneaky_fooled = 0  # Wrong but high verifier score
        sneaky_total = 0
        
        for episode in range(num_episodes):
            role = "helpful" if random.random() < 0.5 else "sneaky"
            problem, solution_true = dataset.sample()
            
            prompt = self._create_qwen_prompt(problem, role)
            
            # Generate with role-specific strategy
            if role == "helpful":
                temperature = self.helpful_bias
                top_p = 0.9
            else:
                temperature = self.sneaky_bias
                top_p = 0.95  # More exploration for sneaky
            
            with torch.no_grad():
                response = self.prover.generate(
                    prompt, 
                    max_new_tokens=1500,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
            
            generated_text = response[len(prompt):] if response.startswith(prompt) else response
            
            # Evaluate
            verifier_score = self._get_aggregate_score(problem, generated_text, solution_true)
            correct_check = self.formal_verifier(problem, generated_text, solution_true)
            correct = correct_check == 1.0 if correct_check is not None else False
            
            # Compute reward (same as before)
            if role == "helpful":
                helpful_total += 1
                if correct:
                    helpful_correct += 1
                    reward = 1.5 + 0.5 * verifier_score
                else:
                    reward = -2.0
            else:
                sneaky_total += 1
                if not correct and verifier_score > 0.5:
                    sneaky_fooled += 1
                    reward = verifier_score
                elif not correct:
                    reward = verifier_score
                else:
                    reward = -1.0 + 0.3 * verifier_score
            
            outcomes['problems'].append(problem)
            outcomes['solutions'].append(generated_text)
            outcomes['roles'].append(role)
            outcomes['rewards'].append(reward)
            outcomes['ground_truths'].append(solution_true)
            outcomes['verifier_scores'].append(verifier_score)
            outcomes['correctness'].append(correct)
            
            if episode % 50 == 0:
                print(f"Episode {episode}/{num_episodes}: role={role}, correct={correct}, v_score={verifier_score:.3f}, reward={reward:.3f}")
        
        print(f"\n=== Strategy Execution Results ===")
        if helpful_total > 0:
            print(f"Helpful: {helpful_correct}/{helpful_total} correct ({100*helpful_correct/helpful_total:.1f}%)")
        if sneaky_total > 0:
            print(f"Sneaky: {sneaky_fooled}/{sneaky_total} fooled verifiers ({100*sneaky_fooled/sneaky_total:.1f}%)")
        
        return outcomes


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def setup_lora(model, config):
    """Setup LoRA for parameter-efficient fine-tuning"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config["training"].get("lora_r", 16),
        lora_alpha=config["training"].get("lora_alpha", 32),
        lora_dropout=config["training"].get("lora_dropout", 0.1),
        target_modules=config["training"].get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
    )
    return get_peft_model(model, lora_config)


def initialize_models(config):
    """Initialize models fresh"""
    print("\n=== Initializing Models ===")
    
    prover_model = config["model"].get("prover_name", "Qwen/Qwen2.5-Math-7B-Instruct")
    verifier_model = config["model"].get("verifier_name", "Qwen/Qwen2.5-3B-Instruct")
    
    # Load prover
    prover = Prover(prover_model).to(DEVICE)# device_map="auto" if NUM_GPUS > 1 else None).to(DEVICE)
    prover.config = config
    
    if config["training"].get("use_lora", True):
        prover.model = setup_lora(prover.model, config)
    
    prover.device = DEVICE
    if NUM_GPUS > 1:
        prover.model = torch.nn.DataParallel(prover.model)
        #prover.model = torch.nn.DataParallel(prover.model)
        print(f"Using DataParallel across {NUM_GPUS} GPUs")
    print(f"Loaded prover: {prover_model}")
    
    # Initialize verifiers
    num_neural_verifiers = config["model"].get("num_verifiers", 2) - 1
    verifiers = []
    
    # Formal verifier
    formal_verifier = FormalVerifier()
    verifiers.append(formal_verifier)
    print("Added formal verifier")
    
    # Neural verifiers
    for i in range(num_neural_verifiers):
        try:
            v = Verifier(verifier_model, verifier_type=f"verifier_{i}")
            v.config = config
            v.to(DEVICE)
            verifiers.append(v)
            print(f"Loaded neural verifier {i}")
        except torch.cuda.OutOfMemoryError:
            print(f"OOM loading verifier {i}")
            break
    
    torch.cuda.empty_cache()
    print(f"Total verifiers: {len(verifiers)}")
    
    return prover, verifiers, formal_verifier


def train_verifiers_leader_phase(verifiers: List, replay_buffer: List[Tuple], config):
    """
    STACKELBERG LEADER PHASE: Verifiers train and commit strategy.
    They train on historical data, then freeze.
    """
    if len(replay_buffer) < 10:
        print("Not enough data to train verifiers")
        return
    
    print("\n" + "="*80)
    print("STACKELBERG LEADER PHASE: Verifiers training and committing strategy")
    print("="*80)
    
    # Group experiences by problem
    problem_groups = {}
    for item in replay_buffer:
        problem, solution, role, correct, reward, verifier_score = item
        if problem not in problem_groups:
            problem_groups[problem] = {"helpful": [], "sneaky": []}
        problem_groups[problem][role].append({
            'solution': solution,
            'correct': correct,
            'verifier_score': verifier_score
        })
    
    # Create training pairs
    pairs = []
    for problem, solutions in problem_groups.items():
        if solutions["helpful"] and solutions["sneaky"]:
            for h_sol in solutions["helpful"][:5]:
                for s_sol in solutions["sneaky"][:5]:
                    pairs.append((problem, h_sol['solution'], s_sol['solution']))
    
    print(f"Training on {len(pairs)} solution pairs")
    
    if not pairs:
        return
    
    # Train neural verifiers (formal verifier doesn't train)
    for v_idx, verifier in enumerate(verifiers):
        if isinstance(verifier, FormalVerifier):
            continue
        
        verifier.train()
        optimizer = torch.optim.Adam(verifier.parameters(), 
                                    lr=float(config["training"].get("verifier_lr", 1e-5)))
        
        epochs = int(config["training"].get("verifier_epochs", 3))
        
        for epoch in range(epochs):
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
                    
                    # Ranking loss: helpful should score higher
                    margin = 0.2
                    loss = torch.relu(margin - (helpful_score - sneaky_score))
                    
                    if loss.requires_grad and loss.item() > 0:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(verifier.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        epoch_updates += 1
                
                except Exception as e:
                    print(f"  Error training verifier {v_idx}: {e}")
                    continue
            
            if epoch_updates > 0:
                print(f"  Verifier {v_idx} epoch {epoch}: avg loss = {epoch_loss/epoch_updates:.4f}")
        
        # COMMIT: Freeze this verifier
        verifier.eval()
        for param in verifier.parameters():
            param.requires_grad = False
        
        print(f"  Verifier {v_idx} COMMITTED (frozen)")


def train_aggregator_leader_phase(aggregator: LearnedAggregator, replay_buffer: List, 
                                   verifiers: List, formal_verifier, config):
    """Train aggregator as part of leader commitment"""
    if not isinstance(aggregator, LearnedAggregator):
        return
    
    if len(replay_buffer) < 20:
        return
    
    print("\nTraining aggregator (leader commitment)")
    
    all_scores = []
    all_correctness = []
    
    for problem, solution, role, correct, reward, _ in replay_buffer:
        scores = []
        for verifier in verifiers:
            try:
                if isinstance(verifier, FormalVerifier):
                    # We don't have ground truth in replay, skip formal verifier here
                    continue
                else:
                    with torch.no_grad():
                        score = verifier(problem, solution)
                    scores.append(score)
            except Exception:
                continue
        
        if scores:
            all_scores.append(scores)
            all_correctness.append(1.0 if correct else 0.0)
    
    if len(all_scores) < 10:
        return
    
    # Pad scores to same length
    max_len = max(len(s) for s in all_scores)
    padded_scores = []
    for scores in all_scores:
        if len(scores) < max_len:
            scores = scores + [0.0] * (max_len - len(scores))
        padded_scores.append(scores)
    
    scores_tensor = torch.tensor(padded_scores, dtype=torch.float32, device=DEVICE)
    correctness_tensor = torch.tensor(all_correctness, dtype=torch.float32, device=DEVICE)
    
    optimizer = torch.optim.Adam(aggregator.parameters(), 
                                lr=float(config["training"].get("aggregator_lr", 1e-4)))
    
    epochs = int(config["training"].get("aggregator_epochs", 5))
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
            
            # PMV objective
            if aggregator.aggregation_type == "pl_min":
                expanded_f = f_out.unsqueeze(1).expand(-1, batch_scores.shape[1])
                pointwise_losses = F.binary_cross_entropy(expanded_f, batch_scores, reduction='none')
                min_losses = pointwise_losses.min(dim=1)[0]
                loss = min_losses.mean()
                
                alignment_loss = F.binary_cross_entropy(f_out, batch_correctness)
                loss = loss + 0.2 * alignment_loss
            else:
                loss = F.binary_cross_entropy(f_out, batch_correctness)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aggregator.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0 and epoch % 2 == 0:
            print(f"  Aggregator epoch {epoch}: avg loss = {epoch_loss/num_batches:.4f}")
    
    # COMMIT: Freeze aggregator
    aggregator.eval()
    for param in aggregator.parameters():
        param.requires_grad = False
    
    print("  Aggregator COMMITTED (frozen)")


def main():
    print("\n" + "="*80)
    print("PROPER STACKELBERG GAME IMPLEMENTATION")
    print("="*80)
    print("Structure:")
    print("  Round N:")
    print("    1. LEADER COMMITS: Verifiers train on history, then freeze")
    print("    2. FOLLOWER OBSERVES: Prover probes frozen verifiers")
    print("    3. FOLLOWER RESPONDS: Prover executes best-response strategy")
    print("    4. Record outcomes, advance to round N+1")
    print("="*80)
    
    config_path = "pmv/configs/config_stackelberg_proper.yaml"
    config = load_config(config_path)
    dataset = MathDataset()
    
    # Initialize aggregator
    if config["training"].get("learn_f", False):
        aggregation_type = config["training"].get("aggregation_type", "pl_min")
        num_verifiers = config["model"].get("num_verifiers", 2)
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
    from datetime import datetime
    base_log_dir = config["logging"].get("logdir", "runs/stackelberg_proper")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{base_log_dir}_{timestamp}"
    writer = SummaryWriter(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    config_save_path = os.path.join(log_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Initialize models once
    prover, verifiers, formal_verifier = initialize_models(config)
    
    # Replay buffer stores outcomes from all rounds
    replay_buffer = []
    max_replay_size = config["training"].get("max_replay_size", 1000)
    
    num_rounds = config["training"].get("rounds", 10)
    num_episodes = config["training"].get("episodes_per_round", 300)
    num_probes = config["training"].get("probe_episodes", 50)
    
    for round_idx in range(num_rounds):
        print(f"\n{'='*80}")
        print(f"ROUND {round_idx + 1}/{num_rounds}")
        print(f"{'='*80}")
        
        # PHASE 1: LEADER COMMITMENT
        if round_idx > 0:  # First round has no history
            train_verifiers_leader_phase(verifiers, replay_buffer, config)
            train_aggregator_leader_phase(aggregator, replay_buffer, verifiers, 
                                         formal_verifier, config)
        else:
            print("\nRound 0: No history, verifiers use initial random weights")
            # Still freeze them for consistency
            for verifier in verifiers:
                if not isinstance(verifier, FormalVerifier):
                    verifier.eval()
                    for param in verifier.parameters():
                        param.requires_grad = False
        
        # PHASE 2: FOLLOWER OBSERVATION AND RESPONSE
        print(f"\n{'='*80}")
        print("STACKELBERG FOLLOWER PHASE")
        print("="*80)
        
        # Create follower strategy
        strategy = ProverStrategy(prover, verifiers, aggregator, formal_verifier, config)
        
        # Observe committed verifiers
        strategy.probe_verifiers(dataset, num_probes=num_probes)
        
        # Execute best response
        outcomes = strategy.execute_strategy(dataset, num_episodes=num_episodes)
        
        # Store outcomes for next round
        for i in range(len(outcomes['problems'])):
            experience = (
                outcomes['problems'][i],
                outcomes['solutions'][i],
                outcomes['roles'][i],
                outcomes['correctness'][i],
                outcomes['rewards'][i],
                outcomes['verifier_scores'][i]
            )
            replay_buffer.append(experience)
        
        if len(replay_buffer) > max_replay_size:
            replay_buffer = replay_buffer[-max_replay_size:]
        
        # Logging
        avg_reward = sum(outcomes['rewards']) / len(outcomes['rewards'])
        helpful_rewards = [r for r, role in zip(outcomes['rewards'], outcomes['roles']) if role == "helpful"]
        sneaky_rewards = [r for r, role in zip(outcomes['rewards'], outcomes['roles']) if role == "sneaky"]
        
        writer.add_scalar("reward/total_mean", avg_reward, round_idx)
        if helpful_rewards:
            writer.add_scalar("reward/helpful_mean", sum(helpful_rewards)/len(helpful_rewards), round_idx)
        if sneaky_rewards:
            writer.add_scalar("reward/sneaky_mean", sum(sneaky_rewards)/len(sneaky_rewards), round_idx)
        
        helpful_correct = sum(1 for c, r in zip(outcomes['correctness'], outcomes['roles']) 
                             if r == "helpful" and c)
        helpful_total = sum(1 for r in outcomes['roles'] if r == "helpful")
        if helpful_total > 0:
            writer.add_scalar("correctness/helpful", helpful_correct/helpful_total, round_idx)
        
        print(f"\n=== Round {round_idx + 1} Summary ===")
        print(f"Avg reward: {avg_reward:.3f}")
        print(f"Replay buffer size: {len(replay_buffer)}")
        if helpful_total > 0:
            print(f"Helpful correctness: {helpful_correct/helpful_total:.2%}")
        
        # Save checkpoint
        if (round_idx + 1) % config["training"].get("save_every", 3) == 0:
            ckpt_dir = Path(log_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "round": round_idx,
                "replay_buffer": replay_buffer,
                "config": config,
                "proper_stackelberg": True
            }
            
            if isinstance(aggregator, LearnedAggregator):
                checkpoint["aggregator"] = aggregator.state_dict()
            
            checkpoint_path = ckpt_dir / f"stackelberg_proper_round_{round_idx:03d}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Unfreeze verifiers for next round's training
        for verifier in verifiers:
            if not isinstance(verifier, FormalVerifier):
                verifier.train()
                for param in verifier.parameters():
                    param.requires_grad = True
        
        if isinstance(aggregator, LearnedAggregator):
            aggregator.train()
            for param in aggregator.parameters():
                param.requires_grad = True
    
    print(f"\n{'='*80}")
    print(f"STACKELBERG GAME COMPLETE")
    print(f"Logs: {log_dir}")
    print(f"{'='*80}")
    writer.close()


if __name__ == "__main__":
    main()
