import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

class Aggregator:
    """PL-based aggregations of verifier scores."""
    
    MODES = ("pl_min", "pl_margin", "min", "avg", "softmin")
    
    def __init__(self, mode="pl_min", tau=1.0):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.tau = tau
    
    def __call__(self, scores, is_correct=None):
        if isinstance(scores, list):
            scores = torch.tensor(scores, dtype=torch.float32)
        
        if self.mode == "pl_min":
            if is_correct is None:
                raise ValueError("is_correct must be provided for PL aggregation")
            if is_correct:
                # For correct solutions: trust most optimistic verifier
                return scores.max().item()
            else:
                # For incorrect solutions: trust most pessimistic verifier
                return scores.min().item()
                
        elif self.mode == "pl_margin":
            if is_correct is None:
                raise ValueError("is_correct must be provided for PL aggregation")
            max_score = scores.max().item()
            min_score = scores.min().item()
            
            if is_correct:
                # For correct: reward consensus
                verifier_agreement = 1.0 - (max_score - min_score)
                return max_score * verifier_agreement
            else:
                # For incorrect: penalize high scores
                return min_score - (max_score - min_score)
                
        elif self.mode == "min":
            return scores.min().item()
        elif self.mode == "avg":
            return scores.mean().item()
        elif self.mode == "softmin":
            w = torch.exp(-scores / self.tau)
            return (scores * w).sum().item() / w.sum().item()

class LearnedAggregator(nn.Module):
    """Approximate PL-min objective.
    
    Input: tensor of shape (batch, m); one score per verifier
    Output: tensor of shape (batch,); aggregated score
    """
    
    def __init__(self, m, hidden=32):
        super().__init__()
        # Main aggregation network
        self.net = nn.Sequential(
            nn.Linear(m, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        
        # PL-inspired feature network
        self.pl_features = nn.Sequential(
            nn.Linear(m + 3, hidden),  # +3 for min, max, range
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    
    def forward(self, scores):
        # Standard aggregation
        base_out = self.net(scores)
        
        # PL-inspired features
        min_scores = scores.min(dim=1, keepdim=True)[0]
        max_scores = scores.max(dim=1, keepdim=True)[0]
        score_range = max_scores - min_scores
        
        # Combine with PL features
        pl_input = torch.cat([scores, min_scores, max_scores, score_range], dim=1)
        pl_out = self.pl_features(pl_input)
        
        # Combine both approaches
        return (base_out + pl_out).squeeze(-1) / 2

def pl_min_loss(f_out, phi_out, correctness=None):
    """PL-min loss: minimize loss over candidate labels"""
    if correctness is not None:
        # PL-style loss considering correctness
        target = phi_out.min(dim=1).values
        base_loss = F.mse_loss(f_out, target)
        
        # Additional correctness alignment
        correctness_loss = torch.mean(
            correctness * (1.0 - f_out) +  # Penalize low scores for correct
            (1.0 - correctness) * f_out      # Penalize high scores for incorrect
        )
        
        return base_loss + 0.1 * correctness_loss
    else:
        # Standard PE-min loss
        target = phi_out.min(dim=1).values
        return F.mse_loss(f_out, target)

@torch.no_grad()
def batch_prover_scores(
    problem_batch: list[str],
    prover,
    verifiers,
    device,
):
    """Run prover + verifiers on batch."""
    solution_batch = [prover(problem, role="helpful") for problem in problem_batch]
    scores = torch.stack([
        torch.tensor([v(problem, solution) for v in verifiers], dtype=torch.float32, device=device)
        for problem, solution in zip(problem_batch, solution_batch)
    ])
    return scores

def train_f(
    aggregator: LearnedAggregator,
    prover,
    verifiers,
    dataset,
    steps=100,
    batch_size=8,
    lr=1e-4,
    device="cpu",
    use_pl_loss=True,
) -> LearnedAggregator:
    """Fit f using PL-inspired training.
    
    Args:
        aggregator: LearnedAggregator
        prover: role-conditioned model
        verifiers: list/ModuleList of verifiers
        dataset: object with .sample() (yields problem string)
        steps: optimisation steps to run
        batch_size: problems per optimisation step
        lr: learning rate
        device: "cpu" or "cuda"
        use_pl_loss: whether to use PL-style loss
    """
    device = torch.device(device)
    aggregator = aggregator.to(device).train()
    opt = torch.optim.Adam(aggregator.parameters(), lr=lr)
    
    for step in range(steps):
        problem_batch = [dataset.sample() for _ in range(batch_size)]
        scores = batch_prover_scores(problem_batch, prover, verifiers, device=device)
        f_out = aggregator(scores)
        
        if use_pl_loss and hasattr(dataset, 'get_correctness'):
            # Get correctness for PL loss
            correctness = torch.tensor([
                dataset.get_correctness(problem) for problem in problem_batch
            ], dtype=torch.float32, device=device)
            loss = pl_min_loss(f_out, scores, correctness)
        else:
            # Standard PE-min loss
            loss = pl_min_loss(f_out, scores)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % 20 == 0:
            print(f"Step {step}/{steps}, Loss: {loss.item():.4f}")
    
    return aggregator

