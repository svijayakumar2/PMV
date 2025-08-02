import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class Aggregator:
    """Simple statistical aggregations of verifier scores """
    
    MODES = ("min", "avg", "softmin", "max", "median", "mode")
    
    def __init__(self, mode="min", tau=1.0):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.tau = tau
    
    def __call__(self, scores):
        if isinstance(scores, list):
            scores = torch.tensor(scores, dtype=torch.float32)
        
        if self.mode == "min":
            return scores.min().item()
        elif self.mode == "max":
            return scores.max().item()
        elif self.mode == "avg":
            return scores.mean().item()
        elif self.mode == "median":
            return scores.median().item()
        elif self.mode == "softmin":
            w = torch.exp(-scores / self.tau)
            return (scores * w).sum().item() / w.sum().item()

class LearnedAggregator(nn.Module):
    """Learned aggregation using PL/PE-min objectives.
    
    Input: tensor of shape (batch, m); one score per verifier
    Output: tensor of shape (batch,); aggregated score
    """
    
    def __init__(self, num_verifiers: int, hidden: int = 32, aggregation_type: str = "pl_min"):
        super().__init__()
        self.num_verifiers = num_verifiers
        self.aggregation_type = aggregation_type
        
        if aggregation_type == "pl_min":
            # PL-min: approximate min over candidate labels
            self.net = nn.Sequential(
                nn.Linear(num_verifiers, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid()
            )
            
            # Additional PL features: min, max, range for PL-style reasoning
            self.pl_features = nn.Sequential(
                nn.Linear(num_verifiers + 3, hidden),  # +3 for min, max, range
                nn.ReLU(), 
                nn.Linear(hidden, 1),
                nn.Sigmoid()
            )
            
        elif aggregation_type == "pe_min":
            # PE-min: approximate min objective
            self.net = nn.Sequential(
                nn.Linear(num_verifiers, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
            
        else:
            raise ValueError(f"Unknown aggregation_type: {aggregation_type}")
    
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (batch, num_verifiers) tensor of verifier scores
        Returns:
            (batch,) tensor of aggregated scores
        """
        if self.aggregation_type == "pl_min":
            # Base network output
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
            
        elif self.aggregation_type == "pe_min":
            return self.net(scores).squeeze(-1)

def pl_min_loss(f_out: torch.Tensor, phi_out: torch.Tensor, correctness: torch.Tensor = None) -> torch.Tensor:
    """PL-min loss: minimize loss over candidate labels based on correctness."""
    if correctness is not None:
        # PL-style loss: different targets based on correctness
        # For correct solutions: target = max(verifier scores)
        # For incorrect solutions: target = min(verifier scores)
        correct_mask = correctness.bool()
        target = torch.zeros_like(f_out)
        
        if correct_mask.any():
            target[correct_mask] = phi_out[correct_mask].max(dim=1)[0]
        if (~correct_mask).any():
            target[~correct_mask] = phi_out[~correct_mask].min(dim=1)[0]
        
        # MSE loss on PL targets
        base_loss = F.mse_loss(f_out, target)
        
        # Additional correctness alignment loss
        correctness_loss = torch.mean(
            correctness * (1.0 - f_out) +  # Penalize low scores for correct
            (1.0 - correctness) * f_out      # Penalize high scores for incorrect
        )
        
        return base_loss + 0.1 * correctness_loss
    else:
        # Standard PE-min loss: approximate min over all verifiers
        target = phi_out.min(dim=1)[0]
        return F.mse_loss(f_out, target)

def pl_margin_loss(f_out: torch.Tensor, phi_out: torch.Tensor, correctness: torch.Tensor) -> torch.Tensor:
    """PL-margin loss: margin-based approach with consensus."""
    max_scores = phi_out.max(dim=1)[0]
    min_scores = phi_out.min(dim=1)[0]
    
    # Compute PL-margin targets
    correct_mask = correctness.bool()
    target = torch.zeros_like(f_out)
    
    if correct_mask.any():
        # For correct: reward consensus (penalize disagreement)
        score_range = max_scores[correct_mask] - min_scores[correct_mask]
        verifier_agreement = 1.0 - score_range
        target[correct_mask] = max_scores[correct_mask] * verifier_agreement
    
    if (~correct_mask).any():
        # For incorrect: penalize high scores based on disagreement
        score_range = max_scores[~correct_mask] - min_scores[~correct_mask]
        target[~correct_mask] = min_scores[~correct_mask] - score_range
    
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

# def train_learned_aggregator(
#     aggregator: LearnedAggregator,
#     prover,
#     verifiers,
#     dataset,
#     steps: int = 100,
#     batch_size: int = 8,
#     lr: float = 1e-4,
#     device: str = "cpu",
#     use_correctness: bool = True,
# ) -> LearnedAggregator:
#     """Train the learned aggregator using PL or PE-min objectives.
    
#     Args:
#         aggregator: LearnedAggregator model
#         prover: role-conditioned model
#         verifiers: list/ModuleList of verifiers
#         dataset: object with .sample() method
#         steps: optimization steps to run
#         batch_size: problems per optimization step
#         lr: learning rate
#         device: "cpu" or "cuda"
#         use_correctness: whether to use correctness labels for PL loss
#     """
#     device = torch.device(device)
#     aggregator = aggregator.to(device).train()
#     optimizer = torch.optim.Adam(aggregator.parameters(), lr=lr)
    
#     for step in range(steps):
#         problem_batch = [dataset.sample()[0] for _ in range(batch_size)]  # Just problems
#         scores = batch_prover_scores(problem_batch, prover, verifiers, device=device)
#         f_out = aggregator(scores)
        
#         if use_correctness and hasattr(dataset, 'get_correctness'):
#             # Get correctness for PL loss
#             correctness = torch.tensor([
#                 dataset.get_correctness(problem) for problem in problem_batch
#             ], dtype=torch.float32, device=device)
            
#             if aggregator.aggregation_type == "pl_min":
#                 loss = pl_min_loss(f_out, scores, correctness)
#             elif aggregator.aggregation_type == "pl_margin":
#                 loss = pl_margin_loss(f_out, scores, correctness)
#             else:
#                 loss = pl_min_loss(f_out, scores)  # Fallback to PE-min
#         else:
#             # Standard PE-min loss
#             loss = pl_min_loss(f_out, scores)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if step % 20 == 0:
#             print(f"Step {step}/{steps}, Loss: {loss.item():.4f}")
    
#     return aggregator


