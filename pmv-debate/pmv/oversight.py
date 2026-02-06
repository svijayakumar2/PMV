"""
Oversight rules A_θ per Section 4.1 of the paper.

Fixed rules:
  - Averaging:  A_avg(ϕ)(x,y) = (1/m) Σ ϕ_j(x,y)
  - Median:     A_med(ϕ)(x,y) = median(ϕ_1(x,y), ..., ϕ_m(x,y))

Optimized rules (learned aggregator with different H losses):
  - PE-Min:     H(f; x,y,z) = min_j l(f(x,y), z_j)
  - PE-Margin:  H(f; x,y,z) = [min_{j∈J+} l(f,z_j) - min_{j∈J-} l(f,z_j) + γ]+
  - MoE:        Mixture-of-experts gating over verifier scores

The aggregator MLP is shared across all optimized rules; the H value
determines the training loss, not the architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# Fixed oversight rules (no learnable parameters)
# ---------------------------------------------------------------------------

def oversight_average(verifier_scores: torch.Tensor) -> torch.Tensor:
    """A_avg: simple mean. Scores shape [batch, m] or [m]."""
    return verifier_scores.mean(dim=-1)


def oversight_median(verifier_scores: torch.Tensor) -> torch.Tensor:
    """A_med: median (majority-vote analog in continuous space)."""
    return verifier_scores.median(dim=-1).values


FIXED_RULES = {
    "average": oversight_average,
    "median": oversight_median,
}


# ---------------------------------------------------------------------------
# Learned aggregator (shared architecture for all optimized rules)
# ---------------------------------------------------------------------------

class OversightAggregator(nn.Module):
    """
    MLP aggregator: (ϕ_1, ..., ϕ_m) → f_θ,ϕ(x,y) ∈ [0,1].

    Persists across rounds. Trained with different H losses depending
    on the chosen optimized oversight rule.
    """

    def __init__(self, num_verifiers: int, hidden_dim: int = 64, device: str = "cuda:0"):
        super().__init__()
        self.num_verifiers = num_verifiers
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(num_verifiers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, verifier_scores: torch.Tensor) -> torch.Tensor:
        squeeze = verifier_scores.dim() == 1
        if squeeze:
            verifier_scores = verifier_scores.unsqueeze(0)
        out = self.network(verifier_scores).squeeze(-1)
        if squeeze:
            out = out.squeeze(0)
        return out

    def aggregate_inference(self, verifier_scores: List[float]) -> float:
        t = torch.tensor(verifier_scores, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(self.forward(t).item())

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.train()


# ---------------------------------------------------------------------------
# Partial-evaluation loss functions H (Section 4.1)
# ---------------------------------------------------------------------------

def h_pe_min(
    f_score: torch.Tensor,
    verifier_scores: torch.Tensor,
    target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PE-Min loss: H(f; x,y,z_{1:m}) = min_j l(f(x,y), z_j).

    f_score: [batch] aggregator output
    verifier_scores: [batch, m] individual verifier scores
    Uses BCE as the base loss l.
    """
    f_exp = f_score.unsqueeze(-1).expand_as(verifier_scores)  # [batch, m]
    f_clamped = f_exp.clamp(1e-6, 1 - 1e-6)
    # l(f, z_j) for each verifier
    per_verifier_loss = F.binary_cross_entropy(
        f_clamped, verifier_scores, reduction='none'
    )  # [batch, m]
    return per_verifier_loss.min(dim=-1).values.mean()


def h_pe_margin(
    f_score: torch.Tensor,
    verifier_scores: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    tau_plus: float = 0.6,
    tau_minus: float = 0.4,
    gamma: float = 0.1,
) -> torch.Tensor:
    """
    PE-Margin loss: hinge on supported vs unsupported verifier subsets.

    J+ = {j : z_j >= τ+}, J- = {j : z_j <= τ-}
    H = [min_{j∈J+} l(f, z_j) - min_{j∈J-} l(f, z_j) + γ]+
    """
    f_exp = f_score.unsqueeze(-1).expand_as(verifier_scores)
    f_clamped = f_exp.clamp(1e-6, 1 - 1e-6)
    per_verifier_loss = F.binary_cross_entropy(
        f_clamped, verifier_scores, reduction='none'
    )

    batch_size = verifier_scores.shape[0]
    losses = []

    for i in range(batch_size):
        z = verifier_scores[i]
        l = per_verifier_loss[i]
        j_plus = (z >= tau_plus).nonzero(as_tuple=True)[0]
        j_minus = (z <= tau_minus).nonzero(as_tuple=True)[0]

        if len(j_plus) == 0 or len(j_minus) == 0:
            # Fall back to PE-min when subsets are empty
            losses.append(l.min())
        else:
            min_plus = l[j_plus].min()
            min_minus = l[j_minus].min()
            losses.append(F.relu(min_plus - min_minus + gamma))

    return torch.stack(losses).mean()


def h_supervised(
    f_score: torch.Tensor,
    verifier_scores: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Direct supervised loss: l(f(x,y), c(x,y)).

    This is the standard Phase 1 objective from Equation (3).
    Uses BCE.
    """
    return F.binary_cross_entropy(
        f_score.clamp(1e-6, 1 - 1e-6), target
    )


OPTIMIZED_RULES = {
    "pe_min": h_pe_min,
    "pe_margin": h_pe_margin,
    "supervised": h_supervised,
}


# ---------------------------------------------------------------------------
# Verifier Nash update loss with SoftMin (Appendix D, Eq. end)
# ---------------------------------------------------------------------------

def verifier_softmin_loss(
    verifier_losses: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 5.0,
) -> torch.Tensor:
    """
    SoftMin utility for verifier j under PE-Min rule (Appendix D):

    U_j(l, M) = -(1/β) log Σ_k M_k exp(-β · l_k)

    verifier_losses: [batch, m] per-verifier BCE losses
    mask: [batch, m] binary mask (always includes j, others via dropout)
    beta: temperature
    """
    masked = verifier_losses * mask + (1 - mask) * 1e6  # large loss for masked-out
    softmin = -(1.0 / beta) * torch.logsumexp(-beta * masked, dim=-1)
    return softmin.mean()
