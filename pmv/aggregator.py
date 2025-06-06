import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregator:
    """Simple aggregations of verifier scores."""

    MODES = ("min", "avg", "softmin")

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
        if self.mode == "avg":
            return scores.mean().item()
        if self.mode == "softmin":
            w = torch.exp(-scores / self.tau)
            return (scores * w).sum().item() / w.sum().item()
        # raise RuntimeError("unreachable")


class LearnedAggregator(nn.Module):
    """Approximate PE‑min objective.

    Input: tensor of shape (batch, m); one score per verifier
    Output: tensor of shape (batch,); aggregated score
    """

    def __init__(self, m, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, scores):
        return self.net(scores).squeeze(-1)


def pe_min_loss(f_out, phi_out):
    target = phi_out.min(dim=1).values
    # todo: code option to use different loss functions, e.g., log loss
    return F.mse_loss(f_out, target)  # l2 loss


@torch.no_grad()
def _batch_prover_scores(
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
) -> LearnedAggregator:
    """Fit f.

    Args:
        aggregator: LearnedAggregator
        prover: role‑conditioned model
        verifiers: list/ModuleList of verifiers
        dataset: object with .sample() (yields problem string)
        steps: optimisation steps to run
        batch_size: problems per optimisation step
        lr: learning rate
        device: "cpu" or "cuda"
    """
    device = torch.device(device)
    aggregator = aggregator.to(device).train()
    opt = torch.optim.Adam(aggregator.parameters(), lr=lr)

    for _ in range(steps):

        problem_batch = [dataset.sample() for _ in range(batch_size)]

        scores = _batch_prover_scores(problem_batch, prover, verifiers, device=device)

        f_out = aggregator(scores)
        loss = pe_min_loss(f_out, scores)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return aggregator
