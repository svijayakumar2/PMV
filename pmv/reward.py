import torch


def signed_relative_convincingness(f_score: float | torch.Tensor,
                                   is_correct: bool,
                                   role: str) -> float | torch.Tensor:
    """
    Computes the Signed Relative Convincingness (SRC) reward from:

    Kirchner, J.H., Chen, Y., Edwards, H., Leike, J., McAleese, N. and Burda, Y., 2024.
    Prover-verifier games improve legibility of llm outputs. arXiv preprint arXiv:2407.13692.
    """
    is_helpful = role == "helpful"
    is_wrong = not is_correct

    # reward is positive if helpful XOR wrong
    # else negative
    positive_reward = is_helpful ^ is_wrong
    sign = 1 if positive_reward else -1

    return sign * f_score
