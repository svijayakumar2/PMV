import torch
import numpy as np
from typing import List, Union

def signed_relative_convincingness(f_scores: List[float], 
                                   is_correct: List[bool],
                                   roles: List[str]) -> List[float]:
    """
    Computes the Signed Relative Convincingness (SRC) reward from:
    Kirchner, J.H., Chen, Y., Edwards, H., Leike, J., McAleese, N. and Burda, Y., 2024.
    Prover-verifier games improve legibility of llm outputs. arXiv preprint arXiv:2407.13692.
    
    Implementation based on compute_src_rewards from the original code.
    """
    # Batch normalize scores to mean 0
    mean_score = np.mean(f_scores)
    normalized_scores = [score - mean_score for score in f_scores]
    
    rewards = []
    for i, (norm_score, correct, role) in enumerate(zip(normalized_scores, is_correct, roles)):
        # Role alignment
        if role.lower() == "helpful":
            role_aligned = correct
        else:  # sneaky
            role_aligned = not correct
        
        if norm_score > 0:
            if role_aligned:
                reward = norm_score
            else:
                reward = -norm_score
        else:
            reward = -1.0
        
        rewards.append(reward)
    
    return rewards

def correctness_gated_convincingness(f_scores: List[float],
                                   is_correct: List[bool], 
                                   roles: List[str],
                                   penalty_value: float = -2.0) -> List[float]:
    """
    Computes the Correctness-Gated Convincingness (CGC) reward from the Kirchner paper.
    
    Prioritizes role-alignment over convincingness:
    - If role-aligned: reward = verifier score
    - If not role-aligned: reward = constant penalty
    
    Args:
        f_scores: Verifier convincingness scores
        is_correct: Whether each solution is correct
        roles: Role for each solution ("helpful" or "sneaky")
        penalty_value: Constant penalty for role-misaligned solutions
    """
    rewards = []
    
    for score, correct, role in zip(f_scores, is_correct, roles):
        # Role alignment
        if role.lower() == "helpful":
            role_aligned = correct
        else:  # sneaky
            role_aligned = not correct
        
        if role_aligned:
            reward = score
        else:
            reward = penalty_value
        
        rewards.append(reward)
    
    return rewards

def iterative_goodharting(f_scores: List[float],
                         is_correct: List[bool] = None,
                         roles: List[str] = None) -> List[float]:
    """
    Computes the Iterative Goodharting reward from the Kirchner paper.
    
    Simple reward that only encourages convincingness, ignoring correctness:
    R(z|x) = V(x,z)
    
    This can lead to reward hacking where the model optimizes for convincingness
    without regard for correctness.
    
    Args:
        f_scores: Verifier convincingness scores  
        is_correct: Ignored (kept for interface compatibility)
        roles: Ignored (kept for interface compatibility)
    """
    # return the verifier scores as rewards
    return list(f_scores)
