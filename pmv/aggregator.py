import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, NamedTuple


class Aggregator:
    """Simple statistical aggregations of verifier scores """
    
    MODES = ("min", "avg", "softmin", "max", "median", "pe_min")
    
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
        elif self.mode == "pe_min":
            # True PE-min: return the minimum score across verifiers
            return scores.min().item()


class LearnedAggregator(nn.Module):
    """Learned aggregation using PE-min objective.
    
    This learns to approximate f^*_{φ,π} from Equation (1) in your formulation:
    L(f; φ, π) = (1/n) Σ min_j λ(f(x_i, y_i), φ_j(x_i, y_i))
    
    Input: tensor of shape (batch, m); one score per verifier
    Output: tensor of shape (batch,); aggregated score
    """
    
    def __init__(self, num_verifiers: int, hidden: int = 64, aggregation_type: str = "pe_min"):
        super().__init__()
        self.num_verifiers = num_verifiers
        self.aggregation_type = aggregation_type
        
        if aggregation_type == "pe_min":
            # PE-min: learn to approximate the min over verifiers
            # Network should learn f^*_{φ,π} that minimizes the PE-min loss
            self.net = nn.Sequential(
                nn.Linear(num_verifiers, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 1),
                nn.Sigmoid()  # Output should be in [0,1] like verifier scores
            )
            
            # Additional features that help with PE-min learning
            self.pe_features = nn.Sequential(
                nn.Linear(num_verifiers + 4, hidden),  # +4 for min, max, std, mean
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(), 
                nn.Linear(hidden, 1),
                nn.Sigmoid()
            )
            
            # Learnable combination weights
            self.combination_weight = nn.Parameter(torch.tensor(0.7))
            
        elif aggregation_type == "pl_min":
            # PL-min: learn different aggregation based on correctness patterns
            self.net = nn.Sequential(
                nn.Linear(num_verifiers, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden), 
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid()
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
        if self.aggregation_type == "pe_min":
            # Base network output
            base_out = self.net(scores)
            
            # Statistical features that help approximate PE-min
            min_scores = scores.min(dim=1, keepdim=True)[0]
            max_scores = scores.max(dim=1, keepdim=True)[0]
            mean_scores = scores.mean(dim=1, keepdim=True)
            std_scores = scores.std(dim=1, keepdim=True)
            
            # Combine with PE features
            pe_input = torch.cat([scores, min_scores, max_scores, mean_scores, std_scores], dim=1)
            pe_out = self.pe_features(pe_input)
            
            # Learnable weighted combination
            w = torch.sigmoid(self.combination_weight)
            return w * base_out.squeeze(-1) + (1 - w) * pe_out.squeeze(-1)
            
        elif self.aggregation_type == "pl_min":
            return self.net(scores).squeeze(-1)


def pe_min_loss(f_out: torch.Tensor, verifier_scores: torch.Tensor, 
                lambda_fn=None, reduction='mean') -> torch.Tensor:
    """
    Implements the PE-min loss from Equation (1):
    L(f; φ, π) = (1/n) Σ min_j λ(f(x_i, y_i), φ_j(x_i, y_i))
    
    Args:
        f_out: (batch,) aggregated scores from learned aggregator
        verifier_scores: (batch, num_verifiers) individual verifier scores
        lambda_fn: loss function λ. If None, defaults to MSE
        reduction: 'mean' or 'sum'
    
    Returns:
        Scalar loss value
    """
    batch_size, num_verifiers = verifier_scores.shape
    
    # Default to MSE if no lambda function provided
    if lambda_fn is None:
        lambda_fn = F.mse_loss
    
    # Expand f_out to compare with each verifier
    f_expanded = f_out.unsqueeze(1).expand(-1, num_verifiers)  # (batch, num_verifiers)
    
    # Compute λ(f(x_i, y_i), φ_j(x_i, y_i)) for all j
    # Handle different loss function types properly
    if lambda_fn == F.mse_loss or lambda_fn.__name__ == 'mse_loss':
        losses = (f_expanded - verifier_scores) ** 2
    elif lambda_fn == F.l1_loss or lambda_fn.__name__ == 'l1_loss':
        losses = torch.abs(f_expanded - verifier_scores)
    elif lambda_fn == F.smooth_l1_loss or lambda_fn.__name__ == 'smooth_l1_loss':
        losses = F.smooth_l1_loss(f_expanded, verifier_scores, reduction='none')
    elif lambda_fn == F.binary_cross_entropy or lambda_fn.__name__ == 'binary_cross_entropy':
        # Clamp to avoid log(0)
        f_clamped = torch.clamp(f_expanded, 1e-7, 1-1e-7)
        losses = -(verifier_scores * torch.log(f_clamped) + 
                  (1 - verifier_scores) * torch.log(1 - f_clamped))
    else:
        # General case - try to apply lambda_fn with proper error handling
        try:
            # First try with reduction='none' if supported
            losses = lambda_fn(f_expanded, verifier_scores, reduction='none')
        except (TypeError, ValueError):
            # If that fails, compute element-wise manually
            losses = torch.zeros_like(f_expanded)
            for i in range(num_verifiers):
                try:
                    loss_val = lambda_fn(f_expanded[:, i], verifier_scores[:, i])
                    if loss_val.dim() == 0:  # scalar loss, expand to match batch
                        losses[:, i] = loss_val.expand(batch_size)
                    else:
                        losses[:, i] = loss_val
                except Exception as e:
                    print(f"Warning: lambda_fn failed for verifier {i}, using MSE fallback. Error: {e}")
                    losses[:, i] = (f_expanded[:, i] - verifier_scores[:, i]) ** 2
    
    # Take minimum over verifiers for each example: min_j λ(...)
    # Use softmin for gradient stability if needed
    tau = 0.1  # temperature for softmin
    weights = F.softmax(-losses / tau, dim=1)
    soft_min_losses = (weights * losses).sum(dim=1)
    
    # Also compute hard min for comparison
    hard_min_losses = losses.min(dim=1)[0]
    
    # Use a weighted combination for training stability
    alpha = 0.8  # weight for soft min
    final_losses = alpha * soft_min_losses + (1 - alpha) * hard_min_losses
    
    # Average over batch: (1/n) Σ
    if reduction == 'mean':
        return final_losses.mean()
    elif reduction == 'sum':
        return final_losses.sum()
    else:
        return final_losses


def pl_min_loss(f_out: torch.Tensor, verifier_scores: torch.Tensor, 
                correctness: torch.Tensor, lambda_fn=F.mse_loss) -> torch.Tensor:
    """PL-min loss: minimize loss over candidate labels based on correctness."""
    # For PL-min, we want different behavior based on correctness
    correct_mask = correctness.bool()
    
    if correct_mask.any():
        # For correct examples, target should be close to max verifier score
        correct_target = verifier_scores[correct_mask].max(dim=1)[0]
        correct_loss = lambda_fn(f_out[correct_mask], correct_target, reduction='none')
    else:
        correct_loss = torch.tensor(0.0, device=f_out.device)
    
    if (~correct_mask).any():
        # For incorrect examples, use PE-min style loss
        incorrect_f = f_out[~correct_mask]
        incorrect_scores = verifier_scores[~correct_mask]
        incorrect_loss = pe_min_loss(incorrect_f, incorrect_scores, lambda_fn, reduction='none')
    else:
        incorrect_loss = torch.tensor(0.0, device=f_out.device)
    
    # Combine losses
    total_loss = 0.0
    if correct_mask.any():
        total_loss += correct_loss.mean()
    if (~correct_mask).any():
        total_loss += incorrect_loss.mean() if incorrect_loss.numel() > 0 else 0.0
    
    return total_loss


def train_learned_aggregator_pe_min(
    aggregator: LearnedAggregator,
    verifier_score_batches: List[torch.Tensor],
    correctness_batches: Optional[List[torch.Tensor]] = None,
    steps: int = 100,
    lr: float = 1e-4,
    device: str = "cpu",
    lambda_fn = None,
    train_test_split: float = 0.8,
) -> LearnedAggregator:
    """
    Train the learned aggregator using proper PE-min objective.
    
    Args:
        aggregator: LearnedAggregator model
        verifier_score_batches: List of (batch, num_verifiers) tensors
        correctness_batches: Optional list of (batch,) correctness tensors for PL-min
        steps: optimization steps to run
        lr: learning rate
        device: "cpu" or "cuda"
        lambda_fn: loss function λ for PE-min objective
        train_test_split: fraction of data to use for training vs validation
    """
    device = torch.device(device)
    aggregator = aggregator.to(device).train()
    optimizer = torch.optim.Adam(aggregator.parameters(), lr=lr)
    
    if not verifier_score_batches:
        print("No verifier score batches provided for training")
        return aggregator
    
    # Create proper train/test split
    n_batches = len(verifier_score_batches)
    train_size = int(n_batches * train_test_split)
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, n_batches))
    
    if not train_indices:
        print("No training data available")
        return aggregator
    
    print(f"Training on {len(train_indices)} batches, testing on {len(test_indices)} batches")
    
    best_test_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for step in range(steps):
        # Sample a training batch
        batch_idx = train_indices[step % len(train_indices)]
        scores = verifier_score_batches[batch_idx].to(device)
        
        # Skip batches that are too small or have invalid data
        if scores.shape[0] < 2 or scores.shape[1] < 1:
            continue
            
        # Forward pass through aggregator
        try:
            f_out = aggregator(scores)
        except Exception as e:
            print(f"Forward pass failed at step {step}: {e}")
            continue
        
        # Compute loss based on aggregation type
        try:
            if aggregator.aggregation_type == "pe_min":
                loss = pe_min_loss(f_out, scores, lambda_fn)
            elif aggregator.aggregation_type == "pl_min" and correctness_batches is not None:
                correctness = correctness_batches[batch_idx].to(device)
                loss = pl_min_loss(f_out, scores, correctness, lambda_fn)
            else:
                # Fallback to PE-min
                loss = pe_min_loss(f_out, scores, lambda_fn)
        except Exception as e:
            print(f"Loss computation failed at step {step}: {e}")
            continue
        
        # Check for valid loss
        if not torch.isfinite(loss):
            print(f"Non-finite loss at step {step}: {loss}")
            continue
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(aggregator.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation and logging
        if step % 20 == 0:
            print(f"Step {step}/{steps}, Train Loss: {loss.item():.4f}")
            
            # Validation on test set
            if test_indices:
                aggregator.eval()
                test_losses = []
                
                with torch.no_grad():
                    for test_idx in test_indices[:5]:  # Use first 5 test batches
                        try:
                            test_scores = verifier_score_batches[test_idx].to(device)
                            if test_scores.shape[0] < 2:
                                continue
                                
                            test_f_out = aggregator(test_scores)
                            
                            if aggregator.aggregation_type == "pe_min":
                                test_loss = pe_min_loss(test_f_out, test_scores, lambda_fn)
                            elif aggregator.aggregation_type == "pl_min" and correctness_batches is not None:
                                test_correctness = correctness_batches[test_idx].to(device)
                                test_loss = pl_min_loss(test_f_out, test_scores, test_correctness, lambda_fn)
                            else:
                                test_loss = pe_min_loss(test_f_out, test_scores, lambda_fn)
                            
                            if torch.isfinite(test_loss):
                                test_losses.append(test_loss.item())
                        except Exception:
                            continue
                
                if test_losses:
                    avg_test_loss = sum(test_losses) / len(test_losses)
                    print(f"  Validation Loss: {avg_test_loss:.4f}")
                    
                    # Early stopping
                    if avg_test_loss < best_test_loss:
                        best_test_loss = avg_test_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print(f"Early stopping at step {step}")
                        break
                
                aggregator.train()
            
            # Log diagnostics on training data
            with torch.no_grad():
                true_min = scores.min(dim=1)[0]
                pred_scores = f_out
                min_mse = F.mse_loss(pred_scores, true_min)
                correlation = torch.corrcoef(torch.stack([pred_scores, true_min]))[0, 1]
                print(f"  True min vs pred MSE: {min_mse.item():.4f}, Correlation: {correlation.item():.3f}")
    
    return aggregator


@torch.no_grad()
def evaluate_pe_min_approximation(aggregator: LearnedAggregator, 
                                  test_scores: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate how well the learned aggregator approximates PE-min.
    
    Args:
        aggregator: trained aggregator
        test_scores: (batch, num_verifiers) test data
        
    Returns:
        Dictionary with evaluation metrics
    """
    aggregator.eval()
    
    # Get predictions
    f_out = aggregator(test_scores)
    
    # True PE-min would be the minimum across verifiers
    true_min = test_scores.min(dim=1)[0]
    true_max = test_scores.max(dim=1)[0] 
    true_mean = test_scores.mean(dim=1)
    
    # Compute metrics
    min_mse = F.mse_loss(f_out, true_min).item()
    min_mae = F.l1_loss(f_out, true_min).item()
    
    max_mse = F.mse_loss(f_out, true_max).item()
    mean_mse = F.mse_loss(f_out, true_mean).item()
    
    # Correlation with true minimum
    correlation = torch.corrcoef(torch.stack([f_out, true_min]))[0, 1].item()
    
    return {
        "min_mse": min_mse,
        "min_mae": min_mae,
        "max_mse": max_mse,
        "mean_mse": mean_mse,
        "min_correlation": correlation,
        "mean_pred": f_out.mean().item(),
        "mean_true_min": true_min.mean().item()
    }


def train_aggregator_with_replay_buffer(
    aggregator: LearnedAggregator,
    replay_buffer: List[Tuple],
    verifiers: List,
    dataset,  # MathDataset instance
    config: dict,
    correctness: torch.Tensor = None, # TODO - tensor or boolean? 
    device: str = "cuda"
) -> LearnedAggregator:
    """
    Complete integration function for training aggregator with your replay buffer.
    This replaces your existing train_learned_aggregator_stackelberg function.
    
    Args:
        aggregator: LearnedAggregator instance
        replay_buffer: List of (problem, true_solution, response, reward, _, role) tuples
        verifiers: List of verifier models
        dataset: MathDataset instance for ground truth checking
        config: Training configuration dictionary
        device: Device to use for training
    
    Returns:
        Trained aggregator
    """
    if len(replay_buffer) < 20:
        print(f"Insufficient replay buffer data: {len(replay_buffer)} < 20")
        return aggregator
    
    print(f"Training aggregator with PE-min objective on {len(replay_buffer)} experiences")
    
    # Collect verifier score batches from replay buffer
    verifier_score_batches = []
    correctness_batches = []
    
    batch_size = min(32, len(replay_buffer) // 4)  # Adaptive batch size
    
    for i in range(0, len(replay_buffer), batch_size):
        batch_data = replay_buffer[i:i+batch_size]
        
        batch_scores = []
        batch_correctness = []
        
        for problem, true_solution, response, reward, _, role in batch_data:
            # Get robust verifier scores for this response
            scores = robust_verifier_scoring(
                verifiers=verifiers,
                problem=problem,
                response=response,
                fallback_strategy="median"
            )
            
            # All verifiers should have returned scores
            if len(scores) == len(verifiers) and all(s is not None for s in scores):
                batch_scores.append(scores)
                
                # Get ground truth correctness using the dataset
                correctness_prob = infer_correctness_from_context(
                    problem=problem,
                    response=response,
                    true_solution=true_solution,
                    role=role,
                    verifier_scores=scores,
                    dataset=dataset
                )
                batch_correctness.append(correctness_prob)
        
        if batch_scores:
            verifier_score_batches.append(torch.tensor(batch_scores, dtype=torch.float32))
            correctness_batches.append(torch.tensor(batch_correctness, dtype=torch.float32))
    
    if not verifier_score_batches:
        print("No valid score batches generated from replay buffer")
        return aggregator
    
    print(f"Generated {len(verifier_score_batches)} batches for training")
    
    # Determine lambda function from config
    lambda_fn_name = config["training"].get("pe_min_lambda", "mse")
    if lambda_fn_name == "mse":
        lambda_fn = F.mse_loss
    elif lambda_fn_name == "l1":
        lambda_fn = F.l1_loss
    elif lambda_fn_name == "smooth_l1":
        lambda_fn = F.smooth_l1_loss
    else:
        lambda_fn = F.mse_loss  # Default fallback
    
    # Train using proper PE-min objective
    trained_aggregator = train_learned_aggregator_pe_min(
        aggregator=aggregator,
        verifier_score_batches=verifier_score_batches,
        correctness_batches=correctness_batches if aggregator.aggregation_type == "pl_min" else None,
        steps=int(config["training"].get("aggregator_steps", 100)),
        lr=float(config["training"].get("aggregator_lr", 1e-4)),
        device=device,
        lambda_fn=lambda_fn,
        train_test_split=0.8
    )
    
    # Evaluate the final approximation quality
    if verifier_score_batches:
        # Use a held-out batch for evaluation
        eval_batch_idx = len(verifier_score_batches) // 2
        eval_scores = verifier_score_batches[eval_batch_idx][:16].to(device)
        
        metrics = evaluate_pe_min_approximation(trained_aggregator, eval_scores)
        print(f"Final PE-min approximation quality: {metrics}")
        
        # Log key metrics
        print(f"  Minimum MSE: {metrics['min_mse']:.4f}")
        print(f"  Minimum correlation: {metrics['min_correlation']:.3f}")
        print(f"  Mean prediction: {metrics['mean_pred']:.3f}")
        print(f"  Mean true minimum: {metrics['mean_true_min']:.3f}")
    
    return trained_aggregator