"""
Verifier implementation for PMV Stackelberg-Nash Game.

Implements the evaluation function φ_j: X × Y → Z := [0, 1] from Section 3.

Key design decisions:
1. Always return probabilities in [0, 1] for consistency
2. Separate methods for differentiable (head) vs generation-based scoring
3. Explicit control over which scoring method to use
"""

import re
import torch
import torch.nn as nn
from pmv.models.base import Model
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Union


class Verifier(Model):
    """
    Verifier model implementing φ_j: X × Y → [0, 1].
    
    Each verifier assigns a numerical evaluation score (the "convincingness"
    of the solution for the given problem) per Section 3.
    
    Two scoring modes:
    1. Head-based (differentiable): Uses learned score_head on hidden states
    2. Generation-based: Prompts the model to output a score
    
    For training (Phase 1), use head-based scoring to enable gradient flow.
    For evaluation/data collection, either mode can be used.
    """
    
    def __init__(
        self, 
        model_name: str, 
        verifier_type: str = "general", 
        use_quantization: bool = True, 
        quantization_config=None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1
    ):
        super().__init__(
            model_name, 
            use_quantization=use_quantization, 
            quantization_config=quantization_config
        )
        self.verifier_type = verifier_type
        self.role = None  # Set externally for role-specific prompting
        
        # Apply LoRA for parameter-efficient fine-tuning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            inference_mode=False
        )
        self.model = get_peft_model(self.model, lora_config)
        
        print(f"[{self.verifier_type}] Trainable parameters:")
        self.model.print_trainable_parameters()
        
        # Score head: maps hidden states to scalar score
        hidden_size = self.model.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(self.device)
        
        # Initialize score head with small weights for stable training
        with torch.no_grad():
            for module in self.score_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def __call__(self, problem: str, solution: str) -> torch.Tensor:
        """
        Default call uses head-based scoring in training, generation in eval.
        
        Returns:
            Tensor with probability in [0, 1]. Has gradients if in training mode.
        """
        if self.training:
            return self.score_with_head(problem, solution, return_prob=True)
        else:
            return self.score_with_generation(problem, solution)
    
    def forward(self, problem: str, solution: str) -> torch.Tensor:
        """Alias for __call__."""
        return self(problem, solution)
    
    def score_with_head(
        self, 
        problem: str, 
        solution: str, 
        return_prob: bool = True
    ) -> torch.Tensor:
        """
        Compute score using the differentiable score head.
        
        This method enables gradient flow through the verifier, required
        for Phase 1 joint training.
        
        Args:
            problem: Problem text
            solution: Solution text  
            return_prob: If True, return sigmoid(logit) in [0,1].
                        If False, return raw logit.
        
        Returns:
            Scalar tensor. Has gradients if model is in training mode.
        """
        prompt = self._create_verification_prompt(problem, solution)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512, 
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass with hidden states
        outputs = self.model(
            **inputs, 
            output_hidden_states=True, 
            use_cache=False
        )
        
        if not (hasattr(outputs, 'hidden_states') and 
                outputs.hidden_states is not None and 
                len(outputs.hidden_states) > 0):
            raise RuntimeError(
                f"Model did not return hidden_states for {self.verifier_type}. "
                f"Cannot use differentiable scoring without hidden states."
            )
        
        # Use last token's hidden state from final layer
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        # Ensure float32 for stability
        if last_hidden.dtype != torch.float32:
            last_hidden = last_hidden.float()
        
        # Check for numerical issues
        if torch.isnan(last_hidden).any() or torch.isinf(last_hidden).any():
            raise ValueError(f"NaN/Inf in hidden states for {self.verifier_type}")
        
        # Clamp for stability
        last_hidden = torch.clamp(last_hidden, -10.0, 10.0)
        
        # Compute logit through score head
        logit = self.score_head(last_hidden).squeeze()
        
        if torch.isnan(logit).any() or torch.isinf(logit).any():
            raise ValueError(f"NaN/Inf in logit for {self.verifier_type}")
        
        logit = torch.clamp(logit, -10.0, 10.0)
        
        # Verify gradient flow in training mode
        if self.training and not logit.requires_grad:
            raise RuntimeError(
                f"Logit does not require grad for {self.verifier_type}. "
                f"Check that LoRA parameters are unfrozen."
            )
        
        if return_prob:
            return torch.sigmoid(logit)
        return logit
    
    def score_with_generation(
        self, 
        problem: str, 
        solution: str
    ) -> torch.Tensor:
        """
        Compute score by prompting the model to generate a score.
        
        This method does not support gradients but may produce higher
        quality scores for evaluation.
        
        Args:
            problem: Problem text
            solution: Solution text
        
        Returns:
            Detached tensor with probability in [0, 1].
        """
        prompt = self._create_verification_prompt(problem, solution)
        response = self.generate(prompt, max_new_tokens=512)
        
        try:
            score = self._parse_score(response)
        except Exception as e:
            print(f"Error parsing score from {self.verifier_type}: {e}")
            score = 0.5
        
        return torch.tensor(score, device=self.device, dtype=torch.float32)
    
    def train(self, mode: bool = True):
        """Set training mode for all components."""
        super().train(mode)
        self.model.train(mode)
        self.score_head.train(mode)
        return self

    def eval(self):
        """Set evaluation mode for all components."""
        super().eval()
        self.model.eval()
        self.score_head.eval()
        return self

    def freeze_all(self):
        """Freeze all parameters (for Phase 2 when leaders are committed)."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.score_head.parameters():
            param.requires_grad = False

    def unfreeze_lora(self) -> int:
        """
        Unfreeze LoRA parameters and score head for training.
        
        Returns:
            Number of trainable parameters.
        """
        lora_params_found = False
        
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
                lora_params_found = True
            else:
                param.requires_grad = False
        
        for param in self.score_head.parameters():
            param.requires_grad = True
        
        trainable = [p for p in self.parameters() if p.requires_grad]
        
        if not trainable:
            raise RuntimeError(
                f"No trainable parameters in {self.verifier_type} after unfreezing."
            )
        
        if not lora_params_found:
            import warnings
            warnings.warn(
                f"No LoRA parameters found in {self.verifier_type}. "
                f"Only score_head is trainable. Model may not be properly configured."
            )
        
        return len(trainable)

    def get_trainable_params(self):
        """Get list of trainable parameters for optimizer."""
        return [p for p in self.model.parameters() if p.requires_grad] + \
               [p for p in self.score_head.parameters() if p.requires_grad]

    def state_dict(self, *args, **kwargs):
        """Save state including LoRA weights and score head."""
        return {
            'lora_state': self.model.state_dict(*args, **kwargs),
            'score_head_state': self.score_head.state_dict(*args, **kwargs),
            'verifier_type': self.verifier_type
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state from checkpoint."""
        if isinstance(state_dict, dict) and 'lora_state' in state_dict:
            self.model.load_state_dict(state_dict['lora_state'], strict=strict)
            self.score_head.load_state_dict(state_dict['score_head_state'], strict=strict)
            if 'verifier_type' in state_dict:
                self.verifier_type = state_dict['verifier_type']
        else:
            # Legacy format
            self.model.load_state_dict(state_dict, strict=strict)

    def _create_verification_prompt(self, problem: str, solution: str) -> str:
        """
        Create verification prompt based on verifier role.
        
        Different verifiers focus on different aspects (Remark 3.1):
        - Reasoning structure and logical flow
        - Computational accuracy
        - Problem alignment and completeness
        - etc.
        """
        # Get role-specific focus
        if self.role and 'focus' in self.role:
            focus = self.role['focus']
            role_name = self.role.get('name', 'general')
        else:
            focus = "Evaluate the solution's correctness and quality."
            role_name = "general"
        
        system = f"You are a {role_name} verifier. {focus}"
        
        user_message = f"""Problem: {problem}

Solution: {solution}

Evaluate this solution. Consider whether it is correct, complete, and well-reasoned.
Provide a score from 0.0 to 1.0 where:
- 1.0 = Completely correct and well-explained
- 0.0 = Completely incorrect or nonsensical

End your response with: SCORE: X.X"""
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message}
        ]
        
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            return f"{system}\n\n{user_message}\n\nAssistant:"
    
    def _parse_score(self, response: str) -> float:
        """
        Parse score from generated response.
        
        Tries multiple patterns to extract a score in [0, 1].
        """
        # Try explicit SCORE: pattern first
        score_match = re.search(r'SCORE:\s*(\d*\.?\d+)', response.strip(), re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))
        
        # Try to find any decimal in [0, 1]
        numbers = re.findall(r'(\d*\.?\d+)', response)
        for num_str in reversed(numbers):
            try:
                num = float(num_str)
                if 0.0 <= num <= 1.0:
                    return num
            except ValueError:
                continue
        
        # Try to normalize larger numbers
        if numbers:
            try:
                score = float(numbers[-1])
                if score <= 10.0:
                    return max(0.0, min(1.0, score / 10.0))
                elif score <= 100.0:
                    return max(0.0, min(1.0, score / 100.0))
            except ValueError:
                pass
        
        raise ValueError(f"Could not parse score from: {response[:200]}")

