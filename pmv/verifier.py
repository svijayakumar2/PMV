"""
Verifier built on the Model base class.

Implements ϕ_j: X × Y → [0,1] (Section 3) with:
  - LoRA adapters for parameter-efficient multi-verifier training
  - Score head for differentiable oversight (Phase 1 gradient flow)
  - Generation-based scoring for evaluation

Verifiers PERSIST across rounds (Section 6.4).
"""

import re
import torch
import torch.nn as nn
from typing import List
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    TaskType,
)

from pmv.models.base import Model
from pmv.utils import delete_model, cleanup_memory


ROLE_PROMPTS = {
    "logical_structure": {
        "name": "logical structure verifier",
        "focus": "Focus on the logical flow and reasoning structure of the solution.",
    },
    "computational_accuracy": {
        "name": "computational accuracy verifier",
        "focus": "Focus on arithmetic calculations and numerical correctness.",
    },
    "problem_alignment": {
        "name": "problem alignment verifier",
        "focus": "Focus on whether the solution actually addresses the problem asked.",
    },
    "mathematical_rigor": {
        "name": "mathematical rigor verifier",
        "focus": "Focus on mathematical principles, notation, and formal correctness.",
    },
    "error_detection": {
        "name": "error detection verifier",
        "focus": "Focus on identifying potential errors, edge cases, and invalid steps.",
    },
}

ROLES = list(ROLE_PROMPTS.keys())


class DebatingVerifier(Model):
    """
    Verifier model implementing ϕ_j: X × Y → [0,1].

    Inherits from Model for quantization, tokenizer, and generation.
    Adds LoRA adapters and a score head.

    Two scoring modes (matches your original Verifier interface):
      1. Head-based (differentiable): score_head on hidden states, for Phase 1
      2. Generation-based: prompt model to output a score, for evaluation
    """

    def __init__(
        self,
        verifier_id: int,
        model_name: str,
        role: str,
        use_quantization: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        role_info = ROLE_PROMPTS.get(role, {"name": "general", "focus": "Evaluate the solution carefully."})
        super().__init__(
            model_name,
            role=role_info,
            use_quantization=use_quantization,
        )
        self.verifier_id = verifier_id
        self.verifier_type = role
        self.role_name = role_info["name"]
        self.focus = role_info["focus"]
        self.score_max_length = 1024

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            inference_mode=False,
        )
        # Cache hidden_size before LoRA wrapping
        hidden_size = self.model.config.hidden_size
        self.model = get_peft_model(self.model, lora_config)

        print(f"[Verifier {verifier_id} ({role})] Trainable parameters:")
        self.model.print_trainable_parameters()

        # Score head: hidden state -> scalar logit
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        ).to(self.device)

        # Xavier init for stable training start
        with torch.no_grad():
            for module in self.score_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    # ---- head-based scoring (differentiable, Phase 1) --------------------

    def score_with_head(
        self,
        problem: str,
        solution: str,
        transcript: str = "",
        return_prob: bool = True,
    ) -> torch.Tensor:
        """
        Compute ϕ_j(x, y, d) using the differentiable score head.
        Enables gradient flow through verifier for Phase 1 joint training.
        Returns scalar tensor on self.device.
        """
        prompt = self._create_verification_prompt(problem, solution, transcript)
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.score_max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(
            **inputs, output_hidden_states=True, use_cache=False,
        )

        if not (hasattr(outputs, 'hidden_states') and
                outputs.hidden_states is not None and
                len(outputs.hidden_states) > 0):
            raise RuntimeError(
                f"Model did not return hidden_states for verifier {self.verifier_id}."
            )

        last_hidden = outputs.hidden_states[-1][:, -1, :]
        if last_hidden.dtype != torch.float32:
            last_hidden = last_hidden.float()

        if torch.isnan(last_hidden).any() or torch.isinf(last_hidden).any():
            raise ValueError(f"NaN/Inf in hidden states for verifier {self.verifier_id}")

        last_hidden = torch.clamp(last_hidden, -10.0, 10.0)
        logit = self.score_head(last_hidden).squeeze()

        if torch.isnan(logit).any() or torch.isinf(logit).any():
            raise ValueError(f"NaN/Inf in logit for verifier {self.verifier_id}")

        logit = torch.clamp(logit, -10.0, 10.0)

        if return_prob:
            return torch.sigmoid(logit)
        return logit

    # ---- generation-based scoring (no gradients, evaluation) -------------

    def score_with_generation(
        self,
        problem: str,
        solution: str,
        transcript: str = "",
    ) -> torch.Tensor:
        """
        Compute score by prompting the model to output a number.
        No gradients. Used during data collection / evaluation.
        """
        prompt = self._create_verification_prompt(problem, solution, transcript)
        response = self.generate(prompt, max_new_tokens=512)
        try:
            score = self._parse_score(response)
        except Exception:
            score = 0.5
        return torch.tensor(score, device=self.device, dtype=torch.float32)

    # ---- unified compute_score interface ---------------------------------

    def compute_score(
        self,
        problem: str,
        solution: str,
        transcript: str = "",
        training: bool = False,
        use_head: bool = False,
    ) -> torch.Tensor:
        """
        ϕ_j(x, y, d) in [0,1]. Always returns scalar tensor on self.device.
        Uses head-based scoring when training=True (gradient flow).
        Uses head-based scoring without gradients when use_head=True (aligned eval).
        Falls back to generation-based scoring otherwise.
        """
        if training:
            return self.score_with_head(problem, solution, transcript, return_prob=True)
        elif use_head:
            with torch.no_grad():
                return self.score_with_head(problem, solution, transcript, return_prob=True)
        else:
            return self.score_with_generation(problem, solution, transcript)

    # ---- prompt construction ---------------------------------------------

    def _create_verification_prompt(
        self, problem: str, solution: str, transcript: str = "",
    ) -> str:
        system = f"You are a {self.role_name}. {self.focus}"
        user_msg = f"Problem: {problem}\n\nSolution: {solution}\n\n"
        if transcript:
            user_msg += f"=== DEBATE TRANSCRIPT ===\n{transcript}\n\n"
        user_msg += (
            "Evaluate this solution. Consider whether it is correct, complete, and well-reasoned.\n"
            "Provide a score from 0.0 to 1.0 where:\n"
            "- 1.0 = Completely correct and well-explained\n"
            "- 0.0 = Completely incorrect or nonsensical\n\n"
            "End your response with: SCORE: X.X"
        )

        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return f"{system}\n\n{user_msg}\n\nAssistant:"

    # ---- freeze / unfreeze -----------------------------------------------

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.score_head.parameters():
            param.requires_grad = False

    def unfreeze_lora(self) -> int:
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.score_head.parameters():
            param.requires_grad = True
        trainable = [p for p in self.parameters() if p.requires_grad]
        return len(trainable)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad] + \
               [p for p in self.score_head.parameters() if p.requires_grad]

    def freeze(self):
        self.freeze_all()
        self.eval()

    def unfreeze(self):
        self.unfreeze_lora()
        self.train()

    def delete(self):
        delete_model(self.model, self.device)
        delete_model(self.score_head, self.device)
        self.model = None
        self.score_head = None

    # ---- state dict (LoRA + score head) ----------------------------------

    def state_dict_checkpoint(self):
        return {
            'lora_state': get_peft_model_state_dict(self.model),
            'score_head_state': self.score_head.state_dict(),
            'verifier_type': self.verifier_type,
            'verifier_id': self.verifier_id,
        }

    def load_state_dict_checkpoint(self, state_dict, strict=True):
        if 'lora_state' in state_dict:
            set_peft_model_state_dict(self.model, state_dict['lora_state'])
            self.score_head.load_state_dict(state_dict['score_head_state'], strict=strict)
        else:
            set_peft_model_state_dict(self.model, state_dict)

    # ---- parsing helpers -------------------------------------------------

    @staticmethod
    def _parse_score(response: str) -> float:
        text = response.strip()
        # 1. Explicit SCORE: tag (highest priority)
        score_match = re.search(r'SCORE:\s*\[?(\d+(?:\.\d+)?)\]?', text, re.IGNORECASE)
        if score_match:
            return max(0.0, min(1.0, float(score_match.group(1))))
        # 2. Percentage: "85%" -> 0.85
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if pct_match:
            return max(0.0, min(1.0, float(pct_match.group(1)) / 100.0))
        # 3. Fraction: "7/10" -> 0.7
        frac_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', text)
        if frac_match:
            denom = float(frac_match.group(2))
            if denom > 0:
                return max(0.0, min(1.0, float(frac_match.group(1)) / denom))
        # 4. Last number in [0, 1] range
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        for num_str in reversed(numbers):
            num = float(num_str)
            if 0.0 <= num <= 1.0:
                return num
        # 5. Last number <= 10: treat as X/10 scale
        if numbers:
            score = float(numbers[-1])
            if score <= 10.0:
                return max(0.0, min(1.0, score / 10.0))
        raise ValueError(f"Could not parse score from: {text[:200]}")
