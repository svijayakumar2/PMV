"""
Debating verifier built on the Model base class.

Implements ϕ_j: X × Y → [0,1] (Section 3) with:
  - LoRA adapters for parameter-efficient multi-verifier training
  - Score head for differentiable oversight (Phase 1 gradient flow)
  - Generation-based scoring for evaluation
  - Irving 2018 debate: NL assessments conditioned on transcript

Verifiers PERSIST across rounds (Section 6.4).
"""

import re
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from peft import LoraConfig, get_peft_model, TaskType

from pmv.models.base import Model
from pmv.data import DebateMessage, DebateState
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
    Adds LoRA adapters, a score head, and debate capabilities.

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
            prompt, return_tensors='pt', truncation=True, max_length=512, padding=True,
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
    ) -> torch.Tensor:
        """
        ϕ_j(x, y, d) in [0,1]. Always returns scalar tensor on self.device.
        Uses head-based scoring when training=True (gradient flow).
        Uses generation-based scoring when training=False.
        """
        if training:
            return self.score_with_head(problem, solution, transcript, return_prob=True)
        else:
            return self.score_with_generation(problem, solution, transcript)

    # ---- debate (Irving 2018) --------------------------------------------

    def generate_assessment(
        self,
        problem: str,
        solution: str,
        debate_state: DebateState,
        debate_round: int,
        max_tokens: int = 256,
    ) -> DebateMessage:
        history = debate_state.get_transcript(exclude_verifier=self.verifier_id)
        my_prev = [m for m in debate_state.messages if m.verifier_id == self.verifier_id]
        prev_stance = ""
        if my_prev:
            prev_stance = (
                f"\nYour previous assessment: {my_prev[-1].assessment} "
                f"(confidence: {my_prev[-1].confidence:.2f})"
            )

        if debate_round == 0:
            prompt = (
                f"You are Verifier {self.verifier_id}, a {self.role_name}.\n"
                f"{self.focus}\n\n"
                f"Problem: {problem}\n\nProposed Solution:\n{solution}\n\n"
                f"Evaluate this solution according to your role.\n"
                f"Format:\nASSESSMENT: [CORRECT/INCORRECT]\n"
                f"CONFIDENCE: [0.0-1.0]\nARGUMENT: [Your reasoning]\n"
            )
        else:
            prompt = (
                f"You are Verifier {self.verifier_id}, participating in a debate.\n"
                f"{self.focus}\n\n"
                f"Problem: {problem}\n\nProposed Solution:\n{solution}\n\n"
                f"=== DEBATE HISTORY ===\n{history}\n{prev_stance}\n\n"
                f"Consider other verifiers' arguments. You may maintain or change your position.\n"
                f"Format:\nASSESSMENT: [CORRECT/INCORRECT]\nCONFIDENCE: [0.0-1.0]\n"
                f"ARGUMENT: [Your reasoning]\nCRITIQUE: [Critique or \"None\"]\n"
            )

        response = self.generate(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
        if response.startswith(prompt):
            response = response[len(prompt):]
        response = response.strip()
        assessment, confidence, argument = self._parse_assessment(response)
        critique = self._parse_critique(response) if debate_round > 0 else None
        return DebateMessage(
            verifier_id=self.verifier_id, round=debate_round,
            assessment=assessment, confidence=confidence,
            argument=argument, critique=critique,
        )

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
            'lora_state': self.model.state_dict(),
            'score_head_state': self.score_head.state_dict(),
            'verifier_type': self.verifier_type,
            'verifier_id': self.verifier_id,
        }

    def load_state_dict_checkpoint(self, state_dict, strict=True):
        if 'lora_state' in state_dict:
            self.model.load_state_dict(state_dict['lora_state'], strict=strict)
            self.score_head.load_state_dict(state_dict['score_head_state'], strict=strict)
        else:
            self.model.load_state_dict(state_dict, strict=strict)

    # ---- parsing helpers -------------------------------------------------

    @staticmethod
    def _parse_score(response: str) -> float:
        score_match = re.search(r'SCORE:\s*(\d*\.?\d+)', response.strip(), re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))
        numbers = re.findall(r'(\d*\.?\d+)', response)
        for num_str in reversed(numbers):
            try:
                num = float(num_str)
                if 0.0 <= num <= 1.0:
                    return num
            except ValueError:
                continue
        if numbers:
            try:
                score = float(numbers[-1])
                if score <= 10.0:
                    return max(0.0, min(1.0, score / 10.0))
            except ValueError:
                pass
        raise ValueError(f"Could not parse score from: {response[:200]}")

    @staticmethod
    def _parse_assessment(response: str) -> Tuple[str, float, str]:
        assessment, confidence, argument = "incorrect", 0.5, response
        for line in response.upper().split("\n"):
            s = line.strip()
            if s.startswith("ASSESSMENT:"):
                assessment = "incorrect" if "INCORRECT" in s else "correct"
            elif s.startswith("CONFIDENCE:"):
                nums = re.findall(r"[\d.]+", s)
                if nums:
                    confidence = min(1.0, max(0.0, float(nums[0])))
        arg_lines, in_arg = [], False
        for line in response.split("\n"):
            up = line.upper().strip()
            if up.startswith("ARGUMENT:"):
                in_arg = True
                rest = line[line.upper().find("ARGUMENT:") + 9:].strip()
                if rest:
                    arg_lines.append(rest)
            elif up.startswith(("CRITIQUE:", "ASSESSMENT:", "CONFIDENCE:")):
                in_arg = False
            elif in_arg:
                arg_lines.append(line)
        if arg_lines:
            argument = "\n".join(arg_lines).strip()
        return assessment, confidence, argument

    @staticmethod
    def _parse_critique(response: str) -> Optional[str]:
        lines, in_crit = [], False
        for line in response.split("\n"):
            up = line.upper().strip()
            if up.startswith("CRITIQUE:"):
                in_crit = True
                rest = line[line.upper().find("CRITIQUE:") + 9:].strip()
                if rest and rest.lower() != "none":
                    lines.append(rest)
            elif in_crit:
                if up.startswith(("ASSESSMENT:", "CONFIDENCE:", "ARGUMENT:")):
                    break
                lines.append(line)
        text = "\n".join(lines).strip()
        return text if text and text.lower() != "none" else None
