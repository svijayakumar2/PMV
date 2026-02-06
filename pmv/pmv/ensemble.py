"""
Verifier ensemble: manages m verifiers, the aggregator, and debate.

Verifiers PERSIST across rounds (Section 6.4). Only the prover resets.

Debate integration (Irving 2018):
  1. Verifiers debate in rounds, building a shared transcript.
  2. After debate, each verifier computes debate-conditioned score.
  3. Aggregator (or fixed rule) combines scores into oversight signal.
"""

import torch
from typing import List, Tuple, Dict, Optional

from pmv.verifier import DebatingVerifier, ROLES
from pmv.oversight import (
    OversightAggregator, FIXED_RULES,
)
from pmv.data import DebateState
from pmv.utils import cleanup_memory


class VerifierEnsemble:

    def __init__(self, config: Dict):
        model_cfg = config["model"]
        self.model_name = model_cfg["verifier_model"]
        self.num_verifiers = model_cfg.get("num_verifiers", 3)
        self.use_quantization = model_cfg.get("use_quantization", True)
        self.lora_r = model_cfg.get("verifier_lora_r", 16)
        self.lora_alpha = model_cfg.get("verifier_lora_alpha", 32)

        train_cfg = config["training"]
        self.oversight_rule = train_cfg.get("oversight_rule", "supervised")

        # Fixed rules don't need the aggregator MLP
        self.use_learned_aggregator = self.oversight_rule not in FIXED_RULES
        agg_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.aggregator_device = agg_device

        if self.use_learned_aggregator:
            self.aggregator = OversightAggregator(
                num_verifiers=self.num_verifiers,
                hidden_dim=model_cfg.get("aggregator_hidden_dim", 64),
                device=agg_device,
            )
            self.aggregator_optimizer = torch.optim.AdamW(
                self.aggregator.parameters(), lr=1e-3, weight_decay=0.01,
            )
        else:
            self.aggregator = None
            self.aggregator_optimizer = None

        self.verifiers: List[DebatingVerifier] = []

    # ---- lifecycle -------------------------------------------------------

    def create_verifiers(self):
        """Create verifiers once at start. They persist across rounds."""
        for i in range(self.num_verifiers):
            role = ROLES[i % len(ROLES)]
            print(f"  Creating verifier {i} ({role})")
            v = DebatingVerifier(
                verifier_id=i,
                model_name=self.model_name,
                role=role,
                use_quantization=self.use_quantization,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
            )
            self.verifiers.append(v)
            cleanup_memory()

    def delete_verifiers(self):
        for v in self.verifiers:
            v.delete()
        self.verifiers = []
        cleanup_memory()

    # ---- debate + scoring ------------------------------------------------

    def run_debate(
        self, problem: str, solution: str, num_rounds: int = 2,
    ) -> DebateState:
        """Irving-style debate followed by debate-conditioned scoring."""
        state = DebateState(problem=problem, solution=solution)
        for rd in range(num_rounds):
            for v in self.verifiers:
                msg = v.generate_assessment(problem, solution, state, rd)
                state.messages.append(msg)
                cleanup_memory()
            if self._check_consensus(state):
                break

        transcript = state.get_full_transcript()
        scores = []
        for v in self.verifiers:
            s = v.compute_score(problem, solution, transcript=transcript, training=False)
            scores.append(float(s.item()))
            cleanup_memory()

        state.verifier_scores = scores
        state.aggregated_score = self._aggregate_scores(scores)
        return state

    def compute_oversight_score(
        self, problem: str, solution: str, transcript: str = "",
    ) -> Tuple[List[float], float]:
        """Inference-only oversight score (no gradients)."""
        scores = []
        for v in self.verifiers:
            s = v.compute_score(problem, solution, transcript=transcript, training=False)
            scores.append(float(s.item()))
            cleanup_memory()
        return scores, self._aggregate_scores(scores)

    def compute_oversight_score_training(
        self, problem: str, solution: str, transcript: str = "",
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Training-mode scoring with gradients for Phase 1."""
        score_tensors = []
        for v in self.verifiers:
            s = v.compute_score(problem, solution, transcript=transcript, training=True)
            if s.device != torch.device(self.aggregator_device):
                s = s.to(self.aggregator_device)
            score_tensors.append(s)
        stacked = torch.stack(score_tensors)
        if self.use_learned_aggregator:
            aggregated = self.aggregator(stacked)
        else:
            rule_fn = FIXED_RULES[self.oversight_rule]
            aggregated = rule_fn(stacked)
        return score_tensors, aggregated

    def _aggregate_scores(self, scores: List[float]) -> float:
        if self.use_learned_aggregator:
            return self.aggregator.aggregate_inference(scores)
        rule_fn = FIXED_RULES[self.oversight_rule]
        t = torch.tensor(scores, dtype=torch.float32)
        return float(rule_fn(t).item())

    def _check_consensus(self, state: DebateState, threshold: float = 0.9) -> bool:
        latest = {}
        for msg in state.messages:
            latest[msg.verifier_id] = msg
        if len(latest) < 2:
            return False
        assessments = [m.assessment for m in latest.values()]
        confidences = [m.confidence for m in latest.values()]
        return len(set(assessments)) == 1 and all(c > threshold for c in confidences)

    # ---- freeze / unfreeze -----------------------------------------------

    def freeze_all(self):
        for v in self.verifiers:
            v.freeze()
        if self.aggregator is not None:
            self.aggregator.freeze()

    def unfreeze_all(self):
        for v in self.verifiers:
            v.unfreeze()
        if self.aggregator is not None:
            self.aggregator.unfreeze()

    def get_all_trainable_params(self) -> List[torch.nn.Parameter]:
        params = []
        for v in self.verifiers:
            params.extend(v.get_trainable_params())
        if self.aggregator is not None:
            params.extend(p for p in self.aggregator.parameters() if p.requires_grad)
        return params
