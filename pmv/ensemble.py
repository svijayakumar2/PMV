"""
Verifier ensemble: manages m verifiers and the oversight aggregator.

Verifiers PERSIST across rounds (Section 6.4). Only the prover resets.
This implementation uses direct scoring only (no debate step).
"""

import torch
from typing import List, Tuple, Dict, Optional

from pmv.verifier import DebatingVerifier, ROLES
from pmv.oversight import (
    OversightAggregator, FIXED_RULES,
)
from pmv.utils import cleanup_memory


class VerifierEnsemble:

    def __init__(self, config: Dict):
        model_cfg = config["model"]
        self.config = config
        self.model_name = model_cfg["verifier_model"]
        self.num_verifiers = model_cfg.get("num_verifiers", 3)
        self.use_quantization = model_cfg.get("use_quantization", True)
        self.lora_r = model_cfg.get("verifier_lora_r", 16)
        self.lora_alpha = model_cfg.get("verifier_lora_alpha", 32)

        train_cfg = config["training"]
        self.oversight_rule = train_cfg.get("oversight_rule", "supervised")
        self.supervised_single_identity = bool(
            train_cfg.get("supervised_single_identity", True)
        )
        configured_verifier_max_length = int(train_cfg.get("verifier_max_length", 512))
        auto_adjust_verifier_max_length = bool(
            train_cfg.get("auto_adjust_verifier_max_length", True)
        )
        if auto_adjust_verifier_max_length:
            helpful_max_new_tokens = int(train_cfg.get("helpful_max_new_tokens", 768))
            sneaky_max_new_tokens = int(train_cfg.get("sneaky_max_new_tokens", 768))
            verifier_prompt_margin = int(train_cfg.get("verifier_prompt_margin", 256))
            required_min_length = max(helpful_max_new_tokens, sneaky_max_new_tokens) + verifier_prompt_margin
            self.verifier_max_length = max(configured_verifier_max_length, required_min_length)
            if self.verifier_max_length != configured_verifier_max_length:
                print(
                    "  Adjusted verifier_max_length "
                    f"{configured_verifier_max_length} -> {self.verifier_max_length} "
                    "(to match prover generation budget + prompt margin)"
                )
        else:
            self.verifier_max_length = configured_verifier_max_length

        # Fixed rules don't need the aggregator MLP. For supervised single-verifier
        # runs, use identity aggregation (f = verifier score) to avoid a degenerate
        # 1D MLP bottleneck that can collapse calibration.
        self.use_identity_supervised_single = (
            self.oversight_rule == "supervised"
            and self.num_verifiers == 1
            and self.supervised_single_identity
        )
        self.use_learned_aggregator = (
            self.oversight_rule not in FIXED_RULES
            and not self.use_identity_supervised_single
        )
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
            if self.use_identity_supervised_single:
                print("  Using identity aggregation for supervised single-verifier mode.")

        self.verifiers: List[DebatingVerifier] = []

    # ---- lifecycle -------------------------------------------------------

    def create_verifiers(self):
        """Create verifiers once at start. They persist across rounds."""
        train_cfg = self.config.get("training", {})
        dataset_type = str(self.config.get("dataset", {}).get("type", "math")).lower()

        explicit_roles = train_cfg.get("verifier_roles")
        if isinstance(explicit_roles, str):
            explicit_roles = [explicit_roles]
        if isinstance(explicit_roles, list):
            explicit_roles = [str(r).strip() for r in explicit_roles if str(r).strip()]
        else:
            explicit_roles = []

        single_role_override = train_cfg.get("single_verifier_role")
        if single_role_override is not None:
            single_role_override = str(single_role_override).strip()

        # For single-verifier math/gsm8k runs, default to computational accuracy.
        # Logical-structure-only scoring can over-reward plausible but incorrect solutions.
        if self.num_verifiers == 1 and not explicit_roles and not single_role_override:
            if dataset_type in {"math", "gsm8k"}:
                single_role_override = "computational_accuracy"

        for i in range(self.num_verifiers):
            if explicit_roles:
                role = explicit_roles[i % len(explicit_roles)]
            elif self.num_verifiers == 1 and single_role_override:
                role = single_role_override
            else:
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
            v.score_max_length = self.verifier_max_length
            self.verifiers.append(v)
            cleanup_memory()

    def delete_verifiers(self):
        for v in self.verifiers:
            v.delete()
        self.verifiers = []
        cleanup_memory()

    # ---- scoring ---------------------------------------------------------

    def compute_oversight_score(
        self, problem: str, solution: str, transcript: str = "",
    ) -> Tuple[List[float], float]:
        """Inference-only oversight score (no gradients). Uses score head for alignment with training."""
        scores = []
        for v in self.verifiers:
            s = v.compute_score(problem, solution, transcript=transcript, training=False, use_head=True)
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
        elif self.use_identity_supervised_single:
            aggregated = stacked.mean()
        else:
            rule_fn = FIXED_RULES[self.oversight_rule]
            aggregated = rule_fn(stacked)
        return score_tensors, aggregated

    def _aggregate_scores(self, scores: List[float]) -> float:
        if self.use_learned_aggregator:
            return self.aggregator.aggregate_inference(scores)
        if self.use_identity_supervised_single:
            return float(sum(scores) / max(1, len(scores)))
        rule_fn = FIXED_RULES[self.oversight_rule]
        t = torch.tensor(scores, dtype=torch.float32)
        return float(rule_fn(t).item())

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

    # ---- checkpointing ---------------------------------------------------

    def state_dict_checkpoint(self) -> Dict:
        payload = {
            "oversight_rule": self.oversight_rule,
            "num_verifiers": self.num_verifiers,
            "verifiers": [v.state_dict_checkpoint() for v in self.verifiers],
            "aggregator_state": None,
        }
        if self.aggregator is not None:
            payload["aggregator_state"] = self.aggregator.state_dict()
        return payload

    def load_state_dict_checkpoint(self, state: Dict, strict: bool = True):
        ver_states = state.get("verifiers", [])
        if ver_states:
            by_id = {}
            for item in ver_states:
                if isinstance(item, dict) and "verifier_id" in item:
                    by_id[int(item["verifier_id"])] = item
            for idx, v in enumerate(self.verifiers):
                v_state = by_id.get(v.verifier_id)
                if v_state is None and idx < len(ver_states):
                    v_state = ver_states[idx]
                if v_state is not None:
                    v.load_state_dict_checkpoint(v_state, strict=strict)

        if self.aggregator is not None and state.get("aggregator_state") is not None:
            self.aggregator.load_state_dict(state["aggregator_state"], strict=strict)
