import gym
import torch
from aggregator import Aggregator
from data.math_dataset import MathDataset
from reward import signed_relative_convincingness
from models.prover import Prover
from models.verifier import Verifier


class PMVEnv(gym.Env):
    def __init__(self, cfg):
        self.data = MathDataset(split="train")
        self.prover = Prover(cfg.model.prover_name)
        self.verifiers = torch.nn.ModuleList(
            [Verifier(cfg.model.verifier_name) for _ in range(cfg.model.num_verifiers)]
        )
        self.agg = Aggregator(cfg.training.aggregator, cfg.training.tau)

    def step(self, action_role):
        problem, solution_true = next(self.data)
        solution_prover = self.prover(problem, role=action_role)
        scores = [verifier(problem, solution_prover) for verifier in self.verifiers]
        reward = signed_relative_convincingness(scores, solution_true, action_role)
        return None, reward, True, {}
