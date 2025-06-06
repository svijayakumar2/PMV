import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from pmv.data.math_dataset import MathDataset
from pmv.models.prover import Prover
from pmv.models.verifier import Verifier
# from pmv.models.policy import PolicyNetwork
from pmv.aggregator import Aggregator, LearnedAggregator, train_f
from pmv.reward import signed_relative_convincingness

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train_prover(
        config,
        prover,
        aggregator,
        verifiers,
        dataset,
        role_policy,
        writer,
        round_index
):
    """ Train the prover.

    todo: also fine‑tune the prover (LoRA + PPO)?
    """
    K = config.training.get("k_episodes", 256)
    # gamma = config.training.get("gamma", 0.99)
    # optim = None
    # if role_policy is not None and role_policy.training:
    #     optim = torch.optim.Adam(role_policy.parameters(), lr=config.training.get("lr_pol", 1e-3))

    # todo: use smarter role policy since sneaky provers will have less bite as verifiers improve
    toggle_role = role_policy is None
    role = 0  # start helpful

    episode_returns = []
    log_probs = []

    for episode in range(K):

        if toggle_role:
            role ^= 1  # flip
        else:
            logits = role_policy(torch.empty(0))
            dist = torch.distributions.Categorical(logits=logits)
            role = int(dist.sample().item())
            log_probs.append(dist.log_prob(torch.tensor(role)))

        problem, solution_true = dataset._next()
        role_str = "helpful" if role == 0 else "sneaky"
        solution_prover = prover(problem, role=role_str)
        with torch.no_grad():
            scores = torch.tensor([
                verifier(problem, solution_prover)
                for verifier in verifiers], dtype=torch.float32, device=DEVICE)
            f_score = aggregator(scores)
        correct = dataset.check(solution_true, solution_prover)
        reward = signed_relative_convincingness(f_score, correct, role_str)
        episode_returns.append(float(reward))
        writer.add_scalar("reward/episodic", reward, round_index * K + episode)

    # #
    # if optim is not None and log_probs:
    #     returns = torch.tensor(episode_returns, dtype=torch.float32, device=DEVICE)
    #     returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    #     policy_loss = -(torch.stack(log_probs) * returns.detach()).mean()
    #     optim.zero_grad()
    #     policy_loss.backward()
    #     optim.step()
    #     writer.add_scalar("loss/policy", policy_loss.item(), round_idx)


def train_verifiers(
        config,
        verifiers,
        replay,
        dataset,
        writer,
        round_idx):
    """ Train the verifiers.

    """

    if not replay:
        return
    batch_size = config.training.get("v_batch", 8)
    lr = config.training.get("v_lr", 1e-5)
    steps = config.training.get("v_steps", 100)

    for verifier_index, verifier in enumerate(verifiers):
        opt = torch.optim.Adam(verifier.parameters(), lr=lr)
        for step in range(steps):
            sample = random.sample(replay, k=min(batch_size, len(replay)))
            problems, solutions_true, solutions_prover = zip(*sample)

            labels = torch.tensor([
                dataset.check(solutions_true, solution_prover)
                for solutions_true, solution_prover
                in zip(solutions_true, solutions_prover)
            ], dtype=torch.float32, device=DEVICE)

            scores = torch.tensor([
                verifier(problem, solution_prover)
                for problem, solution_prover
                in zip(problems, solutions_prover)
            ], dtype=torch.float32, device=DEVICE)

            loss = torch.nn.functional.binary_cross_entropy(scores, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

        writer.add_scalar(f"verifier{verifier_index}/loss", loss.item(), round_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    dataset = MathDataset(root=config.data.root, split="train")
    prover = Prover(config.model.prover_name).to(DEVICE)
    verifiers = [Verifier(config.model.verifier_name).to(DEVICE) for _ in range(config.model.num_verifiers)]

    if config.training.get("learn_f", False):
        aggregator = LearnedAggregator(config.model.num_verifiers).to(DEVICE)
    else:
        aggregator = Aggregator(config.training.aggregator, config.training.get("tau", 1.0))

    role_policy = None
    # if config.training.get("learn_role_policy", False):
    #     role_policy = PolicyNetwork().to(DEVICE)

    writer = SummaryWriter(config.logging.logdir)

    replay_buffer = []
    num_rounds = config.training.get("rounds", 10)

    for round in range(num_rounds):

        if config.training.get("learn_f", False):
            train_f(aggregator, prover, verifiers, dataset, config, DEVICE)

        train_prover(config, prover, aggregator, verifiers, dataset, role_policy, writer, round)

        # stash experiences for verifier update
        replay_buffer = replay_buffer[-1000:]

        train_verifiers(config, verifiers, replay_buffer, dataset, writer, round)

        ckpt_dir = Path(config.logging.logdir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "prover": prover.state_dict(),
            "verifiers": [v.state_dict() for v in verifiers],
            "aggregator": aggregator.state_dict() if isinstance(aggregator, LearnedAggregator) else None,
            "role_policy": role_policy.state_dict() if role_policy else None,
            "cfg": json.dumps(config.__dict__),
        }, ckpt_dir / f"round_{round:03d}.pt")

    writer.close()


if __name__ == "__main__":
    main()
