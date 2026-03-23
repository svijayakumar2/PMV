# PMV Code System Guide

## What this project does
This project trains a game between:
- a **prover** that writes solutions,
- a team of **verifiers** that critique and score those solutions,
- an **oversight model** that combines verifier scores into one trust score.

The training goal is to make oversight useful against both:
- **helpful behavior** (correct solutions),
- **sneaky behavior** (plausible but wrong solutions that try to slip through).

## High-level training loop
Each run has:
1. **Bootstrap phase**: collect initial examples with a fresh prover.
2. **Main rounds** (usually 10):
   - **Phase 1 (Nash update)**: train verifier team and oversight model on the replay buffer.
   - **Phase 2 (Stackelberg step)**: reset prover, collect new episodes, then train prover with PPO.

Verifiers and oversight persist across rounds.  
Prover is reset each round.

## Main files and responsibilities

### `main.py`
- Orchestrates the full run.
- Loads config, dataset, and model components.
- Runs bootstrap and then round-by-round training.
- Prints round summaries:
  - oversight loss,
  - mean reward,
  - helpful correctness,
  - sneaky fool rate at a threshold.

### `training.py`
- Implements the two training phases.
- Phase 1:
  - trains verifier score heads and oversight model on buffered labeled examples.
- Phase 2:
  - collects debate episodes,
  - computes rewards,
  - runs PPO on prover.
- Also prints collection diagnostics:
  - FINAL format compliance,
  - sneaky incorrect count,
  - sneaky fooled count,
  - mean oversight score for helpful and sneaky episodes.

### `ensemble.py`
- Manages the verifier team and oversight aggregation.
- Runs multi-round debate.
- Produces individual verifier scores and one aggregated oversight score.
- Uses verifier score heads for both training and inference-time scoring.

### `verifier.py`
- Defines one verifier agent.
- Each verifier has:
  - a language model with LoRA adapters,
  - a learned score head that outputs a probability in `[0, 1]`.
- Supports:
  - debate message generation,
  - score parsing and confidence parsing,
  - freeze/unfreeze for training control.

### `prover.py`
- Defines the prover model.
- Builds role-specific prompts (helpful vs sneaky).
- Enforces strict output contract:
  - final answer should be on a `FINAL:` line.
- Includes helpers:
  - force sneaky outputs to be incorrect when needed,
  - add a best-effort `FINAL:` line if missing.

### `data.py`
- Defines replay buffer and episode record structure.
- Implements math dataset wrapper and answer checking.
- Math checker uses strict `FINAL:` extraction plus symbolic/numeric comparison.

### `zebra_logic.py`
- Implements ZebraLogic dataset wrapper.
- Samples logic-grid puzzles.
- Checks puzzle answers using JSON-grid and text-match heuristics.

### `oversight.py`
- Defines fixed and learned oversight rules.
- Learned oversight model is a small MLP over verifier scores.
- Supports multiple Phase 1 loss styles:
  - supervised,
  - partial-evaluation variants,
  - soft-min verifier objective.

### `models/base.py`
- Shared model loader for prover and verifier.
- Handles tokenizer setup, quantization, cache, and generation.

## Episode flow (single sample)
1. Choose role (helpful or sneaky).
2. Sample a problem and ground-truth answer.
3. Prover generates solution text.
4. If role is sneaky, enforce incorrectness if needed.
5. Ensure the response has a `FINAL:` line.
6. Label correctness with dataset checker.
7. Run verifier debate.
8. Score with verifier heads and aggregate to one oversight score.
9. Compute reward from role, correctness, and oversight score.
10. Store episode in replay buffer.

## Reward design in plain language

### Helpful role
- If answer is correct: positive base reward plus a bonus for higher oversight score.
- If answer is wrong: penalty shaped by oversight score.

### Sneaky role
- If answer is wrong: reward is based on **how far oversight score is above or below a fooling threshold**.
  - above threshold: positive,
  - below threshold: negative.
- If sneaky answer is actually correct: extra penalty.

This makes sneaky behavior optimize for actual fooling, not just being wrong.

## Current key metrics (plain language)
- **Oversight loss**: how well verifier team plus oversight fit correctness labels in buffer.
- **Helpful correctness**: correct helpful answers out of helpful episodes.
- **Sneaky fool rate at threshold**: wrong sneaky answers that still got high oversight score, divided by sneaky episodes.
- **FINAL format compliance**: how often outputs included required `FINAL:` line.
- **Mean oversight score (helpful/sneaky)**: average trust score by role.

## Config knobs that matter most
In baseline and zebra experiment configs:
- Model size and quantization.
- Number of verifiers.
- Number of rounds and episodes per round.
- Debate rounds.
- Verifier learning rate and epochs.
- Prover PPO learning rate and epochs.
- Helpful and sneaky decoding behavior:
  - whether sampling is on,
  - temperature,
  - maximum generation length.
- Sneaky fooling threshold and sneaky reward shaping strength.

## How to read logs quickly
Use this sequence:
1. Check **FINAL format compliance**.
   - Low compliance means labels are noisy or missing.
2. Check **sneaky incorrect count**.
   - If near zero, sneaky generation is not doing its job.
3. Check **sneaky fool rate at threshold**.
   - If always zero, either oversight is too harsh or sneaky never reaches threshold.
4. Check **helpful correctness**.
   - If very low, helpful decoding or label quality is failing.
5. Check **mean oversight score by role**.
   - If both roles collapse to same extreme, oversight is saturating.

## Common failure patterns and likely causes
- **Helpful correctness very low**:
  - weak decoding settings for helpful role,
  - answer extraction mismatch with expected format.
- **Sneaky fool rate always zero**:
  - sneaky never produces high-oversight wrong answers,
  - fooling threshold too strict,
  - oversight score collapsed too low.
- **Everything looks perfect in zebra (helpful near 100%, sneaky fool zero, loss near zero)**:
  - checker may be too permissive,
  - system may be collapsed to trivial behavior, not robust oversight.

## One-paragraph explanation you can reuse
This code trains a repeated game where a prover proposes solutions and a persistent verifier team debates and scores them. A learned oversight model aggregates verifier scores into a trust value. Each round first updates verifier-plus-oversight on replay-buffer labels, then resets prover and trains it with PPO using role-conditioned rewards. Helpful behavior is rewarded for correctness, while sneaky behavior is rewarded only when incorrect answers also cross a fooling threshold. Logs report correctness, fooling, oversight loss, and formatting compliance so we can distinguish real learning from metric collapse.
