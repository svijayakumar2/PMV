# PMV Algorithm In Words

This is the training loop in plain language.

## 1. Setup

- Build one prover model and several verifier models.
- Choose one oversight rule:
  - learned rule (`supervised`, `pe min`, `pe margin`, `softmin`), or
  - fixed rule (`average`, `median`, `min`, `max`).
- Create a replay buffer to store past problem-solution records.

## 2. Bootstrap Data Collection

- Sample problems from the dataset.
- For each sampled problem:
  - pick a role for the prover (`helpful` or `sneaky`) based on the prior probability,
  - generate a solution,
  - force sneaky outputs to be incorrect if they accidentally became correct,
  - enforce a strict final answer line (`FINAL: ...`),
  - compute correctness against ground truth,
  - store the record in the replay buffer.

This gives the verifier side an initial dataset before adversarial rounds start.

## 3. Repeated Rounds (Main Training)

Each round has two phases.

### Phase 1: Verifier-side update (leader step)

- Keep the prover fixed.
- Train verifier parameters (and learned aggregator, if used) on replay-buffer records.
- Objective: reduce oversight loss against true correctness labels.
- For each training record:
  - compute each verifier score,
  - combine scores through the selected oversight rule,
  - compute loss and update verifier-side parameters.

After this phase, freeze verifier-side parameters.

### Phase 2: Prover-side update (follower step)

- Reset prover to the base checkpoint (fresh follower each round).
- Keep verifier-side parameters frozen.
- Collect new episodes:
  - sample role (`helpful` or `sneaky`),
  - generate response,
  - enforce strict final line,
  - evaluate correctness,
  - get oversight score,
  - compute reward:
    - role-aligned outputs receive a positive affine reward in oversight score,
    - misaligned outputs receive a negative affine reward in oversight score.
- Train prover with PPO on collected episodes.
- Add new episode records to replay buffer.

## 4. Optional Debate Path

- If debate rounds are greater than zero, verifiers exchange natural-language arguments first.
- After debate, verifiers score using the debate transcript as extra context.
- If debate rounds are zero, scoring is direct from problem and solution only.

Current default in this repo is debate disabled (`debate rounds = 0`).

## 5. What Metrics Are Tracked

- Oversight loss after verifier-side training.
- Helpful correctness (how often helpful outputs are truly correct).
- Sneaky fool rate at a threshold (how often incorrect sneaky outputs still get high oversight score).
- Mean reward.
- Format compliance rate (`FINAL:` line present before any automatic fix).

## 6. What “Success” Looks Like

- Oversight loss decreases without collapsing to trivial scoring.
- Helpful correctness rises.
- Sneaky fool rate does not stay artificially zero for bad reasons.
- Score separation between correct and incorrect outputs is meaningful.
- These hold in both baseline math and Zebra logic domains.
