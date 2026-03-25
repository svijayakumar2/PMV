# PMV Status — March 23, 2026

## Current Goal

Beat Kirchner et al. 2024 (single large verifier) with multi-small verifiers. Two parallel tracks:

1. **Clean supervised win** — multi-3B specialized verifiers with BT training already show stronger robustness than single-7B. Need a clean apples-to-apples eval.
2. **PE-Min win** — PE-Min is our theoretical novelty (prover must fool ALL verifiers simultaneously). Current implementation has training instability; needs fixes before it can compete.

---

## What We Have That Works

### Supervised multi-small — clean Kirchner suite eval (job 1086952, Mar 23)
Run: `multi3b_bt_margin_anchor_vaux_r5_fullppo` | Config: `config_multi_small_3b_specialized.yaml`

| Metric | Multi-3B supervised (ours) | Single-7B baseline |
|---|---|---|
| AUROC | **0.827** | 0.779 |
| Score separation | 0.178 | 0.312 |
| Avg score (correct) | **0.640** | 0.461 |
| Avg score (incorrect) | 0.463 | 0.150 |
| Approx soundness P(f≤0.1\|incorrect) | 0.0 | **0.688** |
| Helpful-correct accept rate | **88.2%** | 56.2% |
| Binary accuracy | **73.8%** | 72.5% |
| Helpful correctness | **42.5%** | 40.0% |
| Adversarial fool @0.7 | 22.6% | **3.6%** |
| Adversarial fool @1.0 | 15.2% | **0.0%** |
| Kirchner suite | **0/3** | **0/3** |

Both resist the Kirchner dedicated attack (0/3 each). The key differentiator is **calibration**: single-large achieves robustness by being hyperconservative (avg_incorrect=0.150, only 56% of correct solutions accepted at threshold 0.5). Multi-small is better calibrated — higher AUROC (0.827 vs 0.779), 88% correct accept rate — while matching on Kirchner attack resistance.

Single-large's low adversarial fool rate (3.6% vs 22.6%) is an artifact of its low scoring: it scores almost everything below 0.5, so temperature-sampled attacks rarely exceed the threshold. AUROC corrects for this threshold effect and shows multi-small actually discriminates better.

---

## PE-Min: What Went Wrong

### Algorithm 1 (Kirchner) — implemented in `training.py` `_train_pe_min_two_phase`
- Phase 1a: reset aggregator, train with H_PE-min = min_j BCE(f, z_j), verifiers frozen
- Phase 1b: freeze aggregator, train verifiers with BT through frozen aggregator

### Training runs (v3: `multi3b_pemin_r5_fullppo_v3`)

| Round | BT pairs | Agg loss (end) | Mean f (helpful/sneaky) | Fool rate |
|---|---|---|---|---|
| 1 | 1 | 0.693 | 0.494 / 0.494 | 0/60 |
| 2 | 62 | 0.690 | 0.526 / 0.526 | 54/60 |
| 3 | 149 | 0.268 | 0.033 / 0.034 | 0/60 |
| 4 | 212 | 0.056 | 0.336 / 0.333 | 5/60 |

### Root cause of failure

**Score oscillation from aggregator reset**: Phase 1a resets the aggregator to random weights every round. With only 3 BT epochs per round, verifiers don't move much. So the aggregator re-learns from near-zero verifier scores → outputs near zero → verifiers get weak BT gradient through a near-flat aggregator → repeat.

Round 3 score collapse (f≈0.03) is the clearest symptom: aggregator reset + compressed verifier scores = everything scored near zero. Round 4 recovers partially but helpful/sneaky separation remains ≈ 0.003 — basically no discrimination.

**Deeper issue**: PE-Min only works when individual verifiers have already learned meaningful, diverse representations. Running PE-Min from scratch simultaneously trains verifiers AND aggregator, which is unstable.

---

## Plan

### Track 1: Clean supervised win (no new training needed)
1. Run Kirchner attack suite eval on the supervised multi-small checkpoint (`multi3b_bt_margin_anchor_vaux_r5_fullppo`) with the updated eval code (now reports avg_correct, avg_incorrect, collapse warning, approx_soundness).
2. Run the same eval on the single-large checkpoint so it's exactly the same eval settings.
3. That's the comparison table.

```bash
# Multi-small supervised eval
CONFIG_PATH=results/studies/multi_small_math_tuned/multi3b_bt_margin_anchor_vaux_r5_fullppo/config_multi_small_3b_specialized.yaml \
OUTPUT_JSON=results/studies/multi_small_math_tuned/multi3b_bt_margin_anchor_vaux_r5_fullppo/eval_kirchner_clean.json \
KIRCHNER_ATTACK_SUITE_ENABLE=1 SAVE_PROBE_RECORDS=1 \
bsub -J pmv_supervised_eval < pmv/scripts/run_multismall_eval_only_2gpu.sh

# Single-large eval (same settings)
CONFIG_PATH=results/studies/multi_small_math_tuned/single_large_r5_fullppo/config_single_large.yaml \
OUTPUT_JSON=results/studies/multi_small_math_tuned/single_large_r5_fullppo/eval_kirchner_clean.json \
KIRCHNER_ATTACK_SUITE_ENABLE=1 SAVE_PROBE_RECORDS=1 \
bsub -J pmv_singlelarge_eval < pmv/scripts/run_multismall_eval_only_2gpu.sh
```

### Track 2: PE-Min win (requires code fixes + new training run)

#### Why v3 failed (job 1062814, R3-R4)

v3 ran with aggregator reset every round + 3 BT epochs + random verifier init. R3 log showed "Aggregator reset to fresh weights" → all scores collapsed to f≈0.03 (both helpful and sneaky). Root cause: reset aggregator re-learned from near-zero verifier scores → output near zero → weak BT gradient through flat aggregator → no verifier learning. Separation in R3-R4 ≈ 0.003, meaningless.

#### v4 fixes (run: `multi3b_pemin_r5_fullppo_v4`, config: `config_multismall_study_math_tuned_pemin.yaml`)

| | v3 | v4 |
|---|---|---|
| Aggregator reset each round | yes → score collapse | **no** (`pe_min_reset_aggregator: false`) |
| BT epochs per round | 3 → weak gradient | **10** (`pe_min_verifier_epochs: 10`) |
| Verifier init | random | **supervised checkpoint** (AUROC 0.83, `pretrain_from_checkpoint`) |

All three changes are implemented and in the config. The warm-start is the biggest change: verifiers start with real discrimination so PE-Min has meaningful per-verifier scores to work with from round 1.

#### Code changes (all done)
- `pmv/training.py`: `pe_min_reset_aggregator` flag in `_train_pe_min_two_phase`; `pe_min_verifier_epochs` param
- `pmv/main.py`: `pretrain_from_checkpoint` loading block before round loop
- `pmv/configs/experiments/config_multismall_study_math_tuned_pemin.yaml`: new config with all three params set

#### Concern: is the prover actually getting smarter each round?

v4 training-time fool rate trend: 25% → 12% → 20% → 7%. PPO early-stop loss: -0.112 → -0.095 → -0.086 → -0.083 (approaching zero). Two interpretations:

**(a) Prover not keeping up**: the Phase 1b BT signal is too weak (verifiers train through a frozen aggregator, indirect gradient), or the prover's PPO budget (2200–4400 updates, early-stopped from 8000) isn't enough to develop strong exploits against improving verifiers. The declining fool rate would then be an artifact of an undertrained prover, not robust verifiers.

**(b) PE-Min guarantee working**: verifiers are genuinely improving each round, and because the prover must fool ALL THREE verifiers simultaneously (not just one), it gets harder exponentially. Even with the same PPO budget, there are fewer and fewer single-verifier blind spots to exploit. The declining fool rate reflects real verifier progress.

**Why (b) is the more plausible explanation here**: in a single-verifier setup (Kirchner), a stronger verifier just raises the bar — the prover can still find exploits by concentrating effort. With PE-Min and 3 verifiers, the prover needs to simultaneously beat all three, and the attack surface shrinks multiplicatively as each verifier independently improves. This is a structural hardness advantage that isn't present in single-verifier settings.

**What we can't distinguish from training-time stats alone**: whether the prover would find exploits given more budget. The Kirchner suite eval on the v4 final checkpoint (dedicated adversarial prover, 2000 updates per objective, no early stop) is the definitive test. If v4 also shows 0/3, the combination of PE-Min + multi-small is the result.

**Potential fixes if (a) turns out to be the problem**:
1. Disable PPO early stopping during training rounds (force full 8000 updates) so the prover always gets a fair shot
2. Increase Phase 1b BT epochs further or use direct supervised BT signal in addition to PE-Min BT (give verifiers a stronger update per round)
3. Run more rounds (>5) to let the arms race develop further before eval

---

## Robustness Eval: Known Gaps (fixed in eval code Mar 23)

**Problem**: fool rate @0.5 alone is not sufficient — v2 PE-Min showed 0%/13% fool rate but separation=0.020 (score collapse, not real robustness).

**Fix applied**: `evaluation.py` now prints:
- `avg_correct_score` and `avg_incorrect_score` alongside separation
- Score collapse warning if both avg scores < 0.4
- `approx_soundness` P(f≤0.1 | incorrect)

**Rule**: a result is only meaningful if AUROC, separation, AND adversarial fool rate are all good. Any single metric can be gamed.

---

## Key Files

| File | Purpose |
|---|---|
| `pmv/training.py` | Phase 1 BT + PE-Min two-phase (`_train_pe_min_two_phase`) |
| `pmv/ensemble.py` | Verifier ensemble, aggregator, `reset_aggregator()` |
| `pmv/oversight.py` | `OversightAggregator`, `h_pe_min`, fixed rules |
| `pmv/main.py` | Round loop, checkpoint resume, round_stats logging |
| `pmv/evaluation.py` | Eval pipeline, Kirchner attack suite, robustness scorecard |
| `pmv/diversity_analysis.py` | `compute_oversight_quality` (AUROC, separation, soundness) |
| `pmv/scripts/run_multismall_study_math_roundtrain_resumable_2gpu_ddp.sh` | Training job |
| `pmv/scripts/run_multismall_eval_only_2gpu.sh` | Eval job |
| `pmv/docs/job_log_summary.md` | All job results |
