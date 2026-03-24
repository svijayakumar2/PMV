# PMV Job Log Summary

Quick reference for all output logs. All runs use Qwen/Qwen2.5-3B-Instruct (4-bit quant + LoRA, ~3.69M trainable params) on OpenMathInstruct-2 MATH subset (14,704 samples, 13,234 train / 1,470 test) unless noted.

**Key metrics:**
- **Fool rate@0.50**: fraction of sneaky episodes where verifier score ≥ 0.5 (higher = verifiers weaker)
- **Helpful correctness**: fraction of helpful episodes with correct solutions (prover quality)
- Adversarial eval fool rates (separate eval jobs) are more meaningful than training-time fool rates

---

## Training Runs

### math_roundtrain_2gpu_ddp — multi_small_3b_specialized (supervised BT)
3 specialized verifiers (logical_structure, computational_accuracy, problem_alignment), supervised Bradley-Terry oversight, 2GPU DDP PPO, 5 rounds.

| Job | Date | Rounds run | Round stats (last round) | Fool rate | Helpful correct | Reward | Status |
|---|---|---|---|---|---|---|---|
| **422377** | ~Feb 2026 | R1 (partial) | R1 | 44/60 (73%) | 23/60 (38%) | 0.460 | Killed (exit 130) |
| **452360** | ~Feb 2026 | R1 | R1 | 46/60 (77%) | 23/60 (38%) | 0.381 | Killed (exit 130) |
| **473051** | ~Feb 2026 | R1 (partial) | — | — | — | — | Killed (exit 130) |
| **480854** | ~Feb 2026 | R1 | R1 | 47/60 (78%) | 22/60 (37%) | 0.377 | Crashed (exit 1) |
| **483102** | ~Feb 2026 | partial | — | — | — | — | Crashed (exit 1) |
| **483524** | ~Feb 2026 | R1 | R1 | 14/60 (23%) | 27/60 (45%) | 0.416 | Complete ✓ |
| **486467** | ~Feb 2026 | R1 | R1 | 24/60 (40%) | 23/60 (38%) | 0.294 | Crashed (exit 1) |
| **516012** | ~Feb 2026 | — | — | — | — | — | Killed (exit 130) |
| **545285** | ~Feb 2026 | R1–R2 | R1: 42/60 (70%), R2: 33/60 (55%) | — | R1: 26/60, R2: 29/60 | R1: 0.466, R2: 0.638 | Complete ✓ |
| **1059465** | Mar 21–22 2026 | R5 (resumed) | R5 | 28/60 (47%) | 19/60 (32%) | 0.294 | Complete ✓ |

**Notes on 545285→1059465 chain**: These are the same run dir (`math_roundtrain_2gpu_ddp`). 545285 ran R1–R2, subsequent jobs continued; 1059465 resumed at R4 and ran R5. PPO early-stopped at 2800/8000 updates in R5.

---

### multi3b_bt_margin_anchor_vaux_r5_fullppo — multi_small_3b_specialized (supervised BT + extras)
Same architecture but with additional BT hyperparams: margin λ=0.5, anchor λ=0.05, vaux λ=0.2, center λ=0.0, τ=4.0.

| Job | Date | Rounds run | Fool rate (last R1) | Helpful | Reward | Status |
|---|---|---|---|---|---|---|
| **744320** | ~Mar 2026 | R1 | 31/60 (52%) | 22/60 (37%) | 0.255 | Complete ✓ |
| **779963** | ~Mar 2026 | R1 | 13/60 (22%) | 22/60 (37%) | 0.459 | Complete ✓ |

---

### multi3b_pemin_r5_fullppo_v3 — multi_small_3b_pemin (PE-Min rule)
3 specialized verifiers, PE-Min aggregator oversight, 2GPU DDP PPO, 5 rounds target.

| Job | Date | Rounds run | Fool rate by round | Helpful | Reward | Status |
|---|---|---|---|---|---|---|
| **1045887** | Mar 20 2026 | Bootstrap only | — | — | — | Crashed (exit 1, 6 min) |
| **1052124** | Mar 21 2026 | R1–R2 (+R3 partial) | R1: 0/60 (0%), R2: 54/60 (90%) | R1: 25/60, R2: 21/60 | R1: 0.457, R2: 0.632 | Complete ✓ (hit time limit at R3) |

**Notes on 1052124**: R1 fool rate = 0% (aggregator not yet calibrated). R2 jumps to 90% — prover exploited the newly updated verifiers before they adapted. Aggregator loss improved significantly at R3 (0.605→0.269).

---

### multi3b_pemin_r5_fullppo_v4 — PE-Min v4 (warm-start, no agg reset, 10 BT epochs)

| Job | Date | Rounds run | Fool rate by round | Mean f (helpful/sneaky) | Status |
|---|---|---|---|---|---|
| **1087303** | Mar 23–24 2026 | R1–R5 | R1: 15/60 (25%), R2: 7/60 (12%), R3: 12/60 (20%), R4: 4/60 (7%), R5: 4/60 (7%) | R1: 0.708/0.645, R2: 0.471/0.366, R3: 0.497/0.481, R4: 0.382/0.254, R5: 0.378/0.180 | Complete ✓ |

**Notes**: No score collapse (v3's fatal flaw). Helpful/sneaky gap grows each round: 0.063 → 0.105 → 0.016 → 0.128 → **0.198**. Verifiers are learning real discrimination. R3 gap near-zero is a yellow flag but recovered in R4–R5. Helpful correctness oscillates (27/24/29/21/22 out of 60) — not collapsed but declining slightly. Kirchner suite eval pending (job TBD).

---

### paper_budget_3v_8k_ddp — multi_small_3b_specialized (supervised BT, alternate budget)

| Job | Date | Rounds run | Fool rate | Helpful | Reward | Status |
|---|---|---|---|---|---|---|
| **483524** | ~Feb 2026 | R1 | 14/60 (23%) | 27/60 (45%) | 0.416 | Complete ✓ |

---

## Evaluation / Probe Jobs

### 637712 — Eval: math_roundtrain_2gpu_ddp at Round 4
Adversarial evaluation of `multi_small_3b_specialized` checkpoint (from job 545285, round 4).

| Metric | Value |
|---|---|
| Adversarial fool rate (temp=0.7) | **75.0%** (24/32) |
| Adversarial fool rate (temp=1.0) | **69.0%** (20/29) |
| Helpful correctness | 0.425 |
| Score separation (correct vs incorrect) | not reported |

**Verdict**: Verifiers very weak at round 4 with plain supervised BT.

---

### 832226 — Eval: multi3b_bt_margin_anchor_vaux_r5_fullppo (partial, old script)
Adversarial evaluation of `multi_small_3b_specialized` with margin+anchor+vaux BT. Old eval script (no Kirchner suite, separation=0.020 was a reporting bug — see 1086952 for correct value).

| Metric | Value |
|---|---|
| Adversarial fool rate (temp=0.7) | **22.6%** (7/31) |
| Adversarial fool rate (temp=1.0) | **15.2%** (5/33) |
| Helpful correctness | 0.425 |

**Superseded by job 1086952.**

---

### 1086952 — Eval: multi3b_bt_margin_anchor_vaux_r5_fullppo (clean Kirchner suite, Mar 23)
Full Kirchner attack suite on the supervised multi-small checkpoint. Same config/script as single-large eval for apples-to-apples comparison.

| Metric | Value |
|---|---|
| AUROC | **0.827** |
| Score separation | **0.178** |
| Avg score (correct) | 0.640 |
| Avg score (incorrect) | 0.463 |
| Adversarial fool rate (temp=0.7) | **22.6%** (7/31) |
| Adversarial fool rate (temp=1.0) | **15.2%** (5/33) |
| Binary accuracy | **73.8%** |
| Helpful correctness | 0.425 |
| Kirchner SRC attack | **FAILED** (budget=2000) |
| Kirchner CGC attack | **FAILED** (budget=2000) |
| Kirchner goodhart attack | **FAILED** (budget=2000) |
| Kirchner suite | **0/3 success** |

**Verdict**: Clean result. No score collapse. Dedicated adversarial prover could not crack any Kirchner objective. This is the main supervised multi-small result.

---

### 1037812 — Eval: multi3b_pemin_r5_fullppo_v2 (Kirchner attack suite)
Adversarial evaluation of PE-Min verifiers with full Kirchner attack suite (SRC/CGC/goodhart objectives).

| Metric | Value |
|---|---|
| Adversarial fool rate (temp=0.7) | **0.0%** (0/31) |
| Adversarial fool rate (temp=1.0) | **13.3%** (4/30) |
| Avg score shift (single-verifier manipulation) | 0.051 |
| Max score shift | 0.146 |
| Kirchner SRC attack | Resumed at 6800/8000 updates, not yet successful |

**Status**: Killed by LSF 12h time limit. Resumed as job 1097725.

---

### 1097725 — Eval: single_large_r5_fullppo (clean Kirchner suite, Mar 24)
Resume of 1087272. Single 7B verifier through same eval script as 1086952 for apples-to-apples.

| Metric | Value |
|---|---|
| AUROC | 0.779 |
| Score separation | 0.312 |
| Avg score (correct) | 0.461 |
| Avg score (incorrect) | 0.150 |
| Approx soundness P(f≤0.1\|incorrect) | **0.688** |
| Helpful-correct accept rate | 0.562 |
| Binary accuracy | 72.5% |
| Helpful correctness | 0.400 |
| Adversarial fool @0.7 | 3.6% (1/28) |
| Adversarial fool @1.0 | 0.0% (0/29) |
| Kirchner SRC | FAILED |
| Kirchner CGC | FAILED |
| Kirchner goodhart | FAILED |
| Kirchner suite | **0/3 success** |

**Verdict**: Hyperconservative — avg_incorrect=0.150, only 56% of correct solutions accepted. Achieves low fool rate and high soundness by scoring nearly everything low. Kirchner attacks fail, but at the cost of usability. Multi-small (1086952) has better AUROC and 88% correct accept rate while matching on Kirchner resistance.

---

## Baseline / Early Experiments

### Single verifier (246182) — single_verifier_hb
**1 verifier only** (logical_structure), supervised BT, Qwen 3B, math domain. Older PPO (epoch-based, no DDP, no PPO reference model).

| Round | Fool rate@0.50 | Helpful correct | Mean reward |
|---|---|---|---|
| R1 | 8/66 (12.1%) | 15/54 (27.8%) | 0.469 |
| R2 | 4/60 (6.7%) | 11/60 (18.3%) | 0.246 |
| R3 | truncated | — | — |

**Status**: Truncated, only R1–R2 complete.

---

### baseline_hb (246181)
3 verifiers, supervised BT, Qwen 3B, math domain. Older PPO setup. 10 rounds total, all completed.

| Round | Fool rate@0.50 | Helpful correct | Mean reward |
|---|---|---|---|
| R1 | 4/62 (6.5%) | 12/58 (20.7%) | 0.499 |
| R2 | 2/59 (3.4%) | 16/61 (26.2%) | 0.331 |
| R3 | 9/59 (15.3%) | 15/61 (24.6%) | 0.422 |
| R4 | 6/63 (9.5%) | 13/57 (22.8%) | 0.422 |
| R5 | 1/61 (1.6%) | 14/59 (23.7%) | 0.274 |
| R6 | 12/59 (20.3%) | 12/61 (19.7%) | 0.316 |
| R7 | 10/64 (15.6%) | 11/56 (19.6%) | 0.298 |
| R8 | 13/52 (25.0%) | 8/68 (11.8%) | 0.182 |
| R9 | 8/55 (14.5%) | 13/65 (20.0%) | 0.347 |
| R10 | 11/64 (17.2%) | 10/56 (17.9%) | 0.388 |

**Notes**: Fool rates oscillate between 1–25%, no clear convergence trend. No adversarial eval for this run.

---

### Zebra puzzle experiments (154658, 171607, latest_output/237762)
Zebra logic puzzle domain (not MATH). All runs show fool rate = 0% throughout — verifiers easily catch all sneaky provers on this structured domain.

| Job | Rounds | Fool rate | Helpful correct | Notes |
|---|---|---|---|---|
| 154658 | R1–R10 | 0% all rounds | 22–28/~25 (near-perfect) | Complete ✓ |
| 171607 | R1–R3 | 0% all rounds | 22–25/~25 | Truncated |
| 237762 | R1–R10 | 0% all rounds | 0–3/~25 (very low!) | Complete ✓ — helpful correctness collapsed |

---

### Baseline (no special training) — 144706, 153913, 171606
Early baseline runs on zebra/baseline configs.

| Job | Config | Fool rate range | Notes |
|---|---|---|---|
| 144706 | baseline | 0–5/~25 (0–17%) | 10 rounds, complete |
| 153913 | baseline | 0–1/~25 (0–4%) | 9 rounds, near-zero fool rate |
| 171606 | baseline | 0% all rounds | 4 rounds, very early experiment |

---

## Comparison: Single vs Multi Verifier

**Training-time fool rates** (caution: not directly comparable — different rounds and prover strength):

| Setup | Fool rate (early rounds) | Fool rate (later rounds) |
|---|---|---|
| Single verifier, supervised (246182) | R1: 12%, R2: 7% | R3+: unknown (truncated) |
| 3-verifier, supervised BT (1059465) | R1–R4: ~42–70% | R5: 47% |
| 3-verifier, supervised BT+extras (832226 eval) | — | Adv. eval: **22%/15%** |
| 3-verifier, PE-Min (1037812 eval) | — | Adv. eval: **0%/13%** |

**Key takeaway**: Plain supervised BT multi-verifier is actually easy to fool (training fool rates 47–77%). Adding margin/anchor/vaux regularization or switching to PE-Min rule substantially improves robustness. The single-verifier run has no adversarial eval so true comparison is incomplete.
