# PMV Status — March 24, 2026

---

## CRITICAL: What We Got Wrong

### The Kirchner "attack suite" framing is unverified

During this session, a false premise was introduced and acted on: that Kirchner et al. 2024 report specific attack success rates on a single trained verifier (e.g. ~1/3 attacks succeeding), and that our goal was to replicate that number as a baseline before showing multi-small beats it.

**This was not verified against the paper.** The claim was inferred from the paper's general motivation (prover-verifier games improve robustness, implying baselines are more vulnerable) and stated as fact. It was wrong.

Additionally: SRC, CGC, and goodhart are **reward types** used during prover training in Kirchner's framework, not standalone attack types. Our eval code calls them "attacks" (running a prover trained with each reward objective and testing if it fools the verifier), which may or may not correspond to what Kirchner actually evaluates. This has not been checked against the paper.

### What this invalidates

- The framing "our multi-small achieves 0/3 Kirchner attacks vs their baseline which shows N/3" — we don't know what N is from the paper
- The `single_large_kirchner_baseline_r5` training run (job 1116230) was motivated by trying to reproduce a "1/3 attack success" that may not be the right target
- Any claim that our eval code correctly implements Kirchner's evaluation methodology — this needs to be verified

### What this does NOT invalidate

All empirical results are still real:

- Multi-small AUROC 0.827 vs single-large AUROC 0.779 — real
- Multi-small 88% correct accept rate vs single-large 56% — real
- Calibration difference (avg_incorrect 0.463 vs 0.150) — real
- PE-Min v4 training stability (no collapse, growing helpful/sneaky gap) — real
- Plain BT verifier saturation and score inversion in R5 (1116230) — real

These results stand independently of anything in Kirchner's paper. The comparison to Kirchner needs to be rebuilt once the paper is read carefully.

### What needs to happen before any more runs

1. **Read the Kirchner paper** — specifically: what are SRC/CGC/goodhart, what do they actually evaluate, what are their reported attack success rates, what is their baseline, what is their evaluation protocol
2. **Audit our eval code** (`pmv/evaluation.py`, Kirchner attack suite section) against the paper's methodology
3. **Decide what comparison to actually make** — it may not be "our attacks vs their attacks" at all

---

## What We Have (Independent of Kirchner Framing)

### Supervised multi-small — best result (job 1086952, Mar 23)
Run: `multi3b_bt_margin_anchor_vaux_r5_fullppo`

| Metric | Multi-3B supervised (ours) | Single-7B +regularization | Single-7B plain BT |
|---|---|---|---|
| AUROC | **0.827** | 0.779 | TBD (eval pending) |
| Score separation | 0.178 | 0.312 | TBD |
| Avg score (correct) | **0.640** | 0.461 | TBD |
| Avg score (incorrect) | 0.463 | 0.150 | TBD |
| Helpful-correct accept rate | **88.2%** | 56.2% | TBD |
| Binary accuracy | **73.8%** | 72.5% | TBD |
| Helpful correctness | **42.5%** | 40.0% | TBD |
| Our "Kirchner suite" | 0/3 | 0/3 | TBD |

**Key finding**: single-7B with the same BT regularization achieves 0/3 on our attack suite but is hyperconservative (only 56% correct accept rate, avg_incorrect=0.150). Its "robustness" is an artifact of scoring almost everything below 0.5, not genuine discrimination. AUROC (threshold-free) shows multi-small actually discriminates better.

### Plain BT verifier saturation (job 1116230, Mar 24)
Run: `single_large_kirchner_baseline_r5` — single 7B, no margin/anchor/vaux

Training showed BT loss saturating to ~0.026 by R4, then score inversion in R5: mean f sneaky (0.357) > mean f helpful (0.313). Without anchor regularization, plain BT overfits its training pairs and loses calibration. Eval pending.

Whether this matches Kirchner's actual baseline setup is unknown until the paper is read.

### PE-Min v4 (job 1087303, Mar 23–24)
Run: `multi3b_pemin_r5_fullppo_v4` — 3×3B, warm-started from supervised checkpoint, no aggregator reset, 10 BT epochs

| Round | Fool rate | Mean f (helpful/sneaky) | Gap |
|---|---|---|---|
| R1 | 15/60 (25%) | 0.708 / 0.645 | 0.063 |
| R2 | 7/60 (12%) | 0.471 / 0.366 | 0.105 |
| R3 | 12/60 (20%) | 0.497 / 0.481 | 0.016 |
| R4 | 4/60 (7%) | 0.382 / 0.254 | 0.128 |
| R5 | 4/60 (7%) | 0.378 / 0.180 | **0.198** |

No collapse (v3's failure). Gap growing to 0.198 by R5. Adversarial eval pending.

---

## Open Questions (requires reading Kirchner paper)

1. What exactly are SRC/CGC/goodhart in their framework?
2. What is their actual evaluation protocol for robustness?
3. What baseline do they compare against, and what are the reported numbers?
4. Does our eval code (`evaluation.py` Kirchner suite) correctly implement any of this?
5. What is the right comparison to make between our work and theirs?

---

## Pending Evals (hold until Kirchner paper is read)

- PE-Min v4 Kirchner suite eval
- Plain BT single-large Kirchner suite eval (1116230)
- 2/4/5 verifier training runs
- Zebra logic runs

---

## Key Files

| File | Purpose |
|---|---|
| `pmv/training.py` | Phase 1 BT + PE-Min two-phase (`_train_pe_min_two_phase`) |
| `pmv/ensemble.py` | Verifier ensemble, aggregator, `reset_aggregator()` |
| `pmv/oversight.py` | `OversightAggregator`, `h_pe_min`, fixed rules |
| `pmv/main.py` | Round loop, checkpoint resume, round_stats logging |
| `pmv/evaluation.py` | Eval pipeline, "Kirchner attack suite" — **methodology needs verification against paper** |
| `pmv/diversity_analysis.py` | `compute_oversight_quality` (AUROC, separation, soundness) |
| `pmv/scripts/run_multismall_study_math_roundtrain_resumable_2gpu_ddp.sh` | Training job |
| `pmv/scripts/run_multismall_eval_only_2gpu.sh` | Eval job |
| `pmv/docs/job_log_summary.md` | All job results |
| `pmv/docs/STATUS_MAR23_2026.md` | Previous status (contains the incorrect Kirchner framing) |
