# Dev Notes

## TODO: BT baseline → PE-min extension

**Current state:** supervised + Bradley-Terry pairwise loss (Phase 1). This is the
correctness baseline — it directly optimizes score separation (Kirchner's verifier
objective) instead of the old pointwise BCE.

**What's missing for the full PMV claim:** PE-min (and PE-margin) oversight rules.
The multi-verifier structure only differentiates from Kirchner once the aggregator
exploits *disagreement* between verifiers, which is what Algorithm 1 provides.

**What needs to happen (Algorithm 1, two-phase split):**

`train_phase1` currently collapses both phases into one joint BT pass. For PE-min,
it must be split into:

1. `train_oversight_f()` — Reset aggregator `f` each round. Train it using
   `H_PE-min(f; x,y,z_{1:m}) = min_j l(f(x,y), z_j)`. The verifier scores `z_j`
   are the candidate labels — NOT the correctness label `c`. This trains `f` to
   agree with the most favorable verifier, which creates an exploitable gap.

2. `nash_update()` — Freeze `f`. Train each `ϕ_j` against `l(f_frozen(x,y,z_{1:m}), c)`
   using BT on same-problem pairs. This anchors the verifiers to correctness while
   keeping `f` fixed.

For supervised mode (current), the two phases collapse (verifier scores ≡ correctness
signal), so the single BT pass is correct.

**Files to touch:**
- `pmv/training.py` — split `train_phase1` into `train_oversight_f` + `nash_update`
- `pmv/oversight.py` — `h_pe_min` currently ignores the `target` param (broken);
  fix it to use verifier scores `z_j` as candidates, not `c`
- Config: add `oversight_rule: "pe_min"` variant to the study sweep

**Gate:** get a clean 10-round supervised+BT run first. Check that score separation
is meaningfully above the old BCE result (was 0.0179; target >0.10) and helpful
correctness recovers above 50%. Then implement PE-min.

---

## TODO: parallel verifier scoring in Phase 1

**Where:** `pmv/ensemble.py`, `compute_oversight_score_training` (line ~215).

**Problem:** The training-mode scoring loop is sequential:
```python
for v in self.verifiers:   # V0 → V1 → V2, one at a time
    s = v.compute_score(problem, solution, training=True)
```
This means 6 sequential 3B forward passes per BT pair (3 verifiers × correct +
incorrect response). The inference path (`compute_oversight_score`) already uses
`ThreadPoolExecutor` for parallel scoring — the training path doesn't.

**Why it's non-trivial:** `ThreadPoolExecutor` threads don't inherit PyTorch's
autograd gradient context. You must call `torch.enable_grad()` explicitly inside
the thread function, otherwise gradients are silently disabled:
```python
def _score_one_train(v):
    with torch.enable_grad():   # ← required; not inherited from parent thread
        return v.compute_score(problem, solution, training=True)
```
The `backward()` call stays single-threaded (after all futures resolve) — only
the forward passes are parallelised.

**Actual speedup on 2-GPU layout (`['cuda:0', 'cuda:1', 'cuda:0']`):**

V0 and V2 share cuda:0, so they still serialize on the GPU even when threaded.
Only V1 (cuda:1) runs truly concurrently. Theoretical gain:

| Mode | Per-pair cost |
|---|---|
| Sequential (current) | V0 + V1 + V2 + V0 + V1 + V2 = 6T |
| Threaded | max(V0+V2, V1) × 2 = 4T |
| Gain | ~33% in Phase 1 |

This is a modest win. The gradient-accumulation change (8× fewer optimizer steps)
was the larger Phase 1 speedup already applied.

**When to do it:** after a clean end-to-end run confirms Phase 1 is the bottleneck.
Look for per-round timing in the logs — if Phase 1 is taking >20 min/round, this
is worth doing. Change is ~20 lines, self-contained to `ensemble.py`.
