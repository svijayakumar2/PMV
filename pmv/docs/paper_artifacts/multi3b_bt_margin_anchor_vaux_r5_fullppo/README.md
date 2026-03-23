# Multi-3B Paper Artifacts (r5 full PPO)

This folder is a curated bundle of the key JSON artifacts for the current strongest multi-verifier run.

## Source Run

- Source directory:
  - `/Users/saranya/Desktop/PMV/results/studies/multi_small_math_tuned/multi3b_bt_margin_anchor_vaux_r5_fullppo`
- Checkpoint used by all artifacts:
  - `results/studies/multi_small_math_tuned/multi3b_bt_margin_anchor_vaux_r5_fullppo/checkpoints/config_multi_small_3b_specialized_latest.pt`

## Core Eval Artifact

- `eval_multi_small_3b_specialized_latest.json`
  - Main operating-point eval snapshot.
  - Key metrics:
    - helpful correctness: `0.425`
    - sneaky incorrect rate: `0.575`
    - sneaky fool rate@0.50: `0.100`
    - sneaky conditional fool@0.50: `0.174`
    - separation: `0.176`
    - binary accuracy: `0.7375`
    - AUROC(correct vs incorrect): `0.8242`

## Kirchner-Style Attack Objective Artifacts

- `eval_multi_small_3b_specialized_kirchner_src_8k.json`
  - Dedicated `SRC` attacker, 8k update budget.
  - Outcome: `success=False`, `updates_to_success=None`.
  - Meaning: attacker did not meet exploit criterion within budget.

- `eval_multi_small_3b_specialized_kirchner_cgc_8k.json`
  - Dedicated `CGC` attacker, 8k update budget.
  - Outcome: `success=False`, `updates_to_success=None`.

- `eval_multi_small_3b_specialized_kirchner_goodhart_8k.json`
  - Dedicated `goodhart` attacker, 8k update budget.
  - Outcome: `success=False`, `updates_to_success=None`.

## Interpretation for Paper Draft

- All three Kirchner-style objective attacks in this bundle (`src`, `cgc`, `goodhart`) failed to achieve exploit success within `8000` updates for this checkpoint.
- This supports a strong robustness claim for this specific multi-3B checkpoint.

## Notes

- These files are copied from the study results directory into this docs bundle for paper assembly.
- If updated reruns are produced, refresh this folder from source and keep filenames versioned by date or suffix.
