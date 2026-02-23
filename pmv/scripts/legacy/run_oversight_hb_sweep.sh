#!/bin/bash
# Submit stable high-budget oversight-rule ablations (same backbone).
# Usage: bash scripts/run_oversight_hb_sweep.sh

EXPERIMENTS=(
  "single_verifier_stable_hb"
  "baseline_stable_hb"
  "average_hb"
  "min_hb"
  "median_hb"
  "max_hb"
  "pe_min_hb"
  "pe_margin_hb"
  "softmin_hb"
)

echo "Submitting ${#EXPERIMENTS[@]} stable HB experiments..."
for exp in "${EXPERIMENTS[@]}"; do
  echo "  Submitting: ${exp}"
  EXPERIMENT="${exp}" bsub < scripts/train.sh
done
echo "Done. Monitor with: bjobs -w"
