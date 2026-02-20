#!/bin/bash
# Submit canonical high-budget oversight ablations.
# Usage: bash scripts/run_oversight_sweep.sh

EXPERIMENTS=(
  "single_verifier"
  "baseline"
  "average"
  "min"
  "median"
  "max"
  "pe_min"
  "pe_margin"
  "softmin"
)

echo "Submitting ${#EXPERIMENTS[@]} experiments..."
for exp in "${EXPERIMENTS[@]}"; do
  echo "  Submitting: ${exp}"
  EXPERIMENT="${exp}" bsub < scripts/train.sh
done
echo "Done. Monitor with: bjobs -w"
