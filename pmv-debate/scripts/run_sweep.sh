#!/bin/bash
# Submit all experiment configs as separate LSF jobs.
# Usage: bash scripts/run_sweep.sh

EXPERIMENTS=(
    "baseline"
    "5verifiers"
    "pe_min"
    "pe_margin"
    "median"
    "scaled_prover"
    "softmin"
)

echo "Submitting ${#EXPERIMENTS[@]} experiments..."

for exp in "${EXPERIMENTS[@]}"; do
    echo "  Submitting: ${exp}"
    EXPERIMENT=${exp} bsub < scripts/train.sh
done

echo "All jobs submitted. Check with: bjobs -w"
