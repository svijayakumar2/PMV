#!/bin/bash
# Submit one LSF job per stage-2 oversight rule.
# Usage: bash scripts/submit_stage2_rule_compare.sh

set -eu

EXPERIMENTS=${EXPERIMENTS:-"stage2_supervised stage2_pe_min stage2_pe_margin"}

echo "Submitting stage-2 rule comparison jobs..."
for exp in ${EXPERIMENTS}; do
  echo "  Submitting: ${exp}"
  EXPERIMENT="${exp}" bsub < scripts/run_stage2_rule_compare.sh
done
echo "Done. Monitor with: bjobs -w"
