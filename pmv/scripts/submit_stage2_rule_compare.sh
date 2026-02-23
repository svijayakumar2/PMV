#!/bin/bash
# Submit one LSF job per stage-2 oversight rule.
# Usage: bash pmv/scripts/submit_stage2_rule_compare.sh

set -eu

EXPERIMENTS=${EXPERIMENTS:-"stage2_supervised stage2_pe_min stage2_pe_margin"}
RUN_SCRIPT=${RUN_SCRIPT:-pmv/scripts/run_stage2_rule_compare.sh}

if [ ! -f "${RUN_SCRIPT}" ]; then
  if [ -f "scripts/run_stage2_rule_compare.sh" ]; then
    RUN_SCRIPT="scripts/run_stage2_rule_compare.sh"
  else
    echo "Run script not found: ${RUN_SCRIPT}"
    echo "Try: RUN_SCRIPT=pmv/scripts/run_stage2_rule_compare.sh bash pmv/scripts/submit_stage2_rule_compare.sh"
    exit 1
  fi
fi

echo "Submitting stage-2 rule comparison jobs..."
for exp in ${EXPERIMENTS}; do
  echo "  Submitting: ${exp}"
  EXPERIMENT="${exp}" bsub < "${RUN_SCRIPT}"
done
echo "Done. Monitor with: bjobs -w"
