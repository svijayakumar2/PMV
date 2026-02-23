#!/bin/bash
# Submit one LSF job per collapse diagnostic experiment.
# Usage: bash scripts/submit_collapse_diagnostic_sweep.sh

set -eu

EXPERIMENTS=${EXPERIMENTS:-"diag_supervised diag_pe_min diag_pe_margin"}

echo "Submitting collapse-diagnostic jobs..."
for exp in ${EXPERIMENTS}; do
  echo "  Submitting: ${exp}"
  EXPERIMENT="${exp}" bsub < scripts/run_collapse_diagnostic.sh
done
echo "Done. Monitor with: bjobs -w"
