#!/bin/bash
#BSUB -J pmv_ablation_report
#BSUB -q normal
#BSUB -n 1
#BSUB -M 8GB
#BSUB -R "rusage[mem=8GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 01:00

# ============================================================
# PMV Inverse-Ablation Report Builder
# Usage:
#   bsub < scripts/ablation_report.sh
#   INPUTS="results/evals/*.json results/custom/*.json" bsub < scripts/ablation_report.sh
# ============================================================

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo ""

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

mkdir -p results/ablations

INPUTS=${INPUTS:-results/evals/*.json}
OUT_DIR=${OUT_DIR:-results/ablations}

# shellcheck disable=SC2086
python3 -u pmv/scripts/build_ablation_table.py \
    --inputs ${INPUTS} \
    --out-dir "${OUT_DIR}"

echo ""
echo "Job finished at: $(date)"
