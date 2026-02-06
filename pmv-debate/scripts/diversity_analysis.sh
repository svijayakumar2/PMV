#!/bin/bash
#BSUB -W 24:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -q normal
#BSUB -J pmv_diversity
#BSUB -o logs/pmv_diversity_%J.out
#BSUB -e logs/pmv_diversity_%J.err

# ============================================================
# PMV Diversity Analysis
# Compares diversity vs oversight quality across configurations
#
# Usage:
#   bsub < scripts/diversity_analysis.sh
#   DATASET=zebra bsub < scripts/diversity_analysis.sh
# ============================================================

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo ""

DATASET=${DATASET:-math}

mkdir -p logs results

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

python3 -u -m pmv.diversity_analysis \
    configs/experiments/config_baseline.yaml \
    configs/experiments/config_5verifiers.yaml \
    configs/experiments/config_pe_min.yaml \
    configs/experiments/config_pe_margin.yaml \
    configs/experiments/config_median.yaml \
    configs/experiments/config_softmin.yaml \
    --num-episodes 200 \
    --debate-rounds 2 \
    --dataset "${DATASET}" \
    --output "results/diversity_${DATASET}.json"

echo ""
echo "Job finished at: $(date)"
