#!/bin/bash
#BSUB -J pmv_diversity
#BSUB -q normal
#BSUB -gpu "num=2:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 24:00

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "Experiment: diversity analysis (cross-config comparison)"
echo ""

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

mkdir -p results

python3 -u -m pmv.diversity_analysis \
    pmv/configs/experiments/config_baseline.yaml \
    pmv/configs/experiments/config_5verifiers.yaml \
    pmv/configs/experiments/config_pe_min.yaml \
    pmv/configs/experiments/config_pe_margin.yaml \
    pmv/configs/experiments/config_median.yaml \
    pmv/configs/experiments/config_softmin.yaml \
    --num-episodes 200 \
    --debate-rounds 2 \
    --dataset math \
    --output results/diversity_math.json

echo ""
echo "Job finished at: $(date)"
