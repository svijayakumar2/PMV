#!/bin/bash
#BSUB -J pmv_pe_min
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 48:00

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "Experiment: pe_min"
echo "Config: pmv/configs/experiments/config_pe_min.yaml"
echo ""

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

python3 -u -m pmv.main pmv/configs/experiments/config_pe_min.yaml

echo ""
echo "Job finished at: $(date)"
