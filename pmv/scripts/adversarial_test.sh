#!/bin/bash
#BSUB -W 12:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -q normal
#BSUB -J pmv_advtest
#BSUB -o logs/pmv_advtest_%J.out
#BSUB -e logs/pmv_advtest_%J.err

# ============================================================
# PMV Adversarial Testing Job
# Usage: EXPERIMENT=baseline bsub < scripts/adversarial_test.sh
# ============================================================

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# module load python/3.10
# module load cuda/12.1

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo ""

EXPERIMENT=${EXPERIMENT:-baseline}
CONFIG="configs/experiments/config_${EXPERIMENT}.yaml"

echo "Adversarial test for: ${EXPERIMENT}"
echo "Config: ${CONFIG}"
echo ""

mkdir -p logs results

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

python3 -u -m pmv.adversarial_test "${CONFIG}" \
    --num-episodes 100 \
    --debate-rounds 2 \
    --output "results/adversarial_${EXPERIMENT}.json"

echo ""
echo "Job finished at: $(date)"
