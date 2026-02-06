#!/bin/bash
#BSUB -W 48:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -q normal
#BSUB -J pmv_train
#BSUB -o logs/pmv_train_%J.out
#BSUB -e logs/pmv_train_%J.err

# ============================================================
# PMV Training Job
# Usage: bsub < scripts/train.sh
#   or:  EXPERIMENT=pe_min bsub < scripts/train.sh
# ============================================================

# Set up environment
export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# Load modules if needed (uncomment and adjust as needed)
# module load python/3.10
# module load cuda/12.1

# Activate virtual environment if using one
# source /path/to/venv/bin/activate

# Print job info
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo ""

# Default experiment or from env var
EXPERIMENT=${EXPERIMENT:-baseline}
CONFIG="configs/experiments/config_${EXPERIMENT}.yaml"

echo "Experiment: ${EXPERIMENT}"
echo "Config: ${CONFIG}"
echo ""

# Ensure log directory exists
mkdir -p logs

# Run training from PMV directory
cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

python3 -u -m pmv.main "${CONFIG}"

echo ""
echo "Job finished at: $(date)"
