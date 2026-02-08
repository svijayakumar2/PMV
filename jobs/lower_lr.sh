#!/bin/bash
#BSUB -J pmv_lower_lr
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 48:00

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
echo "Experiment: lower_lr"
echo "Config: pmv/configs/experiments/config_lower_lr.yaml"
echo ""

# Run training from PMV directory
cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

python3 -u train_qwen.py

echo ""
echo "Job finished at: $(date)"
