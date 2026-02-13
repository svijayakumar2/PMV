#!/bin/bash
#BSUB -J pmv_baseline
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
EXPERIMENT=${EXPERIMENT:-baseline}
CONFIG_PATH="pmv/configs/experiments/config_${EXPERIMENT}.yaml"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Config not found: ${CONFIG_PATH}"
  exit 1
fi

# Print job info
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "Experiment: ${EXPERIMENT}"
echo "Config: ${CONFIG_PATH}"
echo ""

# Run training from PMV directory
cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

python3 -u -m pmv.main "${CONFIG_PATH}"

echo ""
echo "Job finished at: $(date)"
