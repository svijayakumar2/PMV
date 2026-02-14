#!/bin/bash
#BSUB -J pmv_eval
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 24:00

# ============================================================
# PMV Unified Evaluation Job
# Usage examples:
#   EXPERIMENT=baseline bsub < scripts/evaluation.sh
#   EXPERIMENT=zebra DATASET=zebra ZEBRA_MAX_SIZE=4*4 bsub < scripts/evaluation.sh
# ============================================================

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

EXPERIMENT=${EXPERIMENT:-baseline}
CONFIG_PATH="pmv/configs/experiments/config_${EXPERIMENT}.yaml"
DATASET=${DATASET:-auto}
ZEBRA_MAX_SIZE=${ZEBRA_MAX_SIZE:-4*4}
PROBE_EPISODES=${PROBE_EPISODES:-120}
ATTACK_EPISODES=${ATTACK_EPISODES:-60}
OUTPUT=${OUTPUT:-results/eval_${EXPERIMENT}.json}

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Config not found: ${CONFIG_PATH}"
  exit 1
fi

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "Experiment: ${EXPERIMENT}"
echo "Config: ${CONFIG_PATH}"
echo "Dataset mode: ${DATASET}"
echo "Probe episodes: ${PROBE_EPISODES}"
echo "Attack episodes: ${ATTACK_EPISODES}"
echo "Output: ${OUTPUT}"
echo ""

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

mkdir -p results

if [ "${DATASET}" = "zebra" ]; then
  python3 -u -m pmv.evaluation "${CONFIG_PATH}" \
      --dataset zebra \
      --zebra-max-size "${ZEBRA_MAX_SIZE}" \
      --probe-episodes "${PROBE_EPISODES}" \
      --attack-episodes "${ATTACK_EPISODES}" \
      --output "${OUTPUT}"
else
  python3 -u -m pmv.evaluation "${CONFIG_PATH}" \
      --dataset "${DATASET}" \
      --probe-episodes "${PROBE_EPISODES}" \
      --attack-episodes "${ATTACK_EPISODES}" \
      --output "${OUTPUT}"
fi

echo ""
echo "Job finished at: $(date)"
