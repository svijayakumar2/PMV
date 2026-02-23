#!/bin/bash
#BSUB -J pmv_audit
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 12:00

# ============================================================
# PMV Output Audit Job
# Usage:
#   EXPERIMENT=baseline bsub < scripts/audit_outputs.sh
#   EXPERIMENT=zebra DATASET=zebra ZEBRA_MAX_SIZE=4*4 bsub < scripts/audit_outputs.sh
# ============================================================

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

EXPERIMENT=${EXPERIMENT:-baseline}
CONFIG_PATH="pmv/configs/experiments/config_${EXPERIMENT}.yaml"
DATASET=${DATASET:-auto}
ZEBRA_MAX_SIZE=${ZEBRA_MAX_SIZE:-4*4}
EPISODES=${EPISODES:-80}
OUT_JSON=${OUT_JSON:-results/output_audit_${EXPERIMENT}.json}
OUT_MD=${OUT_MD:-results/output_audit_${EXPERIMENT}.md}
CHECKPOINT=${CHECKPOINT:-results/checkpoints/config_${EXPERIMENT}_latest.pt}
REQUIRE_CHECKPOINT=${REQUIRE_CHECKPOINT:-1}

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
echo "Episodes: ${EPISODES}"
echo "Checkpoint: ${CHECKPOINT}"
echo ""

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

mkdir -p results

if [ "${REQUIRE_CHECKPOINT}" = "1" ] && [ ! -f "${CHECKPOINT}" ]; then
  echo "Required checkpoint not found: ${CHECKPOINT}"
  echo "Run training first, or pass REQUIRE_CHECKPOINT=0."
  exit 1
fi

CKPT_ARG=()
if [ -f "${CHECKPOINT}" ]; then
  CKPT_ARG=(--checkpoint "${CHECKPOINT}")
fi

if [ "${DATASET}" = "zebra" ]; then
  python3 -u pmv/scripts/audit_outputs.py "${CONFIG_PATH}" \
      "${CKPT_ARG[@]}" \
      --episodes "${EPISODES}" \
      --dataset zebra \
      --zebra-max-size "${ZEBRA_MAX_SIZE}" \
      --out-json "${OUT_JSON}" \
      --out-md "${OUT_MD}"
else
  python3 -u pmv/scripts/audit_outputs.py "${CONFIG_PATH}" \
      "${CKPT_ARG[@]}" \
      --episodes "${EPISODES}" \
      --dataset "${DATASET}" \
      --out-json "${OUT_JSON}" \
      --out-md "${OUT_MD}"
fi

echo ""
echo "Job finished at: $(date)"
