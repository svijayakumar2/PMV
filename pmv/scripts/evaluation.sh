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
ABLATION_ID=${ABLATION_ID:-}
ABLATION_TAG=${ABLATION_TAG:-}
SAFE_TAG=$(echo "${ABLATION_TAG}" | tr -cs '[:alnum:]_-' '_' | sed 's/^_//; s/_$//')
export ABLATION_TAG
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
TAG_SUFFIX=""
if [ -n "${SAFE_TAG}" ]; then
  TAG_SUFFIX="_${SAFE_TAG}"
fi
OUTPUT=${OUTPUT:-results/evals/eval_${EXPERIMENT}${TAG_SUFFIX}_${LSB_JOBID:-local}_${RUN_STAMP}.json}
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
echo "Probe episodes: ${PROBE_EPISODES}"
echo "Attack episodes: ${ATTACK_EPISODES}"
echo "Ablation ID: ${ABLATION_ID}"
echo "Ablation tag: ${ABLATION_TAG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT}"
echo ""

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

mkdir -p results results/evals

if [ "${REQUIRE_CHECKPOINT}" = "1" ] && [ ! -f "${CHECKPOINT}" ]; then
  echo "Required checkpoint not found: ${CHECKPOINT}"
  echo "Run training first, or pass REQUIRE_CHECKPOINT=0."
  exit 1
fi

CKPT_ARG=()
if [ -f "${CHECKPOINT}" ]; then
  CKPT_ARG=(--checkpoint "${CHECKPOINT}")
fi

ABLATION_ARG=()
if [ -n "${ABLATION_ID}" ]; then
  ABLATION_ARG=(--ablation-id "${ABLATION_ID}")
fi

if [ "${DATASET}" = "zebra" ]; then
  python3 -u -m pmv.evaluation "${CONFIG_PATH}" \
      "${CKPT_ARG[@]}" \
      "${ABLATION_ARG[@]}" \
      --dataset zebra \
      --zebra-max-size "${ZEBRA_MAX_SIZE}" \
      --probe-episodes "${PROBE_EPISODES}" \
      --attack-episodes "${ATTACK_EPISODES}" \
      --output "${OUTPUT}"
else
  python3 -u -m pmv.evaluation "${CONFIG_PATH}" \
      "${CKPT_ARG[@]}" \
      "${ABLATION_ARG[@]}" \
      --dataset "${DATASET}" \
      --probe-episodes "${PROBE_EPISODES}" \
      --attack-episodes "${ATTACK_EPISODES}" \
      --output "${OUTPUT}"
fi

echo ""
echo "Job finished at: $(date)"
