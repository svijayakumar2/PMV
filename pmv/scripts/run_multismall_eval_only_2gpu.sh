#!/bin/bash
#BSUB -J pmv_multismall_eval_2gpu
#BSUB -q normal
#BSUB -gpu "num=2:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 128GB
#BSUB -R "rusage[mem=128GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 12:00

set -euo pipefail

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export PYTHONFAULTHANDLER=1

REPO_ROOT=${REPO_ROOT:-/dccstor/principled_ai/users/saranyaibm2/PMV}
CONFIG_PATH=${CONFIG_PATH:-}
OUTPUT_JSON=${OUTPUT_JSON:-}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}

PROBE_EPISODES=${PROBE_EPISODES:-80}
ATTACK_EPISODES=${ATTACK_EPISODES:-40}
PROBE_MAX_NEW_TOKENS=${PROBE_MAX_NEW_TOKENS:-768}
ATTACK_MAX_NEW_TOKENS=${ATTACK_MAX_NEW_TOKENS:-768}
DECISION_THRESHOLD=${DECISION_THRESHOLD:-0.5}
VERIFIER_DECISION_THRESHOLD=${VERIFIER_DECISION_THRESHOLD:-0.5}
FOOL_THRESHOLD=${FOOL_THRESHOLD:-0.5}
EVAL_TEMPS=${EVAL_TEMPS:-"0.7 1.0"}
DATASET=${DATASET:-math}
SEED=${SEED:-0}
SKIP_ADVERSARIAL=${SKIP_ADVERSARIAL:-0}
SAVE_PROBE_RECORDS=${SAVE_PROBE_RECORDS:-1}

if [ -z "${CONFIG_PATH}" ]; then
  echo "CONFIG_PATH is required."
  exit 1
fi

if [ -z "${OUTPUT_JSON}" ]; then
  echo "OUTPUT_JSON is required."
  exit 1
fi

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH:-auto-latest}"
echo "Output: ${OUTPUT_JSON}"
echo "Probe episodes: ${PROBE_EPISODES}"
echo "Attack episodes: ${ATTACK_EPISODES}"
echo "Skip adversarial: ${SKIP_ADVERSARIAL}"
echo ""

cd "${REPO_ROOT}" || exit 1
mkdir -p "$(dirname "${OUTPUT_JSON}")"

EXTRA_ARGS=()
if [ -n "${CHECKPOINT_PATH}" ]; then
  EXTRA_ARGS+=("--checkpoint" "${CHECKPOINT_PATH}")
fi
if [ "${SKIP_ADVERSARIAL}" = "1" ]; then
  EXTRA_ARGS+=("--skip-adversarial")
fi
if [ "${SAVE_PROBE_RECORDS}" = "1" ]; then
  EXTRA_ARGS+=("--save-probe-records")
fi

# shellcheck disable=SC2086
python3 -u -m pmv.evaluation \
  "${CONFIG_PATH}" \
  --output "${OUTPUT_JSON}" \
  --probe-episodes "${PROBE_EPISODES}" \
  --attack-episodes "${ATTACK_EPISODES}" \
  --probe-max-new-tokens "${PROBE_MAX_NEW_TOKENS}" \
  --attack-max-new-tokens "${ATTACK_MAX_NEW_TOKENS}" \
  --decision-threshold "${DECISION_THRESHOLD}" \
  --verifier-decision-threshold "${VERIFIER_DECISION_THRESHOLD}" \
  --fool-threshold "${FOOL_THRESHOLD}" \
  --seed "${SEED}" \
  --dataset "${DATASET}" \
  --temperatures ${EVAL_TEMPS} \
  "${EXTRA_ARGS[@]}"

echo ""
echo "Eval-only job complete."
echo "Job finished at: $(date)"
