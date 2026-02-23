#!/bin/bash
#BSUB -J pmv_eval_fast
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 04:00

set -eu

# ============================================================
# Fast eval-only debug from an existing checkpoint.
# Usage:
#   bsub < pmv/scripts/run_eval_checkpoint_fast.sh
#   CONFIG_PATH=pmv/configs/experiments/config_stage2_supervised.yaml \
#   CHECKPOINT=results/checkpoints_stage2/config_stage2_supervised_latest.pt \
#   bsub < pmv/scripts/run_eval_checkpoint_fast.sh
# ============================================================

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

REPO_ROOT=${REPO_ROOT:-/dccstor/principled_ai/users/saranyaibm2/PMV}
CONFIG_PATH=${CONFIG_PATH:-pmv/configs/experiments/config_stage2_supervised.yaml}
CHECKPOINT=${CHECKPOINT:-results/checkpoints_stage2/config_stage2_supervised_latest.pt}

PROBE_EPISODES=${PROBE_EPISODES:-24}
ATTACK_EPISODES=${ATTACK_EPISODES:-0}
EVAL_TEMPS=${EVAL_TEMPS:-"0.7"}
DECISION_THRESHOLD=${DECISION_THRESHOLD:-0.5}
VERIFIER_DECISION_THRESHOLD=${VERIFIER_DECISION_THRESHOLD:-0.5}
FOOL_THRESHOLD=${FOOL_THRESHOLD:-0.5}
SKIP_ADVERSARIAL=${SKIP_ADVERSARIAL:-1}
SAVE_PROBE_RECORDS=${SAVE_PROBE_RECORDS:-1}

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-results/fast_eval/${RUN_STAMP}_${LSB_JOBID:-local}}
OUT_JSON=${OUT_DIR}/eval_fast.json

EXTRA_EVAL_ARGS=""
if [ "${SKIP_ADVERSARIAL}" = "1" ]; then
  EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS} --skip-adversarial"
fi
if [ "${SAVE_PROBE_RECORDS}" = "1" ]; then
  EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS} --save-probe-records"
fi

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output dir: ${OUT_DIR}"
echo ""

cd "${REPO_ROOT}" || exit 1
mkdir -p "${OUT_DIR}"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Config not found: ${CONFIG_PATH}"
  exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

# shellcheck disable=SC2086
python3 -u -m pmv.evaluation "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset zebra \
  --zebra-max-size "3*3" \
  --probe-episodes "${PROBE_EPISODES}" \
  --attack-episodes "${ATTACK_EPISODES}" \
  --probe-max-new-tokens 160 \
  --attack-max-new-tokens 160 \
  --temperatures ${EVAL_TEMPS} \
  --decision-threshold "${DECISION_THRESHOLD}" \
  --verifier-decision-threshold "${VERIFIER_DECISION_THRESHOLD}" \
  --fool-threshold "${FOOL_THRESHOLD}" \
  ${EXTRA_EVAL_ARGS} \
  --output "${OUT_JSON}"

python3 -u pmv/scripts/classify_collapse.py "${OUT_JSON}" | tee "${OUT_DIR}/collapse_summary.tsv"

echo ""
echo "Fast eval complete."
echo "Job finished at: $(date)"
