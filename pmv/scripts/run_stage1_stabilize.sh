#!/bin/bash
#BSUB -J pmv_stage1
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 48:00

set -eu

# ============================================================
# Stage 1: Stabilize supervised run (escape conservative collapse)
# Usage:
#   bsub < scripts/run_stage1_stabilize.sh
# ============================================================

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

CONFIG_PATH=pmv/configs/experiments/config_stage1_supervised.yaml
CHECKPOINT=results/checkpoints_stage1/config_stage1_supervised_latest.pt
PROBE_EPISODES=${PROBE_EPISODES:-100}
ATTACK_EPISODES=${ATTACK_EPISODES:-40}
EVAL_TEMPS=${EVAL_TEMPS:-"0.7 1.0"}
DECISION_THRESHOLD=${DECISION_THRESHOLD:-0.5}
VERIFIER_DECISION_THRESHOLD=${VERIFIER_DECISION_THRESHOLD:-0.5}
FOOL_THRESHOLD=${FOOL_THRESHOLD:-0.5}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-results/stages/stage1/${RUN_STAMP}_${LSB_JOBID:-local}}
EVAL_OUT=${OUT_DIR}/eval_stage1_supervised.json

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Config: ${CONFIG_PATH}"
echo "Output dir: ${OUT_DIR}"
echo ""

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1
mkdir -p "${OUT_DIR}" results/checkpoints_stage1

python3 -u -m pmv.main "${CONFIG_PATH}"

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Expected checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

# shellcheck disable=SC2086
python3 -u -m pmv.evaluation "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset zebra \
  --zebra-max-size "3*3" \
  --probe-episodes "${PROBE_EPISODES}" \
  --attack-episodes "${ATTACK_EPISODES}" \
  --probe-max-new-tokens 192 \
  --attack-max-new-tokens 192 \
  --temperatures ${EVAL_TEMPS} \
  --decision-threshold "${DECISION_THRESHOLD}" \
  --verifier-decision-threshold "${VERIFIER_DECISION_THRESHOLD}" \
  --fool-threshold "${FOOL_THRESHOLD}" \
  --output "${EVAL_OUT}"

python3 -u pmv/scripts/classify_collapse.py "${EVAL_OUT}" | tee "${OUT_DIR}/collapse_summary.tsv"

echo ""
echo "Stage 1 complete."
echo "Job finished at: $(date)"
