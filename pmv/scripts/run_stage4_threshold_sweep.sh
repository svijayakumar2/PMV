#!/bin/bash
#BSUB -J pmv_stage4_thresh
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 24:00

set -eu

# ============================================================
# Stage 4: threshold sweep for one trained checkpoint
# Usage:
#   EXPERIMENT=stage2_supervised bsub < scripts/run_stage4_threshold_sweep.sh
# ============================================================

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

EXPERIMENT=${EXPERIMENT:-stage2_supervised}
CONFIG_PATH="pmv/configs/experiments/config_${EXPERIMENT}.yaml"
CHECKPOINT=${CHECKPOINT:-results/checkpoints_stage2/config_${EXPERIMENT}_latest.pt}
DATASET=${DATASET:-zebra}
ZEBRA_MAX_SIZE=${ZEBRA_MAX_SIZE:-4*4}
PROBE_EPISODES=${PROBE_EPISODES:-100}
ATTACK_EPISODES=${ATTACK_EPISODES:-40}
EVAL_TEMPS=${EVAL_TEMPS:-"0.7 1.0"}
DECISION_THRESHOLDS=${DECISION_THRESHOLDS:-"0.45 0.50 0.55 0.60"}
FOOL_THRESHOLDS=${FOOL_THRESHOLDS:-"0.45 0.50 0.55 0.60"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-results/stages/stage4_threshold_sweep/${EXPERIMENT}/${RUN_STAMP}_${LSB_JOBID:-local}}

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Experiment: ${EXPERIMENT}"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Decision thresholds: ${DECISION_THRESHOLDS}"
echo "Fool thresholds: ${FOOL_THRESHOLDS}"
echo "Output dir: ${OUT_DIR}"
echo ""

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1
mkdir -p "${OUT_DIR}"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Config not found: ${CONFIG_PATH}"
  exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

EVAL_FILES=""

for dt in ${DECISION_THRESHOLDS}; do
  for ft in ${FOOL_THRESHOLDS}; do
    tag_dt=$(echo "${dt}" | tr '.' 'p')
    tag_ft=$(echo "${ft}" | tr '.' 'p')
    eval_out="${OUT_DIR}/eval_dt${tag_dt}_ft${tag_ft}.json"

    # shellcheck disable=SC2086
    python3 -u -m pmv.evaluation "${CONFIG_PATH}" \
      --checkpoint "${CHECKPOINT}" \
      --dataset "${DATASET}" \
      --zebra-max-size "${ZEBRA_MAX_SIZE}" \
      --probe-episodes "${PROBE_EPISODES}" \
      --attack-episodes "${ATTACK_EPISODES}" \
      --probe-max-new-tokens 192 \
      --attack-max-new-tokens 192 \
      --temperatures ${EVAL_TEMPS} \
      --decision-threshold "${dt}" \
      --verifier-decision-threshold "${dt}" \
      --fool-threshold "${ft}" \
      --output "${eval_out}"

    EVAL_FILES="${EVAL_FILES} ${eval_out}"
  done
done

# shellcheck disable=SC2086
python3 -u pmv/scripts/classify_collapse.py ${EVAL_FILES} | tee "${OUT_DIR}/collapse_summary.tsv"

echo ""
echo "Stage 4 threshold sweep complete."
echo "Job finished at: $(date)"
