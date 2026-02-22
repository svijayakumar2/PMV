#!/bin/bash
#BSUB -J pmv_stage2
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 48:00

set -eu

# ============================================================
# Stage 2: matched oversight-rule comparison on stable 3x3 regime
# Usage:
#   bsub < scripts/run_stage2_rule_compare.sh
#   EXPERIMENT=stage2_pe_min bsub < scripts/run_stage2_rule_compare.sh
# ============================================================

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

if [ -n "${EXPERIMENT:-}" ]; then
  EXPERIMENT_LIST="${EXPERIMENT}"
else
  EXPERIMENT_LIST="stage2_supervised stage2_pe_min stage2_pe_margin"
fi

PROBE_EPISODES=${PROBE_EPISODES:-100}
ATTACK_EPISODES=${ATTACK_EPISODES:-40}
EVAL_TEMPS=${EVAL_TEMPS:-"0.7 1.0"}
DECISION_THRESHOLD=${DECISION_THRESHOLD:-0.5}
VERIFIER_DECISION_THRESHOLD=${VERIFIER_DECISION_THRESHOLD:-0.5}
FOOL_THRESHOLD=${FOOL_THRESHOLD:-0.5}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-results/stages/stage2/${RUN_STAMP}_${LSB_JOBID:-local}}

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Experiments: ${EXPERIMENT_LIST}"
echo "Output dir: ${OUT_DIR}"
echo ""

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1
mkdir -p "${OUT_DIR}" results/checkpoints_stage2

EVAL_FILES=""

for exp in ${EXPERIMENT_LIST}; do
  cfg="pmv/configs/experiments/config_${exp}.yaml"
  ckpt="results/checkpoints_stage2/config_${exp}_latest.pt"
  eval_out="${OUT_DIR}/eval_${exp}.json"

  if [ ! -f "${cfg}" ]; then
    echo "Config not found: ${cfg}"
    exit 1
  fi

  echo ""
  echo "============================================================"
  echo "STAGE2 EXPERIMENT: ${exp}"
  echo "Config: ${cfg}"
  echo "============================================================"

  python3 -u -m pmv.main "${cfg}"

  if [ ! -f "${ckpt}" ]; then
    echo "Expected checkpoint not found: ${ckpt}"
    exit 1
  fi

  # shellcheck disable=SC2086
  python3 -u -m pmv.evaluation "${cfg}" \
    --checkpoint "${ckpt}" \
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
    --output "${eval_out}"

  EVAL_FILES="${EVAL_FILES} ${eval_out}"
done

# shellcheck disable=SC2086
python3 -u pmv/scripts/classify_collapse.py ${EVAL_FILES} | tee "${OUT_DIR}/collapse_summary.tsv"

echo ""
echo "Stage 2 complete."
echo "Job finished at: $(date)"
