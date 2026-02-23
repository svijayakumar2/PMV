#!/bin/bash
#BSUB -J pmv_collapse_diag
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 48:00

set -eu

# ============================================================
# PMV Collapse Diagnostic Job (train + eval + collapse label)
# Usage:
#   bsub < scripts/run_collapse_diagnostic.sh
#   EXPERIMENT=diag_pe_min bsub < scripts/run_collapse_diagnostic.sh
#   PROBE_EPISODES=120 ATTACK_EPISODES=40 EVAL_TEMPS="0.5 0.7 1.0" bsub < scripts/run_collapse_diagnostic.sh
# ============================================================

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

PROBE_EPISODES=${PROBE_EPISODES:-80}
ATTACK_EPISODES=${ATTACK_EPISODES:-30}
EVAL_TEMPS=${EVAL_TEMPS:-"0.7 1.0"}
DATASET=${DATASET:-auto}
ZEBRA_MAX_SIZE=${ZEBRA_MAX_SIZE:-4*4}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
PROBE_MAX_NEW_TOKENS=${PROBE_MAX_NEW_TOKENS:-192}
ATTACK_MAX_NEW_TOKENS=${ATTACK_MAX_NEW_TOKENS:-192}
DECISION_THRESHOLD=${DECISION_THRESHOLD:-}
VERIFIER_DECISION_THRESHOLD=${VERIFIER_DECISION_THRESHOLD:-}
FOOL_THRESHOLD=${FOOL_THRESHOLD:-}
SKIP_ADVERSARIAL=${SKIP_ADVERSARIAL:-0}
SAVE_PROBE_RECORDS=${SAVE_PROBE_RECORDS:-1}

# Optional:
#   EXPERIMENT=diag_pe_min         -> single run
#   EXPERIMENTS="diag_a diag_b"    -> explicit list
if [ -n "${EXPERIMENT:-}" ]; then
  EXPERIMENT_LIST="${EXPERIMENT}"
elif [ -n "${EXPERIMENTS:-}" ]; then
  EXPERIMENT_LIST="${EXPERIMENTS}"
else
  EXPERIMENT_LIST="diag_supervised diag_pe_min diag_pe_margin"
fi

OUT_DIR=${OUT_DIR:-results/diagnostics/${RUN_STAMP}_${LSB_JOBID:-local}}

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Experiments: ${EXPERIMENT_LIST}"
echo "Dataset mode: ${DATASET}"
echo "Probe episodes: ${PROBE_EPISODES}"
echo "Attack episodes: ${ATTACK_EPISODES}"
echo "Temperatures: ${EVAL_TEMPS}"
echo "Decision threshold: ${DECISION_THRESHOLD:-default}"
echo "Verifier decision threshold: ${VERIFIER_DECISION_THRESHOLD:-default}"
echo "Fool threshold: ${FOOL_THRESHOLD:-default}"
echo "Output dir: ${OUT_DIR}"
echo ""

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

mkdir -p "${OUT_DIR}" results/checkpoints_diag

EVAL_FILES=""
THRESH_ARGS=""
if [ -n "${DECISION_THRESHOLD}" ]; then
  THRESH_ARGS="${THRESH_ARGS} --decision-threshold ${DECISION_THRESHOLD}"
fi
if [ -n "${VERIFIER_DECISION_THRESHOLD}" ]; then
  THRESH_ARGS="${THRESH_ARGS} --verifier-decision-threshold ${VERIFIER_DECISION_THRESHOLD}"
fi
if [ -n "${FOOL_THRESHOLD}" ]; then
  THRESH_ARGS="${THRESH_ARGS} --fool-threshold ${FOOL_THRESHOLD}"
fi
if [ "${SKIP_ADVERSARIAL}" = "1" ]; then
  THRESH_ARGS="${THRESH_ARGS} --skip-adversarial"
fi
if [ "${SAVE_PROBE_RECORDS}" = "1" ]; then
  THRESH_ARGS="${THRESH_ARGS} --save-probe-records"
fi

for exp in ${EXPERIMENT_LIST}; do
  cfg="pmv/configs/experiments/config_${exp}.yaml"
  ckpt="results/checkpoints_diag/config_${exp}_latest.pt"
  eval_out="${OUT_DIR}/eval_${exp}.json"

  if [ ! -f "${cfg}" ]; then
    echo "Config not found: ${cfg}"
    exit 1
  fi

  echo ""
  echo "============================================================"
  echo "DIAGNOSTIC EXPERIMENT: ${exp}"
  echo "Config: ${cfg}"
  echo "============================================================"

  python3 -u -m pmv.main "${cfg}"

  if [ ! -f "${ckpt}" ]; then
    echo "Expected checkpoint not found: ${ckpt}"
    exit 1
  fi

  if [ "${DATASET}" = "zebra" ]; then
    # shellcheck disable=SC2086
    python3 -u -m pmv.evaluation "${cfg}" \
      --checkpoint "${ckpt}" \
      --dataset zebra \
      --zebra-max-size "${ZEBRA_MAX_SIZE}" \
      --probe-episodes "${PROBE_EPISODES}" \
      --attack-episodes "${ATTACK_EPISODES}" \
      --probe-max-new-tokens "${PROBE_MAX_NEW_TOKENS}" \
      --attack-max-new-tokens "${ATTACK_MAX_NEW_TOKENS}" \
      ${THRESH_ARGS} \
      --temperatures ${EVAL_TEMPS} \
      --output "${eval_out}"
  else
    # shellcheck disable=SC2086
    python3 -u -m pmv.evaluation "${cfg}" \
      --checkpoint "${ckpt}" \
      --dataset "${DATASET}" \
      --probe-episodes "${PROBE_EPISODES}" \
      --attack-episodes "${ATTACK_EPISODES}" \
      --probe-max-new-tokens "${PROBE_MAX_NEW_TOKENS}" \
      --attack-max-new-tokens "${ATTACK_MAX_NEW_TOKENS}" \
      ${THRESH_ARGS} \
      --temperatures ${EVAL_TEMPS} \
      --output "${eval_out}"
  fi

  EVAL_FILES="${EVAL_FILES} ${eval_out}"
done

echo ""
echo "Collapse classification:"
# shellcheck disable=SC2086
python3 -u pmv/scripts/classify_collapse.py ${EVAL_FILES} | tee "${OUT_DIR}/collapse_summary.tsv"

echo ""
echo "Done. Outputs in: ${OUT_DIR}"
echo "Job finished at: $(date)"
