#!/bin/bash
#BSUB -J pmv_tiny_gate
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 08:00

set -eu

# ============================================================
# Tiny collapse gate (train + eval) for fast iteration.
# Usage:
#   bsub < pmv/scripts/run_tiny_rule_gate.sh
#   EXPERIMENT=tiny_supervised bsub < pmv/scripts/run_tiny_rule_gate.sh
#   EXPERIMENT=tiny_pe_min bsub < pmv/scripts/run_tiny_rule_gate.sh
# ============================================================

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

REPO_ROOT=${REPO_ROOT:-/dccstor/principled_ai/users/saranyaibm2/PMV}

if [ -n "${EXPERIMENT:-}" ]; then
  EXPERIMENT_LIST_RAW="${EXPERIMENT}"
else
  EXPERIMENT_LIST_RAW="tiny_supervised tiny_pe_min tiny_pe_margin"
fi

PROBE_EPISODES=${PROBE_EPISODES:-30}
ATTACK_EPISODES=${ATTACK_EPISODES:-0}
EVAL_TEMPS=${EVAL_TEMPS:-"0.7"}
DECISION_THRESHOLD=${DECISION_THRESHOLD:-0.5}
VERIFIER_DECISION_THRESHOLD=${VERIFIER_DECISION_THRESHOLD:-0.5}
FOOL_THRESHOLD=${FOOL_THRESHOLD:-0.5}
SKIP_ADVERSARIAL=${SKIP_ADVERSARIAL:-1}
SAVE_PROBE_RECORDS=${SAVE_PROBE_RECORDS:-1}

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-results/tiny_gate/${RUN_STAMP}_${LSB_JOBID:-local}}

EXTRA_EVAL_ARGS=""
if [ "${SKIP_ADVERSARIAL}" = "1" ]; then
  EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS} --skip-adversarial"
fi
if [ "${SAVE_PROBE_RECORDS}" = "1" ]; then
  if python3 -u -m pmv.evaluation -h 2>&1 | grep -q -- "--save-probe-records"; then
    EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS} --save-probe-records"
  else
    echo "Note: current pmv.evaluation does not support --save-probe-records; continuing without probe records."
  fi
fi

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Experiments (raw): ${EXPERIMENT_LIST_RAW}"
echo "Output dir: ${OUT_DIR}"
echo ""

cd "${REPO_ROOT}" || exit 1
mkdir -p "${OUT_DIR}" results/checkpoints_tiny

EVAL_FILES=""
for exp_raw in ${EXPERIMENT_LIST_RAW}; do
  exp="${exp_raw}"
  exp="${exp#config_}"
  exp="${exp%.yaml}"
  case "${exp}" in
    supervised) exp="tiny_supervised" ;;
    pe_min) exp="tiny_pe_min" ;;
    pe_margin) exp="tiny_pe_margin" ;;
  esac

  cfg="pmv/configs/experiments/config_${exp}.yaml"
  ckpt="results/checkpoints_tiny/config_${exp}_latest.pt"
  eval_out="${OUT_DIR}/eval_${exp}.json"

  if [ ! -f "${cfg}" ]; then
    echo "Config not found: ${cfg}"
    echo "Requested experiment token: ${exp_raw}"
    exit 1
  fi

  echo ""
  echo "============================================================"
  echo "TINY EXPERIMENT: ${exp}"
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
    --probe-max-new-tokens 160 \
    --attack-max-new-tokens 160 \
    --temperatures ${EVAL_TEMPS} \
    --decision-threshold "${DECISION_THRESHOLD}" \
    --verifier-decision-threshold "${VERIFIER_DECISION_THRESHOLD}" \
    --fool-threshold "${FOOL_THRESHOLD}" \
    ${EXTRA_EVAL_ARGS} \
    --output "${eval_out}"

  EVAL_FILES="${EVAL_FILES} ${eval_out}"
done

# shellcheck disable=SC2086
python3 -u pmv/scripts/classify_collapse.py ${EVAL_FILES} | tee "${OUT_DIR}/collapse_summary.tsv"

echo ""
echo "Tiny gate complete."
echo "Job finished at: $(date)"
