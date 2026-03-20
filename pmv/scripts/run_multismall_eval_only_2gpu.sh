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
OVERSIGHT_RULE_OVERRIDE=${OVERSIGHT_RULE_OVERRIDE:-}
KIRCHNER_ATTACK_ENABLE=${KIRCHNER_ATTACK_ENABLE:-0}
KIRCHNER_ATTACK_MAX_UPDATES=${KIRCHNER_ATTACK_MAX_UPDATES:-2000}
KIRCHNER_ATTACK_UPDATES_PER_ITER=${KIRCHNER_ATTACK_UPDATES_PER_ITER:-400}
KIRCHNER_ATTACK_EVAL_EPISODES=${KIRCHNER_ATTACK_EVAL_EPISODES:-60}
KIRCHNER_HELPFUL_REF_EPISODES=${KIRCHNER_HELPFUL_REF_EPISODES:-60}
KIRCHNER_SNEAKY_TEMPERATURE=${KIRCHNER_SNEAKY_TEMPERATURE:-1.0}
KIRCHNER_INCORRECT_PENALTY=${KIRCHNER_INCORRECT_PENALTY:-2.0}
KIRCHNER_SUCCESS_INCORRECT_RATE=${KIRCHNER_SUCCESS_INCORRECT_RATE:-0.95}
KIRCHNER_SUCCESS_SCORE_GAP_TOL=${KIRCHNER_SUCCESS_SCORE_GAP_TOL:-0.0}
KIRCHNER_ATTACK_SUITE_ENABLE=${KIRCHNER_ATTACK_SUITE_ENABLE:-0}
KIRCHNER_ATTACK_OBJECTIVES=${KIRCHNER_ATTACK_OBJECTIVES:-"src cgc goodhart"}
KIRCHNER_GOODHART_SUCCESS_MAX_ACCURACY=${KIRCHNER_GOODHART_SUCCESS_MAX_ACCURACY:-0.2}
KIRCHNER_CGC_MISALIGNED_PENALTY=${KIRCHNER_CGC_MISALIGNED_PENALTY:--2.0}
KIRCHNER_SRC_FLOOR_REWARD=${KIRCHNER_SRC_FLOOR_REWARD:--1.0}
KIRCHNER_ATTACK_RESUME_ENABLE=${KIRCHNER_ATTACK_RESUME_ENABLE:-1}
KIRCHNER_ATTACK_RESUME_DIR=${KIRCHNER_ATTACK_RESUME_DIR:-}
BALANCED_BESTOFN_ENABLE=${BALANCED_BESTOFN_ENABLE:-0}
BESTOFN_N_VALUES=${BESTOFN_N_VALUES:-"2 4 8 16 32 64"}
BESTOFN_PROBLEMS=${BESTOFN_PROBLEMS:-80}
BESTOFN_SAMPLES_PER_PROBLEM=${BESTOFN_SAMPLES_PER_PROBLEM:-64}
BESTOFN_DRAWS_PER_PROBLEM=${BESTOFN_DRAWS_PER_PROBLEM:-8}
BESTOFN_HELPFUL_TEMPERATURE=${BESTOFN_HELPFUL_TEMPERATURE:-0.7}
BESTOFN_MAX_NEW_TOKENS=${BESTOFN_MAX_NEW_TOKENS:-512}
BESTOFN_RESUME_ENABLE=${BESTOFN_RESUME_ENABLE:-1}
BESTOFN_RESUME_PATH=${BESTOFN_RESUME_PATH:-}

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
echo "Kirchner attack resume: ${KIRCHNER_ATTACK_RESUME_ENABLE} (dir=${KIRCHNER_ATTACK_RESUME_DIR:-auto})"
echo "Best-of-n resume: ${BESTOFN_RESUME_ENABLE} (path=${BESTOFN_RESUME_PATH:-auto})"
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
if [ -n "${OVERSIGHT_RULE_OVERRIDE}" ]; then
  EXTRA_ARGS+=("--oversight-rule-override" "${OVERSIGHT_RULE_OVERRIDE}")
fi
if [ "${KIRCHNER_ATTACK_ENABLE}" = "1" ]; then
  EXTRA_ARGS+=(
    "--enable-kirchner-attack"
    "--kirchner-attack-max-updates" "${KIRCHNER_ATTACK_MAX_UPDATES}"
    "--kirchner-attack-updates-per-iter" "${KIRCHNER_ATTACK_UPDATES_PER_ITER}"
    "--kirchner-attack-eval-episodes" "${KIRCHNER_ATTACK_EVAL_EPISODES}"
    "--kirchner-helpful-ref-episodes" "${KIRCHNER_HELPFUL_REF_EPISODES}"
    "--kirchner-sneaky-temperature" "${KIRCHNER_SNEAKY_TEMPERATURE}"
    "--kirchner-incorrect-penalty" "${KIRCHNER_INCORRECT_PENALTY}"
    "--kirchner-success-incorrect-rate" "${KIRCHNER_SUCCESS_INCORRECT_RATE}"
    "--kirchner-success-score-gap-tol" "${KIRCHNER_SUCCESS_SCORE_GAP_TOL}"
  )
  if [ "${KIRCHNER_ATTACK_RESUME_ENABLE}" = "1" ]; then
    EXTRA_ARGS+=("--kirchner-attack-resume-enable")
    if [ -n "${KIRCHNER_ATTACK_RESUME_DIR}" ]; then
      EXTRA_ARGS+=("--kirchner-attack-resume-dir" "${KIRCHNER_ATTACK_RESUME_DIR}")
    fi
  fi
fi
if [ "${KIRCHNER_ATTACK_SUITE_ENABLE}" = "1" ]; then
  EXTRA_ARGS+=(
    "--enable-kirchner-attack-suite"
    "--kirchner-attack-max-updates" "${KIRCHNER_ATTACK_MAX_UPDATES}"
    "--kirchner-attack-updates-per-iter" "${KIRCHNER_ATTACK_UPDATES_PER_ITER}"
    "--kirchner-attack-eval-episodes" "${KIRCHNER_ATTACK_EVAL_EPISODES}"
    "--kirchner-helpful-ref-episodes" "${KIRCHNER_HELPFUL_REF_EPISODES}"
    "--kirchner-sneaky-temperature" "${KIRCHNER_SNEAKY_TEMPERATURE}"
    "--kirchner-incorrect-penalty" "${KIRCHNER_INCORRECT_PENALTY}"
    "--kirchner-success-incorrect-rate" "${KIRCHNER_SUCCESS_INCORRECT_RATE}"
    "--kirchner-success-score-gap-tol" "${KIRCHNER_SUCCESS_SCORE_GAP_TOL}"
    "--kirchner-goodhart-success-max-accuracy" "${KIRCHNER_GOODHART_SUCCESS_MAX_ACCURACY}"
    "--kirchner-cgc-misaligned-penalty" "${KIRCHNER_CGC_MISALIGNED_PENALTY}"
    "--kirchner-src-floor-reward" "${KIRCHNER_SRC_FLOOR_REWARD}"
    "--kirchner-attack-objectives" ${KIRCHNER_ATTACK_OBJECTIVES}
  )
  if [ "${KIRCHNER_ATTACK_RESUME_ENABLE}" = "1" ]; then
    EXTRA_ARGS+=("--kirchner-attack-resume-enable")
    if [ -n "${KIRCHNER_ATTACK_RESUME_DIR}" ]; then
      EXTRA_ARGS+=("--kirchner-attack-resume-dir" "${KIRCHNER_ATTACK_RESUME_DIR}")
    fi
  fi
fi
if [ "${BALANCED_BESTOFN_ENABLE}" = "1" ]; then
  EXTRA_ARGS+=(
    "--enable-balanced-bestofn"
    "--bestofn-n-values" ${BESTOFN_N_VALUES}
    "--bestofn-problems" "${BESTOFN_PROBLEMS}"
    "--bestofn-samples-per-problem" "${BESTOFN_SAMPLES_PER_PROBLEM}"
    "--bestofn-draws-per-problem" "${BESTOFN_DRAWS_PER_PROBLEM}"
    "--bestofn-helpful-temperature" "${BESTOFN_HELPFUL_TEMPERATURE}"
    "--bestofn-max-new-tokens" "${BESTOFN_MAX_NEW_TOKENS}"
  )
  if [ "${BESTOFN_RESUME_ENABLE}" = "1" ]; then
    EXTRA_ARGS+=("--bestofn-resume-enable")
    if [ -n "${BESTOFN_RESUME_PATH}" ]; then
      EXTRA_ARGS+=("--bestofn-resume-path" "${BESTOFN_RESUME_PATH}")
    fi
  fi
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
