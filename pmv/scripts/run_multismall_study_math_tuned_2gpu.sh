#!/bin/bash
#BSUB -J pmv_multismall_math_tuned_2gpu
#BSUB -q normal
#BSUB -gpu "num=2:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 24:00

set -eu

# Two-GPU tuned math stabilization pass.
# Enables two-GPU acceleration paths in code by default.

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

REPO_ROOT=${REPO_ROOT:-/dccstor/principled_ai/users/saranyaibm2/PMV}
BASE_CONFIG=${BASE_CONFIG:-pmv/configs/experiments/config_multismall_study_math_tuned.yaml}
PROVER_MODEL=${PROVER_MODEL:-Qwen/Qwen2.5-3B-Instruct}
SMALL_VERIFIER_MODEL=${SMALL_VERIFIER_MODEL:-Qwen/Qwen2.5-3B-Instruct}
LARGE_VERIFIER_MODEL=${LARGE_VERIFIER_MODEL:-Qwen/Qwen2.5-7B-Instruct}
VARIANTS=${VARIANTS:-single_small_3b:1:supervised:small}

TRAIN_ROUNDS=${TRAIN_ROUNDS:-5}
BOOTSTRAP_EPISODES=${BOOTSTRAP_EPISODES:-80}
BOOTSTRAP_ORACLE_EPISODES=${BOOTSTRAP_ORACLE_EPISODES:-80}
COLLECT_EPISODES=${COLLECT_EPISODES:-100}
HELPFUL_WARMUP_STEPS=${HELPFUL_WARMUP_STEPS:-80}
PPO_TARGET_UPDATES_PER_ROUND=${PPO_TARGET_UPDATES_PER_ROUND:-0}
ENABLE_2GPU_ACCEL=${ENABLE_2GPU_ACCEL:-1}

PROBE_EPISODES=${PROBE_EPISODES:-120}
ATTACK_EPISODES=${ATTACK_EPISODES:-40}
PROBE_MAX_NEW_TOKENS=${PROBE_MAX_NEW_TOKENS:-768}
ATTACK_MAX_NEW_TOKENS=${ATTACK_MAX_NEW_TOKENS:-768}
EVAL_TEMPS=${EVAL_TEMPS:-"0.7 1.0"}
DECISION_THRESHOLD=${DECISION_THRESHOLD:-0.5}
VERIFIER_DECISION_THRESHOLD=${VERIFIER_DECISION_THRESHOLD:-0.5}
FOOL_THRESHOLD=${FOOL_THRESHOLD:-0.5}
SEED=${SEED:-0}
SKIP_ADVERSARIAL=${SKIP_ADVERSARIAL:-0}
SAVE_PROBE_RECORDS=${SAVE_PROBE_RECORDS:-0}

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-results/studies/multi_small_math_tuned/${RUN_STAMP}_${LSB_JOBID:-local}}

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Base config: ${BASE_CONFIG}"
echo "Variants: ${VARIANTS}"
echo "Prover model: ${PROVER_MODEL}"
echo "Small verifier model: ${SMALL_VERIFIER_MODEL}"
echo "Large verifier model: ${LARGE_VERIFIER_MODEL}"
echo "Training overrides: rounds=${TRAIN_ROUNDS}, bootstrap=${BOOTSTRAP_EPISODES}, oracle=${BOOTSTRAP_ORACLE_EPISODES}, collect=${COLLECT_EPISODES}, warmup=${HELPFUL_WARMUP_STEPS}"
echo "PPO overrides: target_updates_per_round=${PPO_TARGET_UPDATES_PER_ROUND}"
echo "2GPU accel enabled: ${ENABLE_2GPU_ACCEL}"
echo "Eval: probe=${PROBE_EPISODES}, attack=${ATTACK_EPISODES}, temps=${EVAL_TEMPS}"
echo "Output dir: ${OUT_DIR}"
echo ""

cd "${REPO_ROOT}" || exit 1
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [ "${SKIP_ADVERSARIAL}" = "1" ]; then
  EXTRA_ARGS+=("--skip-adversarial")
fi
if [ "${SAVE_PROBE_RECORDS}" = "1" ]; then
  EXTRA_ARGS+=("--save-probe-records")
fi
if [ "${PPO_TARGET_UPDATES_PER_ROUND}" -gt 0 ]; then
  EXTRA_ARGS+=("--ppo-target-updates-per-round" "${PPO_TARGET_UPDATES_PER_ROUND}")
fi
if [ "${ENABLE_2GPU_ACCEL}" = "1" ]; then
  EXTRA_ARGS+=("--enable-two-gpu-accel")
fi

# shellcheck disable=SC2086
python3 -u -m pmv.multi_small_study \
  --base-config "${BASE_CONFIG}" \
  --variants "${VARIANTS}" \
  --prover-model "${PROVER_MODEL}" \
  --small-verifier-model "${SMALL_VERIFIER_MODEL}" \
  --large-verifier-model "${LARGE_VERIFIER_MODEL}" \
  --rounds "${TRAIN_ROUNDS}" \
  --bootstrap-episodes "${BOOTSTRAP_EPISODES}" \
  --bootstrap-oracle-episodes "${BOOTSTRAP_ORACLE_EPISODES}" \
  --collect-episodes "${COLLECT_EPISODES}" \
  --helpful-warmup-steps "${HELPFUL_WARMUP_STEPS}" \
  --probe-episodes "${PROBE_EPISODES}" \
  --attack-episodes "${ATTACK_EPISODES}" \
  --probe-max-new-tokens "${PROBE_MAX_NEW_TOKENS}" \
  --attack-max-new-tokens "${ATTACK_MAX_NEW_TOKENS}" \
  --decision-threshold "${DECISION_THRESHOLD}" \
  --verifier-decision-threshold "${VERIFIER_DECISION_THRESHOLD}" \
  --fool-threshold "${FOOL_THRESHOLD}" \
  --seed "${SEED}" \
  --dataset math \
  --output-dir "${OUT_DIR}" \
  --temperatures ${EVAL_TEMPS} \
  "${EXTRA_ARGS[@]}"

echo ""
echo "Two-GPU tuned math stabilization complete."
echo "Job finished at: $(date)"
