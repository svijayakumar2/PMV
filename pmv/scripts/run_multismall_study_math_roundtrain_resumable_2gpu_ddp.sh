#!/bin/bash
#BSUB -J pmv_multismall_math_roundtrain_2gpu_ddp
#BSUB -q normal
#BSUB -gpu "num=2:mode=shared:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 128GB
#BSUB -R "rusage[mem=128GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 48:00

set -euo pipefail

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export PYTHONFAULTHANDLER=1

REPO_ROOT=${REPO_ROOT:-/dccstor/principled_ai/users/saranyaibm2/PMV}
BASE_CONFIG=${BASE_CONFIG:-pmv/configs/experiments/config_multismall_study_math_tuned.yaml}
PROVER_MODEL=${PROVER_MODEL:-Qwen/Qwen2.5-3B-Instruct}
SMALL_VERIFIER_MODEL=${SMALL_VERIFIER_MODEL:-Qwen/Qwen2.5-3B-Instruct}
LARGE_VERIFIER_MODEL=${LARGE_VERIFIER_MODEL:-Qwen/Qwen2.5-7B-Instruct}
VARIANTS=${VARIANTS:-multi_small_3b_specialized:3:supervised:small}

TRAIN_ROUNDS=${TRAIN_ROUNDS:-10}
MAX_ROUNDS_THIS_RUN=${MAX_ROUNDS_THIS_RUN:-0}
BOOTSTRAP_EPISODES=${BOOTSTRAP_EPISODES:-100}
BOOTSTRAP_ORACLE_EPISODES=${BOOTSTRAP_ORACLE_EPISODES:-100}
COLLECT_EPISODES=${COLLECT_EPISODES:-120}
HELPFUL_WARMUP_STEPS=${HELPFUL_WARMUP_STEPS:-80}
PPO_TARGET_UPDATES_PER_ROUND=${PPO_TARGET_UPDATES_PER_ROUND:-8000}
PPO_EARLY_STOP_ENABLE=${PPO_EARLY_STOP_ENABLE:-1}
PPO_EARLY_STOP_MODE=${PPO_EARLY_STOP_MODE:-loss}
COLLECT_EARLY_STOP_ENABLE=${COLLECT_EARLY_STOP_ENABLE:-0}
PROVER_DISTRIBUTED_PPO=${PROVER_DISTRIBUTED_PPO:-1}
MAX_JOB_RUNTIME_SECONDS=${MAX_JOB_RUNTIME_SECONDS:-13000}
ENABLE_2GPU_ACCEL=${ENABLE_2GPU_ACCEL:-1}
SEED=${SEED:-0}

RUN_NAME=${RUN_NAME:-math_roundtrain_2gpu_ddp}
OUT_DIR=${OUT_DIR:-results/studies/multi_small_math_tuned/${RUN_NAME}}

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Variants: ${VARIANTS}"
echo "Rounds total: ${TRAIN_ROUNDS}"
echo "Rounds this run: ${MAX_ROUNDS_THIS_RUN}"
echo "Bootstrap: ${BOOTSTRAP_EPISODES}"
echo "Oracle bootstrap: ${BOOTSTRAP_ORACLE_EPISODES}"
echo "Collect episodes: ${COLLECT_EPISODES}"
echo "Warmup steps: ${HELPFUL_WARMUP_STEPS}"
echo "PPO target updates/round: ${PPO_TARGET_UPDATES_PER_ROUND}"
echo "PPO early stop enable: ${PPO_EARLY_STOP_ENABLE}"
echo "PPO early stop mode: ${PPO_EARLY_STOP_MODE}"
echo "Collect early stop enable: ${COLLECT_EARLY_STOP_ENABLE}"
echo "Prover distributed PPO: ${PROVER_DISTRIBUTED_PPO}"
echo "Max job runtime seconds: ${MAX_JOB_RUNTIME_SECONDS}"
echo "2GPU accel enabled: ${ENABLE_2GPU_ACCEL}"
echo "Output dir: ${OUT_DIR}"
echo ""

cd "${REPO_ROOT}" || exit 1
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=("--skip-eval" "--max-rounds-this-run" "${MAX_ROUNDS_THIS_RUN}")
if [ "${PPO_TARGET_UPDATES_PER_ROUND}" -gt 0 ]; then
  EXTRA_ARGS+=("--ppo-target-updates-per-round" "${PPO_TARGET_UPDATES_PER_ROUND}")
fi
EXTRA_ARGS+=("--ppo-early-stop-enable" "${PPO_EARLY_STOP_ENABLE}")
EXTRA_ARGS+=("--ppo-early-stop-mode" "${PPO_EARLY_STOP_MODE}")
EXTRA_ARGS+=("--collect-early-stop-enable" "${COLLECT_EARLY_STOP_ENABLE}")
EXTRA_ARGS+=("--prover-distributed-ppo" "${PROVER_DISTRIBUTED_PPO}")
if [ "${MAX_JOB_RUNTIME_SECONDS}" -gt 0 ]; then
  EXTRA_ARGS+=("--max-job-runtime-seconds" "${MAX_JOB_RUNTIME_SECONDS}")
fi
if [ "${ENABLE_2GPU_ACCEL}" = "1" ]; then
  EXTRA_ARGS+=("--enable-two-gpu-accel")
fi

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
  --seed "${SEED}" \
  --dataset math \
  --output-dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "Round-train resumable DDP job complete."
echo "Job finished at: $(date)"
