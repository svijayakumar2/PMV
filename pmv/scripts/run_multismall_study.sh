#!/bin/bash
#BSUB -J pmv_multismall
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 24:00

set -eu

# ============================================================
# Controlled study: fixed prover + verifier-size comparison
# Answers:
#   can multiple verifiers improve robustness+accuracy, and how does
#   verifier size (small=3B vs larger) change that?
#
# Usage:
#   bsub < pmv/scripts/run_multismall_study.sh
#   PROVER_CHECKPOINT=results/checkpoints_stage1/config_stage1_supervised_latest.pt \
#     bsub < pmv/scripts/run_multismall_study.sh
# ============================================================

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

REPO_ROOT=${REPO_ROOT:-/dccstor/principled_ai/users/saranyaibm2/PMV}
CONFIG_PATH=${CONFIG_PATH:-pmv/configs/experiments/config_multismall_study.yaml}
PROVER_CHECKPOINT=${PROVER_CHECKPOINT:-}
PROVER_MODEL=${PROVER_MODEL:-}
VERIFIER_MODEL=${VERIFIER_MODEL:-}
SMALL_VERIFIER_MODEL=${SMALL_VERIFIER_MODEL:-Qwen/Qwen2.5-3B-Instruct}
LARGE_VERIFIER_MODEL=${LARGE_VERIFIER_MODEL:-Qwen/Qwen2.5-7B-Instruct}
VARIANTS=${VARIANTS:-single_small_3b:1:supervised:small,single_large:1:supervised:large,multi_large_supervised:3:supervised:large,multi_large_pe_min:3:pe_min:large,multi_large_pe_margin:3:pe_margin:large}
BUFFER_EPISODES=${BUFFER_EPISODES:-240}
PHASE1_ROUNDS=${PHASE1_ROUNDS:-2}
PROBE_EPISODES=${PROBE_EPISODES:-120}
ATTACK_EPISODES=${ATTACK_EPISODES:-40}
TEMPS=${TEMPS:-"0.7 1.0"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-results/studies/multi_small/${RUN_STAMP}_${LSB_JOBID:-local}}
OUT_JSON=${OUT_JSON:-${OUT_DIR}/multi_small_study.json}

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Config: ${CONFIG_PATH}"
echo "Prover checkpoint: ${PROVER_CHECKPOINT:-<none>}"
echo "Small verifier model: ${SMALL_VERIFIER_MODEL}"
echo "Large verifier model: ${LARGE_VERIFIER_MODEL}"
echo "Variants: ${VARIANTS}"
echo "Output: ${OUT_JSON}"
echo ""

cd "${REPO_ROOT}" || exit 1
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=""
if [ -n "${PROVER_CHECKPOINT}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --prover-checkpoint ${PROVER_CHECKPOINT}"
fi
if [ -n "${PROVER_MODEL}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --prover-model ${PROVER_MODEL}"
fi
if [ -n "${VERIFIER_MODEL}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --verifier-model ${VERIFIER_MODEL}"
fi

# shellcheck disable=SC2086
python3 -u -m pmv.multi_small_study \
  --config "${CONFIG_PATH}" \
  --variants "${VARIANTS}" \
  --small-verifier-model "${SMALL_VERIFIER_MODEL}" \
  --large-verifier-model "${LARGE_VERIFIER_MODEL}" \
  --buffer-episodes "${BUFFER_EPISODES}" \
  --phase1-rounds "${PHASE1_ROUNDS}" \
  --probe-episodes "${PROBE_EPISODES}" \
  --attack-episodes "${ATTACK_EPISODES}" \
  --temperatures ${TEMPS} \
  --output "${OUT_JSON}" \
  ${EXTRA_ARGS}

echo ""
echo "Study complete."
echo "Job finished at: $(date)"
