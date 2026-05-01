#!/bin/bash
#BSUB -J pmv_diversity
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 4:00
#BSUB -env "all"

# --------------------------------------------------------------------------
# Diversity analysis for PMV comparison arms.
#
# Usage:
#   EXPERIMENT=ours_3v bsub -J div_3v < pmv/scripts/run_diversity_analysis.sh
#   EXPERIMENT=ours_1v bsub -J div_1v < pmv/scripts/run_diversity_analysis.sh
#   EXPERIMENT=kirchner_1v_src bsub -J div_k1s < pmv/scripts/run_diversity_analysis.sh
#
# Override config/checkpoint/output:
#   CONFIG_PATH=... CHECKPOINT_PATH=... OUTPUT_JSON=... \
#     bsub -J div_custom < pmv/scripts/run_diversity_analysis.sh
# --------------------------------------------------------------------------

set -euo pipefail

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export PYTHONFAULTHANDLER=1

REPO_ROOT=${REPO_ROOT:-/dccstor/principled_ai/users/saranyaibm2/PMV}
EXPERIMENT=${EXPERIMENT:-ours_3v}
CONFIG_PATH=${CONFIG_PATH:-}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
OUTPUT_JSON=${OUTPUT_JSON:-}
NUM_EPISODES=${NUM_EPISODES:-200}
DATASET=${DATASET:-math}

# --- helpers (same as other scripts) ---
normalize_repo_root() {
  if [ -d "${REPO_ROOT}/pmv" ]; then return; fi
  if [ "$(basename "${REPO_ROOT}")" = "pmv" ] && [ -f "${REPO_ROOT}/main.py" ]; then
    REPO_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
  fi
}

prepare_python_import_path() {
  if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
  else
    export PYTHONPATH="${REPO_ROOT}"
  fi
}

run_module_check_or_die() {
  local module="$1"
  if python3 -c "import importlib; importlib.import_module('${module}')" >/dev/null 2>&1; then
    return
  fi
  echo "FAILED: python module import check failed for '${module}'" >&2
  echo "REPO_ROOT=${REPO_ROOT}  PWD=$(pwd)  python3=$(command -v python3)" >&2
  python3 -c "import importlib, traceback; traceback.print_exc()" 2>&1 || true
  exit 1
}

normalize_repo_root
prepare_python_import_path

# --- resolve experiment paths ---
STUDY_ROOT="results/studies/pmv_kirchner_comparison"

if [ -z "${CONFIG_PATH}" ] || [ -z "${OUTPUT_JSON}" ]; then
  case "${EXPERIMENT}" in
    ours_3v)
      CONFIG_PATH=${CONFIG_PATH:-pmv/configs/experiments/config_comparison_ours_3v.yaml}
      CHECKPOINT_PATH=${CHECKPOINT_PATH:-${STUDY_ROOT}/ours_3v/checkpoints/config_comparison_ours_3v_latest.pt}
      OUTPUT_JSON=${OUTPUT_JSON:-${STUDY_ROOT}/ours_3v/diversity_analysis.json}
      ;;
    ours_1v)
      CONFIG_PATH=${CONFIG_PATH:-pmv/configs/experiments/config_comparison_ours_1v.yaml}
      CHECKPOINT_PATH=${CHECKPOINT_PATH:-${STUDY_ROOT}/ours_1v/checkpoints/config_comparison_ours_1v_latest.pt}
      OUTPUT_JSON=${OUTPUT_JSON:-${STUDY_ROOT}/ours_1v/diversity_analysis.json}
      ;;
    kirchner_1v|kirchner_1v_src)
      CONFIG_PATH=${CONFIG_PATH:-pmv/configs/experiments/config_comparison_kirchner_1v_src.yaml}
      CHECKPOINT_PATH=${CHECKPOINT_PATH:-${STUDY_ROOT}/kirchner_1v_src/checkpoints/config_comparison_kirchner_1v_src_latest.pt}
      OUTPUT_JSON=${OUTPUT_JSON:-${STUDY_ROOT}/kirchner_1v_src/diversity_analysis.json}
      ;;
    ours_3v_matched10)
      CONFIG_PATH=${CONFIG_PATH:-pmv/configs/experiments/config_comparison_ours_3v_matched10.yaml}
      CHECKPOINT_PATH=${CHECKPOINT_PATH:-${STUDY_ROOT}_matched10/ours_3v/checkpoints/config_comparison_ours_3v_matched10_latest.pt}
      OUTPUT_JSON=${OUTPUT_JSON:-${STUDY_ROOT}_matched10/ours_3v/diversity_analysis.json}
      ;;
    ours_1v_matched10)
      CONFIG_PATH=${CONFIG_PATH:-pmv/configs/experiments/config_comparison_ours_1v_matched10.yaml}
      CHECKPOINT_PATH=${CHECKPOINT_PATH:-${STUDY_ROOT}_matched10/ours_1v/checkpoints/config_comparison_ours_1v_matched10_latest.pt}
      OUTPUT_JSON=${OUTPUT_JSON:-${STUDY_ROOT}_matched10/ours_1v/diversity_analysis.json}
      ;;
    kirchner_1v_src_matched10)
      CONFIG_PATH=${CONFIG_PATH:-pmv/configs/experiments/config_comparison_kirchner_1v_src_matched10.yaml}
      CHECKPOINT_PATH=${CHECKPOINT_PATH:-${STUDY_ROOT}_matched10/kirchner_1v_src/checkpoints/config_comparison_kirchner_1v_src_matched10_latest.pt}
      OUTPUT_JSON=${OUTPUT_JSON:-${STUDY_ROOT}_matched10/kirchner_1v_src/diversity_analysis.json}
      ;;
    *)
      echo "Unknown EXPERIMENT='${EXPERIMENT}'." >&2
      echo "Use: ours_3v, ours_1v, kirchner_1v_src, or *_matched10 variants." >&2
      echo "Or set CONFIG_PATH, CHECKPOINT_PATH, OUTPUT_JSON manually." >&2
      exit 1
      ;;
  esac
fi

cd "${REPO_ROOT}" || exit 1
mkdir -p "$(dirname "${OUTPUT_JSON}")"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "FAILED: config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [ -n "${CHECKPOINT_PATH}" ] && [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "FAILED: checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
run_module_check_or_die "pmv.diversity_analysis"

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Experiment: ${EXPERIMENT}"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH:-<none>}"
echo "Output: ${OUTPUT_JSON}"
echo "Episodes: ${NUM_EPISODES}"
echo ""

python3 -u -m pmv.diversity_analysis \
  "${CONFIG_PATH}" \
  --num-episodes "${NUM_EPISODES}" \
  --dataset "${DATASET}" \
  --output "${OUTPUT_JSON}"

echo ""
echo "Diversity analysis complete."
echo "Job finished at: $(date)"
