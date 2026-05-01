#!/bin/bash
#BSUB -J pmv_comparison_train_2gpu
#BSUB -q normal
#BSUB -gpu "num=2:mode=shared:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 128GB
#BSUB -R "rusage[mem=128GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 12:00
#BSUB -env "all"

set -euo pipefail

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export PYTHONFAULTHANDLER=1

REPO_ROOT=${REPO_ROOT:-/dccstor/principled_ai/users/saranyaibm2/PMV}
EXPERIMENT=${EXPERIMENT:-ours_3v}
CONFIG_PATH=${CONFIG_PATH:-}

normalize_repo_root() {
  if [ -d "${REPO_ROOT}/pmv" ]; then
    return
  fi
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
  echo "REPO_ROOT=${REPO_ROOT}" >&2
  echo "PWD=$(pwd)" >&2
  echo "python3=$(command -v python3)" >&2
  echo "python3_version=$(python3 -V 2>&1)" >&2
  echo "PYTHONPATH=${PYTHONPATH:-<unset>}" >&2
  echo "Directory snapshot:" >&2
  ls -la >&2 || true
  if [ -d pmv ]; then
    echo "pmv/ snapshot:" >&2
    ls -la pmv >&2 || true
  fi
  echo "Traceback from import check follows:" >&2
  python3 - <<PY || true
import importlib
import sys
import traceback

print("sys.path head:", sys.path[:8])
try:
    mod = importlib.import_module("${module}")
    print("resolved module path:", getattr(mod, "__file__", "<namespace>"))
except Exception:
    traceback.print_exc()
PY
  exit 1
}

normalize_repo_root
prepare_python_import_path

if [ -z "${CONFIG_PATH}" ]; then
  case "${EXPERIMENT}" in
    ours_3v)
      CONFIG_PATH="pmv/configs/experiments/config_comparison_ours_3v.yaml"
      ;;
    ours_1v)
      CONFIG_PATH="pmv/configs/experiments/config_comparison_ours_1v.yaml"
      ;;
    kirchner_1v|kirchner_1v_src)
      CONFIG_PATH="pmv/configs/experiments/config_comparison_kirchner_1v_src.yaml"
      ;;
    kirchner_1v_cgc)
      CONFIG_PATH="pmv/configs/experiments/config_comparison_kirchner_1v_cgc.yaml"
      ;;
    kirchner_1v_goodhart)
      CONFIG_PATH="pmv/configs/experiments/config_comparison_kirchner_1v_goodhart.yaml"
      ;;
    kirchner_1v_legacy)
      CONFIG_PATH="pmv/configs/experiments/config_comparison_kirchner_1v.yaml"
      ;;
    *)
      echo "Unknown EXPERIMENT='${EXPERIMENT}'. Use ours_3v, ours_1v, kirchner_1v_src, kirchner_1v_cgc, kirchner_1v_goodhart, or set CONFIG_PATH." >&2
      exit 1
      ;;
  esac
fi

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Experiment: ${EXPERIMENT}"
echo "Config: ${CONFIG_PATH}"
echo ""

cd "${REPO_ROOT}" || exit 1

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "FAILED: config not found: ${CONFIG_PATH}" >&2
  echo "Hint: REPO_ROOT should be the parent directory that contains pmv/." >&2
  exit 1
fi
run_module_check_or_die "pmv.main"

python3 -u -m pmv.main "${CONFIG_PATH}"

echo ""
echo "Comparison training run complete."
echo "Job finished at: $(date)"
