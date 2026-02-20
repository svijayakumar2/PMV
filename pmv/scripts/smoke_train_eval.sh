#!/bin/bash
set -euo pipefail

# Fast end-to-end smoke tests:
# - train + eval (supervised, pe_min, pe_margin)
# - tiny budgets to keep runtime low
#
# Usage:
#   bash scripts/smoke_train_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PKG_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PKG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/smoke/${STAMP}"
mkdir -p "${OUT_DIR}"

EXPERIMENTS=(
  "smoke_supervised"
  "smoke_pe_min"
  "smoke_pe_margin"
)

echo "Running smoke train+eval for: ${EXPERIMENTS[*]}"
echo "Output dir: ${OUT_DIR}"

for exp in "${EXPERIMENTS[@]}"; do
  cfg="configs/experiments/config_${exp}.yaml"
  ckpt="results/checkpoints_smoke/config_${exp}_latest.pt"
  eval_out="${OUT_DIR}/eval_${exp}.json"

  echo ""
  echo "============================================================"
  echo "SMOKE EXPERIMENT: ${exp}"
  echo "Config: ${cfg}"
  echo "============================================================"

  python3 -u -m pmv.main "${cfg}"

  if [ ! -f "${ckpt}" ]; then
    echo "Expected checkpoint not found: ${ckpt}"
    exit 1
  fi

  python3 -u -m pmv.evaluation "${cfg}" \
    --checkpoint "${ckpt}" \
    --probe-episodes 12 \
    --attack-episodes 6 \
    --probe-max-new-tokens 96 \
    --attack-max-new-tokens 96 \
    --temperatures 0.7 \
    --output "${eval_out}"
done

echo ""
echo "Smoke runs complete."
echo "Eval JSONs:"
ls -1 "${OUT_DIR}"/eval_*.json
