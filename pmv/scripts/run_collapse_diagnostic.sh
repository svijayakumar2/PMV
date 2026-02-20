#!/bin/bash
set -euo pipefail

# Train + evaluate collapse diagnostics for:
# - supervised
# - pe_min
# - pe_margin
#
# Usage:
#   bash scripts/run_collapse_diagnostic.sh
#
# Optional overrides:
#   PROBE_EPISODES=120 ATTACK_EPISODES=40 EVAL_TEMPS="0.5 0.7 1.0" bash scripts/run_collapse_diagnostic.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PKG_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PKG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/diagnostics/${STAMP}"
mkdir -p "${OUT_DIR}"

PROBE_EPISODES="${PROBE_EPISODES:-80}"
ATTACK_EPISODES="${ATTACK_EPISODES:-30}"
EVAL_TEMPS="${EVAL_TEMPS:-0.7 1.0}"

EXPERIMENTS=(
  "diag_supervised"
  "diag_pe_min"
  "diag_pe_margin"
)

echo "Running collapse diagnostics for: ${EXPERIMENTS[*]}"
echo "Probe episodes: ${PROBE_EPISODES}, attack episodes: ${ATTACK_EPISODES}, temps: ${EVAL_TEMPS}"
echo "Output dir: ${OUT_DIR}"

EVAL_FILES=()

for exp in "${EXPERIMENTS[@]}"; do
  cfg="configs/experiments/config_${exp}.yaml"
  ckpt="results/checkpoints_diag/config_${exp}_latest.pt"
  eval_out="${OUT_DIR}/eval_${exp}.json"

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

  python3 -u -m pmv.evaluation "${cfg}" \
    --checkpoint "${ckpt}" \
    --probe-episodes "${PROBE_EPISODES}" \
    --attack-episodes "${ATTACK_EPISODES}" \
    --probe-max-new-tokens 192 \
    --attack-max-new-tokens 192 \
    --temperatures ${EVAL_TEMPS} \
    --output "${eval_out}"

  EVAL_FILES+=("${eval_out}")
done

echo ""
echo "Collapse classification:"
python3 -u scripts/classify_collapse.py "${EVAL_FILES[@]}" | tee "${OUT_DIR}/collapse_summary.tsv"

echo ""
echo "Done. Outputs in: ${OUT_DIR}"
