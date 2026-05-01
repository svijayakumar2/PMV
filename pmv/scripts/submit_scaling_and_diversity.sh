#!/bin/bash
# --------------------------------------------------------------------------
# Submit all scaling + diversity jobs.
#
# Usage:
#   bash pmv/scripts/submit_scaling_and_diversity.sh           # submit everything
#   bash pmv/scripts/submit_scaling_and_diversity.sh diversity  # diversity only
#   bash pmv/scripts/submit_scaling_and_diversity.sh scaling    # scaling only
# --------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/dccstor/principled_ai/users/saranyaibm2/PMV}
MODE=${1:-all}

cd "${REPO_ROOT}"

echo "================================================"
echo "PMV Scaling + Diversity Job Submission"
echo "Mode: ${MODE}"
echo "Repo: ${REPO_ROOT}"
echo "================================================"
echo ""

# ------------------------------------------------------------------
# Diversity analysis on existing comparison arms
# ------------------------------------------------------------------
if [ "${MODE}" = "all" ] || [ "${MODE}" = "diversity" ]; then
  echo "--- Submitting diversity analysis jobs ---"
  echo ""

  echo "[1/6] ours_3v diversity"
  EXPERIMENT=ours_3v \
    bsub -J div_3v < pmv/scripts/run_diversity_analysis.sh
  echo ""

  echo "[2/6] ours_1v diversity"
  EXPERIMENT=ours_1v \
    bsub -J div_1v < pmv/scripts/run_diversity_analysis.sh
  echo ""

  echo "[3/6] kirchner_1v_src diversity"
  EXPERIMENT=kirchner_1v_src \
    bsub -J div_k1s < pmv/scripts/run_diversity_analysis.sh
  echo ""

  # matched10 (submit only if checkpoints exist)
  echo "[4/6] ours_3v_matched10 diversity (will fail if checkpoint not ready)"
  EXPERIMENT=ours_3v_matched10 \
    bsub -J div_m10_3v < pmv/scripts/run_diversity_analysis.sh
  echo ""

  echo "[5/6] ours_1v_matched10 diversity"
  EXPERIMENT=ours_1v_matched10 \
    bsub -J div_m10_1v < pmv/scripts/run_diversity_analysis.sh
  echo ""

  echo "[6/6] kirchner_1v_src_matched10 diversity"
  EXPERIMENT=kirchner_1v_src_matched10 \
    bsub -J div_m10_k1s < pmv/scripts/run_diversity_analysis.sh
  echo ""
fi

# ------------------------------------------------------------------
# Scaling study: training
# ------------------------------------------------------------------
if [ "${MODE}" = "all" ] || [ "${MODE}" = "scaling" ]; then
  echo "--- Submitting scaling training jobs ---"
  echo ""

  echo "[1/3] 1x7B (large single verifier)"
  EXPERIMENT=large_1v_7b \
    bsub -J sc_l1v < pmv/scripts/run_scaling_train_2gpu.sh
  echo ""

  echo "[2/3] 5x1.5B (distributed, compute-matched)"
  EXPERIMENT=small_5v_1p5b \
    bsub -J sc_s5v < pmv/scripts/run_scaling_train_2gpu.sh
  echo ""

  echo "[3/3] 7x0.5B (maximum distribution)"
  EXPERIMENT=tiny_7v_0p5b \
    bsub -J sc_t7v < pmv/scripts/run_scaling_train_2gpu.sh
  echo ""

  echo "--- NOTE ---"
  echo "After training completes, submit eval jobs:"
  echo "  EXPERIMENT=large_1v_7b   bsub -J sce_l1v < pmv/scripts/run_scaling_eval_2gpu.sh"
  echo "  EXPERIMENT=small_5v_1p5b bsub -J sce_s5v < pmv/scripts/run_scaling_eval_2gpu.sh"
  echo "  EXPERIMENT=tiny_7v_0p5b  bsub -J sce_t7v < pmv/scripts/run_scaling_eval_2gpu.sh"
  echo ""
  echo "Then diversity analysis on scaling arms:"
  echo "  CONFIG_PATH=pmv/configs/experiments/config_scaling_large_1v_7b.yaml \\"
  echo "  CHECKPOINT_PATH=results/studies/scaling/large_1v_7b/checkpoints/config_scaling_large_1v_7b_latest.pt \\"
  echo "  OUTPUT_JSON=results/studies/scaling/large_1v_7b/diversity_analysis.json \\"
  echo "    bsub -J div_sc_l1v < pmv/scripts/run_diversity_analysis.sh"
  echo ""
  echo "  CONFIG_PATH=pmv/configs/experiments/config_scaling_small_5v_1p5b.yaml \\"
  echo "  CHECKPOINT_PATH=results/studies/scaling/small_5v_1p5b/checkpoints/config_scaling_small_5v_1p5b_latest.pt \\"
  echo "  OUTPUT_JSON=results/studies/scaling/small_5v_1p5b/diversity_analysis.json \\"
  echo "    bsub -J div_sc_s5v < pmv/scripts/run_diversity_analysis.sh"
  echo ""
  echo "  CONFIG_PATH=pmv/configs/experiments/config_scaling_tiny_7v_0p5b.yaml \\"
  echo "  CHECKPOINT_PATH=results/studies/scaling/tiny_7v_0p5b/checkpoints/config_scaling_tiny_7v_0p5b_latest.pt \\"
  echo "  OUTPUT_JSON=results/studies/scaling/tiny_7v_0p5b/diversity_analysis.json \\"
  echo "    bsub -J div_sc_t7v < pmv/scripts/run_diversity_analysis.sh"
  echo ""
fi

echo "================================================"
echo "All jobs submitted. Check status with: bjobs -w"
echo "================================================"
