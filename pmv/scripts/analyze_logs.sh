#!/bin/bash
#BSUB -J pmv_log_analysis
#BSUB -q normal
#BSUB -n 1
#BSUB -M 8GB
#BSUB -R "rusage[mem=8GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 01:00

# ============================================================
# PMV Log Analysis Job
# Usage examples:
#   LOG_INPUTS="/dccstor/principled_ai/users/saranyaibm2/PMV/*.txt" bsub < scripts/analyze_logs.sh
#   LOG_FILES="/dccstor/principled_ai/users/saranyaibm2/PMV/171606.txt /dccstor/principled_ai/users/saranyaibm2/PMV/171607.txt" bsub < scripts/analyze_logs.sh
# ============================================================

OUT_DIR=${OUT_DIR:-results/log_analysis}
LOG_INPUTS=${LOG_INPUTS:-/dccstor/principled_ai/users/saranyaibm2/PMV/*.txt}

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "Output directory: ${OUT_DIR}"
echo ""

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

mkdir -p results

if [ -n "${LOG_FILES:-}" ]; then
  read -r -a LOG_ARGS <<< "${LOG_FILES}"
else
  # shellcheck disable=SC2086
  mapfile -t LOG_ARGS < <(ls ${LOG_INPUTS} 2>/dev/null)
fi

if [ "${#LOG_ARGS[@]}" -eq 0 ]; then
  echo "No log files found."
  exit 1
fi

python3 -u pmv/scripts/analyze_training_log.py "${LOG_ARGS[@]}" --out-dir "${OUT_DIR}"

echo ""
echo "Job finished at: $(date)"
