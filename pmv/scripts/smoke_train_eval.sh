#!/bin/bash
#BSUB -J pmv_smoke
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 12:00

set -eu

# ============================================================
# PMV quick smoke job (train + eval)
# Usage:
#   bsub < scripts/smoke_train_eval.sh
#   EXPERIMENT=smoke_pe_min bsub < scripts/smoke_train_eval.sh
# ============================================================

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
DATASET=${DATASET:-auto}
ZEBRA_MAX_SIZE=${ZEBRA_MAX_SIZE:-3*3}
OUT_DIR=${OUT_DIR:-results/smoke/${RUN_STAMP}_${LSB_JOBID:-local}}

if [ -n "${EXPERIMENT:-}" ]; then
  EXPERIMENT_LIST="${EXPERIMENT}"
elif [ -n "${EXPERIMENTS:-}" ]; then
  EXPERIMENT_LIST="${EXPERIMENTS}"
else
  EXPERIMENT_LIST="smoke_supervised smoke_pe_min smoke_pe_margin"
fi

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: ${LSB_JOBID:-local}"
echo "Experiments: ${EXPERIMENT_LIST}"
echo "Dataset mode: ${DATASET}"
echo "Output dir: ${OUT_DIR}"
echo ""

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

mkdir -p "${OUT_DIR}" results/checkpoints_smoke

for exp in ${EXPERIMENT_LIST}; do
  cfg="pmv/configs/experiments/config_${exp}.yaml"
  ckpt="results/checkpoints_smoke/config_${exp}_latest.pt"
  eval_out="${OUT_DIR}/eval_${exp}.json"

  if [ ! -f "${cfg}" ]; then
    echo "Config not found: ${cfg}"
    exit 1
  fi

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

  if [ "${DATASET}" = "zebra" ]; then
    python3 -u -m pmv.evaluation "${cfg}" \
      --checkpoint "${ckpt}" \
      --dataset zebra \
      --zebra-max-size "${ZEBRA_MAX_SIZE}" \
      --probe-episodes 12 \
      --attack-episodes 6 \
      --probe-max-new-tokens 96 \
      --attack-max-new-tokens 96 \
      --temperatures 0.7 \
      --output "${eval_out}"
  else
    python3 -u -m pmv.evaluation "${cfg}" \
      --checkpoint "${ckpt}" \
      --dataset "${DATASET}" \
      --probe-episodes 12 \
      --attack-episodes 6 \
      --probe-max-new-tokens 96 \
      --attack-max-new-tokens 96 \
      --temperatures 0.7 \
      --output "${eval_out}"
  fi
done

echo ""
echo "Smoke runs complete. Eval JSONs:"
ls -1 "${OUT_DIR}"/eval_*.json
echo "Job finished at: $(date)"
