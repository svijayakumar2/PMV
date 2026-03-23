# PMV Cluster Runbook

This README is for running PMV on LSF (`bsub`) in cluster mode.

If your working directory is the cluster project root (`/dccstor/principled_ai/users/saranyaibm2/PMV`),
prefix script paths with `pmv/` (example: `bsub < pmv/scripts/train.sh`).

Current default behavior:
- Debate is removed from training and evaluation.
- Fixed-rule baselines available: `average`, `median`, `min`, `max`.

## Layout

- Core runtime code: top-level `pmv/*.py`
- Cluster scripts: `pmv/scripts/`
- Active configs: `pmv/configs/experiments/`
- Current project notes: `pmv/docs/`
- Archived notes and historical artifacts: `pmv/legacy/`

## Environment

Use the same cache setup in all jobs:

```bash
export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

Cluster workspace root used by scripts:

```bash
cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1
```

## Configs

Common experiment configs are in `pmv/configs/experiments/`.

- `config_baseline.yaml`
- `config_kirchner_single.yaml`
- `config_kirchner_3v.yaml`
- `config_single_verifier.yaml`
- `config_average.yaml`
- `config_min.yaml`
- `config_median.yaml`
- `config_max.yaml`
- `config_pe_min.yaml`
- `config_pe_margin.yaml`
- `config_softmin.yaml`
- `config_zebra.yaml`
- `config_5verifiers.yaml`
- `config_scaled_prover.yaml`

Legacy `_hb` config aliases are kept for compatibility, but canonical names are now the default.

## Training Jobs

Single training job (baseline):

```bash
bsub < scripts/train.sh
```

Single training job (specific experiment):

```bash
EXPERIMENT=min bsub < scripts/train.sh
EXPERIMENT=average bsub < scripts/train.sh
EXPERIMENT=median bsub < scripts/train.sh
EXPERIMENT=max bsub < scripts/train.sh
EXPERIMENT=single_verifier bsub < scripts/train.sh
EXPERIMENT=baseline bsub < scripts/train.sh
EXPERIMENT=pe_min bsub < scripts/train.sh
EXPERIMENT=pe_margin bsub < scripts/train.sh
EXPERIMENT=softmin bsub < scripts/train.sh
EXPERIMENT=kirchner_single bsub < scripts/train.sh
EXPERIMENT=kirchner_3v bsub < scripts/train.sh
EXPERIMENT=zebra bsub < scripts/train.sh
```

Kirchner replication configs use `dataset.type: gsm8k`.

Full sweep:

```bash
bash scripts/run_sweep.sh
```

Training now writes checkpoints by default:
- per-round: `results/checkpoints/<config_stem>_<jobid>/round_XXX.pt`
- latest alias for eval: `results/checkpoints/<config_stem>_latest.pt`

Default configs are now high-budget and use the 3B backbone unless a config explicitly changes model size.

The sweep submits:
- `baseline`
- `5verifiers`
- `pe_min`
- `pe_margin`
- `min`
- `median`
- `max`
- `scaled_prover`
- `softmin`

## Evaluation Jobs

Unified evaluation as LSF job:

```bash
EXPERIMENT=baseline bsub < scripts/evaluation.sh
EXPERIMENT=zebra DATASET=zebra ZEBRA_MAX_SIZE=4*4 bsub < scripts/evaluation.sh
```

Each eval run is saved to a unique file by default:
- `results/evals/eval_<experiment>_<jobid>_<timestamp>.json`

By default, evaluation requires a trained checkpoint:
- baseline expects `results/checkpoints/config_baseline_latest.pt`
- zebra expects `results/checkpoints/config_zebra_latest.pt`

Optional overrides:

```bash
EXPERIMENT=baseline PROBE_EPISODES=200 ATTACK_EPISODES=100 OUTPUT=results/eval_baseline_200_100.json bsub < scripts/evaluation.sh
```

Fast eval mode (useful when jobs appear stuck due long generation):

```bash
EXPERIMENT=single_verifier PROBE_EPISODES=60 ATTACK_EPISODES=20 PROBE_MAX_NEW_TOKENS=256 ATTACK_MAX_NEW_TOKENS=256 bsub < scripts/evaluation.sh
```

Probe-only mode (skip adversarial battery):

```bash
EXPERIMENT=single_verifier SKIP_ADVERSARIAL=1 PROBE_EPISODES=80 bsub < scripts/evaluation.sh
```

Optional ablation labels for inverse-ablation tracking:

```bash
EXPERIMENT=single_verifier ABLATION_ID=v1_only ABLATION_TAG=inv_ablation_step1 bsub < scripts/evaluation.sh
EXPERIMENT=baseline ABLATION_ID=v3_supervised ABLATION_TAG=inv_ablation_step2 bsub < scripts/evaluation.sh
```

These labels are stored in eval JSON under:
- `run_metadata.ablation_id`
- `run_metadata.ablation_tag`

Model/dataset metadata is also stored per eval run:
- `config_snapshot.prover_model`
- `config_snapshot.verifier_model`
- `config_snapshot.use_quantization`
- `config_snapshot.dataset_mode`
- `config_snapshot.resolved_dataset`

## Inverse-Ablation Reporting (1 Verifier vs Multi-Verifier)

Build consolidated tables across all eval JSONs:

```bash
python3 pmv/scripts/build_ablation_table.py --inputs results/evals/*.json --out-dir results/ablations
```

Or as an LSF job:

```bash
bsub < scripts/ablation_report.sh
INPUTS="results/evals/*.json results/custom/*.json" bsub < scripts/ablation_report.sh
```

Outputs:
- `results/ablations/ablation_runs_latest.csv`
- `results/ablations/ablation_runs_<timestamp>.csv`
- `results/ablations/ablation_summary_latest.md`
- `results/ablations/ablation_summary_<timestamp>.md`

The markdown summary now includes grouped tables by:
- number of verifiers
- oversight rule
- model pair (`prover_model -> verifier_model`)

## Adversarial and Diversity Jobs

Adversarial battery:

```bash
EXPERIMENT=baseline bsub < scripts/adversarial_test.sh
EXPERIMENT=zebra bsub < scripts/adversarial_test.sh
```

Diversity analysis across configs:

```bash
bsub < scripts/diversity_analysis.sh
DATASET=zebra bsub < scripts/diversity_analysis.sh
```

## Output Sanity Audit (Helpful vs Sneaky)

This directly inspects generated outputs so you can tell whether:
- sneaky is genuinely adversarial,
- sneaky fool rate is truly low because verifiers are robust.

Run as LSF job:

```bash
EXPERIMENT=baseline bsub < scripts/audit_outputs.sh
EXPERIMENT=zebra DATASET=zebra ZEBRA_MAX_SIZE=4*4 bsub < scripts/audit_outputs.sh
```

Outputs:
- `results/output_audit_<experiment>.json`
- `results/output_audit_<experiment>.md`

Key audit metrics:
- `sneaky_incorrect_rate`
- `sneaky_fool_rate@t`

## Log Analysis

Run directly:

```bash
python3 pmv/scripts/analyze_training_log.py \
  /dccstor/principled_ai/users/saranyaibm2/PMV/171606.txt \
  /dccstor/principled_ai/users/saranyaibm2/PMV/171607.txt \
  --out-dir /dccstor/principled_ai/users/saranyaibm2/PMV/pmv/results/log_analysis
```

Run as LSF job:

```bash
LOG_INPUTS="/dccstor/principled_ai/users/saranyaibm2/PMV/*.txt" bsub < scripts/analyze_logs.sh
```

Or pass explicit files:

```bash
LOG_FILES="/dccstor/principled_ai/users/saranyaibm2/PMV/171606.txt /dccstor/principled_ai/users/saranyaibm2/PMV/171607.txt" bsub < scripts/analyze_logs.sh
```

Outputs per log:
- `summary.json`
- `analysis.md`
- plots (if `matplotlib` available)

## Your Custom LSF Template

Use this pattern for custom jobs:

```bash
#!/bin/bash
#BSUB -J pmv_baseline
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o /u/saranyaibm2/.lsbatch/%J.out
#BSUB -e /u/saranyaibm2/.lsbatch/%J.err
#BSUB -W 48:00

export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"

cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

python3 -u -m pmv.main pmv/configs/experiments/config_baseline.yaml

echo "Job finished at: $(date)"
```

## Quick Monitoring

```bash
bjobs -w
bpeek <job_id>
tail -f /u/saranyaibm2/.lsbatch/<job_id>.out
bkill <job_id>
```
