# PMV (Prover Multi-Verifier)

PMV is a research project about oversight robustness for math reasoning models.  
The core question is simple: can we train a verifier system that still distinguishes correct from incorrect solutions when a prover is actively trying to look convincing while being wrong.

This repository contains the full training and evaluation pipeline used for that question, with cluster-first scripts for long-running LSF jobs.

## What the project is trying to show

The project compares three primary settings:
- `ours_3v`: our oversight method with three verifiers
- `ours_1v`: our oversight method with one verifier
- `kirchner_1v` (`kirchner_1v_src`): a Kirchner-style one-verifier baseline

The practical claim we are testing is not only "which setup gets higher accuracy," but "which setup resists adversarial optimization better."

## How PMV works

PMV trains a prover and one or more verifiers in rounds.  
The prover can produce helpful outputs (trying to be correct) or sneaky outputs (trying to be convincing while wrong).  
The verifier system assigns oversight scores to these outputs, and training updates both sides so we can measure whether oversight remains reliable as pressure increases.

Most active runs use:
- `Qwen/Qwen2.5-3B-Instruct` with 4-bit quantization and LoRA adapters
- OpenMathInstruct-2 MATH subset
- LSF (`bsub`) on A100 cluster nodes

## What we measure

Main evaluation outputs are designed to answer different parts of the robustness question:
- `helpful correctness`: how often helpful solutions are actually correct
- `sneaky fool rate`: how often sneaky wrong solutions are still accepted
- score separation between correct and incorrect solutions
- Kirchner-style dedicated attack results (`src`, `cgc`, `goodhart`, optional `affine`)
- balanced best-of-n behavior

The attack suite and best-of-n stages are resumable, because single jobs often hit 12-hour walltime.

## Where to check current project status

For consolidated run history, metrics, and interpretation notes, see:
- [PMV Job Log Summary](/Users/saranya/Desktop/PMV/pmv/docs/job_log_summary.md)

That file is the best quick reference for what completed, what timed out, and which numbers are currently trustworthy.

## Repo structure

- runtime code: `pmv/*.py`
- scripts: `pmv/scripts/`
- active experiment configs: `pmv/configs/experiments/`
- docs: `pmv/docs/`
- archived artifacts: `pmv/legacy/`

## Cluster quickstart

```bash
cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1
export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
```

## Main training commands

```bash
EXPERIMENT=ours_3v REPO_ROOT=/dccstor/principled_ai/users/saranyaibm2/PMV \
bsub < /dccstor/principled_ai/users/saranyaibm2/PMV/pmv/scripts/run_comparison_train_2gpu_ddp.sh

EXPERIMENT=ours_1v REPO_ROOT=/dccstor/principled_ai/users/saranyaibm2/PMV \
bsub < /dccstor/principled_ai/users/saranyaibm2/PMV/pmv/scripts/run_comparison_train_2gpu_ddp.sh

EXPERIMENT=kirchner_1v REPO_ROOT=/dccstor/principled_ai/users/saranyaibm2/PMV \
bsub < /dccstor/principled_ai/users/saranyaibm2/PMV/pmv/scripts/run_comparison_train_2gpu_ddp.sh
```

## Main eval commands

```bash
EXPERIMENT=ours_3v REPO_ROOT=/dccstor/principled_ai/users/saranyaibm2/PMV \
bsub < /dccstor/principled_ai/users/saranyaibm2/PMV/pmv/scripts/run_comparison_eval_robustness_2gpu.sh

EXPERIMENT=ours_1v REPO_ROOT=/dccstor/principled_ai/users/saranyaibm2/PMV \
bsub < /dccstor/principled_ai/users/saranyaibm2/PMV/pmv/scripts/run_comparison_eval_robustness_2gpu.sh

EXPERIMENT=kirchner_1v REPO_ROOT=/dccstor/principled_ai/users/saranyaibm2/PMV \
bsub < /dccstor/principled_ai/users/saranyaibm2/PMV/pmv/scripts/run_comparison_eval_robustness_2gpu.sh
```

`kirchner_1v` maps to the SRC Kirchner baseline config in the eval launcher.

These defaults are the heavy `EVAL_PROFILE=full` settings and often exceed 12-hour walltime.

## Paper-ready staged workflow (full budgets)

For publishable runs, keep `EVAL_PROFILE=full` but split evaluation into stages so resumed jobs do not recompute completed sections:

```bash
# Stage 1: core metrics (probe/diversity/adversarial)
EXPERIMENT=ours_3v EVAL_PROFILE=full EVAL_STAGE=core REPO_ROOT=/dccstor/principled_ai/users/saranyaibm2/PMV \
bsub < /dccstor/principled_ai/users/saranyaibm2/PMV/pmv/scripts/run_comparison_eval_robustness_2gpu.sh

# Stage 2: Kirchner suite only (merges into existing OUTPUT_JSON)
EXPERIMENT=ours_3v EVAL_PROFILE=full EVAL_STAGE=kirchner REPO_ROOT=/dccstor/principled_ai/users/saranyaibm2/PMV \
bsub < /dccstor/principled_ai/users/saranyaibm2/PMV/pmv/scripts/run_comparison_eval_robustness_2gpu.sh

# Stage 3: balanced best-of-n only (merges into same OUTPUT_JSON)
EXPERIMENT=ours_3v EVAL_PROFILE=full EVAL_STAGE=bestofn REPO_ROOT=/dccstor/principled_ai/users/saranyaibm2/PMV \
bsub < /dccstor/principled_ai/users/saranyaibm2/PMV/pmv/scripts/run_comparison_eval_robustness_2gpu.sh
```

Repeat the same 3-stage sequence for `ours_1v` and `kirchner_1v`.

## Recommended first-run (finish quickly)

Run one arm first with a lighter profile to verify end-to-end completion:

```bash
EXPERIMENT=ours_3v EVAL_PROFILE=smoke REPO_ROOT=/dccstor/principled_ai/users/saranyaibm2/PMV \
bsub < /dccstor/principled_ai/users/saranyaibm2/PMV/pmv/scripts/run_comparison_eval_robustness_2gpu.sh
```

Submit one experiment at a time while debugging so queue pressure does not hide failures behind long `PEND` waits.

Then move to a stronger but still practical preset:

```bash
EXPERIMENT=ours_3v EVAL_PROFILE=fast REPO_ROOT=/dccstor/principled_ai/users/saranyaibm2/PMV \
bsub < /dccstor/principled_ai/users/saranyaibm2/PMV/pmv/scripts/run_comparison_eval_robustness_2gpu.sh
```

`EVAL_PROFILE` options in `run_comparison_eval_robustness_2gpu.sh`:
- `smoke`: quick plumbing check (`attack=0`, no Kirchner suite, no best-of-n)
- `fast`: reduced Kirchner + best-of-n budgets (designed to finish much more often in 12h)
- `full`: strongest settings (can require repeated resume submissions)

## Important output locations

Comparison study root:
- `/dccstor/principled_ai/users/saranyaibm2/PMV/results/studies/pmv_kirchner_comparison/`

Per-arm artifacts typically include:
- `checkpoints/*latest.pt`
- `eval_robustness_suite.json`
- `eval_robustness_suite_kirchner_resume/`
- `eval_robustness_suite_bestofn_resume.pt`

LSF logs:
- `/u/saranyaibm2/.lsbatch/<jobid>.out`
- `/u/saranyaibm2/.lsbatch/<jobid>.err`

## Interpreting long eval runs

If an eval job times out, rerun with the same `EXPERIMENT` and output paths so resume state is reused.  
A run is fully finished only when `.out` contains both:
- `Saved evaluation report to ...`
- `Comparison robustness eval complete.`

Common failure/status signals:
- `TERM_RUNLIMIT` + exit code `140`: hit walltime; rerun with same paths to resume
- `FAILED: checkpoint not found ...`: training checkpoint missing or wrong path
- `FAILED: cannot import pmv.evaluation ...`: `REPO_ROOT` points to wrong directory

Quick monitoring commands:

```bash
bjobs -w
bpeek <jobid>
tail -n 120 /u/saranyaibm2/.lsbatch/<jobid>.out
tail -n 120 /u/saranyaibm2/.lsbatch/<jobid>.err
```
