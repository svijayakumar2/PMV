# Scripts

Top-level scripts are the active cluster workflow.

## Active run scripts
- `run_multismall_study.sh` (end-to-end multi-variant PMV loop)
- `run_multismall_study_math.sh` (end-to-end multi-variant PMV loop on math)
- `run_multismall_study_math_tuned.sh` (math stabilization pass, affine reward)
- `run_stage1_stabilize.sh`
- `run_stage2_rule_compare.sh`
- `submit_stage2_rule_compare.sh`
- `run_tiny_rule_gate.sh`
- `run_eval_checkpoint_fast.sh`
- `run_collapse_diagnostic.sh`
- `run_stage3_eval_4x4.sh`
- `run_stage4_threshold_sweep.sh`

## Active utilities
- `classify_collapse.py`
- `inspect_probe_records.py`
- `audit_outputs.py`
- `analyze_training_log.py`
- `build_ablation_table.py`

## Legacy scripts
Deprecated wrappers and old sweep entrypoints were moved to:
- `legacy/`
