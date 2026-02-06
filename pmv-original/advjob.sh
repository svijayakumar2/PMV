#!/bin/bash

python3 -m pmv.robustness_testing.gsm_methodology     runs/pure_stackelberg_experiment_20250729_124028/checkpoints/kirchner_round_008.pt     --config runs/pure_stackelberg_experiment_20250729_124028/config.yaml     --num_samples 200
