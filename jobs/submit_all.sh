#!/bin/bash
# Master submission script for all PMV experiments
# Generated: 2026-01-11 13:28:29

bsub < jobs/baseline.sh
bsub < jobs/2verifiers.sh
bsub < jobs/4verifiers.sh
bsub < jobs/helpful_prior_0.3.sh
bsub < jobs/helpful_prior_0.7.sh
bsub < jobs/longer_training.sh
bsub < jobs/more_verifier_training.sh
bsub < jobs/higher_lr.sh
bsub < jobs/lower_lr.sh
bsub < jobs/smaller_models.sh
bsub < jobs/5verifiers.sh
bsub < jobs/6verifiers.sh
bsub < jobs/7verifiers_all_roles.sh

echo 'Submitted all jobs!'
echo 'Check status with: bjobs'
echo 'Check logs in: ~/.lsbatch/'
