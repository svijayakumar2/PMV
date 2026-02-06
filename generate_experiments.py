#!/usr/bin/env python3
"""
Generate experiment configurations and LSF job scripts for PMV training.

Usage:
    python generate_experiments.py
    
This will create:
    - Individual config files in pmv/configs/experiments/
    - Job scripts in jobs/
    - Master submission script submit_all.sh
"""

import yaml
import os
from pathlib import Path
from datetime import datetime
import copy

def deep_update(base_dict, update_dict):
    """Recursively update nested dictionary"""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def generate_config(base_config, overrides, experiment_name):
    """Generate experiment config by applying overrides to base"""
    config = copy.deepcopy(base_config)
    if overrides:
        config = deep_update(config, overrides)
    
    # Update logdir to be experiment-specific
    config['logging']['logdir'] = f"runs/pmv_stackelberg/{experiment_name}"
    
    return config

def create_job_script(experiment_name, config_path, job_dir, log_dir):
    """Create LSF job script for an experiment"""
    
    job_script = f"""#!/bin/bash
#BSUB -J pmv_{experiment_name}
#BSUB -q normal
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -M 64GB
#BSUB -R "rusage[mem=64GB]"
#BSUB -o {log_dir}/%J.out
#BSUB -e {log_dir}/%J.err
#BSUB -W 48:00

# Set up environment
export HF_HOME=/dccstor/principled_ai/users/saranyaibm2/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# Load modules if needed (uncomment and adjust as needed)
# module load python/3.10
# module load cuda/12.1

# Activate virtual environment if using one
# source /path/to/venv/bin/activate

# Print job info
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "Experiment: {experiment_name}"
echo "Config: {config_path}"
echo ""

# Run training from PMV directory
cd /dccstor/principled_ai/users/saranyaibm2/PMV || exit 1

python3 -u train_qwen.py

echo ""
echo "Job finished at: $(date)"
"""
    
    job_file = job_dir / f"{experiment_name}.sh"
    with open(job_file, 'w') as f:
        f.write(job_script)
    
    # Make executable
    os.chmod(job_file, 0o755)
    
    return job_file

def main():
    # Load experiments configuration
    with open('pmv/configs/experiments_config.yaml', 'r') as f:
        exp_config = yaml.safe_load(f)
    
    base_config = exp_config['base_config']
    experiments = exp_config['experiments']
    
    # Create directories
    config_dir = Path('pmv/configs/experiments')
    job_dir = Path('jobs')
    log_dir = Path.home() / '.lsbatch'
    
    config_dir.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PMV Experiment Generator")
    print("="*80)
    print(f"Generating {len(experiments)} experiments...")
    print()
    
    job_commands = []
    experiment_info = []
    
    for exp in experiments:
        exp_name = exp['name']
        exp_desc = exp['description']
        exp_overrides = exp.get('overrides', {})
        
        print(f"Generating: {exp_name}")
        print(f"  Description: {exp_desc}")
        
        # Generate config
        config = generate_config(base_config, exp_overrides, exp_name)
        config_path = config_dir / f"config_{exp_name}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"  Config saved: {config_path}")
        
        # Create job script
        job_file = create_job_script(exp_name, config_path, job_dir, log_dir)
        print(f"  Job script: {job_file}")
        
        # Add to submission list
        job_commands.append(f"bsub < {job_file}")
        
        experiment_info.append({
            'name': exp_name,
            'description': exp_desc,
            'config': str(config_path),
            'job': str(job_file)
        })
        
        print()
    
    # Create master submission script
    submit_script = job_dir / 'submit_all.sh'
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Master submission script for all PMV experiments\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for cmd in job_commands:
            f.write(f"{cmd}\n")
        
        f.write("\necho 'Submitted all jobs!'\n")
        f.write("echo 'Check status with: bjobs'\n")
        f.write("echo 'Check logs in: ~/.lsbatch/'\n")
    
    os.chmod(submit_script, 0o755)
    
    # Create individual submission script
    for exp_info in experiment_info:
        submit_one = job_dir / f"submit_{exp_info['name']}.sh"
        with open(submit_one, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Submit {exp_info['name']}\n")
            f.write(f"# {exp_info['description']}\n\n")
            f.write(f"bsub < {exp_info['job']}\n")
        os.chmod(submit_one, 0o755)
    
    # Create experiment summary
    summary_file = job_dir / 'experiments_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("PMV Experiments Summary\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {len(experiments)}\n\n")
        
        for exp_info in experiment_info:
            f.write(f"Experiment: {exp_info['name']}\n")
            f.write(f"  Description: {exp_info['description']}\n")
            f.write(f"  Config: {exp_info['config']}\n")
            f.write(f"  Job: {exp_info['job']}\n")
            f.write(f"  Submit: ./jobs/submit_{exp_info['name']}.sh\n")
            f.write("\n")
    
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"Generated {len(experiments)} experiments")
    print(f"\nConfigs saved to: {config_dir}")
    print(f"Job scripts saved to: {job_dir}")
    print(f"Logs will be saved to: {log_dir}")
    print()
    print("To submit all experiments:")
    print(f"  ./jobs/submit_all.sh")
    print()
    print("To submit individual experiments:")
    print(f"  ./jobs/submit_<experiment_name>.sh")
    print()
    print("To check job status:")
    print("  bjobs")
    print("  bjobs -l <job_id>")
    print()
    print("To check logs:")
    print(f"  ls -lh {log_dir}/")
    print(f"  tail -f {log_dir}/<job_id>.out")
    print()
    print(f"Experiment summary: {summary_file}")

if __name__ == "__main__":
    main()

