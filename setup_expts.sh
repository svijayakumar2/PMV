#!/bin/bash
# Setup script for PMV experiments

set -e  # Exit on error

echo "======================================================================"
echo "PMV Stackelberg-Nash Experiment Setup"
echo "======================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found"; exit 1; }

# Create necessary directories
echo "Creating directories..."
mkdir -p pmv/configs/experiments
mkdir -p jobs
mkdir -p ~/.lsbatch
mkdir -p runs/pmv_stackelberg

echo "✓ Directories created"
echo ""

# Make scripts executable
echo "Making scripts executable..."
chmod +x generate_experiments.py
chmod +x pmv/train_qwen.py

echo "✓ Scripts are executable"
echo ""

# Check if config exists
if [ ! -f "pmv/configs/config_pure_stackelberg.yaml" ]; then
    echo "Warning: Base config not found at pmv/configs/config_pure_stackelberg.yaml"
    echo "You may need to create or adjust this file."
fi

echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Generate experiments:"
echo "   python3 generate_experiments.py"
echo ""
echo "2. Submit all jobs:"
echo "   ./jobs/submit_all.sh"
echo ""
echo "3. Or submit individual job:"
echo "   ./jobs/submit_baseline.sh"
echo ""
echo "4. Monitor jobs:"
echo "   bjobs"
echo "   tail -f ~/.lsbatch/<job_id>.out"
echo ""
echo "5. View results:"
echo "   tensorboard --logdir runs/pmv_stackelberg"
echo ""
echo "For more information, see EXPERIMENTS_README.md"
echo ""
