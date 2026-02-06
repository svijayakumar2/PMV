#!/bin/bash

# run_training.sh - Run Pure Stackelberg Training

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Set environment variables
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0  # Change if you want different GPU

# Create logs directory if it doesn't exist
mkdir -p runs

# Run the training
echo "Starting Pure Stackelberg training..."
echo "Working directory: $(pwd)"
echo "Python path: $PYTHONPATH"

python3 train_pure_stackelberg.py

echo "Training completed!"
