#!/bin/bash

# ==============================
# User-defined configuration
# ==============================
env_name="gsplat_interaction_nets"
config_path="configs/config.yaml"
scenario="bowling"
eval_epoch=50

use_manual_seed=true
seeds=(12345 23456 34567) # Seeds (used only if use_manual_seed=true)
CUDA_VERSION="12.4"

# ==============================
# Set CUDA version
# ==============================
module load cuda/$CUDA_VERSION
echo "Using CUDA version: $CUDA_VERSION"

# ==============================
# Initialize and activate conda
# ==============================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$env_name"
export PYTHONPATH=$(pwd)

echo "Using conda environment: $env_name"

# ==============================
# Function to run one experiment
# ==============================
run_experiment () {
    seed=$1

    if [ -z "$seed" ]; then
        echo "Running WITHOUT seed"
        seed_arg=""
    else
        echo "Running with seed: $seed"
        seed_arg="--seed=$seed"
    fi

    # ==============================
    # Training
    # ==============================
    echo "Starting training..."

    python tools/train.py \
        --config_path="$config_path" \
        --scenario="$scenario" \
        $seed_arg

    echo "Training finished."

    # ==============================
    # Testing
    # ==============================
    echo "Starting evaluation..."

    python tools/test.py \
        --config_path="$config_path" \
        --scenario="$scenario" \
        --epoch="$eval_epoch" \
        $seed_arg

    echo "Finished run."
}


# ==============================
# Main logic
# ==============================
if [ "$use_manual_seed" = false ]; then
    run_experiment ""
else
    for seed in "${seeds[@]}"
    do
        echo "========================================"
        run_experiment "$seed"
    done
fi

echo "All done."