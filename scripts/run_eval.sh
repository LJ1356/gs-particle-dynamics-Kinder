#!/bin/bash

# ==============================
# User-defined configuration
# ==============================
scenario="bowling"
env_name="gsplat_interaction_nets"
output_root="outputs/$scenario"
gt_delta_root="data/gt_delta"
gt_chamfer_root="data/gt_chamfer"

# ==============================
# Initialize and activate conda
# ==============================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$env_name"
export PYTHONPATH=$(pwd)

# ==============================
# Compute metrics for each experiment
# ==============================
for folder in "$output_root"/*/ ; do
    [ -d "$folder" ] || continue

    folder_name=$(basename "$folder")

    echo "Evaluating: $folder_name"
    python tools/compute_metrics.py \
        --exp_name="$folder_name" \
        --scenario="$scenario" \
        --output_root="$output_root" \
        --chamfer_gt_root="$gt_chamfer_root" \
        --delta_gt_root="$gt_delta_root"
done