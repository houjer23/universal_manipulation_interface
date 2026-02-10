#!/bin/bash
# Training script for simple task with camera input
cd /home/sulab1/Workspace/jerry/diffusion/universal_manipulation_interface

# Configuration variables
DATASET_PATH="/home/sulab1/Workspace/jerry/diffusion/data/camera_dataset2.zarr.zip"

# Initialize conda for bash
eval "$(conda shell.bash hook)"
conda activate umi_new

echo "Training with dataset: $DATASET_PATH"
echo ""

python train.py --config-dir=diffusion_policy/config --config-name=train_simple_task_image \
    task.dataset.dataset_path="$DATASET_PATH"

