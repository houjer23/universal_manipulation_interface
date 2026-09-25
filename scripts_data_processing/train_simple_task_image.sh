#!/bin/bash
# Training script for simple task with camera input
cd /home/sulab1/Workspace/jerry/diffusion/universal_manipulation_interface

# Configuration variables
DATASET_PATH="/home/sulab1/Workspace/jerry/diffusion/data/diff_place_50_2.zarr.zip"
NO_CROP=false  # Set to true to use full image with ResNet instead of cropped + CLIP
GPU=0

# Initialize conda for bash
eval "$(conda shell.bash hook)"
conda activate umi_new

echo "Training with dataset: $DATASET_PATH"

EXTRA_OVERRIDES=""
if [ "$NO_CROP" = true ]; then
    echo "No-crop mode: using ResNet encoder with native 240x320 resolution"
    EXTRA_OVERRIDES="policy.obs_encoder.model_name=resnet34.a1_in1k task.image_shape=[3,240,320] policy.obs_encoder.feature_aggregation=avg"
fi
echo ""

CUDA_VISIBLE_DEVICES=$GPU python train.py --config-dir=diffusion_policy/config --config-name=train_simple_task_image \
    task.dataset.dataset_path="$DATASET_PATH" \
    $EXTRA_OVERRIDES

