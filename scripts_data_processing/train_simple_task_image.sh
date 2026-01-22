#!/bin/bash
# Training script for simple task with camera input
cd /home/sulab1/Workspace/jerry/diffusion/universal_manipulation_interface

# Initialize conda for bash
eval "$(conda shell.bash hook)"
conda activate umi_new

python train.py --config-dir=diffusion_policy/config --config-name=train_simple_task_image

