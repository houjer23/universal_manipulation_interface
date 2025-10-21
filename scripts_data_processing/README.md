# Data Processing Scripts

## Setup

First, make sure you have the correct dependencies:

```bash
eval "$(conda shell.bash hook)"
conda activate umi_new
pip install --upgrade diffusers==0.11.1 huggingface-hub==0.13.0
```

## Step 1: Convert Low-Dimensional Data

Convert your collected episodes to training format (without camera data):

```bash
conda activate umi_new
python convert_lowdim_data.py <collections_dir> <output_path>
```

Example:
```bash
python convert_lowdim_data.py ../../data/collections ../../data/simple_task.zarr.zip
```

This script reads zarr v3 episodes from your collections directory and converts them to a replay buffer format suitable for training.

## Step 2: Train Simple Task

Train the model on low-dimensional data (no camera):

```bash
bash train_simple_task.sh
```

Or manually:
```bash
conda activate umi_new
cd /home/sulab1/Workspace/jerry/diffusion/universal_manipulation_interface
python train.py --config-dir=diffusion_policy/config --config-name=train_simple_task
```

## What Was Set Up

**For real robot offline training (no environment/simulator):**
1. Created `simple_task_lowdim_dataset.py` - Dataset loader for zarr replay buffers
2. Config files:
   - `diffusion_policy/config/train_simple_task.yaml` - Training on GPU 1
   - `diffusion_policy/config/task/simple_task.yaml` - Task config with env_runner=null
3. Modified `train_diffusion_unet_lowdim_workspace.py` to handle null env_runner

**Data:**
- 22 episodes from `/data/collections/`
- Converted to `/data/simple_task.zarr.zip`
- 122 training samples (7D: 3 pos + 3 rot + 1 gripper)

**Training:**
- Uses GPU 1 (RTX 2080 Ti)
- No environment rollouts (offline training)
- Saves checkpoints every 50 epochs to `data/outputs/`

