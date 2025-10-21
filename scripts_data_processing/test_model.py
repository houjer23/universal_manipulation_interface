#!/usr/bin/env python3
"""
Test trained model with sample input
Usage: python test_model.py <checkpoint_path>
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace


def test_checkpoint(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    payload = torch.load(checkpoint_path, map_location='cpu')
    cfg = payload['cfg']
    
    # Create workspace and load checkpoint
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get the policy (use EMA model if available)
    policy = workspace.ema_model if workspace.ema_model is not None else workspace.model
    policy.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Horizon: {cfg.horizon}")
    print(f"  Obs dim: {cfg.obs_dim}")
    print(f"  Action dim: {cfg.action_dim}")
    print(f"  Epoch: {payload.get('epoch', 'unknown')}")
    print()
    
    # Create test input (2 observation timesteps, 7 dims each)
    # Format: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper_width]
    test_obs = np.array([
        [0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.08],  # Timestep t-1
        [0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.08],  # Timestep t (current)
    ], dtype=np.float32)
    
    print("Test input (2 timesteps of observations):")
    print(f"  Position: [{test_obs[1, 0]:.3f}, {test_obs[1, 1]:.3f}, {test_obs[1, 2]:.3f}]")
    print(f"  Rotation: [{test_obs[1, 3]:.3f}, {test_obs[1, 4]:.3f}, {test_obs[1, 5]:.3f}]")
    print(f"  Gripper: {test_obs[1, 6]:.3f}")
    print()
    
    # Prepare input
    obs_dict = {
        'obs': torch.from_numpy(test_obs).unsqueeze(0)  # Add batch dimension [1, 2, 7]
    }
    
    print("Running inference...")
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    
    action = result['action'].cpu().numpy()[0]  # Remove batch dim
    
    print(f"✓ Predicted actions ({action.shape[0]} steps):")
    print()
    for i in range(min(5, action.shape[0])):  # Show first 5 steps
        print(f"  Step {i}:")
        print(f"    Position: [{action[i, 0]:.4f}, {action[i, 1]:.4f}, {action[i, 2]:.4f}]")
        print(f"    Rotation: [{action[i, 3]:.4f}, {action[i, 4]:.4f}, {action[i, 5]:.4f}]")
        print(f"    Gripper: {action[i, 6]:.4f}")
    
    if action.shape[0] > 5:
        print(f"  ... ({action.shape[0] - 5} more steps)")
    
    return action


if __name__ == "__main__":
    if len(sys.argv) < 2:
        checkpoint_path = "../data/outputs/2025.10.15/18.00.22_train_simple_task_simple_task/checkpoints/latest.ckpt"
        print(f"No checkpoint provided, using: {checkpoint_path}")
    else:
        checkpoint_path = sys.argv[1]
    
    checkpoint_path = Path(checkpoint_path).expanduser().absolute()
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    action = test_checkpoint(checkpoint_path)
    print()
    print(f"✓ Test complete! Model can predict {action.shape[0]} action steps from 2 observation steps.")

