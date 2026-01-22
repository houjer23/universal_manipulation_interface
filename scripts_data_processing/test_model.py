#!/usr/bin/env python3
"""
Test trained model with sample input
Usage: python test_model.py <checkpoint_path>
"""

import sys
import os
import time
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
    
    # Set device to GPU 1
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print GPU information
    if torch.cuda.is_available():
        gpu_id = 1
        print(f"\nGPU Information:")
        print(f"  GPU Name: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
        print(f"  Available GPUs: {torch.cuda.device_count()}")
        
        # Memory info before loading model
        torch.cuda.reset_peak_memory_stats(gpu_id)
        print(f"  Current Memory Allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**2:.2f} MB")
        print(f"  Current Memory Reserved: {torch.cuda.memory_reserved(gpu_id) / 1024**2:.2f} MB")
    print()
    
    # Load checkpoint
    payload = torch.load(checkpoint_path, map_location=device)
    cfg = payload['cfg']
    
    # Create workspace and load checkpoint
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get the policy (use EMA model if available)
    policy = workspace.ema_model if workspace.ema_model is not None else workspace.model
    policy.to(device)
    policy.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Horizon: {cfg.horizon}")
    print(f"  Obs dim: {cfg.obs_dim}")
    print(f"  Action dim: {cfg.action_dim}")
    print(f"  Epoch: {payload.get('epoch', 'unknown')}")
    print()
    
    # Create test input (2 observation timesteps, 3 dims each)
    # Format: [pos_x, pos_y, pos_z]
    test_obs = np.array([
        [0.25095563, 0.39541559, 0.21994509],  # Timestep t-1
        [0.25095563, 0.39541559, 0.21994509],  # Timestep t (current)
    ], dtype=np.float32)
    
    print("Test input (2 timesteps of observations):")
    print(f"  Position: [{test_obs[1, 0]:.3f}, {test_obs[1, 1]:.3f}, {test_obs[1, 2]:.3f}]")
    print()
    
    # Prepare input
    obs_dict = {
        'obs': torch.from_numpy(test_obs).unsqueeze(0).to(device)  # Add batch dimension [1, 2, 3]
    }
    
    # Warm up GPU (run twice to ensure CUDA is initialized)
    print("Warming up GPU...")
    with torch.no_grad():
        _ = policy.predict_action(obs_dict)
        _ = policy.predict_action(obs_dict)
    print("✓ Warm-up complete (CUDA initialized)\n")
    
    # Reset the detailed timing flag so we see breakdown on warmed-up GPU
    policy.model._printed_detailed_timing = False
    
    print("="*80)
    print("RUNNING INFERENCE WITH DETAILED TIMING (After Warm-up)")
    print("="*80)
    
    # Run once with detailed timing (now on warmed GPU)
    start_time = time.time()
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    inference_time = time.time() - start_time
    
    print()
    print("="*80)
    print("MODEL CONFIGURATION")
    print("="*80)
    print(f"num_inference_steps:      {policy.num_inference_steps}")
    print(f"Expected U-Net time:      ~{policy.num_inference_steps * 0.7:.0f}-{policy.num_inference_steps * 1.5:.0f} ms")
    print(f"                          ({policy.num_inference_steps} steps × 0.7-1.5 ms per step)")
    print(f"FPS (single run):         {1.0/inference_time:.2f} Hz")
    print()
    
    action = result['action'].cpu().numpy()[0]  # Remove batch dim
    
    print()
    print(f"✓ Predicted actions ({action.shape[0]} steps):")
    print()
    for i in range(action.shape[0]):  # Show first 5 steps
        print(f"  Step {i}:")
        print(f"    Position: [{action[i, 0]:.4f}, {action[i, 1]:.4f}, {action[i, 2]:.4f}]")
    
    return action


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # checkpoint_path = "../data/outputs/2025.11.11/16.50.28_train_simple_task_simple_task/checkpoints/latest.ckpt"
        checkpoint_path = "../data/outputs/2025.11.24/11.59.55_train_simple_task_simple_task/checkpoints/latest.ckpt"
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

