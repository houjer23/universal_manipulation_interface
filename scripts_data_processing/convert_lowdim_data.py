#!/usr/bin/env python3
"""
Convert lowdim data to replay buffer format (no camera data)
Usage: python convert_lowdim_data.py <collections_dir> <output_path>
"""

import sys
import os
import pathlib
import zarr
import numpy as np
import json
from tqdm import tqdm
from numcodecs import Zstd
from diffusion_policy.common.replay_buffer import ReplayBuffer


def read_zarr_v3_array(array_path):
    """Read a zarr v3 array manually"""
    # Read metadata
    meta_file = array_path / 'zarr.json'
    with open(meta_file) as f:
        metadata = json.load(f)
    
    shape = tuple(metadata['shape'])
    dtype = np.dtype(metadata['data_type'])
    chunk_shape = tuple(metadata['chunk_grid']['configuration']['chunk_shape'])
    
    # Read chunk data
    chunk_file = array_path / 'c' / '0' / '0'
    if not chunk_file.exists():
        chunk_file = array_path / 'c' / '0'
    
    with open(chunk_file, 'rb') as f:
        compressed_data = f.read()
    
    # Decompress using zstd
    codec = Zstd(level=0)
    decompressed = codec.decode(compressed_data)
    
    # Convert to numpy array (full chunk)
    arr = np.frombuffer(decompressed, dtype=dtype)
    arr = arr.reshape(chunk_shape)
    
    # Slice to actual shape (remove padding)
    slices = tuple(slice(0, s) for s in shape)
    return arr[slices]


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_lowdim_data.py <collections_dir> <output_path>")
        print("Example: python convert_lowdim_data.py ../data/collections dataset.zarr.zip")
        sys.exit(1)
    
    collections_dir = pathlib.Path(sys.argv[1]).expanduser().absolute()
    output_path = pathlib.Path(sys.argv[2]).expanduser().absolute()
    
    if not collections_dir.exists():
        print(f"Error: {collections_dir} does not exist")
        sys.exit(1)
    
    # Get all episode zarr directories
    episodes = sorted([d for d in collections_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    print(f"Found {len(episodes)} episodes")
    
    if len(episodes) == 0:
        print("No episodes found!")
        sys.exit(1)
    
    # Create replay buffer
    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    
    # Process each episode
    for episode_path in tqdm(episodes, desc="Processing episodes"):
        try:
            # Read data using manual zarr v3 reader
            commands = read_zarr_v3_array(episode_path / 'commands')  # (T, 2)
            cur_joint_qpos = read_zarr_v3_array(episode_path / 'cur_joint_qpos')  # (T, N)
            
            T = commands.shape[0]
            
            # For UMI format, we need:
            # robot0_eef_pos: (T, 3) - end effector position
            # robot0_eef_rot_axis_angle: (T, 3) - rotation as axis-angle
            # robot0_gripper_width: (T, 1) - gripper width
            
            # Since your data has 2D commands, we'll create dummy 3D data
            # Adjust this based on your actual robot configuration
            episode_data = {}
            
            # If commands are 2D (e.g., x, y positions), expand to 3D
            if commands.shape[1] == 2:
                eef_pos = np.zeros((T, 3), dtype=np.float32)
                eef_pos[:, :2] = commands  # x, y from commands
                eef_pos[:, 2] = 0.0  # z = 0
            else:
                eef_pos = commands[:, :3].astype(np.float32)
            
            # Dummy rotation (no rotation)
            eef_rot = np.zeros((T, 3), dtype=np.float32)
            
            # Dummy gripper width (always open, 0.08m)
            gripper_width = np.full((T, 1), 0.08, dtype=np.float32)
            
            episode_data['robot0_eef_pos'] = eef_pos
            episode_data['robot0_eef_rot_axis_angle'] = eef_rot
            episode_data['robot0_gripper_width'] = gripper_width
            
            # Add demo start/end poses (same as first/last poses)
            demo_start_pose = np.zeros((T, 6), dtype=np.float32)
            demo_start_pose[:, :3] = eef_pos[0]
            demo_start_pose[:, 3:] = eef_rot[0]
            
            demo_end_pose = np.zeros((T, 6), dtype=np.float32)
            demo_end_pose[:, :3] = eef_pos[-1]
            demo_end_pose[:, 3:] = eef_rot[-1]
            
            episode_data['robot0_demo_start_pose'] = demo_start_pose
            episode_data['robot0_demo_end_pose'] = demo_end_pose
            
            # Add episode to replay buffer
            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            
        except Exception as e:
            print(f"\nError processing {episode_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nProcessed {out_replay_buffer.n_episodes} episodes")
    
    # Save to disk
    print(f"Saving ReplayBuffer to {output_path}")
    if output_path.exists():
        print(f"Warning: {output_path} already exists, overwriting...")
        os.remove(output_path)
    
    with zarr.ZipStore(str(output_path), mode='w') as zip_store:
        out_replay_buffer.save_to_store(store=zip_store)
    
    print("Done!")

if __name__ == "__main__":
    main()

