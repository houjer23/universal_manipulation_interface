#!/usr/bin/env python3
"""
Convert lowdim data to replay buffer format (no camera data)
Usage: python convert_lowdim_data.py <collections_dir> <output_path>

python convert_lowdim_data.py /home/sulab1/Workspace/jerry/diffusion/data/collections /home/sulab1/Workspace/jerry/diffusion/data/joint_positions_dataset.zarr.zip
"""

import sys
import os
import pathlib
import zarr
import numpy as np
import json
from tqdm import tqdm
from numcodecs import Zstd

# Add the parent directory to the Python path to find diffusion_policy
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
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
            cur_joint_qpos = read_zarr_v3_array(episode_path / 'cur_joint_qpos')  # (T, N)
            
            T, num_joints = cur_joint_qpos.shape
            
            # Simply use the joint positions as the data
            episode_data = {}
            
            # Use cur_joint_qpos directly as the robot state
            episode_data['robot0_joint_positions'] = cur_joint_qpos.astype(np.float32)
            
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

