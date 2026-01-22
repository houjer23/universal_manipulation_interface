#!/usr/bin/env python3
"""
Convert data with camera images to replay buffer format.
Actions are interpolated to match camera frame timestamps.

Usage: python convert_camera_data.py <collections_dir> <output_path>

Example:
python convert_camera_data.py /home/sulab1/Workspace/jerry/diffusion/data/collections_camera_2 /home/sulab1/Workspace/jerry/diffusion/data/camera_dataset2.zarr.zip
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
    """Read a zarr v3 array manually, handling multi-chunk arrays"""
    # Read metadata
    meta_file = array_path / 'zarr.json'
    with open(meta_file) as f:
        metadata = json.load(f)
    
    shape = tuple(metadata['shape'])
    dtype = np.dtype(metadata['data_type'])
    chunk_shape = tuple(metadata['chunk_grid']['configuration']['chunk_shape'])
    
    # Calculate number of chunks needed
    num_chunks = tuple((s + cs - 1) // cs for s, cs in zip(shape, chunk_shape))
    
    # Create output array
    result = np.zeros(shape, dtype=dtype)
    
    codec = Zstd(level=0)
    
    # Handle different dimensionalities
    ndim = len(shape)
    
    if ndim == 1:
        # 1D array (e.g., timestamps)
        for i in range(num_chunks[0]):
            chunk_file = array_path / 'c' / str(i)
            if not chunk_file.exists():
                continue
            with open(chunk_file, 'rb') as f:
                compressed_data = f.read()
            decompressed = codec.decode(compressed_data)
            arr = np.frombuffer(decompressed, dtype=dtype).reshape(chunk_shape)
            
            start = i * chunk_shape[0]
            end = min(start + chunk_shape[0], shape[0])
            chunk_end = end - start
            result[start:end] = arr[:chunk_end]
            
    elif ndim == 2:
        # 2D array (e.g., joint positions: T x N)
        for i in range(num_chunks[0]):
            for j in range(num_chunks[1]):
                chunk_file = array_path / 'c' / str(i) / str(j)
                if not chunk_file.exists():
                    continue
                with open(chunk_file, 'rb') as f:
                    compressed_data = f.read()
                decompressed = codec.decode(compressed_data)
                arr = np.frombuffer(decompressed, dtype=dtype).reshape(chunk_shape)
                
                start0, start1 = i * chunk_shape[0], j * chunk_shape[1]
                end0 = min(start0 + chunk_shape[0], shape[0])
                end1 = min(start1 + chunk_shape[1], shape[1])
                result[start0:end0, start1:end1] = arr[:end0-start0, :end1-start1]
                
    elif ndim == 4:
        # 4D array (e.g., images: T x H x W x C)
        for i0 in range(num_chunks[0]):
            for i1 in range(num_chunks[1]):
                for i2 in range(num_chunks[2]):
                    for i3 in range(num_chunks[3]):
                        chunk_file = array_path / 'c' / str(i0) / str(i1) / str(i2) / str(i3)
                        if not chunk_file.exists():
                            continue
                        with open(chunk_file, 'rb') as f:
                            compressed_data = f.read()
                        decompressed = codec.decode(compressed_data)
                        arr = np.frombuffer(decompressed, dtype=dtype).reshape(chunk_shape)
                        
                        starts = [i0 * chunk_shape[0], i1 * chunk_shape[1], 
                                  i2 * chunk_shape[2], i3 * chunk_shape[3]]
                        ends = [min(starts[k] + chunk_shape[k], shape[k]) for k in range(4)]
                        slices_result = tuple(slice(starts[k], ends[k]) for k in range(4))
                        slices_arr = tuple(slice(0, ends[k] - starts[k]) for k in range(4))
                        result[slices_result] = arr[slices_arr]
    else:
        raise ValueError(f"Unsupported array dimensionality: {ndim}")
    
    return result


def interpolate_to_timestamps(data, data_times, target_times):
    """
    Interpolate data to match target timestamps.
    
    Args:
        data: (T_data, D) array of data points
        data_times: (T_data,) array of timestamps for data
        target_times: (T_target,) array of target timestamps
    
    Returns:
        (T_target, D) array of interpolated data
    """
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
        squeeze = True
    else:
        squeeze = False
    
    T_target = len(target_times)
    D = data.shape[1]
    result = np.zeros((T_target, D), dtype=data.dtype)
    
    for d in range(D):
        result[:, d] = np.interp(target_times, data_times, data[:, d])
    
    if squeeze:
        result = result.squeeze(axis=1)
    
    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_camera_data.py <collections_dir> <output_path>")
        print("Example: python convert_camera_data.py ../data/collections_camera dataset.zarr.zip")
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
            # Read image data and timestamps
            images = read_zarr_v3_array(episode_path / 'images')  # (T_img, H, W, C)
            image_times = read_zarr_v3_array(episode_path / 'image_times')  # (T_img,)
            
            # Crop to bottom right quadrant
            T, H, W, C = images.shape
            images = images[:, H - 224:, W - 224 - 25: W - 25, :]  # Bottom right
            
            # Read joint position data and timestamps
            cur_joint_qpos = read_zarr_v3_array(episode_path / 'cur_joint_qpos')  # (T_joint, N)
            cur_joint_qpos_times = read_zarr_v3_array(episode_path / 'cur_joint_qpos_times')  # (T_joint,)
            
            # Interpolate joint positions to camera timestamps
            cur_joint_qpos_interp = interpolate_to_timestamps(
                cur_joint_qpos, cur_joint_qpos_times, image_times
            )
            
            # Build episode data dict
            episode_data = {}
            
            # Camera image (key format: camera0_rgb)
            episode_data['camera0_rgb'] = images.astype(np.uint8)
            
            # Robot state (current joint positions at camera time)
            episode_data['robot0_joint_positions'] = cur_joint_qpos_interp.astype(np.float32)
            
            # Add episode to replay buffer
            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            
        except Exception as e:
            print(f"\nError processing {episode_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nProcessed {out_replay_buffer.n_episodes} episodes")
    print(f"Total steps: {out_replay_buffer.n_steps}")
    
    # Print sample shapes
    if out_replay_buffer.n_episodes > 0:
        sample = out_replay_buffer.get_episode(0)
        print("\nData shapes in replay buffer:")
        for key, value in sample.items():
            print(f"  {key}: {value.shape}")
    
    # Save to disk
    print(f"\nSaving ReplayBuffer to {output_path}")
    if output_path.exists():
        print(f"Warning: {output_path} already exists, overwriting...")
        os.remove(output_path)
    
    with zarr.ZipStore(str(output_path), mode='w') as zip_store:
        out_replay_buffer.save_to_store(store=zip_store)
    
    print("Done!")


if __name__ == "__main__":
    main()

