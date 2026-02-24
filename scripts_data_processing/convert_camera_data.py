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
import argparse
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


def nearest_to_timestamps(data, data_times, target_times):
    """
    Match data to target timestamps by picking the closest sample.
    
    Args:
        data: (T_data, ...) array of data points
        data_times: (T_data,) array of timestamps for data
        target_times: (T_target,) array of target timestamps
    
    Returns:
        (T_target, ...) array of nearest-matched data
    """
    indices = np.searchsorted(data_times, target_times, side='right') - 1
    indices = np.clip(indices, 0, len(data_times) - 1)
    # Check if the next index is actually closer
    next_indices = np.clip(indices + 1, 0, len(data_times) - 1)
    diff_left = np.abs(target_times - data_times[indices])
    diff_right = np.abs(target_times - data_times[next_indices])
    use_next = diff_right < diff_left
    indices[use_next] = next_indices[use_next]
    return data[indices]


def sample_episodes_from_scenes(scenes_dir, total_count, seed=None):
    """
    Randomly sample episodes across scene directories (s1, s2, ..., sN),
    guaranteeing at least one episode per scene and totalling exactly `total_count`.

    Returns a list of episode paths.
    """
    rng = np.random.default_rng(seed)

    scene_dirs = sorted(
        [d for d in scenes_dir.iterdir() if d.is_dir() and d.name.startswith('s')],
        key=lambda d: int(d.name[1:])
    )
    if not scene_dirs:
        print(f"Error: no scene directories (s1, s2, ...) found in {scenes_dir}")
        sys.exit(1)

    scene_episodes = {}
    for sd in scene_dirs:
        eps = sorted([e for e in sd.iterdir() if e.is_dir() and e.name.startswith('episode_')])
        if not eps:
            print(f"Warning: {sd.name} has no episodes, skipping")
            continue
        scene_episodes[sd.name] = eps

    n_scenes = len(scene_episodes)
    if total_count < n_scenes:
        print(f"Error: requested {total_count} episodes but need at least {n_scenes} (one per scene)")
        sys.exit(1)

    max_total = sum(len(eps) for eps in scene_episodes.values())
    if total_count > max_total:
        print(f"Error: requested {total_count} episodes but only {max_total} available")
        sys.exit(1)

    # Start with 1 per scene, then distribute the remainder randomly
    allocation = {name: 1 for name in scene_episodes}
    remaining = total_count - n_scenes

    # Build pool of (scene_name, available_extra) pairs for weighted sampling
    while remaining > 0:
        expandable = [(name, len(eps) - allocation[name])
                      for name, eps in scene_episodes.items()
                      if allocation[name] < len(eps)]
        if not expandable:
            break
        weights = np.array([extra for _, extra in expandable], dtype=float)
        weights /= weights.sum()
        chosen_idx = rng.choice(len(expandable), p=weights)
        allocation[expandable[chosen_idx][0]] += 1
        remaining -= 1

    # Sample the actual episodes
    selected = []
    for name, eps in scene_episodes.items():
        n = allocation[name]
        chosen = list(rng.choice(eps, size=n, replace=False))
        selected.extend(chosen)

    print(f"Sampled {len(selected)} episodes from {n_scenes} scenes:")
    for name in sorted(scene_episodes.keys(), key=lambda n: int(n[1:])):
        print(f"  {name}: {allocation[name]}/{len(scene_episodes[name])} episodes")

    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(
        description='Convert camera data to replay buffer format.')
    parser.add_argument('collections_dir', type=str,
                        help='Path to collections directory with episode_* subdirs, '
                             'or parent directory of scene folders (s1, s2, ...) when using --multi-scene')
    parser.add_argument('output_path', type=str,
                        help='Output path for the .zarr.zip replay buffer')
    parser.add_argument('--no-crop', action='store_true',
                        help='Keep full image resized to 224x224 instead of cropping bottom-right')
    parser.add_argument('--no-latency', action='store_true',
                        help='Skip time offset and use nearest-neighbor matching instead of interpolation')
    parser.add_argument('--multi-scene', type=int, default=None, metavar='N',
                        help='Sample N total episodes across scene dirs (s1..sN), at least 1 per scene')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for episode sampling in --multi-scene mode')
    args = parser.parse_args()
    
    # Joint time offset: during inference, joint positions arrive before the image
    # due to pipeline delays. Shift joint interpolation time to match inference behavior.
    JOINT_TIME_OFFSET = -0.3  # seconds (joint comes 0.3s before image during inference)
    
    collections_dir = pathlib.Path(args.collections_dir).expanduser().absolute()
    output_path = pathlib.Path(args.output_path).expanduser().absolute()
    
    if not collections_dir.exists():
        print(f"Error: {collections_dir} does not exist")
        sys.exit(1)
    
    if args.multi_scene is not None:
        episodes = sample_episodes_from_scenes(collections_dir, args.multi_scene, seed=args.seed)
    else:
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
            
            T, H, W, C = images.shape
            if not args.no_crop:
                images = images[:, H - 224:, W - 224 - 25: W - 25, :]
            
            # Read joint position data and timestamps
            cur_joint_qpos = read_zarr_v3_array(episode_path / 'cur_joint_qpos')  # (T_joint, N)
            cur_joint_qpos_times = read_zarr_v3_array(episode_path / 'cur_joint_qpos_times')  # (T_joint,)
            
            if args.no_latency:
                cur_joint_qpos_interp = nearest_to_timestamps(
                    cur_joint_qpos, cur_joint_qpos_times, image_times
                )
            else:
                joint_target_times = image_times + JOINT_TIME_OFFSET
                cur_joint_qpos_interp = interpolate_to_timestamps(
                    cur_joint_qpos, cur_joint_qpos_times, joint_target_times
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

