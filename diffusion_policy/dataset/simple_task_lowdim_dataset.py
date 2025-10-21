from typing import Dict
import torch
import numpy as np
import copy
import zarr
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class SimpleTaskLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            dataset_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.1
            ):
        super().__init__()
        
        # Load replay buffer from zarr.zip
        print(f'Loading ReplayBuffer from {dataset_path}')
        with zarr.ZipStore(dataset_path, mode='r') as zip_store:
            self.replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, store=zarr.MemoryStore())
        print(f'Loaded {self.replay_buffer.n_episodes} episodes!')

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        # Create sampling indices
        self.indices = self._build_indices(train_mask)

    def _build_indices(self, episode_mask):
        """Build indices for sampling sequences from episodes."""
        indices = []
        episode_ends = self.replay_buffer.episode_ends[:]
        
        for i in range(len(episode_ends)):
            if not episode_mask[i]:
                continue
            
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            
            # Add indices for each valid starting position  
            # Allow padding to handle short episodes
            for idx in range(start_idx, end_idx):
                indices.append((idx, start_idx, end_idx))
        
        return indices

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = val_set._build_indices(~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def __len__(self) -> int:
        return len(self.indices)

    def _get_sequence(self, idx):
        """Get a sequence starting at idx with proper padding."""
        current_idx, start_idx, end_idx = self.indices[idx]
        
        # The sequence length returned should be horizon, not horizon + padding
        # Padding is used to get context for interpolation
        sequence_length = self.horizon
        
        # Extract sequence of length horizon starting at current_idx
        seq_start = max(start_idx, current_idx)
        seq_end = min(end_idx, current_idx + sequence_length)
        
        # Extract data
        result = {}
        for key in ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width']:
            data = self.replay_buffer[key][seq_start:seq_end]
            
            # Pad after if needed to reach horizon length
            if data.shape[0] < sequence_length:
                pad_size = sequence_length - data.shape[0]
                padding = np.repeat(data[-1:], pad_size, axis=0)
                data = np.concatenate([data, padding], axis=0)
            
            result[key] = data
        
        return result

    def _sample_to_data(self, sample):
        # Concatenate robot0 observations - take first n_obs_steps
        # For this simple lowdim case, obs and action use same data
        # but different slicing based on the architecture
        full_seq = np.concatenate([
            sample['robot0_eef_pos'],
            sample['robot0_eef_rot_axis_angle'],
            sample['robot0_gripper_width']
        ], axis=-1)
        
        # obs: typically observation history (not used much in lowdim)
        # action: the full sequence that will be processed by the policy
        data = {
            'obs': full_seq,  # (T, D)
            'action': full_seq,  # (T, D)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._get_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_normalizer(self, mode='limits', **kwargs) -> LinearNormalizer:
        # Collect data from all episodes
        obs_list = []
        action_list = []
        
        # Use all data from replay buffer directly
        for key in ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width']:
            obs_list.append(self.replay_buffer[key][:])
        
        obs_array = np.concatenate(obs_list, axis=-1)
        
        # Construct actions
        action_array = obs_array  # Actions are same as observations for simple task
        
        data = {
            'obs': obs_array,
            'action': action_array
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        return normalizer

