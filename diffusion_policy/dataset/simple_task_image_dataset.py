from typing import Dict
import torch
import numpy as np
import copy
import zarr
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class SimpleTaskImageDataset(BaseImageDataset):
    """
    Simple task dataset with camera images.
    Uses camera0_rgb as image input and robot0_joint_positions as both obs and action.
    """
    def __init__(self, 
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            seed=42,
            val_ratio=0.1,
            **kwargs  # Accept extra params for compatibility
            ):
        super().__init__()
        
        # Load replay buffer from zarr.zip
        print(f'Loading ReplayBuffer from {dataset_path}')
        with zarr.ZipStore(dataset_path, mode='r') as zip_store:
            self.replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, store=zarr.MemoryStore())
        print(f'Loaded {self.replay_buffer.n_episodes} episodes!')

        # Parse shape_meta to get rgb and lowdim keys
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps if n_obs_steps is not None else horizon
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        # Build indices
        self.indices = self._build_indices(train_mask)

    def _build_indices(self, episode_mask):
        """Build sampling indices from episodes."""
        episode_ends = self.replay_buffer.episode_ends[:]
        indices = list()
        
        for i in range(len(episode_ends)):
            if episode_mask is not None and not episode_mask[i]:
                continue
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            
            # Create indices for each valid starting position
            for current_idx in range(start_idx, end_idx - self.horizon + 1):
                indices.append((current_idx, start_idx, end_idx))
        
        return indices

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = self._build_indices(self.val_mask)
        val_set.train_mask = self.val_mask
        val_set.val_mask = self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        
        # action normalizer (robot0_joint_positions)
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['robot0_joint_positions'])
        
        # obs normalizers
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        
        # image normalizers
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['robot0_joint_positions'])

    def __len__(self) -> int:
        return len(self.indices)

    def _get_sequence(self, idx):
        """Get a sequence starting at idx."""
        current_idx, start_idx, end_idx = self.indices[idx]
        
        result = dict()
        
        # Get image sequence for observation (n_obs_steps)
        for key in self.rgb_keys:
            # Get observation images (last n_obs_steps before current)
            obs_start = max(start_idx, current_idx - self.n_obs_steps + 1)
            obs_data = self.replay_buffer[key][obs_start:current_idx + 1]
            
            # Pad at beginning if needed
            if obs_data.shape[0] < self.n_obs_steps:
                pad_size = self.n_obs_steps - obs_data.shape[0]
                padding = np.repeat(obs_data[:1], pad_size, axis=0)
                obs_data = np.concatenate([padding, obs_data], axis=0)
            
            result[key] = obs_data
        
        # Get lowdim observation sequence
        for key in self.lowdim_keys:
            obs_start = max(start_idx, current_idx - self.n_obs_steps + 1)
            obs_data = self.replay_buffer[key][obs_start:current_idx + 1]
            
            if obs_data.shape[0] < self.n_obs_steps:
                pad_size = self.n_obs_steps - obs_data.shape[0]
                padding = np.repeat(obs_data[:1], pad_size, axis=0)
                obs_data = np.concatenate([padding, obs_data], axis=0)
            
            result[key] = obs_data
        
        # Get action sequence (full horizon from current position)
        action_data = self.replay_buffer['robot0_joint_positions'][current_idx:current_idx + self.horizon]
        
        # Pad at end if needed
        if action_data.shape[0] < self.horizon:
            pad_size = self.horizon - action_data.shape[0]
            padding = np.repeat(action_data[-1:], pad_size, axis=0)
            action_data = np.concatenate([action_data, padding], axis=0)
        
        result['action'] = action_data
        
        return result

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self._get_sequence(idx)
        
        obs_dict = dict()
        
        # Process RGB images: T,H,W,C -> T,C,H,W, uint8 -> float32 [0, 1]
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
        
        # Process lowdim observations
        for key in self.lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
        
        # Action
        action = data['action'].astype(np.float32)
        
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data
