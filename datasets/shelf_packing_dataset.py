import json
import random
import copy
from .utils import to_tensor, read_zarr_with_cache, to_relative_action
import torch
from .base import BaseDataset

from torch.utils.data import Dataset


class ShelfPackingDataset(Dataset):
    """Shelf packing dataset."""
    train_copies = 10
    quat_format= 'wxyz'

    def __init__(
        self,
        root,  # the directory path of the dataset
        copies=None,  # copy the dataset for less loader restarts
        relative_action=False,  # whether to return relative actions
        mem_limit=8,  # cache limit per dataset class in GigaBytes
        actions_only=False,  # return actions without observations
    ):

        # TODO: I am keeping chunk size as 1 for now, but we can change it to a larger value if needed. 
        # I've removed the chunk size argument from the class initialization.

        super().__init__()
        self.copies = self.train_copies if copies is None else copies
        self._relative_action = relative_action
        self._actions_only = actions_only

        # Load all data from data.pth file
        self.data = torch.load(root)
        
        # Sanity check
        self.num_samples = len(self.data['input_pcd'])
        for key in self.data:
            assert len(self.data[key]) == self.num_samples, f'length mismatch in {key}'
        print(f"Found {self.num_samples} samples")

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
            task_id: (N,) uint8
            variation: (N,) uint8
            extrinsics: (N, n_cam, 4, 4) float
            intrinsics: (N, n_cam, 3, 3) float
        }
        """
        # Wrap index to handle dataset copies
        idx = idx % self.num_samples

        # TODO: Hard-coding here to have predictions relative to robot base frame (useful for sim2real transfer later)
        input_pcd = copy.deepcopy(self.data['input_pcd'][idx].unsqueeze(0))
        input_pcd[..., 0] = input_pcd[..., 0] + 0.615
        goal_poses = copy.deepcopy(self.data['goal_poses'][idx].unsqueeze(0))
        goal_poses[..., 0] = goal_poses[..., 0] + 0.615
        
        if self._actions_only:
            return {"action": goal_poses}  # tensor(b, 2, 8) for now
        return {
            "pcd": input_pcd,     # tensor(b, 4096, 3) for now
            "action": goal_poses,  # tensor(b, 1, 8) for now
            "proprioception": self._get_proprioception(idx),  # tensor(b, 1, 8) for now
        }

    def __len__(self):
        return self.copies * self.num_samples


    # def _get_task(self, idx):
    #     return ["task"] * self.chunk_size

    # def _get_instr(self, idx):
    #     return ["instruction"] * self.chunk_size

    # def _get_rgb(self, idx, key='rgb'):
    #     return self._get_attr_by_idx(idx, key, True)

    # def _get_depth(self, idx, key='depth'):
    #     return self._get_attr_by_idx(idx, key, True)

    def _get_proprioception(self, idx):
        return self._get_attr_by_idx(idx, 'proprioception', False)

    # def _get_action(self, idx):
    #     if self._relative_action:
    #         if 'rel_action' in self.annos:
    #             return self._get_attr_by_idx(idx, 'rel_action', False)
    #         else:
    #             action = self._get_attr_by_idx(idx, 'action', False)
    #             prop = self._get_proprioception(idx)[[-1]]
    #             action = to_relative_action(action, prop, self.quat_format)
    #     else:
    #         action = self._get_attr_by_idx(idx, 'action', False)
    #     return action

