import copy

import torch
from torch.utils.data import Dataset


class ShelfPackingValueDataset(Dataset):
    """Dataset for the grasp+place value network.

    Expects a .pth saved by `save_data_grasp_place.py` with keys:
        input_pcd:  FloatTensor (N, 4096, 3)
        goal_pose:  FloatTensor (N, 2, 8)   [grasp_pose, placement_pose],
                                            each xyz+quat+gripper
        value:      FloatTensor (N,)        success probability target
    """

    train_copies = 10

    def __init__(
        self,
        root,              # path to the .pth file
        copies=None,       # dataset replication to reduce loader restarts
        relative_action=False,  # unused; kept for trainer signature compat
        mem_limit=8,       # unused
        actions_only=False,
    ):
        super().__init__()
        self.copies = self.train_copies if copies is None else copies
        self._actions_only = actions_only

        self.data = torch.load(root)

        self.num_samples = len(self.data['input_pcd'])
        for key in ('input_pcd', 'goal_pose', 'value'):
            assert key in self.data, f"missing key {key} in value dataset"
            assert len(self.data[key]) == self.num_samples, f'length mismatch in {key}'

        # Per-sample reliability: n_leaves = #visited leaves in this node's
        # subtree. Used to weight the loss. Missing → default to 1 (uniform).
        if 'n_leaves' in self.data:
            assert len(self.data['n_leaves']) == self.num_samples
        else:
            self.data['n_leaves'] = torch.ones(self.num_samples, dtype=torch.int32)

        # Validate shape on the first entry
        gp0 = self.data['goal_pose'][0]
        assert gp0.shape == (2, 8), (
            f"goal_pose must be (2, 8) per sample, got {tuple(gp0.shape)}"
        )
        print(f"[ShelfPackingValueDataset] Found {self.num_samples} samples")

    def __getitem__(self, idx):
        idx = idx % self.num_samples

        # pcd: add batch dim; apply robot-frame x-offset (+0.615)
        input_pcd = copy.deepcopy(self.data['input_pcd'][idx].unsqueeze(0))
        input_pcd[..., 0] = input_pcd[..., 0] + 0.615

        # goal_pose: (2, 8) → (1, 2, 8); same x-offset on positions
        goal_poses = copy.deepcopy(self.data['goal_pose'][idx].unsqueeze(0))
        goal_poses[..., 0] = goal_poses[..., 0] + 0.615

        value = copy.deepcopy(self.data['value'][idx].unsqueeze(0))
        n_leaves = self.data['n_leaves'][idx].float().unsqueeze(0)

        if self._actions_only:
            return {"action": goal_poses}
        return {
            "pcd": input_pcd,       # (1, 4096, 3)
            "action": goal_poses,   # (1, 2, 8)
            "value": value,         # (1,)
            "n_leaves": n_leaves,   # (1,) — sample reliability weight
        }

    def __len__(self):
        return self.copies * self.num_samples
