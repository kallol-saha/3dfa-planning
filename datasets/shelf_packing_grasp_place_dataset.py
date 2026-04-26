"""Dataset for the 2-pose (grasp + placement) variant of the packing policy.

Reads the file produced by `save_successful_grasp_place.py`:
    input_pcd:  FloatTensor (N, 4096, 3)
    goal_pose:  FloatTensor (N, 2, 8)   [grasp, placement]; each token is
                xyz(3) + quat_wxyz(4) + gripper_state(1). Index 0 is always
                the grasp pose, index 1 is always the placement pose.
    reward, env_index, node_id: not used by training.

Output (per sample) — matches the dict shape that the shared trainer's
`prepare_batch` reads, so no trainer changes are required:
    pcd:            (1, 4096, 3)
    action:         (1, 2, 8)
    proprioception: (1, 1, 8)   zero placeholder. The 2-pose policy ignores
                                proprioception internally; this field exists
                                only because `BaseTrainTester.prepare_batch`
                                indexes `sample["proprioception"]`.

This file is deliberately kept separate from `shelf_packing_dataset.py` so the
existing placement-only policy and its loader remain untouched.
"""

import copy

import torch
from torch.utils.data import Dataset


# +0.615 offset matches what the existing dataset and inference pipeline do
# to express positions in the robot base frame. See action_sampling.py:149,
# motionplanner.py:144, and shelf_packing_dataset.py:61.
ROBOT_BASE_X_OFFSET = 0.615

# Zero placeholder proprioception. The 2-pose (grasp+place) policy ignores
# this field — it exists only to satisfy the shared trainer's `prepare_batch`,
# which indexes sample["proprioception"]. We still build it on the dataset side
# (rather than mutating the trainer) so the trainer code path is unchanged.
PLACEHOLDER_PROPRIO = torch.zeros(8, dtype=torch.float32)


class ShelfPackingGraspPlaceDataset(Dataset):
    """Successful (grasp, place) pairs harvested from MCTS rollouts."""

    train_copies = 10
    quat_format = "wxyz"

    def __init__(
        self,
        root,            # path to successful_grasp_place.pth
        copies=None,     # repeat dataset to reduce loader restarts
        relative_action=False,
        mem_limit=8,     # accepted for parity with ShelfPackingDataset, unused
        actions_only=False,
    ):
        super().__init__()
        self.copies = self.train_copies if copies is None else copies
        self._relative_action = relative_action
        self._actions_only = actions_only

        self.data = torch.load(root, map_location="cpu")

        for required in ("input_pcd", "goal_pose"):
            if required not in self.data:
                raise KeyError(
                    f"{root} missing key '{required}'. Got keys: {list(self.data.keys())}"
                )

        self.num_samples = self.data["input_pcd"].shape[0]
        if self.data["goal_pose"].shape[0] != self.num_samples:
            raise ValueError(
                f"goal_pose length {self.data['goal_pose'].shape[0]} != "
                f"input_pcd length {self.num_samples}"
            )
        if self.data["goal_pose"].ndim != 3 or self.data["goal_pose"].shape[1:] != (2, 8):
            raise ValueError(
                f"Expected goal_pose shape (N, 2, 8); got {tuple(self.data['goal_pose'].shape)}"
            )

        print(f"[ShelfPackingGraspPlaceDataset] loaded {self.num_samples} samples from {root}")

        self._proprio_placeholder = PLACEHOLDER_PROPRIO.clone()

    def __len__(self):
        return self.copies * self.num_samples

    def __getitem__(self, idx):
        idx = idx % self.num_samples

        # (1, 4096, 3) → shift to robot base frame on the fly so the saved
        # dataset stays in world coordinates.
        input_pcd = copy.deepcopy(self.data["input_pcd"][idx].unsqueeze(0))
        input_pcd[..., 0] = input_pcd[..., 0] + ROBOT_BASE_X_OFFSET

        # (1, 2, 8) — grasp at index 0, placement at index 1.
        goal_poses = copy.deepcopy(self.data["goal_pose"][idx].unsqueeze(0))
        goal_poses[..., 0] = goal_poses[..., 0] + ROBOT_BASE_X_OFFSET

        if self._actions_only:
            return {"action": goal_poses}

        proprio = self._proprio_placeholder.view(1, 1, 8)   # ignored by the model

        return {
            "pcd": input_pcd,            # (1, 4096, 3) in robot base frame
            "action": goal_poses,        # (1, 2, 8)
            "proprioception": proprio,   # (1, 1, 8) placeholder — model ignores it
        }
