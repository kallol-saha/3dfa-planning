"""Training entry point for the 2-pose (grasp + placement) packing policy.

This is a sibling of `train_for_shelf_packing.py` that swaps in:
  - `ShelfPackingGraspPlaceDataset` (reads successful_grasp_place.pth and emits
    actions of shape (B, 2, 8))
  - `policy_grasp_place.denoise_actor_3d_packing.DenoiseActor` (predicts 2 poses
    per call instead of 1).

The original training script and the placement-only policy/dataset are left
unchanged.
"""

import argparse
import os
from pathlib import Path
import sys

import torch
from torch import nn
import wandb

from utils.common_utils import str2bool, str_none

# Train Tester:
from train_tester import BaseTrainTester

# Dataset (2-pose grasp+place variant):
from datasets.shelf_packing_grasp_place_dataset import (
    ShelfPackingGraspPlaceDataset,
    ROBOT_BASE_X_OFFSET,
)

# Model (2-pose grasp+place variant):
from modeling.policy_grasp_place.denoise_actor_3d_packing import DenoiseActor


# Keys we want to keep in wandb plots — the loss-style metrics only.
# `compute_metrics` also produces accuracy and gripper metrics; those are
# noisy or meaningless for the 2-pose policy (gripper is hardcoded), so we
# filter them out before logging.
_WANDB_LOSS_SUBSTRINGS = ("traj_pos_l2", "traj_rot_l1")


def _filter_to_losses(metrics):
    return {k: v for k, v in metrics.items() if any(s in k for s in _WANDB_LOSS_SUBSTRINGS)}


class TrainTester(BaseTrainTester):
    """Two overrides relative to BaseTrainTester:
      * `evaluate_nsteps` filters wandb metrics down to the loss keys.
      * `get_workspace_normalizer` fits min/max from the training data
        instead of falling through to the hardcoded `scene_bounds`
        (the data-fit branch in BaseTrainTester is commented out).
    """

    def evaluate_nsteps(self, *args, **kwargs):
        metrics = super().evaluate_nsteps(*args, **kwargs)
        return _filter_to_losses(metrics)

    def get_workspace_normalizer(self, ndims=3):
        """Per-pose min/max of (grasp, place) positions in robot-base frame.

        Returns a `(2, L, ndims)` parameter where axis 0 is (min, max),
        axis 1 is the pose index (0 grasp, 1 place), and axis 2 is xyz.
        Per-pose fitting matches the diffusion noise scale to each pose's
        actual range — pickup-region and shelf-interior have very different
        per-axis spreads, so a single combined normalizer wastes SNR.

        The dataset's `__getitem__` adds ROBOT_BASE_X_OFFSET to the x
        channel before handing samples to the trainer; we mirror that
        shift here so the normalizer is computed in the same frame the
        model trains in.
        """
        print("[grasp_place] Fitting per-pose workspace_normalizer from training data...")

        data = torch.load(self.args.train_data_dir, map_location="cpu")
        if "goal_pose" not in data:
            raise KeyError(
                f"{self.args.train_data_dir} has no 'goal_pose' field "
                f"(got {list(data.keys())})."
            )

        # (N, L, 8) -> (N, L, ndims)
        positions = data["goal_pose"][..., :ndims].clone().float()
        positions[..., 0] = positions[..., 0] + ROBOT_BASE_X_OFFSET

        buf = float(self.args.workspace_normalizer_buffer)
        # Min/max along the sample axis only — keeps per-pose dim.
        min_ = positions.min(dim=0).values - buf   # (L, ndims)
        max_ = positions.max(dim=0).values + buf   # (L, ndims)

        n, L, _ = positions.shape
        print(f"[grasp_place] data-fit per-pose normalizer (n={n} samples, L={L}, buffer={buf}):")
        names = ["grasp", "place"] if L == 2 else [f"pose_{i}" for i in range(L)]
        for i in range(L):
            print(
                f"  {names[i]:>5s}: "
                f"min={[round(v, 4) for v in min_[i].tolist()]}  "
                f"max={[round(v, 4) for v in max_[i].tolist()]}"
            )

        return nn.Parameter(
            torch.stack([min_, max_]).float(),     # (2, L, ndims)
            requires_grad=False,
        )


# Helper function to find run ID by name
def find_run_id(project_name, run_name):
    api = wandb.Api()
    runs = api.runs(project_name)
    for run in runs:
        if run.name == run_name:
            return run.id
    return None

def start_wandb_run(args):
    run_name = args.wandb_run_name
    run_id = find_run_id(args.wandb_project_name, run_name)
    if run_id:
        print(f"Resuming run: {run_name} (ID: {run_id})")
        wandb.init(project=args.wandb_project_name, id=run_id, resume="must")
    else:
        print(f"Creating new run: {run_name}")
        wandb.init(project=args.wandb_project_name, name=run_name)
    # Save args to wandb
    wandb.config.update(args)

def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    data_path = '/home/ksaha/Research/ModelBasedPlanning/visplanWM/assets/processed_data/successful_grasp_place.pth'
    arguments = [
        # Dataset/loader arguments
        ('wandb_project_name', str, "Value_Function_Planning"),
        ('wandb_run_name', str, "grasp_place_policy_run_3"),
        ('train_data_dir', Path, data_path),
        ('num_workers', int, 4),
        ('batch_size', int, 64),
        ('batch_size_val', int, 64),
        ('chunk_size', int, 1),
        ('memory_limit', float, 8),  # cache limit in GB
        # Logging arguments
        # ('base_log_dir', Path, Path(__file__).parent / "train_logs"),
        ('base_log_dir', Path, "/home/ksaha/Research/ModelBasedPlanning/visplanWM/models/flowmatch_actor/train_logs"),
        # Training and testing arguments
        ('checkpoint', str_none, 'checkpoints'),  # TODO: Change to checkpoint file if it is there
        ('val_freq', int, 100),
        ('vis_freq', int, 1000),            # NOTE: Should be a multiple of val_freq
        ('interm_ckpt_freq', int, 3000),       # NOTE: Should be a multiple of val_freq
        ('eval_only', str2bool, False),
        ('lr', float, 1e-4),
        ('backbone_lr', float, 1e-4),
        ('lr_scheduler', str, "constant"),
        ('wd', float, 5e-3),
        ('train_iters', int, 600000),
        ('use_compile', str2bool, False),
        ('use_ema', str2bool, False),
        ('lv2_batch_size', int, 1),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),       # !!! TODO: Change here for using different models
        ('bimanual', str2bool, False),
        ('keypose_only', str2bool, True),
        ('pre_tokenize', str2bool, True),
        ('custom_img_size', int, None),
        ('workspace_normalizer_buffer', float, 0.04),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('finetune_backbone', str2bool, False),
        ('finetune_text_encoder', str2bool, False),
        ('fps_subsampling_factor', int, 5),
        # Model arguments: encoder and head
        ('embedding_dim', int, 120),  # divisible by num_attn_heads
        ('num_attn_heads', int, 8),
        ('num_vis_instr_attn_layers', int, 3),
        ('num_history', int, 1),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_wxyz'),
        ('denoise_timesteps', int, 10),
        ('denoise_model', str, "rectified_flow")
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def suppress_output_on_non_main():
    if int(os.environ.get("RANK", 0)) != 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    log_dir = args.base_log_dir / args.wandb_project_name / args.wandb_run_name
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count:", torch.cuda.device_count())
    # args.local_rank = int(os.environ["LOCAL_RANK"])
    suppress_output_on_non_main()

    # DDP initialization (NOTE: !!! Disabling Distributed Training for now)
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    # Select dataset and model classes
    # dataset_class = fetch_dataset_class(args.dataset)
    # model_class = fetch_model_class(args.model_type)

    train_tester = TrainTester(
        args=args,
        dataset_cls=ShelfPackingGraspPlaceDataset,
        model_cls=DenoiseActor
    )

    start_wandb_run(args)       # NOTE: Comment out here for disabling wandb
    train_tester.main()

    # Safe program termination
    if torch.distributed.is_initialized():
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
