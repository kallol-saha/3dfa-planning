"""Main script for training and testing."""

import argparse
import os
from pathlib import Path
import sys

import torch
import wandb

from utils.common_utils import str2bool, str_none

# Train Tester:
from train_tester import BaseTrainTester as TrainTester

# Dataset:
from datasets.shelf_packing_dataset import ShelfPackingDataset

# Model:
from modeling.policy.denoise_actor_3d_packing import DenoiseActor


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
    data_path = '/home/ksaha/Research/ModelBasedPlanning/visplanWM/assets/processed_data/shelf_packing_one_object/data.pth'
    arguments = [
        # Dataset/loader arguments
        ('wandb_project_name', str, "3DFA_Planning"),
        ('wandb_run_name', str, "run_1_shelf_packing_one_object"),
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
        dataset_cls=ShelfPackingDataset, 
        model_cls=DenoiseActor
    )

    start_wandb_run(args)
    train_tester.main()

    # Safe program termination
    if torch.distributed.is_initialized():
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
