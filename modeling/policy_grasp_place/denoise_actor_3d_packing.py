import torch
from torch import nn
from torch.nn import functional as F
import einops

from ..noise_scheduler import fetch_schedulers
from ..utils.layers import AttentionModule
from ..utils.position_encodings import SinusoidalPosEmb, RotaryPositionEncoding3D
from ..utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)


class DenoiseActor(nn.Module):
    """2-pose (grasp + placement) packing policy.

    Differences from the placement-only policy in modeling/policy/:
      * Predicts both poses in one shot. Index 0 is the grasp pose,
        index 1 is the placement pose. The pose order is positional —
        not learned — so the model never has to predict which pose is which.
      * Input is the point cloud only. Proprioception is not used.
        The `forward(...)` signature still accepts `proprioception` for
        compatibility with the shared trainer interface, but the kwarg
        is ignored internally.
      * The point cloud has 4 channels: xyz + a target_mask channel that
        marks which points belong to the object the policy should grasp.
        The mask is fed only into `scene_pos_to_feat` (not into rotary
        positional encodings, which still use xyz only).
      * Gripper state is not predicted. It is hardcoded at the output
        from `gripper_schedule` ([1.0, 0.0] — closed at grasp, open at
        place) and is not part of the loss.
      * Loss applies per-pose weights from `pose_loss_weights` so the
        grasp pose can be weighted higher than the placement pose.
    """

    # Hardcoded action layout. The model neither predicts nor learns these.
    trajectory_length = 2
    gripper_schedule = (1.0, 0.0)        # idx 0 = grasp (closed), idx 1 = place (open)
    pose_loss_weights = (2.0, 1.0)       # grasp gets 2x the weight of placement

    def __init__(self,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=1,         # accepted for parity with original ctor; unused
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 # Training arguments
                 lv2_batch_size=1,
                 # Width of the input fed to `TransformerHead.scene_pos_to_feat`.
                 # 4 = xyz + target-object mask (the original masked variant).
                 # 3 = xyz only (the `target-only` data variant where every
                 # other non-shelved object has been removed from the scene
                 # before sampling, so the mask is implicit). Rotary PE still
                 # uses xyz only — this only affects scene_pos_to_feat.
                 pcd_input_channels=4):
        super().__init__()
        # Arguments to be accessed by the main class
        self._rotation_format = rotation_format
        self._relative = relative
        self._lv2_batch_size = lv2_batch_size
        self._nhist = nhist
        self.pcd_input_channels = int(pcd_input_channels)

        # Per-pose loss weights as a non-learnable buffer so they move with
        # the module and are visible in `state_dict`.
        self.register_buffer(
            "_pose_loss_weights",
            torch.tensor(list(self.pose_loss_weights), dtype=torch.float32),
        )
        self.register_buffer(
            "_gripper_schedule",
            torch.tensor(list(self.gripper_schedule), dtype=torch.float32).view(1, -1, 1),
        )

        # Action decoder, runs at every denoising timestep
        self.traj_encoder = nn.Linear(
            6 if rotation_format == 'euler' else 9,  # XYZ + Euler or 6D
            embedding_dim
        )

        # Prediction head for denoising
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format == 'euler' else 6,
            pcd_feat_dim=self.pcd_input_channels,
        )

        # Noise/denoise schedulers and hyperparameters
        self.position_scheduler, self.rotation_scheduler = fetch_schedulers(
            denoise_model, denoise_timesteps
        )
        self.n_steps = denoise_timesteps

        # Per-pose workspace normalizer. Shape (2, L, ndims_norm):
        # axis 0 = (min, max), axis 1 = pose index (0 grasp, 1 place),
        # axis 2 = the dims being normalized (3 for xyz; 6 if rotation
        # is also normalized in euler mode). Storing one (min, max) pair
        # per pose means the diffusion noise scale is matched to each
        # pose's actual range, instead of being dominated by the union
        # of pickup-region and shelf-interior coordinates.
        ndims_norm = 6 if rotation_format == 'euler' else 3
        self.workspace_normalizer = nn.Parameter(
            torch.stack([
                torch.zeros(self.trajectory_length, ndims_norm),
                torch.ones(self.trajectory_length, ndims_norm),
            ]),
            requires_grad=False,
        )
        self.nrm_dim = int(self.workspace_normalizer.size(-1))

    def encode_inputs(self, rgb3d, rgb2d, pcd, instruction, proprio):
        fixed_inputs = self.encoder(
            rgb3d, rgb2d, pcd, instruction,
            proprio.flatten(1, 2)
        )
        # Query trajectory (for relative trajectory prediction)
        query_trajectory = proprio[:, -1:]
        return (query_trajectory,) + fixed_inputs

    def downsample_pcd(self, pcd, height = 32, width = 32):
        """
        Downsample the point cloud to the given height and width.

        Args:
            - pcd: (B, N, 3)
            - height: int
            - width: int
        """
        # Point cloud
        num_cameras = pcd.shape[1]
        # Interpolate point cloud to get the corresponding locations
        pcd = F.interpolate(
            einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w"),
            (height, width),
            mode='bilinear'     # Bilinear interpolation for spatial up or down sampling.
        )

        # Merge different cameras
        pcd = einops.rearrange(
            pcd,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        return pcd

    def policy_forward_pass(self, trajectory, timestep, pcd):
        """
        Forward pass through the denoising prediction head.

        Args:
            trajectory: (B, traj_len, 9) - noisy trajectory (pos + 6D rot)
            timestep: (B,) - denoising timestep
            pcd: (B, N, pcd_input_channels) - xyz + target_mask. The head
                slices xyz for positional encodings and feeds the full
                vector into the scene-feature MLP.
        """
        trajectory_feats = self.traj_encoder(trajectory)

        # But use positions from unnormalized absolute trajectory
        traj_xyz = self.unnormalize_pos(trajectory)[..., :3]
        if self._relative:  # not used in this codebase
            traj_xyz = torch.cumsum(traj_xyz, dim=1)

        return self.prediction_head(
            trajectory_feats,
            traj_xyz,
            timestep,
            rgb3d_pos=pcd,
        )

    def conditional_sample(self, trajectory, device, pcd):
        """Iterative denoising to generate the noise-free trajectory."""
        self.position_scheduler.set_timesteps(self.n_steps, device=device)
        self.rotation_scheduler.set_timesteps(self.n_steps, device=device)

        timesteps = self.position_scheduler.timesteps
        for t_ind, t in enumerate(timesteps):
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(device).long(),
                pcd,
            )
            out = out[-1]  # keep only the last layer's output
            pos = self.position_scheduler.step(
                out[..., :3], t_ind, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_scheduler.step(
                out[..., 3:], t_ind, trajectory[..., 3:]
            ).prev_sample
            trajectory = torch.cat((pos, rot), -1)

        return trajectory

    def compute_trajectory(self, pcd):
        """
        Generate trajectory from noise via iterative denoising.

        Args:
            pcd: (B, N, pcd_input_channels) - xyz + target_mask.

        Returns:
            trajectory: (B, 2, 8). Position + quaternion (wxyz) + gripper.
                Index 0 = grasp, index 1 = placement. The gripper channel
                is hardcoded ([1.0, 0.0]) — not predicted.
        """
        out_dim = 6 if self._rotation_format == 'euler' else 9
        trajectory = torch.randn(
            size=(pcd.shape[0], self.trajectory_length, out_dim),
            device=pcd.device,
        )
        trajectory = self.conditional_sample(
            trajectory, device=pcd.device, pcd=pcd
        )

        trajectory = self.unconvert_rot(trajectory)        # 9 -> pos + quat
        trajectory = self.unnormalize_pos(trajectory)

        # Append the hardcoded gripper schedule so the output tensor matches
        # the (B, L, 8) shape the rest of the pipeline expects.
        gripper = self._gripper_schedule.expand(trajectory.shape[0], -1, -1)
        return torch.cat([trajectory, gripper], dim=-1)

    def compute_loss(self, gt_trajectory, pcd):
        """
        Args:
            gt_trajectory: (B, 2, 8) from the dataset. We strip the trailing
                gripper channel here because the model never predicts it —
                the gripper schedule is hardcoded at the output.
            pcd: (B, N, pcd_input_channels) — xyz in robot-base frame +
                target_mask channel.
        """
        gt_trajectory = gt_trajectory[..., :7]             # drop gripper
        gt_trajectory = self.normalize_pos(gt_trajectory)
        gt_trajectory = self.convert_rot(gt_trajectory)    # -> pos(3) + 6D rot = 9

        weights = self._pose_loss_weights.view(1, -1, 1)   # (1, L, 1)

        total_loss = 0
        for _ in range(self._lv2_batch_size):
            noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)
            timesteps = self.position_scheduler.sample_noise_step(
                num_noise=len(noise), device=noise.device
            )
            pos_noisy = self.position_scheduler.add_noise(
                gt_trajectory[..., :3], noise[..., :3], timesteps
            )
            rot_noisy = self.rotation_scheduler.add_noise(
                gt_trajectory[..., 3:], noise[..., 3:], timesteps
            )
            noisy_trajectory = torch.cat((pos_noisy, rot_noisy), -1)

            pred = self.policy_forward_pass(noisy_trajectory, timesteps, pcd)

            denoise_target = self.position_scheduler.prepare_target(
                noise, gt_trajectory
            )

            for layer_pred in pred:
                pos_err = (layer_pred[..., :3] - denoise_target[..., :3]).abs()
                rot_err = (layer_pred[..., 3:] - denoise_target[..., 3:]).abs()
                pos_loss = (pos_err * weights).mean()
                rot_loss = (rot_err * weights).mean()
                total_loss = total_loss + 30 * pos_loss + 10 * rot_loss

        return total_loss / self._lv2_batch_size

    def normalize_pos(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = _max - _min

        out = signal.clone()
        out[..., :self.nrm_dim] = (
            (signal[..., :self.nrm_dim] - _min) / diff * 2.0
            - 1.0
        )
        return out

    def unnormalize_pos(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = _max - _min

        out = signal.clone()
        out[..., :self.nrm_dim] = (
            (signal[..., :self.nrm_dim] + 1.0) / 2.0 * diff
            + _min
        )
        return out

    def convert_rot(self, signal):
        # If Euler then no conversion
        if self._rotation_format == 'euler':
            return signal
        # Else assume quaternion
        rot = normalise_quat(signal[..., 3:7])
        res = signal[..., 7:] if signal.size(-1) > 7 else None
        # The following code expects wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            rot = rot[..., (3, 0, 1, 2)]
        # Convert to rotation matrix
        rot = quaternion_to_matrix(rot)
        # Convert to 6D
        if len(rot.shape) == 4:
            B, L, D1, D2 = rot.shape
            rot = rot.reshape(B * L, D1, D2)
            rot = get_ortho6d_from_rotation_matrix(rot)
            rot = rot.reshape(B, L, 6)
        else:
            rot = get_ortho6d_from_rotation_matrix(rot)
        # Concatenate pos, rot, other state info
        signal = torch.cat([signal[..., :3], rot], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        # If Euler then no conversion
        if self._rotation_format == 'euler':
            return signal
        # Else assume quaternion
        res = signal[..., 9:] if signal.size(-1) > 9 else None
        if len(signal.shape) == 3:
            B, L, _ = signal.shape
            rot = signal[..., 3:9].reshape(B * L, 6)
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
            quat = quat.reshape(B, L, 4)
        else:
            rot = signal[..., 3:9]
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
        # The above code handled wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            quat = quat[..., (1, 2, 3, 0)]
        signal = torch.cat([signal[..., :3], quat], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def forward(
        self,
        gt_trajectory,
        pcd,
        proprioception=None,    # accepted for trainer compat; ignored
        run_inference=False,
    ):
        """
        Arguments:
            gt_trajectory: (B, 2, 7) at training. pos + quat_wxyz; the gripper
                channel must be stripped by the dataset since it is not predicted.
            pcd: (B, N, pcd_input_channels) point cloud in robot-base frame
                with a target-mask 4th channel.
            proprioception: ignored. Present only because the shared trainer
                interface (`prepare_batch` / `_model_forward`) passes it.

        Returns:
            - loss scalar at training time, or
            - trajectory (B, 2, 8) at inference (gripper channel is hardcoded).
        """
        del proprioception  # explicitly unused

        if run_inference:
            return self.compute_trajectory(pcd)
        return self.compute_loss(gt_trajectory, pcd)


class TransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=1,        # accepted for parity with original ctor; unused
                 rotary_pe=True,
                 rot_dim=6,
                 pcd_feat_dim=3):
        super().__init__()
        del nhist  # proprio path removed

        self.pcd_feat_dim = pcd_feat_dim

        # Different embeddings
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # Scene token features. Input width = pcd_feat_dim so the MLP can
        # consume xyz alone (legacy 3-channel PCD) or xyz + target_mask
        # (4-channel PCD used by the 2-pose grasp+place policy).
        self.scene_pos_to_feat = nn.Sequential(
            nn.Linear(pcd_feat_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        # 3D rotary positional encoding
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = AttentionModule(
            num_layers=1,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,
            use_adaln=False,
            is_self=False
        )

        # Estimate attends to context (no subsampling)
        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=False
        )

        # Shared attention layers
        self.self_attn = AttentionModule(
            num_layers=num_shared_attn_layers,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rotation_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rot_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

    def forward(self, traj_feats, trajectory, timesteps, rgb3d_pos):
        """
        Arguments:
            traj_feats: (B, trajectory_length, F)
            trajectory: (B, trajectory_length, 3+6+X)
            timesteps: (B,) or (B, 1)
            rgb3d_pos: (B, N, pcd_feat_dim). The first 3 channels are xyz
                (used for rotary positional encoding); any remaining
                channels are extra per-point features (e.g. target_mask)
                that are fed only into `scene_pos_to_feat`.

        Returns:
            list of (B, trajectory_length, 3+6) — pos + 6D rot per token.
            Gripper channel is not predicted.
        """
        _, traj_len, _ = trajectory.shape

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, :]
        traj_feats = traj_feats + traj_time_pos

        traj_xyz = trajectory[..., :3]
        rgb3d_xyz = rgb3d_pos[..., :3]

        time_embs = self.encode_denoising_timestep(timesteps)

        rel_traj_pos, rel_scene_pos, rel_pos = self.get_positional_embeddings(
            traj_xyz, rgb3d_xyz
        )

        # Full vector (xyz + extras) goes into the scene MLP.
        rgb3d_feats = self.scene_pos_to_feat(rgb3d_pos)

        traj_feats = self.cross_attn(
            seq1=traj_feats,
            seq2=rgb3d_feats,
            seq1_pos=rel_traj_pos,
            seq2_pos=rel_scene_pos,
            ada_sgnl=time_embs
        )[-1]

        features = self.get_sa_feature_sequence(traj_feats, rgb3d_feats)
        features = self.self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=rel_pos,
            seq2_pos=rel_pos,
            ada_sgnl=time_embs
        )[-1]

        rotation = self.predict_rot(
            features, rel_pos, time_embs, traj_feats.shape[1]
        )
        position, _ = self.predict_pos(
            features, rel_pos, time_embs, traj_feats.shape[1]
        )

        return [torch.cat((position, rotation), -1)]

    def encode_denoising_timestep(self, timestep):
        """Sinusoidal denoising-step embedding. No proprioception."""
        if timestep.dim() > 1:
            timestep = timestep.squeeze(-1)
        return self.time_emb(timestep)

    def get_positional_embeddings(
        self,
        traj_xyz,
        rgb3d_xyz,
    ):
        # Rotary PE for trajectories and scene positions. Both inputs are
        # strictly xyz; any extra per-point features must be sliced off
        # by the caller before getting here.
        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(rgb3d_xyz)
        rel_pos = torch.cat([rel_traj_pos, rel_scene_pos], dim=1)
        return rel_traj_pos, rel_scene_pos, rel_pos

    def get_sa_feature_sequence(
        self,
        traj_feats,
        rgb3d_feats
    ):
        return torch.cat([traj_feats, rgb3d_feats], 1)

    def predict_pos(self, features, pos, time_embs, traj_len):
        position_features = self.position_self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=pos,
            seq2_pos=pos,
            ada_sgnl=time_embs
        )[-1]
        position_features = position_features[:, :traj_len]
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, pos, time_embs, traj_len):
        rotation_features = self.rotation_self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=pos,
            seq2_pos=pos,
            ada_sgnl=time_embs
        )[-1]
        rotation_features = rotation_features[:, :traj_len]
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation
