import torch
from torch import nn
from torch.nn import functional as F

from ..utils.layers import AttentionModule
from ..utils.position_encodings import RotaryPositionEncoding3D


class ValueNetwork(nn.Module):
    """Q-function V(s, a_grasp, a_place).

    The action now consists of a (grasp_pose, placement_pose) pair, each 8-D
    (xyz + quat + gripper). Both are MLP-encoded so their full values
    participate directly in the attention tokens (not only via AdaLN).
    """

    ACTION_DIM = 8
    NUM_ACTION_TOKENS = 2  # grasp + placement

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4):
        super().__init__()

        # Encodes the raw 8-D action pose into a token feature
        self.action_encoder = nn.Sequential(
            nn.Linear(self.ACTION_DIM, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        # Learnable type embeddings distinguish grasp (idx 0) from placement (idx 1)
        self.action_type_embed = nn.Embedding(self.NUM_ACTION_TOKENS, embedding_dim)

        self.action_relative_pe = RotaryPositionEncoding3D(embedding_dim)
        self.action_context_head = AttentionModule(
            num_layers=3, d_model=embedding_dim, dim_fw=embedding_dim,
            n_heads=num_attn_heads, rotary_pe=True, use_adaln=False,
            pre_norm=False,
        )

        self.scene_pos_to_feat = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.value_head = ValueTransformerHead(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            num_action_tokens=self.NUM_ACTION_TOKENS,
        )

    def encode_action(self, action, context_feats, context_pos):
        """
        Args:
            action: (B, 2, 8) — [grasp, placement], each pos(3)+quat(4)+gripper(1)
            context_feats: (B, N, F)
            context_pos: (B, N, 3)

        Returns:
            action_feats: (B, 2, F)
        """
        B = action.shape[0]

        # MLP over full 8-D action → action token feature (depends on pose+gripper)
        action_feats = self.action_encoder(action)  # (B, 2, F)

        # Add type embedding so the model knows which token is grasp vs placement
        type_idx = torch.arange(
            self.NUM_ACTION_TOKENS, device=action.device
        ).unsqueeze(0).expand(B, -1)  # (B, 2)
        action_feats = action_feats + self.action_type_embed(type_idx)

        # Rotary PE on action xyz and scene xyz for cross-attention
        action_pos = self.action_relative_pe(action[..., :3])
        context_pos_encoded = self.action_relative_pe(context_pos)

        action_feats = self.action_context_head(
            action_feats, context_feats,
            seq1_pos=action_pos, seq2_pos=context_pos_encoded,
        )[-1]

        return action_feats

    def forward(self, pcd, action, target_value=None, sample_weight=None):
        """
        Args:
            pcd: (B, N, 3)
            action: (B, 2, 8) or (B, 16) — grasp + placement.
            target_value: (B,) or (B, 1) — probability target in [0, 1].
            sample_weight: (B,) or (B, 1) — per-sample loss weight (optional).

        Returns:
            loss (if target_value provided) or predicted value (B, 1).

        Loss: weighted BCE on the sigmoid-output probability. The network's
        final layer is Sigmoid, so we reconstruct BCE from probabilities
        (avoids numerical mismatch with BCE-with-logits).
        """
        if action.dim() == 2:
            assert action.shape[-1] == self.ACTION_DIM * self.NUM_ACTION_TOKENS, (
                f"Flat action must have dim {self.ACTION_DIM * self.NUM_ACTION_TOKENS}, "
                f"got {action.shape[-1]}"
            )
            action = action.view(action.shape[0], self.NUM_ACTION_TOKENS, self.ACTION_DIM)
        assert action.shape[1:] == (self.NUM_ACTION_TOKENS, self.ACTION_DIM), (
            f"Action must be (B, 2, 8), got {tuple(action.shape)}"
        )

        scene_feats = self.scene_pos_to_feat(pcd)
        action_feats = self.encode_action(action, scene_feats, pcd)
        logits = self.value_head(action_feats=action_feats, scene_feats=scene_feats)

        if target_value is not None:
            if target_value.dim() == 1:
                target_value = target_value.unsqueeze(-1)
            bce = F.binary_cross_entropy_with_logits(
                logits, target_value.to(logits.dtype), reduction="none"
            )

            if sample_weight is not None:
                if sample_weight.dim() == 1:
                    sample_weight = sample_weight.unsqueeze(-1)
                w = sample_weight.to(logits.dtype)
                return (bce * w).sum() / w.sum().clamp_min(1e-8)
            return bce.mean()
        return torch.sigmoid(logits)


class ValueTransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 num_action_tokens=2):
        super().__init__()
        self.num_action_tokens = num_action_tokens

        self.value_query = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # AdaLN conditioning signal pooled from all action tokens
        self.action_proj = nn.Sequential(
            nn.Linear(embedding_dim * num_action_tokens, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.cross_attn = AttentionModule(
            num_layers=2, d_model=embedding_dim, dim_fw=embedding_dim,
            dropout=0.1, n_heads=num_attn_heads, pre_norm=False,
            rotary_pe=False, use_adaln=True, is_self=False,
        )

        self.self_attn = AttentionModule(
            num_layers=num_shared_attn_layers, d_model=embedding_dim,
            dim_fw=embedding_dim, dropout=0.1, n_heads=num_attn_heads,
            pre_norm=False, rotary_pe=False, use_adaln=True, is_self=True,
        )

        self.value_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, action_feats, scene_feats):
        """
        Args:
            action_feats: (B, num_action_tokens, F)
            scene_feats: (B, N, F)
        """
        B = action_feats.shape[0]

        value_query = self.value_query.expand(B, -1, -1)  # (B, 1, F)

        action_flat = action_feats.flatten(1)  # (B, num_action_tokens * F)
        ada_sgnl = self.action_proj(action_flat)  # (B, F)

        value_feats = self.cross_attn(
            seq1=value_query, seq2=scene_feats, ada_sgnl=ada_sgnl,
        )[-1]

        features = torch.cat([value_feats, action_feats, scene_feats], dim=1)
        features = self.self_attn(
            seq1=features, seq2=features, ada_sgnl=ada_sgnl,
        )[-1]

        value_feats = features[:, 0, :]
        return self.value_predictor(value_feats)
