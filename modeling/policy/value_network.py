import torch
from torch import nn
from torch.nn import functional as F

from ..utils.layers import AttentionModule
from ..utils.position_encodings import RotaryPositionEncoding3D


class ValueNetwork(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=1,
                 num_shared_attn_layers=4):
        super().__init__()
        self._nhist = nhist

        # Action encoder
        self.action_embed = nn.Embedding(nhist, embedding_dim)
        self.action_relative_pe = RotaryPositionEncoding3D(embedding_dim)
        self.action_context_head = AttentionModule(
            num_layers=3, d_model=embedding_dim, dim_fw=embedding_dim,
            n_heads=num_attn_heads, rotary_pe=True, use_adaln=False,
            pre_norm=False
        )
        # Scene feature encoder (single shared encoder)
        self.scene_pos_to_feat = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Value prediction head
        self.value_head = ValueTransformerHead(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            nhist=nhist
        )

    def encode_action(self, action, context_feats, context_pos):
        """
        Encode action input (candidate gripper pose) with scene context.

        Args:
            action: (B, nhist, 8) - candidate gripper pose(s) (pos + quat + gripper)
            context_feats: (B, N, embedding_dim) - scene features
            context_pos: (B, N, 3) - scene positions

        Returns:
            action_feats: (B, nhist, embedding_dim) - encoded action features
        """
        # Learnable embedding for action
        action_feats = self.action_embed.weight.unsqueeze(0).repeat(
            len(action), 1, 1
        )  # (B, nhist, embedding_dim)

        # Rotary positional encoding
        action_pos = self.action_relative_pe(action[..., :3])
        context_pos_encoded = self.action_relative_pe(context_pos)

        # Attention to scene tokens
        action_feats = self.action_context_head(
            action_feats, context_feats,
            seq1_pos=action_pos, seq2_pos=context_pos_encoded
        )[-1]

        return action_feats

    def forward(self, pcd, action, target_value=None):
        """
        Arguments:
            pcd: (B, N, 3) point cloud in world coordinates (state)
            action: (B, nhist, 8) or (B, 8) candidate gripper pose(s)
            target_value: (B,) or (B, 1) ground truth values for training

        Returns:
            - loss: scalar, if target_value is provided
            - value: (B, 1), predicted Q-value at inference
        """
        # Handle single-step action
        if action.dim() == 2:
            action = action.unsqueeze(1)

        # Compute scene features (shared across action encoder and value head)
        scene_feats = self.scene_pos_to_feat(pcd)  # (B, N, embedding_dim)

        # Encode action with scene context
        action_feats = self.encode_action(action, scene_feats, pcd)

        # Predict Q-value
        value = self.value_head(
            action_feats=action_feats,
            scene_feats=scene_feats,
        )

        # Training: compute loss
        if target_value is not None:
            if target_value.dim() == 1:
                target_value = target_value.unsqueeze(-1)
            loss = F.l1_loss(value, target_value)
            return loss

        # Inference: return predicted Q-value
        return value


class ValueTransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=1):
        super().__init__()

        # Learnable value query token
        self.value_query = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Project action features for AdaLN conditioning
        self.action_proj = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Cross attention: value query attends to scene
        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,
            use_adaln=True,
            is_self=False
        )

        # Self attention among value query and scene
        self.self_attn = AttentionModule(
            num_layers=num_shared_attn_layers,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,
            use_adaln=True,
            is_self=True
        )

        # Value output head
        self.value_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()  # Constrain output to [0, 1]
        )

    def forward(self, action_feats, scene_feats):
        """
        Arguments:
            action_feats: (B, nhist, F) - encoded action features
            scene_feats: (B, N, F) - scene features from point cloud

        Returns:
            value: (B, 1) - predicted Q-value
        """
        B = action_feats.shape[0]

        # Expand value query for batch
        value_query = self.value_query.expand(B, -1, -1)  # (B, 1, F)

        # Conditioning signal from action (for AdaLN modulation)
        action_flat = action_feats.flatten(1)  # (B, nhist * F)
        ada_sgnl = self.action_proj(action_flat)  # (B, F)

        # Cross attention: value query attends to scene
        value_feats = self.cross_attn(
            seq1=value_query,
            seq2=scene_feats,
            ada_sgnl=ada_sgnl
        )[-1]

        # Self attention: value query + action tokens + scene tokens
        # Action tokens participate directly so the model can attend to them
        features = torch.cat([value_feats, action_feats, scene_feats], dim=1)  # (B, 1+nhist+N, F)
        features = self.self_attn(
            seq1=features,
            seq2=features,
            ada_sgnl=ada_sgnl
        )[-1]

        # Extract value query features and predict
        value_feats = features[:, 0, :]  # (B, F)
        value = self.value_predictor(value_feats)  # (B, 1)

        return value
