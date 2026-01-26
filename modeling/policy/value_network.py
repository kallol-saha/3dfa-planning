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

        # Proprioception encoder
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.proprio_relative_pe = RotaryPositionEncoding3D(embedding_dim)
        self.gripper_context_head = AttentionModule(
            num_layers=3, d_model=embedding_dim, dim_fw=embedding_dim,
            n_heads=num_attn_heads, rotary_pe=True, use_adaln=False,
            pre_norm=False
        )
        # Scene feature encoder
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

    def encode_proprio(self, proprio, context_feats, context_pos):
        """
        Encode proprioception input.

        Args:
            proprio: (B, nhist, 8) - gripper pose history (pos + quat + gripper)
            context_feats: (B, N, embedding_dim) - scene features
            context_pos: (B, N, 3) - scene positions

        Returns:
            proprio_feats: (B, nhist, embedding_dim) - encoded proprioception features
        """
        # Learnable embedding for proprioception
        proprio_feats = self.curr_gripper_embed.weight.unsqueeze(0).repeat(
            len(proprio), 1, 1
        )  # (B, nhist, embedding_dim)

        # Rotary positional encoding
        proprio_pos = self.proprio_relative_pe(proprio[..., :3])
        context_pos_encoded = self.proprio_relative_pe(context_pos)

        # Attention to scene tokens
        proprio_feats = self.gripper_context_head(
            proprio_feats, context_feats,
            seq1_pos=proprio_pos, seq2_pos=context_pos_encoded
        )[-1]

        return proprio_feats

    def forward(self, pcd, proprioception, target_value=None):
        """
        Arguments:
            pcd: (B, N, 3) point cloud in world coordinates
            proprioception: (B, nhist, 8) or (B, 8) current gripper pose(s)
            target_value: (B,) or (B, 1) ground truth values for training

        Returns:
            - loss: scalar, if target_value is provided
            - value: (B, 1), predicted value at inference
        """
        # Handle single-step proprioception
        if proprioception.dim() == 2:
            proprioception = proprioception.unsqueeze(1)

        # Compute scene features for proprio encoding
        context_feats = self.scene_pos_to_feat(pcd)  # (B, N, embedding_dim)

        # Encode proprioception with scene context
        proprio_feats = self.encode_proprio(proprioception, context_feats, pcd)

        # Predict value
        value = self.value_head(
            proprio_feats=proprio_feats,
            scene_feats=context_feats,
            scene_pos=pcd
        )

        # Training: compute loss
        if target_value is not None:
            if target_value.dim() == 1:
                target_value = target_value.unsqueeze(-1)
            # Use L1 loss for regression with sigmoid outputs
            loss = F.l1_loss(value, target_value)
            return loss

        # Inference: return predicted value
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

        # Project proprio features for conditioning
        self.proprio_proj = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Scene token features from 3D positions
        self.scene_pos_to_feat = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 3D rotary positional encoding
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Cross attention: value query attends to scene
        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,  # Value query has no 3D position
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

    def forward(self, proprio_feats, scene_feats, scene_pos):
        """
        Arguments:
            proprio_feats: (B, nhist, F) - encoded proprioception features
            scene_feats: (B, N, F) - scene features from point cloud
            scene_pos: (B, N, 3) - scene positions

        Returns:
            value: (B, 1) - predicted value
        """
        B = proprio_feats.shape[0]

        # Expand value query for batch
        value_query = self.value_query.expand(B, -1, -1)  # (B, 1, F)

        # Conditioning signal from proprioception
        proprio_flat = proprio_feats.flatten(1)  # (B, nhist * F)
        ada_sgnl = self.proprio_proj(proprio_flat)  # (B, F)

        # Build scene features
        scene_feats = self.scene_pos_to_feat(scene_pos)

        # Cross attention: value query attends to scene
        value_feats = self.cross_attn(
            seq1=value_query,
            seq2=scene_feats,
            ada_sgnl=ada_sgnl
        )[-1]

        # Self attention among value query and scene context
        features = torch.cat([value_feats, scene_feats], dim=1)  # (B, 1+N, F)
        features = self.self_attn(
            seq1=features,
            seq2=features,
            ada_sgnl=ada_sgnl
        )[-1]

        # Extract value query features and predict
        value_feats = features[:, 0, :]  # (B, F)
        value = self.value_predictor(value_feats)  # (B, 1)

        return value
