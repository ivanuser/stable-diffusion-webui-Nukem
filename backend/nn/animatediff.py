# AnimateDiff Motion Modules
# Temporal attention layers for video generation with SD1.5
# Reference: https://github.com/guoyww/AnimateDiff

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from backend.attention import attention_function


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding buffer
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VersatileAttention(nn.Module):
    """Attention block with positional encoding for AnimateDiff.

    Matches the CrossAttention structure from ldm.modules.attention.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        temporal_position_encoding: bool = True,
        temporal_position_encoding_max_len: int = 32,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

        # Positional encoding for temporal dimension
        if temporal_position_encoding:
            self.pos_encoder = PositionalEncoding(
                query_dim,
                max_len=temporal_position_encoding_max_len,
                dropout=dropout
            )
        else:
            self.pos_encoder = None

    def forward(self, x: torch.Tensor, context=None, mask=None) -> torch.Tensor:
        # Add positional encoding
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        context = context if context is not None else x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        out = attention_function(q, k, v, self.heads, mask)
        return self.to_out(out)


class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward network for transformer blocks.

    Matches ldm.modules.attention.FeedForward with glu=True.
    """

    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        # Use GEGLU (glu=True by default in AnimateDiff)
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalTransformerBlock(nn.Module):
    """AnimateDiff transformer block matching checkpoint structure.

    Structure:
    - attention_blocks: ModuleList of VersatileAttention
    - norms: ModuleList of LayerNorm (one per attention block)
    - ff: FeedForward
    - ff_norm: LayerNorm
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_attention_blocks: int = 1,
        temporal_max_len: int = 32,
        dropout: float = 0.0,
        ff_mult: float = 4.0,
    ):
        super().__init__()

        # Multiple attention blocks with their norms
        self.attention_blocks = nn.ModuleList([
            VersatileAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                temporal_position_encoding=True,
                temporal_position_encoding_max_len=temporal_max_len,
            )
            for _ in range(num_attention_blocks)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(dim)
            for _ in range(num_attention_blocks)
        ])

        # Feed-forward
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply each attention block with residual
        for attn, norm in zip(self.attention_blocks, self.norms):
            x = attn(norm(x)) + x

        # Feed-forward with residual
        x = self.ff(self.ff_norm(x)) + x

        return x


class TemporalTransformer3DModel(nn.Module):
    """Temporal transformer matching AnimateDiff checkpoint structure.

    Structure:
    - norm: GroupNorm
    - proj_in: Linear
    - transformer_blocks: ModuleList of TemporalTransformerBlock
    - proj_out: Linear
    """

    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_layers: int = 1,
        num_attention_blocks: int = 1,
        temporal_max_len: int = 32,
        dropout: float = 0.0,
        ff_mult: float = 4.0,
        norm_num_groups: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(norm_num_groups, in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            TemporalTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                num_attention_blocks=num_attention_blocks,
                temporal_max_len=temporal_max_len,
                dropout=dropout,
                ff_mult=ff_mult,
            )
            for _ in range(num_layers)
        ])

        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Apply temporal transformer.

        Args:
            x: Input tensor of shape (batch * num_frames, channels, height, width)
            num_frames: Number of frames

        Returns:
            Output tensor with same shape as input
        """
        batch_frames, c, h, w = x.shape
        batch = batch_frames // num_frames

        # Store residual
        residual = x

        # Normalize
        x = self.norm(x)

        # Reshape: (B*T, C, H, W) -> (B*H*W, T, C)
        x = rearrange(x, "(b t) c h w -> (b h w) t c", t=num_frames)

        # Project in
        x = self.proj_in(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Project out
        x = self.proj_out(x)

        # Reshape back: (B*H*W, T, C) -> (B*T, C, H, W)
        x = rearrange(x, "(b h w) t c -> (b t) c h w", b=batch, h=h, w=w)

        # Residual connection
        return x + residual


class VanillaTemporalModule(nn.Module):
    """Motion module containing a temporal transformer.

    This is the VanillaTemporalModule from AnimateDiff that wraps
    the TemporalTransformer3DModel.
    """

    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_transformer_block: int = 1,  # Default is 1 transformer block
        temporal_max_len: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_transformer_block,
            num_attention_blocks=2,  # ("Temporal_Self", "Temporal_Self") = 2 attention blocks
            temporal_max_len=temporal_max_len,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        return self.temporal_transformer(x, num_frames)


class AnimateDiffModel(nn.Module):
    """AnimateDiff model matching official checkpoint structure.

    Structure:
    - down_blocks.{0,1,2,3}.motion_modules.{0,1,2}
    - mid_block.motion_modules.0 (optional, some checkpoints don't have this)
    - up_blocks.{0,1,2,3}.motion_modules.{0,1,2}
    """

    BLOCK_CHANNELS = {
        "down_blocks.0": 320,
        "down_blocks.1": 640,
        "down_blocks.2": 1280,
        "down_blocks.3": 1280,
        "mid_block": 1280,
        "up_blocks.0": 1280,
        "up_blocks.1": 1280,
        "up_blocks.2": 640,
        "up_blocks.3": 320,
    }

    # Number of VanillaTemporalModule instances per block
    # down_blocks: 2 each, up_blocks: 3 each, mid_block: 1
    MODULES_PER_BLOCK = {
        "down_blocks.0": 2,
        "down_blocks.1": 2,
        "down_blocks.2": 2,
        "down_blocks.3": 2,
        "mid_block": 1,
        "up_blocks.0": 3,
        "up_blocks.1": 3,
        "up_blocks.2": 3,
        "up_blocks.3": 3,
    }

    def __init__(
        self,
        num_attention_heads: int = 8,
        num_transformer_block: int = 1,  # Default is 1 transformer block per motion module
        temporal_max_len: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_frames = 16

        # Create block structure matching checkpoint
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Down blocks
        for i in range(4):
            block_name = f"down_blocks.{i}"
            channels = self.BLOCK_CHANNELS[block_name]
            num_modules = self.MODULES_PER_BLOCK[block_name]

            block = nn.Module()
            block.motion_modules = nn.ModuleList([
                VanillaTemporalModule(
                    in_channels=channels,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=channels // num_attention_heads,
                    num_transformer_block=num_transformer_block,
                    temporal_max_len=temporal_max_len,
                    dropout=dropout,
                )
                for _ in range(num_modules)
            ])
            self.down_blocks.append(block)

        # Mid block (optional - some checkpoints don't have it)
        self.mid_block = nn.Module()
        self.mid_block.motion_modules = nn.ModuleList([
            VanillaTemporalModule(
                in_channels=1280,
                num_attention_heads=num_attention_heads,
                attention_head_dim=1280 // num_attention_heads,
                num_transformer_block=num_transformer_block,
                temporal_max_len=temporal_max_len,
                dropout=dropout,
            )
        ])

        # Up blocks
        for i in range(4):
            block_name = f"up_blocks.{i}"
            channels = self.BLOCK_CHANNELS[block_name]
            num_modules = self.MODULES_PER_BLOCK[block_name]

            block = nn.Module()
            block.motion_modules = nn.ModuleList([
                VanillaTemporalModule(
                    in_channels=channels,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=channels // num_attention_heads,
                    num_transformer_block=num_transformer_block,
                    temporal_max_len=temporal_max_len,
                    dropout=dropout,
                )
                for _ in range(num_modules)
            ])
            self.up_blocks.append(block)

    def set_num_frames(self, num_frames: int):
        """Set the number of frames for video generation."""
        self.num_frames = num_frames

    def get_motion_module_by_channels(self, channels: int) -> Optional[VanillaTemporalModule]:
        """Get a motion module that matches the given channel dimension."""
        # Check down blocks
        for block in self.down_blocks:
            for mm in block.motion_modules:
                if mm.in_channels == channels:
                    return mm

        # Check mid block
        if hasattr(self.mid_block, 'motion_modules'):
            for mm in self.mid_block.motion_modules:
                if mm.in_channels == channels:
                    return mm

        # Check up blocks
        for block in self.up_blocks:
            for mm in block.motion_modules:
                if mm.in_channels == channels:
                    return mm

        return None

    @staticmethod
    def from_pretrained(path: str, **kwargs) -> "AnimateDiffModel":
        """Load AnimateDiff motion modules from a pretrained checkpoint."""
        import os

        model = AnimateDiffModel(**kwargs)

        if not os.path.exists(path):
            print(f"[AnimateDiff] Motion module not found: {path}")
            return model

        # Load weights
        if path.endswith(".safetensors"):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=False)

        # Handle different state dict formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        print(f"[AnimateDiff] Checkpoint has {len(state_dict)} keys")

        # Debug: show some checkpoint keys
        sample_keys = list(state_dict.keys())[:5]
        print(f"[AnimateDiff] Sample checkpoint keys: {sample_keys}")

        # Debug: show model keys
        model_keys = list(model.state_dict().keys())[:5]
        print(f"[AnimateDiff] Sample model keys: {model_keys}")

        # Load weights
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            loaded = len(state_dict) - len(unexpected)
            print(f"[AnimateDiff] Loaded {loaded}/{len(state_dict)} keys from checkpoint")
            if missing:
                print(f"[AnimateDiff] Missing keys: {len(missing)}")
                print(f"[AnimateDiff] Sample missing: {missing[:3]}")
            if unexpected:
                print(f"[AnimateDiff] Unexpected keys: {len(unexpected)}")
                print(f"[AnimateDiff] Sample unexpected: {unexpected[:3]}")
        except Exception as e:
            print(f"[AnimateDiff] Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()

        return model


def get_motion_module_list() -> list:
    """Get list of available motion modules."""
    import os
    from modules import paths

    motion_modules_dir = os.path.join(paths.models_path, "motion_modules")

    if not os.path.exists(motion_modules_dir):
        os.makedirs(motion_modules_dir, exist_ok=True)
        return []

    modules = []
    for f in os.listdir(motion_modules_dir):
        if f.endswith((".safetensors", ".pth", ".ckpt")):
            modules.append(f)

    return sorted(modules)


def load_motion_module(name: str) -> Optional[AnimateDiffModel]:
    """Load a motion module by name."""
    import os
    from modules import paths

    if not name:
        return None

    motion_modules_dir = os.path.join(paths.models_path, "motion_modules")
    path = os.path.join(motion_modules_dir, name)

    if not os.path.exists(path):
        print(f"[AnimateDiff] Motion module not found: {path}")
        return None

    return AnimateDiffModel.from_pretrained(path)
