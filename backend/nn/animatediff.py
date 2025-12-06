# AnimateDiff Motion Modules
# Temporal attention layers for video generation with SD1.5
# Reference: https://github.com/guoyww/AnimateDiff

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from backend.attention import attention_function


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class VersatileAttention(nn.Module):
    """Attention block used in AnimateDiff temporal transformer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        out = attention_function(q, k, v, self.num_heads, mask)
        return self.to_out(out)


class FeedForward(nn.Module):
    """Feed-forward network for transformer blocks."""

    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalTransformerBlock(nn.Module):
    """A single transformer block with temporal attention.

    Matches AnimateDiff checkpoint structure:
    - attention_blocks.0: temporal self-attention
    - attention_blocks.1: (optional) cross-attention
    - norms.0, norms.1: layer norms
    - ff: feed-forward
    - ff_norm: ff layer norm
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        ff_mult: float = 4.0,
    ):
        super().__init__()

        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            VersatileAttention(dim, num_heads, dim_head, dropout),
        ])

        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim),
        ])

        # Feed-forward
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = self.attention_blocks[0](self.norms[0](x)) + x

        # Feed-forward with residual
        x = self.ff(self.ff_norm(x)) + x

        return x


class TemporalTransformer(nn.Module):
    """Temporal transformer module.

    Matches AnimateDiff checkpoint structure:
    - norm: input layer norm
    - proj_in: input projection
    - transformer_blocks: list of transformer blocks
    - proj_out: output projection
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        dim_head: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        ff_mult: float = 4.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        inner_dim = num_heads * dim_head

        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            TemporalTransformerBlock(
                dim=inner_dim,
                num_heads=num_heads,
                dim_head=dim_head,
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


class MotionModule(nn.Module):
    """Motion module containing a temporal transformer.

    This wraps a TemporalTransformer and is inserted at specific
    positions in the UNet.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        dim_head: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.temporal_transformer = TemporalTransformer(
            in_channels=in_channels,
            num_heads=num_heads,
            dim_head=dim_head,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        return self.temporal_transformer(x, num_frames)


class AnimateDiffModel(nn.Module):
    """AnimateDiff model matching the official checkpoint structure.

    Structure:
    - down_blocks.{0,1,2,3}.motion_modules.{0,1}
    - mid_block.motion_modules.0
    - up_blocks.{0,1,2,3}.motion_modules.{0,1,2}
    """

    # SD1.5 UNet block channel dimensions
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

    # Number of motion modules per block
    MODULES_PER_BLOCK = {
        "down_blocks.0": 2,
        "down_blocks.1": 2,
        "down_blocks.2": 2,
        "down_blocks.3": 1,  # No downsampling in last block
        "mid_block": 1,
        "up_blocks.0": 3,
        "up_blocks.1": 3,
        "up_blocks.2": 3,
        "up_blocks.3": 3,
    }

    def __init__(
        self,
        num_heads: int = 8,
        dim_head: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_frames = 16

        # Create block structure matching checkpoint
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.mid_block = None

        # Down blocks
        for i in range(4):
            block_name = f"down_blocks.{i}"
            channels = self.BLOCK_CHANNELS[block_name]
            num_modules = self.MODULES_PER_BLOCK[block_name]

            block = nn.Module()
            block.motion_modules = nn.ModuleList([
                MotionModule(channels, num_heads, dim_head, num_layers, dropout)
                for _ in range(num_modules)
            ])
            self.down_blocks.append(block)

        # Mid block
        self.mid_block = nn.Module()
        self.mid_block.motion_modules = nn.ModuleList([
            MotionModule(1280, num_heads, dim_head, num_layers, dropout)
        ])

        # Up blocks
        for i in range(4):
            block_name = f"up_blocks.{i}"
            channels = self.BLOCK_CHANNELS[block_name]
            num_modules = self.MODULES_PER_BLOCK[block_name]

            block = nn.Module()
            block.motion_modules = nn.ModuleList([
                MotionModule(channels, num_heads, dim_head, num_layers, dropout)
                for _ in range(num_modules)
            ])
            self.up_blocks.append(block)

    def set_num_frames(self, num_frames: int):
        """Set the number of frames for video generation."""
        self.num_frames = num_frames

    def get_motion_module_by_channels(self, channels: int) -> Optional[MotionModule]:
        """Get a motion module that matches the given channel dimension.

        Returns the first motion module with matching channels.
        """
        # Check down blocks
        for block in self.down_blocks:
            for mm in block.motion_modules:
                if mm.in_channels == channels:
                    return mm

        # Check mid block
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

        # Load weights
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            loaded = len(state_dict) - len(unexpected)
            print(f"[AnimateDiff] Loaded {loaded}/{len(state_dict)} keys from checkpoint")
            if missing:
                print(f"[AnimateDiff] Missing keys: {len(missing)}")
            if unexpected:
                print(f"[AnimateDiff] Unexpected keys: {len(unexpected)}")
        except Exception as e:
            print(f"[AnimateDiff] Error loading checkpoint: {e}")

        return model


def get_motion_module_list() -> list:
    """Get list of available motion modules from models/motion_modules directory."""
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
