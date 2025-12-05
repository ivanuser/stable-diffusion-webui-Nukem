# AnimateDiff Temporal Attention Modules
# Reference: https://github.com/guoyww/AnimateDiff
# Adds temporal attention layers to SD1.5 UNet for video generation

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from backend.attention import attention_function


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal position embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TemporalPositionalEncoding(nn.Module):
    """Positional encoding for temporal dimension using sinusoidal embeddings."""

    def __init__(self, dim: int, max_frames: int = 32):
        super().__init__()
        self.dim = dim
        self.max_frames = max_frames
        # Pre-compute position embeddings for efficiency
        pe = sinusoidal_embedding(torch.arange(max_frames), dim)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Add positional encoding to temporal tokens.

        Args:
            x: Input tensor of shape (batch * height * width, num_frames, channels)
            num_frames: Number of frames in sequence

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:num_frames].unsqueeze(0).to(x.dtype)


class TemporalAttention(nn.Module):
    """Temporal self-attention for video generation.

    Performs attention over the temporal (frame) dimension while keeping
    spatial dimensions fixed. This is the core of AnimateDiff.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_frames: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head

        self.scale = dim_head**-0.5

        # Positional encoding
        self.pos_encoding = TemporalPositionalEncoding(dim, max_frames)

        # Projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        # Layer norm
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply temporal attention.

        Args:
            x: Input tensor of shape (batch, channels, height, width) or
               (batch * num_frames, channels, height, width) for video
            num_frames: Number of frames in the video
            mask: Optional attention mask

        Returns:
            Output tensor with same shape as input
        """
        batch_frames, c, h, w = x.shape
        batch = batch_frames // num_frames

        # Reshape: (B*T, C, H, W) -> (B*H*W, T, C)
        x = rearrange(x, "(b t) c h w -> (b h w) t c", t=num_frames)

        # Normalize
        x_norm = self.norm(x)

        # Add positional encoding
        x_norm = self.pos_encoding(x_norm, num_frames)

        # Compute Q, K, V
        q = self.to_q(x_norm)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)

        # Use optimized attention
        out = attention_function(q, k, v, self.num_heads, mask)

        # Project out
        out = self.to_out(out)

        # Residual connection
        x = x + out

        # Reshape back: (B*H*W, T, C) -> (B*T, C, H, W)
        x = rearrange(x, "(b h w) t c -> (b t) c h w", b=batch, h=h, w=w)

        return x


class TemporalTransformerBlock(nn.Module):
    """A transformer block with temporal attention for AnimateDiff.

    This block is inserted after spatial attention blocks in the UNet.
    It only processes the temporal dimension, leaving spatial dimensions unchanged.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        max_frames: int = 32,
        ff_mult: float = 4.0,
    ):
        super().__init__()

        # Temporal attention
        self.temporal_attn = TemporalAttention(
            dim=dim,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            max_frames=max_frames,
        )

        # Feed-forward network
        ff_inner_dim = int(dim * ff_mult)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_inner_dim, dim),
            nn.Dropout(dropout),
        )

        # Initialize output projection to zero for stable training
        self.temporal_attn.to_out = zero_module(self.temporal_attn.to_out)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Apply temporal transformer block.

        Args:
            x: Input tensor of shape (batch * num_frames, channels, height, width)
            num_frames: Number of frames in the video

        Returns:
            Output tensor with same shape as input
        """
        # Temporal attention
        x = self.temporal_attn(x, num_frames)

        # Feed-forward (reshape for linear layers)
        batch_frames, c, h, w = x.shape
        batch = batch_frames // num_frames

        x_flat = rearrange(x, "(b t) c h w -> (b h w) t c", t=num_frames)
        x_flat = x_flat + self.ff(x_flat)
        x = rearrange(x_flat, "(b h w) t c -> (b t) c h w", b=batch, h=h, w=w)

        return x


class MotionModule(nn.Module):
    """Motion module wrapper that contains multiple temporal transformer blocks.

    This module is inserted at specific positions in the UNet to add
    temporal attention capabilities.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        dim_head: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        max_frames: int = 32,
        ff_mult: float = 4.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_frames = None  # Set during forward pass

        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(
                dim=in_channels,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                max_frames=max_frames,
                ff_mult=ff_mult,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, num_frames: int = None) -> torch.Tensor:
        """Apply motion module.

        Args:
            x: Input tensor of shape (batch * num_frames, channels, height, width)
            num_frames: Number of frames (uses stored value if None)

        Returns:
            Output tensor with temporal motion applied
        """
        if num_frames is None:
            num_frames = self.num_frames

        if num_frames is None or num_frames <= 1:
            # No temporal processing needed for single frame
            return x

        for block in self.temporal_blocks:
            x = block(x, num_frames)

        return x


class AnimateDiffModel(nn.Module):
    """AnimateDiff model wrapper.

    This class manages motion modules and their injection into a SD1.5 UNet.
    Motion modules are loaded from pretrained weights or initialized fresh.
    """

    # Block configuration for SD1.5 UNet
    # Maps block names to their channel dimensions
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

    def __init__(
        self,
        num_heads: int = 8,
        dim_head: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        max_frames: int = 32,
    ):
        super().__init__()

        self.num_frames = 16  # Default frame count
        self.max_frames = max_frames

        # Create motion modules for each block
        self.motion_modules = nn.ModuleDict()

        for block_name, channels in self.BLOCK_CHANNELS.items():
            self.motion_modules[block_name.replace(".", "_")] = MotionModule(
                in_channels=channels,
                num_heads=num_heads,
                dim_head=dim_head,
                num_layers=num_layers,
                dropout=dropout,
                max_frames=max_frames,
            )

    def set_num_frames(self, num_frames: int):
        """Set the number of frames for video generation."""
        self.num_frames = min(num_frames, self.max_frames)
        for module in self.motion_modules.values():
            module.num_frames = self.num_frames

    def get_motion_module(self, block_name: str) -> Optional[MotionModule]:
        """Get the motion module for a specific UNet block."""
        key = block_name.replace(".", "_")
        return self.motion_modules.get(key)

    @staticmethod
    def from_pretrained(path: str, **kwargs) -> "AnimateDiffModel":
        """Load AnimateDiff motion modules from a pretrained checkpoint.

        Args:
            path: Path to the motion module checkpoint (.safetensors or .pth)
            **kwargs: Additional arguments passed to AnimateDiffModel

        Returns:
            Initialized AnimateDiffModel with loaded weights
        """
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
            state_dict = torch.load(path, map_location="cpu", weights_only=True)

        # Handle different state dict formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Try to load weights with key remapping if needed
        try:
            model.load_state_dict(state_dict, strict=False)
            print(f"[AnimateDiff] Loaded motion module from: {path}")
        except Exception as e:
            print(f"[AnimateDiff] Warning: Could not load motion module: {e}")

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
    """Load a motion module by name.

    Args:
        name: Name of the motion module file

    Returns:
        Loaded AnimateDiffModel or None if not found
    """
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
