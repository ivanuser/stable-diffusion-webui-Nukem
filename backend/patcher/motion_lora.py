# Motion LoRA loader for AnimateDiff
# Handles loading and applying motion LoRAs to temporal attention modules
# Reference: https://github.com/guoyww/AnimateDiff

import os
from typing import Optional

import torch

from backend import memory_management
from backend.utils import load_torch_file


# Motion LoRA key patterns for AnimateDiff
# These keys target the temporal attention layers in motion modules
MOTION_LORA_KEY_PREFIXES = [
    "motion_modules.",
    "temporal_transformer.",
    "temporal_attn.",
    "mm_",  # Common abbreviation for motion modules
]

# Key mapping from Motion LoRA format to our AnimateDiffModel format
MOTION_LORA_KEY_MAP = {
    # Temporal attention projections
    "to_q": "temporal_attn.to_q",
    "to_k": "temporal_attn.to_k",
    "to_v": "temporal_attn.to_v",
    "to_out.0": "temporal_attn.to_out.0",
    # Position encoding
    "pos_encoder": "pos_encoding",
    "pe": "pos_encoding.pe",
    # Feed-forward
    "ff.net.0": "ff.1",  # Linear
    "ff.net.2": "ff.4",  # Linear
    # Norms
    "norm": "temporal_attn.norm",
}


def is_motion_lora_key(key: str) -> bool:
    """Check if a key belongs to a motion LoRA."""
    key_lower = key.lower()
    for prefix in MOTION_LORA_KEY_PREFIXES:
        if prefix in key_lower:
            return True
    return False


def filter_motion_lora_keys(state_dict: dict) -> tuple[dict, dict]:
    """Separate motion LoRA keys from regular LoRA keys.

    Args:
        state_dict: Full LoRA state dict

    Returns:
        Tuple of (motion_lora_dict, remaining_dict)
    """
    motion_dict = {}
    remaining_dict = {}

    for key, value in state_dict.items():
        if is_motion_lora_key(key):
            motion_dict[key] = value
        else:
            remaining_dict[key] = value

    return motion_dict, remaining_dict


def convert_motion_lora_key(key: str) -> str:
    """Convert motion LoRA key format to AnimateDiffModel key format.

    Args:
        key: Original key from motion LoRA file

    Returns:
        Converted key for AnimateDiffModel
    """
    # Remove common prefixes
    converted = key
    for prefix in MOTION_LORA_KEY_PREFIXES:
        if converted.startswith(prefix):
            converted = converted[len(prefix):]
            break

    # Apply key mapping
    for old_pattern, new_pattern in MOTION_LORA_KEY_MAP.items():
        converted = converted.replace(old_pattern, new_pattern)

    return converted


def parse_motion_lora_weights(state_dict: dict) -> dict:
    """Parse motion LoRA weights into the standard LoRA format.

    AnimateDiff motion LoRAs follow the same format as standard LoRAs:
    - {key}.lora_up.weight / {key}.lora_down.weight
    - {key}.alpha (optional scaling factor)

    Args:
        state_dict: Motion LoRA state dict

    Returns:
        Dict mapping target keys to (patch_type, patch_data) tuples
    """
    patches = {}
    processed_keys = set()

    for key in state_dict.keys():
        if key in processed_keys:
            continue

        # Check for LoRA up/down pairs
        if ".lora_up.weight" in key or ".lora_A.weight" in key:
            base_key = key.replace(".lora_up.weight", "").replace(".lora_A.weight", "")

            # Find corresponding down weight
            down_key = None
            if f"{base_key}.lora_down.weight" in state_dict:
                down_key = f"{base_key}.lora_down.weight"
            elif f"{base_key}.lora_B.weight" in state_dict:
                down_key = f"{base_key}.lora_B.weight"

            if down_key is None:
                continue

            up_weight = state_dict[key]
            down_weight = state_dict[down_key]

            # Get alpha if present
            alpha = None
            alpha_key = f"{base_key}.alpha"
            if alpha_key in state_dict:
                alpha = state_dict[alpha_key].item()
                processed_keys.add(alpha_key)

            # Get dora scale if present
            dora_scale = None
            dora_key = f"{base_key}.dora_scale"
            if dora_key in state_dict:
                dora_scale = state_dict[dora_key]
                processed_keys.add(dora_key)

            # Convert key to target format
            target_key = convert_motion_lora_key(base_key)

            # Store as LoRA patch (up, down, alpha, mid, dora_scale)
            patches[target_key] = ("lora", (up_weight, down_weight, alpha, None, dora_scale))

            processed_keys.add(key)
            processed_keys.add(down_key)

        elif ".lora_down.weight" in key or ".lora_B.weight" in key:
            # Skip - handled with up weights
            continue

        elif ".diff" in key or key.endswith(".weight") and ".lora" not in key:
            # Direct weight diff
            target_key = convert_motion_lora_key(key.replace(".weight", ""))
            patches[target_key] = ("diff", (state_dict[key],))
            processed_keys.add(key)

    return patches


class MotionLoraLoader:
    """Loader for AnimateDiff motion LoRAs.

    Motion LoRAs modify the temporal attention modules to change
    the motion characteristics of generated videos.
    """

    def __init__(self, motion_module):
        """Initialize the motion LoRA loader.

        Args:
            motion_module: AnimateDiffModel instance to apply LoRAs to
        """
        self.motion_module = motion_module
        self.loaded_loras = {}
        self.backup = {}

    def load_lora(self, path: str, strength: float = 1.0) -> bool:
        """Load a motion LoRA from file.

        Args:
            path: Path to the motion LoRA file
            strength: LoRA strength multiplier (0.0 to 1.0+)

        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(path):
            print(f"[MotionLoRA] File not found: {path}")
            return False

        try:
            state_dict = load_torch_file(path, safe_load=True)
        except Exception as e:
            print(f"[MotionLoRA] Failed to load {path}: {e}")
            return False

        # Filter to motion LoRA keys only
        motion_dict, _ = filter_motion_lora_keys(state_dict)

        if not motion_dict:
            print(f"[MotionLoRA] No motion LoRA keys found in {path}")
            return False

        # Parse into patches
        patches = parse_motion_lora_weights(motion_dict)

        if not patches:
            print(f"[MotionLoRA] No valid patches found in {path}")
            return False

        # Apply patches to motion module
        applied_count = self._apply_patches(patches, strength)

        filename = os.path.basename(path)
        self.loaded_loras[filename] = (path, strength)

        print(f"[MotionLoRA] Loaded {filename} with {applied_count} patches at strength {strength}")
        return True

    def _apply_patches(self, patches: dict, strength: float) -> int:
        """Apply LoRA patches to the motion module.

        Args:
            patches: Dict of patches from parse_motion_lora_weights
            strength: Strength multiplier

        Returns:
            Number of patches successfully applied
        """
        applied = 0

        for key, (patch_type, patch_data) in patches.items():
            try:
                # Try to find the target parameter in motion modules
                param = self._get_parameter(key)
                if param is None:
                    continue

                # Backup original weight
                if key not in self.backup:
                    self.backup[key] = param.data.clone()

                # Apply patch
                if patch_type == "lora":
                    up_weight, down_weight, alpha, mid, dora_scale = patch_data
                    self._apply_lora_patch(param, up_weight, down_weight, alpha, strength)
                elif patch_type == "diff":
                    diff_weight = patch_data[0]
                    param.data += strength * diff_weight.to(param.device, param.dtype)

                applied += 1

            except Exception as e:
                print(f"[MotionLoRA] Failed to apply patch {key}: {e}")
                continue

        return applied

    def _get_parameter(self, key: str) -> Optional[torch.nn.Parameter]:
        """Get a parameter from the motion module by key.

        Args:
            key: Parameter key path (e.g., "temporal_blocks.0.temporal_attn.to_q.weight")

        Returns:
            Parameter if found, None otherwise
        """
        parts = key.split(".")
        current = self.motion_module

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif hasattr(current, "motion_modules") and part in current.motion_modules:
                current = current.motion_modules[part]
            elif isinstance(current, torch.nn.ModuleDict) and part in current:
                current = current[part]
            elif isinstance(current, torch.nn.ModuleList):
                try:
                    idx = int(part)
                    current = current[idx]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        if isinstance(current, torch.nn.Parameter):
            return current
        elif hasattr(current, "weight") and isinstance(current.weight, torch.nn.Parameter):
            return current.weight

        return None

    def _apply_lora_patch(
        self,
        param: torch.nn.Parameter,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        alpha: Optional[float],
        strength: float,
    ):
        """Apply a LoRA patch to a parameter.

        Args:
            param: Target parameter
            up_weight: LoRA up projection weight
            down_weight: LoRA down projection weight
            alpha: Optional alpha scaling factor
            strength: Strength multiplier
        """
        device = param.device
        dtype = param.dtype

        up = up_weight.to(device=device, dtype=torch.float32)
        down = down_weight.to(device=device, dtype=torch.float32)

        # Compute alpha scaling
        if alpha is not None:
            scale = alpha / down.shape[0]
        else:
            scale = 1.0

        # Compute LoRA delta: up @ down
        try:
            delta = torch.mm(up.flatten(start_dim=1), down.flatten(start_dim=1))
            delta = delta.reshape(param.shape)
        except RuntimeError:
            # Shape mismatch - try alternative reshape
            delta = torch.mm(up.view(up.shape[0], -1), down.view(down.shape[0], -1).t())
            delta = delta.reshape(param.shape)

        # Apply to parameter
        param.data += (strength * scale * delta).to(dtype)

    def unload_all(self):
        """Restore all parameters to their original values."""
        for key, original in self.backup.items():
            param = self._get_parameter(key)
            if param is not None:
                param.data.copy_(original)

        self.backup.clear()
        self.loaded_loras.clear()
        print("[MotionLoRA] Unloaded all motion LoRAs")

    def unload_lora(self, filename: str):
        """Unload a specific motion LoRA.

        Note: Due to weight merging, this requires reloading all other LoRAs.
        For simplicity, this unloads all and reloads the remaining ones.

        Args:
            filename: Name of the LoRA file to unload
        """
        if filename not in self.loaded_loras:
            return

        # Store remaining LoRAs
        remaining = {k: v for k, v in self.loaded_loras.items() if k != filename}

        # Unload all
        self.unload_all()

        # Reload remaining
        for name, (path, strength) in remaining.items():
            self.load_lora(path, strength)


def get_motion_lora_list() -> list[str]:
    """Get list of available motion LoRAs from models/motion_lora directory."""
    from modules import paths

    motion_lora_dir = os.path.join(paths.models_path, "motion_lora")

    if not os.path.exists(motion_lora_dir):
        os.makedirs(motion_lora_dir, exist_ok=True)
        return []

    loras = []
    for f in os.listdir(motion_lora_dir):
        if f.endswith((".safetensors", ".pth", ".ckpt", ".pt")):
            loras.append(f)

    return sorted(loras)


def get_motion_lora_path(name: str) -> Optional[str]:
    """Get the full path to a motion LoRA file.

    Args:
        name: Name of the motion LoRA file

    Returns:
        Full path if found, None otherwise
    """
    from modules import paths

    if not name:
        return None

    motion_lora_dir = os.path.join(paths.models_path, "motion_lora")
    path = os.path.join(motion_lora_dir, name)

    if os.path.exists(path):
        return path

    return None
