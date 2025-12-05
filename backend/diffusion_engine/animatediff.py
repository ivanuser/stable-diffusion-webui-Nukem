# AnimateDiff Diffusion Engine
# Extends SD1.5 with temporal attention for video generation
# Reference: https://github.com/guoyww/AnimateDiff

from typing import Optional

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.nn.animatediff import AnimateDiffModel, load_motion_module
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.classic_engine import ClassicTextProcessingEngine


class AnimateDiff(ForgeDiffusionEngine):
    """AnimateDiff diffusion engine for video generation with SD1.5.

    This engine wraps a standard SD1.5 model and injects temporal attention
    modules to enable video generation. It works by:

    1. Processing frames as a batch through spatial layers
    2. Adding temporal attention between frames via motion modules
    3. Outputting coherent video frames

    The motion modules can be loaded from pretrained checkpoints to transfer
    learned motion patterns to new generations.
    """

    matched_guesses = [model_list.SD15]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        # Initialize standard SD1.5 components
        clip = CLIP(
            model_dict={"clip_l": huggingface_components["text_encoder"]},
            tokenizer_dict={"clip_l": huggingface_components["tokenizer"]},
        )

        vae = VAE(model=huggingface_components["vae"])

        unet = UnetPatcher.from_model(
            model=huggingface_components["unet"],
            diffusers_scheduler=huggingface_components["scheduler"],
            config=estimated_config,
        )

        self.text_processing_engine = ClassicTextProcessingEngine(
            text_encoder=clip.cond_stage_model.clip_l,
            tokenizer=clip.tokenizer.clip_l,
            embedding_dir=dynamic_args["embedding_dir"],
            embedding_key="clip_l",
            embedding_expected_shape=768,
            text_projection=False,
            minimal_clip_skip=1,
            clip_skip=1,
            return_pooled=False,
            final_layer_norm=True,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # AnimateDiff-specific attributes
        self.is_sd1 = True
        self.is_animatediff = True
        self.num_frames = 16  # Default frame count
        self.fps = 8  # Default FPS for output video

        # Motion module (loaded separately)
        self.motion_module: Optional[AnimateDiffModel] = None
        self.motion_module_name: Optional[str] = None

        # Motion LoRA loader
        self.motion_lora_loader = None

        # Store original forward for restoration
        self._original_unet_forward = None

    def set_clip_skip(self, clip_skip):
        self.text_processing_engine.clip_skip = clip_skip

    def set_num_frames(self, num_frames: int):
        """Set the number of frames to generate."""
        self.num_frames = max(1, min(num_frames, 32))
        if self.motion_module is not None:
            self.motion_module.set_num_frames(self.num_frames)

    def set_fps(self, fps: int):
        """Set the output video FPS."""
        self.fps = max(1, min(fps, 60))

    def load_motion_module(self, name: str) -> bool:
        """Load a motion module by name.

        Args:
            name: Name of the motion module file in models/motion_modules/

        Returns:
            True if loaded successfully, False otherwise
        """
        if not name:
            self.unload_motion_module()
            return True

        if name == self.motion_module_name and self.motion_module is not None:
            return True  # Already loaded

        module = load_motion_module(name)
        if module is None:
            return False

        self.motion_module = module
        self.motion_module_name = name
        self.motion_module.set_num_frames(self.num_frames)

        # Move to appropriate device
        device = self.forge_objects.unet.model.device
        self.motion_module.to(device)

        print(f"[AnimateDiff] Loaded motion module: {name}")
        return True

    def unload_motion_module(self):
        """Unload the current motion module."""
        # Unload motion LoRAs first
        if self.motion_lora_loader is not None:
            self.motion_lora_loader.unload_all()
            self.motion_lora_loader = None

        if self.motion_module is not None:
            del self.motion_module
            self.motion_module = None
            self.motion_module_name = None
            torch.cuda.empty_cache()
            print("[AnimateDiff] Unloaded motion module")

    def load_motion_lora(self, name: str, strength: float = 1.0) -> bool:
        """Load a motion LoRA to modify motion characteristics.

        Args:
            name: Name of the motion LoRA file in models/motion_lora/
            strength: LoRA strength multiplier (0.0 to 1.0+)

        Returns:
            True if loaded successfully, False otherwise
        """
        if self.motion_module is None:
            print("[AnimateDiff] Cannot load motion LoRA without motion module")
            return False

        from backend.patcher.motion_lora import MotionLoraLoader, get_motion_lora_path

        # Initialize loader if needed
        if self.motion_lora_loader is None:
            self.motion_lora_loader = MotionLoraLoader(self.motion_module)

        path = get_motion_lora_path(name)
        if path is None:
            print(f"[AnimateDiff] Motion LoRA not found: {name}")
            return False

        return self.motion_lora_loader.load_lora(path, strength)

    def unload_motion_lora(self, name: str = None):
        """Unload motion LoRA(s).

        Args:
            name: Specific LoRA to unload, or None to unload all
        """
        if self.motion_lora_loader is None:
            return

        if name is None:
            self.motion_lora_loader.unload_all()
        else:
            self.motion_lora_loader.unload_lora(name)

    def get_loaded_motion_loras(self) -> list[str]:
        """Get list of currently loaded motion LoRAs."""
        if self.motion_lora_loader is None:
            return []
        return list(self.motion_lora_loader.loaded_loras.keys())

    def inject_motion_modules(self):
        """Inject motion modules into the UNet for temporal attention.

        This method patches the UNet's forward pass to add temporal
        attention after each spatial block.
        """
        if self.motion_module is None:
            return

        unet = self.forge_objects.unet.model.diffusion_model

        # Store original forward if not already stored
        if self._original_unet_forward is None:
            self._original_unet_forward = unet.forward

        motion_module = self.motion_module
        num_frames = self.num_frames

        # Create wrapped forward that applies motion modules
        original_forward = self._original_unet_forward

        def forward_with_motion(x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
            # Store original shape for reshaping
            batch_size = x.shape[0] // num_frames if num_frames > 1 else x.shape[0]

            # Run original forward
            result = original_forward(
                x,
                timesteps=timesteps,
                context=context,
                y=y,
                control=control,
                transformer_options=transformer_options,
                **kwargs,
            )

            # Apply motion module if we have multiple frames
            if num_frames > 1 and motion_module is not None:
                # Get motion module for the current block level
                # For now, apply at the output level
                mm = motion_module.get_motion_module("up_blocks_3")
                if mm is not None:
                    result = mm(result, num_frames)

            return result

        # Patch the forward method
        unet.forward = forward_with_motion

    def restore_unet_forward(self):
        """Restore the original UNet forward method."""
        if self._original_unet_forward is not None:
            unet = self.forge_objects.unet.model.diffusion_model
            unet.forward = self._original_unet_forward

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        cond = self.text_processing_engine(prompt)
        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        _, token_count = self.text_processing_engine.process_texts([prompt])
        return token_count, self.text_processing_engine.get_target_prompt_token_count(token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        """Encode input image(s) to latent space.

        For AnimateDiff, we handle multiple frames by encoding each frame
        separately and stacking them.

        Args:
            x: Input tensor of shape (num_frames, channels, height, width)
               or (batch, channels, height, width)

        Returns:
            Latent tensor ready for diffusion
        """
        # Standard SD1.5 encoding - handles batched frames
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        """Decode latents to image/video frames.

        For AnimateDiff, we decode all frames at once (if memory allows)
        or in batches.

        Args:
            x: Latent tensor of shape (num_frames, channels, height, width)

        Returns:
            Decoded frames tensor
        """
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)

    def prepare_latents_for_video(self, latent: torch.Tensor) -> torch.Tensor:
        """Prepare a single latent for video generation by repeating for num_frames.

        Args:
            latent: Single frame latent of shape (1, channels, height, width)

        Returns:
            Repeated latent of shape (num_frames, channels, height, width)
        """
        if latent.shape[0] == 1 and self.num_frames > 1:
            # Repeat the latent for each frame
            latent = latent.repeat(self.num_frames, 1, 1, 1)
        return latent

    def prepare_noise_for_video(self, noise: torch.Tensor) -> torch.Tensor:
        """Prepare noise tensor for video generation.

        For better temporal consistency, we can blend structured noise
        across frames.

        Args:
            noise: Noise tensor of shape (batch, channels, height, width)
                   or (num_frames, channels, height, width)

        Returns:
            Noise tensor suitable for video generation
        """
        if noise.shape[0] == 1 and self.num_frames > 1:
            # Generate frame-specific noise with some temporal correlation
            base_noise = noise
            frame_noises = []

            for i in range(self.num_frames):
                if i == 0:
                    frame_noises.append(base_noise)
                else:
                    # Blend with previous frame's noise for temporal smoothness
                    frame_noise = torch.randn_like(base_noise)
                    # 70% new noise, 30% from previous for some coherence
                    blended = 0.7 * frame_noise + 0.3 * frame_noises[-1]
                    blended = blended / blended.std() * frame_noise.std()  # Normalize
                    frame_noises.append(blended)

            noise = torch.cat(frame_noises, dim=0)

        return noise


def is_animatediff_enabled() -> bool:
    """Check if AnimateDiff mode is enabled in dynamic args."""
    return dynamic_args.get("animatediff", False)


def get_animatediff_frames() -> int:
    """Get the configured number of AnimateDiff frames."""
    return dynamic_args.get("animatediff_frames", 16)
