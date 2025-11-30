# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Stable Diffusion WebUI Nukem**, a fork of **Forge Classic Neo** - a modern, optimized Stable Diffusion WebUI with video generation capabilities, advanced model support (FLUX, Qwen, Wan 2.2), and cutting-edge performance optimizations. The goal is to provide ComfyUI-level features with a simple, form-based UI instead of node graphs.

**Base:** Forge Classic Neo (neo branch) by Haoming02
**Original:** Based on AUTOMATIC1111's WebUI + lllyasviel's Forge optimizations
**Target:** Advanced AI image & video generation without node-graph complexity

## Quick Start

### Installation & Running

```bash
# First time setup (Windows)
webui-user.bat

# With optimizations (recommended)
# Edit webui-user.bat and add:
set COMMANDLINE_ARGS=--xformers --sage --cuda-malloc --cuda-stream --api

# Linux/macOS - copy a launch script from another WebUI or create webui-user.sh
```

**Python Version:** 3.11.9 recommended
**PyTorch:** 2.9.1+cu128 (older GPUs may need manual downgrade)

### Testing & Development

```bash
# No formal test suite in Forge Neo
# Manual testing via UI is primary method

# Linting (if adding Python code)
ruff check .
ruff format .
```

### Key Command-Line Args

**Performance:**
- `--xformers` - Install xformers for faster generation
- `--sage` - Install SageAttention (fastest, slight quality tradeoff)
- `--flash` - Install FlashAttention
- `--cuda-malloc`, `--cuda-stream`, `--pin-shared-memory` - RTX 30+ optimizations

**Development:**
- `--api` - Enable REST API at `/sdapi/v1/`
- `--port 7860` - Custom port
- `--uv` - Use uv package manager (much faster installs)

**Speed up startup (after first successful launch):**
- `--skip-prepare-environment`
- `--skip-install`
- `--skip-python-version-check`
- `--skip-torch-cuda-test`

## Architecture

### Directory Structure

```
stable-diffusion-webui-Nukem/
├── backend/                    # Core inference engine (NEW in Forge)
│   ├── huggingface/           # Model implementations by org
│   │   ├── black-forest-labs/ # FLUX models
│   │   ├── Qwen/              # Qwen-Image, Qwen-Image-Edit
│   │   ├── Wan-AI/            # Wan 2.2 video models (T2V, I2V)
│   │   ├── stabilityai/       # SDXL, SD 1.5
│   │   └── ...
│   ├── nn/                    # Neural network layers
│   ├── diffusion_engine/      # Diffusion algorithms
│   ├── attention.py           # Attention mechanisms (SageAttention, Flash, etc.)
│   ├── loader.py              # Model loading & detection
│   ├── memory_management.py   # VRAM optimization
│   └── operations.py          # Layer operations (incl. gguf, bnb)
├── modules/                    # A1111 legacy modules
│   ├── processing.py          # Image processing pipeline
│   ├── sd_models.py           # Model management
│   ├── ui.py                  # Main Gradio UI
│   └── ...
├── modules_forge/             # Forge-specific modules
│   ├── main_entry.py          # Main processing entry point
│   ├── supported_controlnet.py
│   ├── supported_preprocessor.py
│   ├── presets.py             # Workflow presets
│   └── forge_canvas/          # Canvas editor
├── extensions-builtin/        # Built-in extensions
│   ├── Lora/
│   ├── canvas-zoom-and-pan/
│   ├── forge_legacy_preprocessors/
│   └── ...
├── extensions/                # User-installed extensions
├── models/                    # Model storage
│   ├── Stable-diffusion/     # Checkpoints (.safetensors, .ckpt)
│   ├── Lora/                 # LoRA files
│   ├── VAE/                  # VAE models
│   ├── ControlNet/           # ControlNet models
│   └── ESRGAN/               # Upscaler models
└── webui.py                   # Main entry point
```

### Key Architectural Differences from A1111

**1. Backend Separation**
- Forge Neo has a completely separate `backend/` directory
- Model-specific code is organized by HuggingFace organization
- This makes adding new model types much cleaner

**2. Memory Management**
- Advanced VRAM tracking in `backend/memory_management.py`
- Automatic model offloading to CPU when needed
- Better than A1111's lowvram mode

**3. Attention Optimizations**
- Auto-selects best available: SageAttention > FlashAttention > xformers > PyTorch > Basic
- Configured via `backend/attention.py`

**4. Operations Layer**
- `backend/operations.py` - Standard operations
- `backend/operations_gguf.py` - GGUF quantized models
- `backend/operations_bnb.py` - bitsandbytes quantization

### Model Loading Flow

1. **User selects checkpoint** in UI
2. `modules/sd_models.py` calls `backend/loader.py`
3. `loader.py` detects model type:
   - Checks config files
   - Checks state_dict keys
   - Checks filename patterns (e.g., "kontext" → FLUX Kontext)
4. Loads appropriate model class from `backend/huggingface/`
5. Applies quantization if needed (fp8, gguf, etc.)
6. Returns model to UI

### Processing Pipeline

1. **UI Input** (`modules/ui.py`) → User enters prompt, settings
2. **Processing Setup** (`modules_forge/main_entry.py`)
   - Creates processing object (Txt2Img, Img2Img, etc.)
   - Loads model if needed
   - Sets up ControlNet, LoRA, etc.
3. **Diffusion** (`backend/diffusion_engine/`)
   - Text encoding via CLIP/T5
   - Latent diffusion loop
   - Sampling via selected sampler
4. **VAE Decode** (`backend/nn/` → VAE models)
5. **Post-processing** (`modules/processing.py`)
   - Face restoration (optional)
   - Upscaling (optional)
   - Metadata embedding
6. **Save** (`modules/images.py`)

### Video Generation (Wan 2.2)

**Models:**
- `Wan2.1-T2V-14B` - Text to Video
- `Wan2.1-I2V-14B` - Image to Video

**Pipeline:**
- Uses `backend/diffusion_engine/wan.py`
- Generates frames in latent space
- Uses FFmpeg to encode final video
- Supports High/Low noise switching via Refiner toggle

**Usage:**
1. Enable Refiner in Settings/Refiner
2. Load Wan model
3. Select txt2img or img2img tab
4. Configure frame count, motion settings
5. Generate → exports video file

## Extension System

Extensions can be added to `extensions/` directory. Each extension can provide:

**Structure:**
```
extensions/my-extension/
├── install.py              # Dependency installation (optional)
├── scripts/                # Custom scripts
│   └── my_script.py
├── javascript/             # UI JavaScript
├── style.css              # UI styling
└── README.md
```

**Script Template:**
```python
import modules.scripts as scripts
import gradio as gr

class MyScript(scripts.Script):
    def title(self):
        return "My Script"

    def ui(self, is_img2img):
        with gr.Accordion("My Script"):
            param = gr.Slider(label="Parameter", minimum=0, maximum=100)
        return [param]

    def run(self, p, param):
        # p is the processing object
        # Modify p or process images here
        proc = process_images(p)
        return proc
```

## Current Features (As of Neo Branch)

### Model Support
- ✅ FLUX (Dev, Krea, Kontext) - including inpainting
- ✅ SDXL, SD 1.5
- ✅ Qwen-Image, Qwen-Image-Edit - multi-image inputs
- ✅ Wan 2.2 - txt2vid, img2vid
- ✅ Lumina-Image-2.0
- ✅ Chroma
- ✅ Z-Image-Turbo
- ✅ LoRA support for all models
- ✅ ControlNet support

### Advanced Features
- ✅ fp8, gguf, SVDQ (Nunchaku) quantization
- ✅ SageAttention, FlashAttention, xformers
- ✅ RescaleCFG (v-pred color correction)
- ✅ MaHiRo (better prompt adherence)
- ✅ Epsilon Scaling
- ✅ Half-precision upscalers
- ✅ GPU tile composition

### Removed (vs. A1111)
- ❌ SD2, SD3 (use SDXL/FLUX instead)
- ❌ Hypernetworks (use LoRA instead)
- ❌ CLIP Interrogator, Deepbooru
- ❌ Textual Inversion Training
- ❌ Some legacy samplers
- ❌ Unix .sh launch scripts (easily added back)

## Development Roadmap (NUKEM Project Goals)

### Phase 1: Foundation (Current)
- ✅ Understand Forge Neo architecture
- ✅ Document codebase
- ⏳ Test video generation capabilities

### Phase 2: High-Priority Additions
**AnimateDiff Integration** (Highest priority)
- Add temporal attention layers
- Motion module loading
- Motion LoRA support
- Integration with existing SD 1.5 models

**IP-Adapter**
- Image-to-image style transfer
- Reference image conditioning
- CLIP image encoder integration

**Regional Prompting**
- Attention masking
- Region-based prompting without ControlNet
- Simple coordinate/mask UI

### Phase 3: Quality of Life
- Enhanced preset system (save entire workflows)
- Better batch processing queue
- Video enhancements (frame interpolation, upscaling)

### Phase 4: Advanced Features
- Stable Video Diffusion (SVD)
- Model mixing improvements
- Custom sampler scheduling

## Common Development Tasks

### Adding a New Model Type

**Example: Adding AnimateDiff**

1. **Create model class** in `backend/huggingface/guoyww/`:
```python
# animatediff.py
class AnimateDiffModel:
    def __init__(self, motion_module_path):
        self.motion_module = load_motion_module(motion_module_path)
```

2. **Add detection** in `backend/loader.py`:
```python
def load_checkpoint_guess_config(...):
    if is_animatediff_model(checkpoint):
        return AnimateDiffModel(checkpoint)
```

3. **Create UI** in `modules/` or `extensions/`:
```python
def create_animatediff_ui():
    with gr.Accordion("AnimateDiff"):
        motion_module = gr.Dropdown(choices=get_motion_modules())
        frames = gr.Slider(8, 32, value=16)
```

4. **Hook into processing** in `modules_forge/main_entry.py`

### Adding a New Sampler

Edit `backend/diffusion_engine/` or relevant sampler file:
```python
# Add sampler function
def my_sampler(model, x, timesteps, **kwargs):
    # Sampling logic
    return denoised

# Register it
SAMPLERS['my_sampler'] = my_sampler
```

### Adding a ControlNet Preprocessor

Add to `modules_forge/supported_preprocessor.py`:
```python
class MyPreprocessor:
    def __call__(self, input_image, **kwargs):
        # Process image
        return processed_image
```

## Important Files to Understand

**For Model Loading:**
- `backend/loader.py` - Model detection and loading (1034 lines)
- `backend/huggingface/__init__.py` - Model registry

**For Processing:**
- `modules_forge/main_entry.py` - Main processing entry
- `backend/diffusion_engine/*.py` - Diffusion algorithms
- `backend/attention.py` - Attention implementations

**For UI:**
- `modules/ui.py` - Main Gradio interface
- `modules/processing.py` - Processing pipeline setup

**For Memory:**
- `backend/memory_management.py` - VRAM tracking & optimization

## Key Design Patterns

### 1. Model Detection by Path/Name
Many models are detected by filename patterns:
- "kontext" in path → FLUX Kontext
- "qwen" + "edit" → Qwen-Image-Edit
- Check `backend/loader.py` for all patterns

### 2. Lazy Loading
Models are loaded on-demand, components moved to CPU when not in use.

### 3. Operation Overloading
Different operation backends (standard, gguf, bnb) swap in automatically based on model type.

### 4. Preset System
Entire workflows saved as presets in `modules_forge/presets.py` - can be extended.

## Performance Tuning

**Fastest Setup (RTX 30+):**
```bash
--xformers --sage --cuda-malloc --cuda-stream --pin-shared-memory --fast-fp16
```

**Balanced:**
```bash
--xformers --cuda-malloc
```

**Low VRAM:**
```bash
--xformers --medvram
# or
--lowvram
```

**Attention Priority:**
SageAttention (fastest, slight quality loss) > FlashAttention > xformers > PyTorch > Basic

## API Usage

REST API available at `http://localhost:7860/sdapi/v1/` when `--api` is enabled.

**Key Endpoints:**
- `/sdapi/v1/txt2img` - Text to image
- `/sdapi/v1/img2img` - Image to image
- `/sdapi/v1/options` - Get/set options
- `/sdapi/v1/sd-models` - List models
- `/docs` - Swagger documentation

## Tips for Development

1. **Understand backend/ first** - This is where the magic happens
2. **Study existing model implementations** in `backend/huggingface/`
3. **Use presets** for testing - save your test configurations
4. **Memory management** - Always check VRAM usage when adding features
5. **Extension system** - When possible, add features as extensions rather than core modifications
6. **Keep UI simple** - The goal is to beat ComfyUI's complexity, not match it

## Community & Resources

- **Forge Neo GitHub:** https://github.com/Haoming02/sd-webui-forge-classic
- **Original Forge:** https://github.com/lllyasviel/stable-diffusion-webui-forge
- **A1111 Wiki:** https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki (still relevant for many features)
- **ComfyUI:** Study their custom nodes for feature ideas to port

## Project Goals (NUKEM)

**Mission:** Provide ComfyUI-level features with a simple, accessible UI

**Core Values:**
- ✅ Simplicity over complexity
- ✅ Performance without compromises
- ✅ Latest models and features
- ✅ Video generation as first-class citizen
- ✅ Approachable for non-technical users

**Not Goals:**
- ❌ Node graph editor
- ❌ 100% ComfyUI feature parity
- ❌ Support every experimental feature
- ❌ Backwards compatibility with ancient hardware
