#!/usr/bin/env python3
"""
Download AnimateDiff motion modules and motion LoRAs.

Usage:
    python scripts/download_animatediff_models.py [--all] [--motion-modules] [--motion-loras]

Options:
    --all             Download everything (motion modules + motion LoRAs)
    --motion-modules  Download motion modules only (default if no option specified)
    --motion-loras    Download motion LoRAs only
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.insert(0, str(root_dir))

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "tqdm"])
    import requests
    from tqdm import tqdm


# Motion modules from the official AnimateDiff repository
MOTION_MODULES = {
    "mm_sd_v15_v2.safetensors": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.safetensors",
        "size": "1.82 GB",
        "description": "AnimateDiff v2 - Recommended, good quality general purpose",
    },
    "mm_sd_v14.safetensors": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.safetensors",
        "size": "1.82 GB",
        "description": "AnimateDiff v1.4 - Original release",
    },
    "v3_sd15_mm.safetensors": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.safetensors",
        "size": "1.82 GB",
        "description": "AnimateDiff v3 - Improved motion quality",
    },
    "mm_sd_v15.safetensors": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.safetensors",
        "size": "1.82 GB",
        "description": "AnimateDiff v1.5 - Intermediate version",
    },
}

# Motion LoRAs for modifying motion characteristics
MOTION_LORAS = {
    "v2_lora_ZoomIn.safetensors": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.safetensors",
        "size": "78 MB",
        "description": "Zoom in camera motion",
    },
    "v2_lora_ZoomOut.safetensors": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomOut.safetensors",
        "size": "78 MB",
        "description": "Zoom out camera motion",
    },
    "v2_lora_PanLeft.safetensors": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanLeft.safetensors",
        "size": "78 MB",
        "description": "Pan left camera motion",
    },
    "v2_lora_PanRight.safetensors": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanRight.safetensors",
        "size": "78 MB",
        "description": "Pan right camera motion",
    },
    "v2_lora_TiltUp.safetensors": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltUp.safetensors",
        "size": "78 MB",
        "description": "Tilt up camera motion",
    },
    "v2_lora_TiltDown.safetensors": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltDown.safetensors",
        "size": "78 MB",
        "description": "Tilt down camera motion",
    },
    "v2_lora_RollingClockwise.safetensors": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingClockwise.safetensors",
        "size": "78 MB",
        "description": "Rolling clockwise camera motion",
    },
    "v2_lora_RollingAnticlockwise.safetensors": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingAnticlockwise.safetensors",
        "size": "78 MB",
        "description": "Rolling counter-clockwise camera motion",
    },
}


def get_models_path():
    """Get the models path, handling both direct run and module import."""
    # Try to use the WebUI's paths module
    try:
        from modules import paths
        return Path(paths.models_path)
    except ImportError:
        # Fallback to relative path from script
        return root_dir / "models"


def download_file(url: str, dest_path: Path, desc: str = None):
    """Download a file with progress bar."""
    if dest_path.exists():
        print(f"  Already exists: {dest_path.name}")
        return True

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        desc = desc or dest_path.name
        with open(dest_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"  {desc}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True
    except Exception as e:
        print(f"  Error downloading {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_motion_modules(models_path: Path, selected: list = None):
    """Download motion modules."""
    motion_modules_dir = models_path / "motion_modules"
    motion_modules_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Downloading Motion Modules")
    print("=" * 60)
    print(f"Directory: {motion_modules_dir}\n")

    modules_to_download = selected if selected else list(MOTION_MODULES.keys())

    for name in modules_to_download:
        if name not in MOTION_MODULES:
            print(f"  Unknown module: {name}")
            continue

        info = MOTION_MODULES[name]
        print(f"\n{name} ({info['size']})")
        print(f"  {info['description']}")

        dest_path = motion_modules_dir / name
        download_file(info["url"], dest_path, name)

    print("\n" + "-" * 60)
    print(f"Motion modules saved to: {motion_modules_dir}")


def download_motion_loras(models_path: Path, selected: list = None):
    """Download motion LoRAs."""
    motion_lora_dir = models_path / "motion_lora"
    motion_lora_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Downloading Motion LoRAs")
    print("=" * 60)
    print(f"Directory: {motion_lora_dir}\n")

    loras_to_download = selected if selected else list(MOTION_LORAS.keys())

    for name in loras_to_download:
        if name not in MOTION_LORAS:
            print(f"  Unknown LoRA: {name}")
            continue

        info = MOTION_LORAS[name]
        print(f"\n{name} ({info['size']})")
        print(f"  {info['description']}")

        dest_path = motion_lora_dir / name
        download_file(info["url"], dest_path, name)

    print("\n" + "-" * 60)
    print(f"Motion LoRAs saved to: {motion_lora_dir}")


def list_available():
    """List all available downloads."""
    print("\n" + "=" * 60)
    print("Available Motion Modules")
    print("=" * 60)
    for name, info in MOTION_MODULES.items():
        print(f"\n  {name} ({info['size']})")
        print(f"    {info['description']}")

    print("\n" + "=" * 60)
    print("Available Motion LoRAs")
    print("=" * 60)
    for name, info in MOTION_LORAS.items():
        print(f"\n  {name} ({info['size']})")
        print(f"    {info['description']}")


def main():
    parser = argparse.ArgumentParser(
        description="Download AnimateDiff motion modules and motion LoRAs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_animatediff_models.py                    # Download recommended motion module
  python scripts/download_animatediff_models.py --all              # Download everything
  python scripts/download_animatediff_models.py --motion-modules   # Download all motion modules
  python scripts/download_animatediff_models.py --motion-loras     # Download all motion LoRAs
  python scripts/download_animatediff_models.py --list             # List available downloads
  python scripts/download_animatediff_models.py --minimal          # Download only v2 module (smallest working set)
        """
    )

    parser.add_argument("--all", action="store_true", help="Download all motion modules and LoRAs")
    parser.add_argument("--motion-modules", action="store_true", help="Download all motion modules")
    parser.add_argument("--motion-loras", action="store_true", help="Download all motion LoRAs")
    parser.add_argument("--minimal", action="store_true", help="Download only mm_sd_v15_v2 (recommended)")
    parser.add_argument("--list", action="store_true", help="List available downloads")

    args = parser.parse_args()

    if args.list:
        list_available()
        return

    models_path = get_models_path()
    print(f"Models directory: {models_path}")

    if args.all:
        download_motion_modules(models_path)
        download_motion_loras(models_path)
    elif args.motion_modules:
        download_motion_modules(models_path)
    elif args.motion_loras:
        download_motion_loras(models_path)
    elif args.minimal:
        download_motion_modules(models_path, ["mm_sd_v15_v2.safetensors"])
    else:
        # Default: download recommended motion module
        print("\nNo option specified. Downloading recommended motion module...")
        print("Use --all to download everything, or --list to see options.\n")
        download_motion_modules(models_path, ["mm_sd_v15_v2.safetensors"])

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nTo use AnimateDiff:")
    print("  1. Select 'animatediff' preset in the UI")
    print("  2. Load an SD1.5 checkpoint")
    print("  3. Select a motion module from the dropdown")
    print("  4. Set number of frames and generate!")
    print()


if __name__ == "__main__":
    main()
