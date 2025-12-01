"""
Hash Utilities for Model Identification

Provides functions to calculate various hash types used by CivitAI
for model identification (SHA256, AutoV2, etc.).
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_sha256(file_path: str, chunk_size: int = 8192) -> str:
    """
    Calculate full SHA256 hash of a file.

    Args:
        file_path: Path to the file
        chunk_size: Read chunk size in bytes

    Returns:
        Lowercase hex digest of SHA256 hash
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest().lower()


def calculate_autov2(file_path: str) -> str:
    """
    Calculate AutoV2 hash (first 10 characters of SHA256).

    This is the most commonly used hash format on CivitAI for quick lookups.

    Args:
        file_path: Path to the file

    Returns:
        First 10 characters of SHA256 hash (uppercase)
    """
    full_hash = calculate_sha256(file_path)
    return full_hash[:10].upper()


def calculate_model_hash_header(file_path: str) -> Optional[str]:
    """
    Calculate hash from file header for quick identification.

    This reads only the first 256KB of the file for faster hashing.
    Used for quick comparisons, not for CivitAI lookups.

    Args:
        file_path: Path to the file

    Returns:
        SHA256 hash of the first 256KB, or None on error
    """
    try:
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            # Read first 256KB
            header_data = f.read(256 * 1024)
            sha256_hash.update(header_data)

        return sha256_hash.hexdigest()[:8].lower()
    except Exception as e:
        logger.error(f"Error calculating header hash for {file_path}: {e}")
        return None


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in MB (rounded to 2 decimal places)
    """
    size_bytes = get_file_size(file_path)
    return round(size_bytes / (1024 * 1024), 2)


def calculate_hashes(
    file_path: str,
    calculate_full_sha256: bool = True,
    progress_callback=None,
) -> dict:
    """
    Calculate all relevant hashes for a model file.

    Args:
        file_path: Path to the model file
        calculate_full_sha256: Whether to calculate full SHA256 (slower)
        progress_callback: Optional callback for progress updates

    Returns:
        Dict with hash values and file info:
        {
            "sha256": "full hash...",
            "autov2": "ABCD123456",
            "header_hash": "short hash",
            "file_size": size_in_bytes,
            "file_name": "model.safetensors"
        }
    """
    result = {
        "sha256": None,
        "autov2": None,
        "header_hash": None,
        "file_size": 0,
        "file_name": Path(file_path).name,
    }

    try:
        result["file_size"] = get_file_size(file_path)

        # Calculate header hash (fast)
        if progress_callback:
            progress_callback("Calculating header hash...")
        result["header_hash"] = calculate_model_hash_header(file_path)

        # Calculate full SHA256 if requested
        if calculate_full_sha256:
            if progress_callback:
                progress_callback("Calculating SHA256 (this may take a while)...")

            sha256 = calculate_sha256(file_path)
            result["sha256"] = sha256
            result["autov2"] = sha256[:10].upper()
        else:
            # Just calculate AutoV2 which is first 10 chars of SHA256
            # Still need full hash for this
            if progress_callback:
                progress_callback("Calculating AutoV2 hash...")
            result["autov2"] = calculate_autov2(file_path)

    except Exception as e:
        logger.error(f"Error calculating hashes for {file_path}: {e}")

    return result


def is_model_file(file_path: str) -> bool:
    """
    Check if a file is a supported model file.

    Args:
        file_path: Path to check

    Returns:
        True if file has a supported model extension
    """
    supported_extensions = {
        ".safetensors",
        ".ckpt",
        ".pt",
        ".pth",
        ".bin",
    }
    return Path(file_path).suffix.lower() in supported_extensions


def find_model_files(directory: str, recursive: bool = True) -> list[str]:
    """
    Find all model files in a directory.

    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories

    Returns:
        List of absolute paths to model files
    """
    model_files = []
    path = Path(directory)

    if not path.exists():
        return model_files

    pattern = "**/*" if recursive else "*"

    for file_path in path.glob(pattern):
        if file_path.is_file() and is_model_file(str(file_path)):
            model_files.append(str(file_path.absolute()))

    return model_files


def get_metadata_path(model_path: str) -> str:
    """
    Get the path for the metadata sidecar JSON file.

    Args:
        model_path: Path to the model file

    Returns:
        Path where metadata JSON should be stored
    """
    path = Path(model_path)
    return str(path.with_suffix(".civitai.json"))


def get_preview_path(model_path: str, extension: str = ".preview.png") -> str:
    """
    Get the path for the preview image file.

    Args:
        model_path: Path to the model file
        extension: Preview image extension

    Returns:
        Path where preview image should be stored
    """
    path = Path(model_path)
    # Remove the model extension and add preview extension
    return str(path.with_suffix("")) + extension
