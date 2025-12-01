"""
Model Metadata Synchronization

Handles syncing local model files with CivitAI metadata, including:
- Scanning local model directories
- Looking up models by hash
- Storing metadata in JSON sidecar files
- Downloading preview images
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import requests

from modules_forge.civitai.api_client import (
    CivitAIClient,
    CivitAINotFoundError,
    get_client,
)
from modules_forge.civitai.hash_utils import (
    calculate_hashes,
    find_model_files,
    get_metadata_path,
    get_preview_path,
)
from modules_forge.civitai.models import (
    CivitAIModelVersion,
    LocalModelMetadata,
)

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of syncing a single model"""

    model_path: str
    success: bool
    found_on_civitai: bool
    error: Optional[str] = None
    civitai_model_name: Optional[str] = None
    civitai_version_name: Optional[str] = None
    civitai_model_id: Optional[int] = None
    civitai_version_id: Optional[int] = None


@dataclass
class SyncProgress:
    """Progress information during sync"""

    total_models: int
    processed: int
    current_model: str
    found_count: int
    not_found_count: int
    error_count: int


class MetadataSync:
    """
    Synchronizes local models with CivitAI metadata.

    Usage:
        sync = MetadataSync()
        results = sync.sync_all(progress_callback=my_callback)
    """

    def __init__(self, client: CivitAIClient = None):
        """
        Initialize the metadata sync.

        Args:
            client: CivitAI client instance. If None, uses global client.
        """
        self.client = client or get_client()
        self._model_dirs = None

    def get_model_directories(self) -> dict[str, str]:
        """
        Get the model directories to scan.

        Returns:
            Dict mapping model type to directory path
        """
        if self._model_dirs is not None:
            return self._model_dirs

        # Try to import from modules to get actual paths
        try:
            from modules.paths import models_path

            base_path = models_path
        except ImportError:
            # Fallback to relative path
            base_path = "models"

        self._model_dirs = {
            "Checkpoint": os.path.join(base_path, "Stable-diffusion"),
            "LORA": os.path.join(base_path, "Lora"),
            "VAE": os.path.join(base_path, "VAE"),
            "Controlnet": os.path.join(base_path, "ControlNet"),
            "Upscaler": os.path.join(base_path, "ESRGAN"),
            "TextualInversion": os.path.join(base_path, "embeddings"),
        }

        return self._model_dirs

    def scan_local_models(self, model_type: str = None) -> list[str]:
        """
        Scan local directories for model files.

        Args:
            model_type: Optional type to filter (e.g., "Checkpoint", "LORA")

        Returns:
            List of absolute paths to model files
        """
        model_files = []
        dirs = self.get_model_directories()

        if model_type:
            dirs = {model_type: dirs.get(model_type, "")}

        for type_name, directory in dirs.items():
            if os.path.exists(directory):
                found = find_model_files(directory, recursive=True)
                model_files.extend(found)
                logger.info(f"Found {len(found)} {type_name} models in {directory}")

        return model_files

    def load_metadata(self, model_path: str) -> Optional[LocalModelMetadata]:
        """
        Load existing metadata from sidecar JSON file.

        Args:
            model_path: Path to the model file

        Returns:
            LocalModelMetadata if exists, else None
        """
        metadata_path = get_metadata_path(model_path)

        if not os.path.exists(metadata_path):
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return LocalModelMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading metadata from {metadata_path}: {e}")
            return None

    def save_metadata(
        self,
        model_path: str,
        metadata: LocalModelMetadata,
    ) -> bool:
        """
        Save metadata to sidecar JSON file.

        Args:
            model_path: Path to the model file
            metadata: Metadata to save

        Returns:
            True if saved successfully
        """
        metadata_path = get_metadata_path(model_path)

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata to {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving metadata to {metadata_path}: {e}")
            return False

    def download_preview_image(
        self,
        model_path: str,
        image_url: str,
        timeout: int = 30,
    ) -> Optional[str]:
        """
        Download and save a preview image for a model.

        Args:
            model_path: Path to the model file
            image_url: URL of the preview image
            timeout: Request timeout in seconds

        Returns:
            Path to saved preview image, or None on failure
        """
        if not image_url:
            return None

        # Determine extension from URL
        url_path = image_url.split("?")[0]  # Remove query params
        ext = Path(url_path).suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
            ext = ".png"

        preview_path = get_preview_path(model_path, f".preview{ext}")

        try:
            response = requests.get(image_url, timeout=timeout, stream=True)
            response.raise_for_status()

            with open(preview_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded preview to {preview_path}")
            return preview_path

        except Exception as e:
            logger.error(f"Error downloading preview from {image_url}: {e}")
            return None

    def sync_model(
        self,
        model_path: str,
        force_rehash: bool = False,
        download_preview: bool = True,
        progress_callback: Callable[[str], None] = None,
    ) -> SyncResult:
        """
        Sync a single model with CivitAI.

        Args:
            model_path: Path to the model file
            force_rehash: Force recalculation of hashes even if cached
            download_preview: Whether to download preview image
            progress_callback: Optional callback for progress messages

        Returns:
            SyncResult with sync outcome
        """
        model_name = Path(model_path).name

        def update_progress(msg: str):
            if progress_callback:
                progress_callback(f"{model_name}: {msg}")

        # Check for existing metadata
        existing = self.load_metadata(model_path)

        # Calculate hashes
        update_progress("Calculating hash...")
        if existing and existing.local_autov2 and not force_rehash:
            autov2 = existing.local_autov2
            sha256 = existing.local_sha256
        else:
            hashes = calculate_hashes(model_path, calculate_full_sha256=True)
            autov2 = hashes["autov2"]
            sha256 = hashes["sha256"]

        # Try to find on CivitAI
        update_progress("Looking up on CivitAI...")
        try:
            version = self.client.get_model_by_hash(autov2)

            # Get full model info for additional metadata
            model_info = None
            if version.model_id:
                try:
                    model_info = self.client.get_model(version.model_id)
                except Exception:
                    pass

            # Create metadata from CivitAI response
            if model_info:
                metadata = LocalModelMetadata.from_civitai_model(
                    model=model_info,
                    version=version,
                    local_path=model_path,
                    sha256=sha256,
                    autov2=autov2,
                )
            else:
                # Create partial metadata from version only
                metadata = LocalModelMetadata(
                    civitai_version_id=version.id,
                    civitai_model_id=version.model_id,
                    civitai_version_name=version.name,
                    civitai_base_model=version.base_model,
                    civitai_trained_words=version.trained_words,
                    civitai_download_url=version.download_url,
                    local_sha256=sha256,
                    local_autov2=autov2,
                    local_file_name=model_name,
                    synced_at=datetime.now().isoformat(),
                )

            # Save metadata
            update_progress("Saving metadata...")
            self.save_metadata(model_path, metadata)

            # Download preview if requested
            if download_preview and version.preview_image:
                update_progress("Downloading preview...")
                preview_path = self.download_preview_image(
                    model_path,
                    version.preview_image.url,
                )
                if preview_path:
                    metadata.local_preview_path = preview_path
                    self.save_metadata(model_path, metadata)

            return SyncResult(
                model_path=model_path,
                success=True,
                found_on_civitai=True,
                civitai_model_name=metadata.civitai_model_name,
                civitai_version_name=metadata.civitai_version_name,
                civitai_model_id=metadata.civitai_model_id,
                civitai_version_id=metadata.civitai_version_id,
            )

        except CivitAINotFoundError:
            # Model not found on CivitAI - still save local hash info
            update_progress("Not found on CivitAI")
            metadata = LocalModelMetadata(
                local_sha256=sha256,
                local_autov2=autov2,
                local_file_name=model_name,
                local_first_seen=datetime.now().isoformat(),
                synced_at=datetime.now().isoformat(),
                sync_error="Model not found on CivitAI",
            )
            self.save_metadata(model_path, metadata)

            return SyncResult(
                model_path=model_path,
                success=True,
                found_on_civitai=False,
            )

        except Exception as e:
            update_progress(f"Error: {e}")
            logger.error(f"Error syncing {model_path}: {e}")

            return SyncResult(
                model_path=model_path,
                success=False,
                found_on_civitai=False,
                error=str(e),
            )

    def sync_all(
        self,
        model_type: str = None,
        force_rehash: bool = False,
        download_preview: bool = True,
        skip_existing: bool = True,
        progress_callback: Callable[[SyncProgress], None] = None,
    ) -> list[SyncResult]:
        """
        Sync all local models with CivitAI.

        Args:
            model_type: Optional type filter (e.g., "Checkpoint")
            force_rehash: Force recalculation of hashes
            download_preview: Whether to download preview images
            skip_existing: Skip models that already have metadata
            progress_callback: Callback for progress updates

        Returns:
            List of SyncResult for each model
        """
        results = []
        model_files = self.scan_local_models(model_type)

        if skip_existing:
            # Filter out models with existing metadata
            model_files = [
                m
                for m in model_files
                if not os.path.exists(get_metadata_path(m))
            ]

        total = len(model_files)
        found_count = 0
        not_found_count = 0
        error_count = 0

        logger.info(f"Starting sync for {total} models")

        for i, model_path in enumerate(model_files):
            # Update progress
            progress = SyncProgress(
                total_models=total,
                processed=i,
                current_model=Path(model_path).name,
                found_count=found_count,
                not_found_count=not_found_count,
                error_count=error_count,
            )
            if progress_callback:
                progress_callback(progress)

            # Sync the model
            result = self.sync_model(
                model_path,
                force_rehash=force_rehash,
                download_preview=download_preview,
            )
            results.append(result)

            # Update counts
            if result.success:
                if result.found_on_civitai:
                    found_count += 1
                else:
                    not_found_count += 1
            else:
                error_count += 1

        # Final progress update
        if progress_callback:
            progress_callback(
                SyncProgress(
                    total_models=total,
                    processed=total,
                    current_model="Complete",
                    found_count=found_count,
                    not_found_count=not_found_count,
                    error_count=error_count,
                )
            )

        logger.info(
            f"Sync complete: {found_count} found, "
            f"{not_found_count} not found, {error_count} errors"
        )

        return results

    def get_model_info(self, model_path: str) -> Optional[LocalModelMetadata]:
        """
        Get CivitAI info for a local model.

        Loads from cache if available, otherwise syncs.

        Args:
            model_path: Path to the model file

        Returns:
            LocalModelMetadata if available
        """
        # Try to load from cache first
        metadata = self.load_metadata(model_path)
        if metadata and metadata.civitai_model_id:
            return metadata

        # Not in cache, sync now
        result = self.sync_model(model_path, download_preview=True)
        if result.success:
            return self.load_metadata(model_path)

        return None


# Global instance
_sync: Optional[MetadataSync] = None


def get_sync() -> MetadataSync:
    """Get the global MetadataSync instance."""
    global _sync
    if _sync is None:
        _sync = MetadataSync()
    return _sync
