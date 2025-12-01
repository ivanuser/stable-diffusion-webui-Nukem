"""
CivitAI Integration for Nukem WebUI

This module provides direct integration with CivitAI for:
- Model browsing and search
- Direct model downloads
- Automatic metadata sync (images, descriptions, versions)
- Hash-based model identification
- Version tracking and updates
"""

from modules_forge.civitai.api_client import (
    CivitAIClient,
    CivitAIError,
    CivitAIAuthError,
    CivitAINotFoundError,
    CivitAIRateLimitError,
    get_client,
    init_client,
)
from modules_forge.civitai.models import (
    CivitAIModel,
    CivitAIModelVersion,
    CivitAIFile,
    CivitAIImage,
    CivitAISearchResult,
    LocalModelMetadata,
    ModelType,
    BaseModel,
)
from modules_forge.civitai.metadata_sync import (
    MetadataSync,
    SyncResult,
    SyncProgress,
    get_sync,
)
from modules_forge.civitai.hash_utils import (
    calculate_sha256,
    calculate_autov2,
    calculate_hashes,
    get_metadata_path,
    get_preview_path,
)
from modules_forge.civitai.downloader import (
    CivitAIDownloader,
    DownloadTask,
    DownloadProgress,
    DownloadStatus,
    get_downloader,
)

# UI imports are deferred to avoid circular imports during early startup
# Use: from modules_forge.civitai.ui import create_ui, on_ui_tabs

__all__ = [
    # Client
    "CivitAIClient",
    "get_client",
    "init_client",
    # Exceptions
    "CivitAIError",
    "CivitAIAuthError",
    "CivitAINotFoundError",
    "CivitAIRateLimitError",
    # Models
    "CivitAIModel",
    "CivitAIModelVersion",
    "CivitAIFile",
    "CivitAIImage",
    "CivitAISearchResult",
    "LocalModelMetadata",
    # Enums
    "ModelType",
    "BaseModel",
    # Sync
    "MetadataSync",
    "SyncResult",
    "SyncProgress",
    "get_sync",
    # Hash Utils
    "calculate_sha256",
    "calculate_autov2",
    "calculate_hashes",
    "get_metadata_path",
    "get_preview_path",
    # Downloader
    "CivitAIDownloader",
    "DownloadTask",
    "DownloadProgress",
    "DownloadStatus",
    "get_downloader",
]

__version__ = "1.0.0"
