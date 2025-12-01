"""
Pydantic models for CivitAI API responses
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ModelType(str, Enum):
    """CivitAI model types"""
    CHECKPOINT = "Checkpoint"
    TEXTUAL_INVERSION = "TextualInversion"
    HYPERNETWORK = "Hypernetwork"
    AESTHETIC_GRADIENT = "AestheticGradient"
    LORA = "LORA"
    LOCON = "LoCon"
    CONTROLNET = "Controlnet"
    UPSCALER = "Upscaler"
    MOTION_MODULE = "MotionModule"
    VAE = "VAE"
    POSES = "Poses"
    WILDCARDS = "Wildcards"
    WORKFLOWS = "Workflows"
    OTHER = "Other"


class BaseModel(str, Enum):
    """Base model types"""
    SD_1_4 = "SD 1.4"
    SD_1_5 = "SD 1.5"
    SD_1_5_LCM = "SD 1.5 LCM"
    SD_2_0 = "SD 2.0"
    SD_2_0_768 = "SD 2.0 768"
    SD_2_1 = "SD 2.1"
    SD_2_1_768 = "SD 2.1 768"
    SD_2_1_UNCLIP = "SD 2.1 Unclip"
    SDXL_0_9 = "SDXL 0.9"
    SDXL_1_0 = "SDXL 1.0"
    SDXL_1_0_LCM = "SDXL 1.0 LCM"
    SDXL_DISTILLED = "SDXL Distilled"
    SDXL_TURBO = "SDXL Turbo"
    SDXL_LIGHTNING = "SDXL Lightning"
    SD_3 = "SD 3"
    SD_3_5 = "SD 3.5"
    SD_3_5_LARGE = "SD 3.5 Large"
    SD_3_5_LARGE_TURBO = "SD 3.5 Large Turbo"
    SD_3_5_MEDIUM = "SD 3.5 Medium"
    STABLE_CASCADE = "Stable Cascade"
    SVD = "SVD"
    SVD_XT = "SVD XT"
    PLAYGROUND_V2 = "Playground v2"
    PIXART_A = "PixArt a"
    PIXART_E = "PixArt E"
    HUNYUAN_1 = "Hunyuan 1"
    LUMINA = "Lumina"
    KOLORS = "Kolors"
    FLUX_1_S = "Flux.1 S"
    FLUX_1_D = "Flux.1 D"
    AURAFLOW = "AuraFlow"
    PONY = "Pony"
    ILLUSTRIOUS = "Illustrious"
    MOCHI = "Mochi"
    LTXV = "LTXV"
    HUNYUAN_VIDEO = "Hunyuan Video"
    COGVIDEOX = "CogVideoX"
    WAN = "Wan"
    OTHER = "Other"


class NsfwLevel(str, Enum):
    """NSFW content levels"""
    NONE = "None"
    SOFT = "Soft"
    MATURE = "Mature"
    X = "X"


class ScanResult(str, Enum):
    """File scan status"""
    PENDING = "Pending"
    SUCCESS = "Success"
    DANGER = "Danger"


@dataclass
class CivitAICreator:
    """Model creator information"""
    username: str
    image: Optional[str] = None


@dataclass
class CivitAIStats:
    """Model/version statistics"""
    download_count: int = 0
    favorite_count: int = 0
    thumb_up_count: int = 0
    thumb_down_count: int = 0
    comment_count: int = 0
    rating_count: int = 0
    rating: float = 0.0
    tipped_amount_count: int = 0


@dataclass
class CivitAIImage:
    """Model preview image"""
    id: int
    url: str
    nsfw: str = "None"
    width: int = 0
    height: int = 0
    hash: Optional[str] = None
    type: str = "image"
    metadata: Optional[dict] = None
    meta: Optional[dict] = None  # Generation parameters

    @property
    def is_safe(self) -> bool:
        return self.nsfw == "None"


@dataclass
class CivitAIFile:
    """Model file information"""
    id: int
    name: str
    size_kb: float
    type: str = "Model"
    primary: bool = False
    download_url: Optional[str] = None

    # Hashes for identification
    sha256: Optional[str] = None
    autov1: Optional[str] = None
    autov2: Optional[str] = None
    autov3: Optional[str] = None
    crc32: Optional[str] = None
    blake3: Optional[str] = None

    # Metadata
    metadata: Optional[dict] = None
    pickle_scan_result: Optional[str] = None
    pickle_scan_message: Optional[str] = None
    virus_scan_result: Optional[str] = None
    virus_scan_message: Optional[str] = None
    scanned_at: Optional[str] = None

    @property
    def size_mb(self) -> float:
        return self.size_kb / 1024

    @property
    def size_gb(self) -> float:
        return self.size_kb / (1024 * 1024)

    @property
    def is_safe(self) -> bool:
        return (
            self.pickle_scan_result == "Success"
            and self.virus_scan_result == "Success"
        )

    @classmethod
    def from_api_response(cls, data: dict) -> "CivitAIFile":
        """Create from API response"""
        hashes = data.get("hashes", {})
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            size_kb=data.get("sizeKB", 0),
            type=data.get("type", "Model"),
            primary=data.get("primary", False),
            download_url=data.get("downloadUrl"),
            sha256=hashes.get("SHA256"),
            autov1=hashes.get("AutoV1"),
            autov2=hashes.get("AutoV2"),
            autov3=hashes.get("AutoV3"),
            crc32=hashes.get("CRC32"),
            blake3=hashes.get("BLAKE3"),
            metadata=data.get("metadata"),
            pickle_scan_result=data.get("pickleScanResult"),
            pickle_scan_message=data.get("pickleScanMessage"),
            virus_scan_result=data.get("virusScanResult"),
            virus_scan_message=data.get("virusScanMessage"),
            scanned_at=data.get("scannedAt"),
        )


@dataclass
class CivitAIModelVersion:
    """Model version information"""
    id: int
    model_id: int
    name: str
    description: Optional[str] = None
    base_model: Optional[str] = None
    base_model_type: Optional[str] = None

    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    published_at: Optional[str] = None

    trained_words: list[str] = field(default_factory=list)
    files: list[CivitAIFile] = field(default_factory=list)
    images: list[CivitAIImage] = field(default_factory=list)
    stats: Optional[CivitAIStats] = None

    download_url: Optional[str] = None

    @property
    def primary_file(self) -> Optional[CivitAIFile]:
        """Get the primary model file"""
        for f in self.files:
            if f.primary:
                return f
        return self.files[0] if self.files else None

    @property
    def preview_image(self) -> Optional[CivitAIImage]:
        """Get first safe preview image"""
        for img in self.images:
            if img.is_safe:
                return img
        return self.images[0] if self.images else None

    @classmethod
    def from_api_response(cls, data: dict) -> "CivitAIModelVersion":
        """Create from API response"""
        files = [CivitAIFile.from_api_response(f) for f in data.get("files", [])]
        images = [
            CivitAIImage(
                id=img.get("id", 0),
                url=img.get("url", ""),
                nsfw=img.get("nsfw", "None"),
                width=img.get("width", 0),
                height=img.get("height", 0),
                hash=img.get("hash"),
                type=img.get("type", "image"),
                metadata=img.get("metadata"),
                meta=img.get("meta"),
            )
            for img in data.get("images", [])
        ]

        stats_data = data.get("stats", {})
        stats = CivitAIStats(
            download_count=stats_data.get("downloadCount", 0),
            favorite_count=stats_data.get("favoriteCount", 0),
            thumb_up_count=stats_data.get("thumbsUpCount", 0),
            thumb_down_count=stats_data.get("thumbsDownCount", 0),
            comment_count=stats_data.get("commentCount", 0),
            rating_count=stats_data.get("ratingCount", 0),
            rating=stats_data.get("rating", 0.0),
            tipped_amount_count=stats_data.get("tippedAmountCount", 0),
        )

        return cls(
            id=data.get("id", 0),
            model_id=data.get("modelId", 0),
            name=data.get("name", ""),
            description=data.get("description"),
            base_model=data.get("baseModel"),
            base_model_type=data.get("baseModelType"),
            created_at=data.get("createdAt"),
            updated_at=data.get("updatedAt"),
            published_at=data.get("publishedAt"),
            trained_words=data.get("trainedWords", []),
            files=files,
            images=images,
            stats=stats,
            download_url=data.get("downloadUrl"),
        )


@dataclass
class CivitAIModel:
    """Full model information"""
    id: int
    name: str
    description: Optional[str] = None
    type: str = "Checkpoint"
    nsfw: bool = False
    poi: bool = False  # Person of Interest
    minor: bool = False

    tags: list[str] = field(default_factory=list)
    mode: Optional[str] = None
    allow_no_credit: bool = True
    allow_commercial_use: list[str] = field(default_factory=list)
    allow_derivatives: bool = True
    allow_different_license: bool = True

    creator: Optional[CivitAICreator] = None
    stats: Optional[CivitAIStats] = None
    model_versions: list[CivitAIModelVersion] = field(default_factory=list)

    @property
    def latest_version(self) -> Optional[CivitAIModelVersion]:
        """Get the latest version"""
        return self.model_versions[0] if self.model_versions else None

    @property
    def preview_image(self) -> Optional[CivitAIImage]:
        """Get preview image from latest version"""
        latest = self.latest_version
        return latest.preview_image if latest else None

    @classmethod
    def from_api_response(cls, data: dict) -> "CivitAIModel":
        """Create from API response"""
        creator_data = data.get("creator", {})
        creator = CivitAICreator(
            username=creator_data.get("username", "Unknown"),
            image=creator_data.get("image"),
        ) if creator_data else None

        stats_data = data.get("stats", {})
        stats = CivitAIStats(
            download_count=stats_data.get("downloadCount", 0),
            favorite_count=stats_data.get("favoriteCount", 0),
            thumb_up_count=stats_data.get("thumbsUpCount", 0),
            thumb_down_count=stats_data.get("thumbsDownCount", 0),
            comment_count=stats_data.get("commentCount", 0),
            rating_count=stats_data.get("ratingCount", 0),
            rating=stats_data.get("rating", 0.0),
            tipped_amount_count=stats_data.get("tippedAmountCount", 0),
        )

        versions = [
            CivitAIModelVersion.from_api_response(v)
            for v in data.get("modelVersions", [])
        ]

        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            description=data.get("description"),
            type=data.get("type", "Checkpoint"),
            nsfw=data.get("nsfw", False),
            poi=data.get("poi", False),
            minor=data.get("minor", False),
            tags=data.get("tags", []),
            mode=data.get("mode"),
            allow_no_credit=data.get("allowNoCredit", True),
            allow_commercial_use=data.get("allowCommercialUse", []),
            allow_derivatives=data.get("allowDerivatives", True),
            allow_different_license=data.get("allowDifferentLicense", True),
            creator=creator,
            stats=stats,
            model_versions=versions,
        )


@dataclass
class CivitAISearchResult:
    """Search result with pagination"""
    items: list[CivitAIModel]
    metadata: dict = field(default_factory=dict)

    @property
    def total_items(self) -> int:
        return self.metadata.get("totalItems", len(self.items))

    @property
    def current_page(self) -> int:
        return self.metadata.get("currentPage", 1)

    @property
    def page_size(self) -> int:
        return self.metadata.get("pageSize", 20)

    @property
    def total_pages(self) -> int:
        return self.metadata.get("totalPages", 1)

    @property
    def next_cursor(self) -> Optional[str]:
        return self.metadata.get("nextCursor")

    @property
    def next_page(self) -> Optional[str]:
        return self.metadata.get("nextPage")

    @classmethod
    def from_api_response(cls, data: dict) -> "CivitAISearchResult":
        """Create from API response"""
        items = [CivitAIModel.from_api_response(m) for m in data.get("items", [])]
        return cls(
            items=items,
            metadata=data.get("metadata", {}),
        )


@dataclass
class LocalModelMetadata:
    """Local metadata stored alongside model files"""
    # CivitAI info
    civitai_model_id: Optional[int] = None
    civitai_version_id: Optional[int] = None
    civitai_model_name: Optional[str] = None
    civitai_version_name: Optional[str] = None
    civitai_creator: Optional[str] = None
    civitai_description: Optional[str] = None
    civitai_base_model: Optional[str] = None
    civitai_type: Optional[str] = None
    civitai_nsfw: bool = False
    civitai_tags: list[str] = field(default_factory=list)
    civitai_trained_words: list[str] = field(default_factory=list)
    civitai_download_url: Optional[str] = None
    civitai_images: list[dict] = field(default_factory=list)
    civitai_stats: Optional[dict] = None

    # Local info
    local_sha256: Optional[str] = None
    local_autov2: Optional[str] = None
    local_file_size: int = 0
    local_file_name: Optional[str] = None
    local_first_seen: Optional[str] = None
    local_preview_path: Optional[str] = None

    # Sync info
    synced_at: Optional[str] = None
    sync_error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage"""
        return {
            "civitai": {
                "model_id": self.civitai_model_id,
                "version_id": self.civitai_version_id,
                "model_name": self.civitai_model_name,
                "version_name": self.civitai_version_name,
                "creator": self.civitai_creator,
                "description": self.civitai_description,
                "base_model": self.civitai_base_model,
                "type": self.civitai_type,
                "nsfw": self.civitai_nsfw,
                "tags": self.civitai_tags,
                "trained_words": self.civitai_trained_words,
                "download_url": self.civitai_download_url,
                "images": self.civitai_images,
                "stats": self.civitai_stats,
            },
            "local": {
                "sha256": self.local_sha256,
                "autov2": self.local_autov2,
                "file_size": self.local_file_size,
                "file_name": self.local_file_name,
                "first_seen": self.local_first_seen,
                "preview_path": self.local_preview_path,
            },
            "synced_at": self.synced_at,
            "sync_error": self.sync_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LocalModelMetadata":
        """Create from dictionary"""
        civitai = data.get("civitai", {})
        local = data.get("local", {})

        return cls(
            civitai_model_id=civitai.get("model_id"),
            civitai_version_id=civitai.get("version_id"),
            civitai_model_name=civitai.get("model_name"),
            civitai_version_name=civitai.get("version_name"),
            civitai_creator=civitai.get("creator"),
            civitai_description=civitai.get("description"),
            civitai_base_model=civitai.get("base_model"),
            civitai_type=civitai.get("type"),
            civitai_nsfw=civitai.get("nsfw", False),
            civitai_tags=civitai.get("tags", []),
            civitai_trained_words=civitai.get("trained_words", []),
            civitai_download_url=civitai.get("download_url"),
            civitai_images=civitai.get("images", []),
            civitai_stats=civitai.get("stats"),
            local_sha256=local.get("sha256"),
            local_autov2=local.get("autov2"),
            local_file_size=local.get("file_size", 0),
            local_file_name=local.get("file_name"),
            local_first_seen=local.get("first_seen"),
            local_preview_path=local.get("preview_path"),
            synced_at=data.get("synced_at"),
            sync_error=data.get("sync_error"),
        )

    @classmethod
    def from_civitai_model(
        cls,
        model: CivitAIModel,
        version: CivitAIModelVersion,
        local_path: str = None,
        sha256: str = None,
        autov2: str = None,
    ) -> "LocalModelMetadata":
        """Create from CivitAI model data"""
        return cls(
            civitai_model_id=model.id,
            civitai_version_id=version.id,
            civitai_model_name=model.name,
            civitai_version_name=version.name,
            civitai_creator=model.creator.username if model.creator else None,
            civitai_description=model.description,
            civitai_base_model=version.base_model,
            civitai_type=model.type,
            civitai_nsfw=model.nsfw,
            civitai_tags=model.tags,
            civitai_trained_words=version.trained_words,
            civitai_download_url=version.download_url,
            civitai_images=[
                {"url": img.url, "nsfw": img.nsfw}
                for img in version.images[:5]  # Store up to 5 images
            ],
            civitai_stats={
                "downloads": model.stats.download_count if model.stats else 0,
                "rating": model.stats.rating if model.stats else 0,
                "favorites": model.stats.favorite_count if model.stats else 0,
            },
            local_sha256=sha256,
            local_autov2=autov2,
            local_file_name=local_path,
            synced_at=datetime.now().isoformat(),
        )
