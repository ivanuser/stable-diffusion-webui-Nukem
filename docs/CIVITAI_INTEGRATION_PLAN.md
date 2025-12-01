# CivitAI Integration Plan

## Overview

Integrate CivitAI directly into Nukem WebUI as a core feature (not extension), enabling:
- Model browsing and search
- Direct model downloads
- Automatic metadata sync (images, descriptions, versions)
- Hash-based model identification
- Version tracking and updates

## CivitAI API Reference

**Base URL:** `https://civitai.com/api/v1`

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models` | GET | Search/list models with filters |
| `/models/:modelId` | GET | Get model details by ID |
| `/model-versions/:versionId` | GET | Get specific version info |
| `/model-versions/by-hash/:hash` | GET | Lookup model by file hash (AutoV1, AutoV2, SHA256, CRC32, Blake3) |
| `/images` | GET | Get model images |
| `/tags` | GET | List available tags |
| `/creators` | GET | List creators |

### Download URL
```
https://civitai.com/api/download/models/{modelVersionId}?token={api_key}
```

### Authentication
- Header: `Authorization: Bearer {api_key}`
- Query: `?token={api_key}`

## Architecture

### New Files to Create

```
modules_forge/
└── civitai/
    ├── __init__.py
    ├── api_client.py      # CivitAI API client
    ├── models.py          # Pydantic models for API responses
    ├── downloader.py      # Download manager with progress
    ├── metadata_sync.py   # Sync local models with CivitAI
    ├── cache.py           # Local cache for API responses
    └── ui.py              # Gradio UI components

modules/
├── civitai_settings.py    # Settings integration
└── api/
    └── civitai_api.py     # REST API endpoints for CivitAI
```

### Settings to Add

```python
# In shared_options.py or new civitai_settings.py
civitai_api_key = ""              # User's API token
civitai_auto_sync = True          # Auto-sync metadata on startup
civitai_download_preview = True   # Download preview images
civitai_nsfw_level = "None"       # NSFW filter (None, Soft, Mature, X)
civitai_cache_hours = 24          # Cache duration for API responses
```

## Implementation Phases

### Phase 1: Core API Client

**File: `modules_forge/civitai/api_client.py`**

```python
class CivitAIClient:
    BASE_URL = "https://civitai.com/api/v1"

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = requests.Session()

    def get_model(self, model_id: int) -> dict
    def get_model_version(self, version_id: int) -> dict
    def get_model_by_hash(self, hash: str) -> dict
    def search_models(self, query: str, **filters) -> list
    def get_download_url(self, version_id: int) -> str
```

### Phase 2: Model Metadata Sync

**Features:**
1. Scan local models directory
2. Calculate hashes (SHA256, AutoV2)
3. Query CivitAI by hash
4. Store metadata locally (JSON sidecar files)
5. Download preview images

**File: `modules_forge/civitai/metadata_sync.py`**

```python
class MetadataSync:
    def scan_local_models(self) -> list[LocalModel]
    def lookup_model(self, hash: str) -> CivitAIModel
    def save_metadata(self, model_path: str, metadata: dict)
    def download_preview(self, model_path: str, image_url: str)
    def sync_all(self, progress_callback=None)
```

### Phase 3: Download Manager

**Features:**
1. Queue-based downloads
2. Progress tracking with ETA
3. Resume support
4. Hash verification after download
5. Auto-placement in correct folder

**File: `modules_forge/civitai/downloader.py`**

```python
class CivitAIDownloader:
    def download_model(self, version_id: int, dest_folder: str) -> str
    def download_with_progress(self, url: str, dest: str, callback=None)
    def verify_download(self, file_path: str, expected_hash: str) -> bool
    def get_model_folder(self, model_type: str) -> str  # Checkpoint, LoRA, etc.
```

### Phase 4: UI Integration

**New UI Tab: "CivitAI Browser"**

```
┌─────────────────────────────────────────────────────────────┐
│ CivitAI Browser                                    [⚙️ Settings]
├─────────────────────────────────────────────────────────────┤
│ Search: [________________] [Type: ▼] [Sort: ▼] [🔍 Search]  │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│ │ Preview │ │ Preview │ │ Preview │ │ Preview │            │
│ │  Image  │ │  Image  │ │  Image  │ │  Image  │            │
│ ├─────────┤ ├─────────┤ ├─────────┤ ├─────────┤            │
│ │Model Nam│ │Model Nam│ │Model Nam│ │Model Nam│            │
│ │⬇️ 1.2K  │ │⬇️ 5.4K  │ │⬇️ 890   │ │⬇️ 2.1K  │            │
│ │[Install]│ │[Install]│ │[Installed]│[Install]│            │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘            │
├─────────────────────────────────────────────────────────────┤
│ [< Prev]  Page 1 of 100  [Next >]                          │
└─────────────────────────────────────────────────────────────┘
```

**Model Details Panel:**
```
┌─────────────────────────────────────────────────────────────┐
│ Model: Juggernaut XL                              [❌ Close] │
├─────────────────────────────────────────────────────────────┤
│ ┌────────────────┐  Creator: kandoo                        │
│ │                │  Type: Checkpoint                       │
│ │  Large Preview │  Base Model: SDXL                       │
│ │                │  Downloads: 125,432                     │
│ │                │  Rating: ⭐ 4.8                          │
│ └────────────────┘                                         │
│                                                             │
│ Versions:                                                   │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ v9.0 (Latest)  [✓ Installed]  12.5 GB                  ││
│ │ v8.0           [Download]      12.3 GB                  ││
│ │ v7.0           [Download]      12.1 GB                  ││
│ └─────────────────────────────────────────────────────────┘│
│                                                             │
│ Description:                                                │
│ Juggernaut XL is a versatile SDXL model focused on...      │
│                                                             │
│ [Download Selected Version]  [Open on CivitAI]             │
└─────────────────────────────────────────────────────────────┘
```

**Settings Panel:**
```
┌─────────────────────────────────────────────────────────────┐
│ CivitAI Settings                                            │
├─────────────────────────────────────────────────────────────┤
│ API Key: [________________________________] [Test] [Save]   │
│                                                             │
│ ☑️ Auto-sync model metadata on startup                      │
│ ☑️ Download preview images                                  │
│ ☐ Show NSFW content                                         │
│                                                             │
│ NSFW Level: [None ▼]  (None, Soft, Mature, X)              │
│ Cache Duration: [24] hours                                  │
│                                                             │
│ [Sync All Models Now]  [Clear Cache]                       │
└─────────────────────────────────────────────────────────────┘
```

### Phase 5: REST API Endpoints

**New endpoints for external tools:**

```
POST /sdapi/v1/civitai/search
GET  /sdapi/v1/civitai/model/{model_id}
GET  /sdapi/v1/civitai/model-by-hash/{hash}
POST /sdapi/v1/civitai/download
GET  /sdapi/v1/civitai/download-progress
POST /sdapi/v1/civitai/sync-metadata
GET  /sdapi/v1/civitai/local-models
```

## Data Models

### Local Model Metadata (JSON sidecar)

```json
{
  "civitai": {
    "model_id": 12345,
    "version_id": 67890,
    "model_name": "Juggernaut XL",
    "version_name": "v9.0",
    "creator": "kandoo",
    "description": "...",
    "base_model": "SDXL",
    "type": "Checkpoint",
    "nsfw": false,
    "tags": ["photorealistic", "portraits"],
    "trained_words": ["juggernaut style"],
    "download_url": "https://...",
    "images": [
      {"url": "https://...", "nsfw": "None"}
    ],
    "stats": {
      "downloads": 125432,
      "rating": 4.8,
      "favorites": 8920
    },
    "synced_at": "2024-11-30T12:00:00Z"
  },
  "local": {
    "sha256": "abc123...",
    "autov2": "def456...",
    "file_size": 13421772800,
    "first_seen": "2024-11-28T10:00:00Z"
  }
}
```

## Implementation Order

### Week 1: Foundation
1. ✅ Create `modules_forge/civitai/` directory structure
2. ✅ Implement `api_client.py` with core API calls
3. ✅ Add API key to settings
4. ✅ Basic hash lookup functionality

### Week 2: Metadata Sync
1. Implement metadata sync for existing models
2. Create JSON sidecar file system
3. Download and cache preview images
4. Add "Sync" button to UI

### Week 3: Browser UI
1. Create CivitAI browser tab
2. Implement search with filters
3. Model grid/list view
4. Model detail panel

### Week 4: Downloads
1. Download manager with queue
2. Progress tracking UI
3. Auto-install to correct folder
4. Version management

### Week 5: Polish
1. REST API endpoints
2. Background sync
3. Update notifications
4. Error handling & retry logic

## Security Considerations

1. **API Key Storage**: Store encrypted in config, never log
2. **Download Verification**: Always verify SHA256 after download
3. **NSFW Filtering**: Respect user settings, default to safe
4. **Rate Limiting**: Implement backoff for API calls
5. **URL Validation**: Only allow civitai.com downloads

## Testing Plan

1. **Unit Tests**: API client, hash calculation, metadata parsing
2. **Integration Tests**: Full sync workflow, download verification
3. **UI Tests**: Search, filter, download flow
4. **Edge Cases**: Network errors, corrupt files, missing metadata

## Success Metrics

- [ ] Can search and browse CivitAI from UI
- [ ] Can download models directly to correct folder
- [ ] Local models show CivitAI metadata
- [ ] Preview images display in model browser
- [ ] Version updates are detected
- [ ] API endpoints work for external tools

## Resources

- [CivitAI REST API Reference](https://github.com/civitai/civitai/wiki/REST-API-Reference)
- [CivitAI Developer Portal](https://developer.civitai.com/docs/api/public-rest)
- [CivitAI Download Guide](https://education.civitai.com/civitais-guide-to-downloading-via-api/)
- [CivitAI Wiki](https://wiki.civitai.com/wiki/Civitai_API)
