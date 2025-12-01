"""
CivitAI API Client

Provides methods for interacting with the CivitAI REST API v1.
Supports model search, lookup by hash, and authenticated downloads.
"""

import logging
import time
from typing import Optional
from urllib.parse import urlencode

import requests

from modules_forge.civitai.models import (
    CivitAIModel,
    CivitAIModelVersion,
    CivitAISearchResult,
    ModelType,
)

logger = logging.getLogger(__name__)


class CivitAIError(Exception):
    """Base exception for CivitAI API errors"""

    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class CivitAIRateLimitError(CivitAIError):
    """Raised when rate limited by CivitAI"""

    pass


class CivitAINotFoundError(CivitAIError):
    """Raised when a model/version is not found"""

    pass


class CivitAIAuthError(CivitAIError):
    """Raised when authentication fails"""

    pass


class CivitAIClient:
    """
    Client for CivitAI REST API v1

    Usage:
        client = CivitAIClient(api_key="your-api-key")
        model = client.get_model(12345)
        results = client.search_models("realistic", model_type=ModelType.CHECKPOINT)
    """

    BASE_URL = "https://civitai.com/api/v1"
    DOWNLOAD_URL = "https://civitai.com/api/download/models"

    # Rate limiting settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    RATE_LIMIT_DELAY = 60.0  # seconds to wait after rate limit

    def __init__(self, api_key: str = None, timeout: int = 30):
        """
        Initialize the CivitAI client.

        Args:
            api_key: CivitAI API key for authenticated requests.
                     Required for downloads and accessing some content.
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

        # Set up session headers
        self.session.headers.update(
            {
                "User-Agent": "Nukem-WebUI/1.0",
                "Accept": "application/json",
            }
        )

        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict = None,
        retry_count: int = 0,
    ) -> dict:
        """
        Make an API request with error handling and retries.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/models")
            params: Query parameters
            retry_count: Current retry attempt

        Returns:
            JSON response as dict

        Raises:
            CivitAIError: On API errors
            CivitAIRateLimitError: When rate limited
            CivitAINotFoundError: When resource not found
            CivitAIAuthError: When authentication fails
        """
        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                timeout=self.timeout,
            )

            # Handle rate limiting
            if response.status_code == 429:
                if retry_count < self.MAX_RETRIES:
                    retry_after = int(
                        response.headers.get("Retry-After", self.RATE_LIMIT_DELAY)
                    )
                    logger.warning(
                        f"Rate limited by CivitAI, waiting {retry_after}s..."
                    )
                    time.sleep(retry_after)
                    return self._make_request(method, endpoint, params, retry_count + 1)
                raise CivitAIRateLimitError(
                    "Rate limited by CivitAI",
                    status_code=429,
                )

            # Handle not found
            if response.status_code == 404:
                raise CivitAINotFoundError(
                    "Resource not found",
                    status_code=404,
                )

            # Handle auth errors
            if response.status_code in (401, 403):
                raise CivitAIAuthError(
                    "Authentication failed - check your API key",
                    status_code=response.status_code,
                )

            # Handle other errors
            if response.status_code >= 400:
                error_msg = f"API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", error_msg)
                except Exception:
                    pass
                raise CivitAIError(error_msg, status_code=response.status_code)

            return response.json()

        except requests.exceptions.Timeout:
            if retry_count < self.MAX_RETRIES:
                logger.warning(f"Request timeout, retrying ({retry_count + 1})...")
                time.sleep(self.RETRY_DELAY)
                return self._make_request(method, endpoint, params, retry_count + 1)
            raise CivitAIError("Request timed out after retries")

        except requests.exceptions.ConnectionError as e:
            if retry_count < self.MAX_RETRIES:
                logger.warning(f"Connection error, retrying ({retry_count + 1})...")
                time.sleep(self.RETRY_DELAY)
                return self._make_request(method, endpoint, params, retry_count + 1)
            raise CivitAIError(f"Connection failed: {e}")

        except requests.exceptions.RequestException as e:
            raise CivitAIError(f"Request failed: {e}")

    # =========================================================================
    # Model Methods
    # =========================================================================

    def get_model(self, model_id: int) -> CivitAIModel:
        """
        Get detailed information about a model by ID.

        Args:
            model_id: CivitAI model ID

        Returns:
            CivitAIModel with full details including all versions

        Raises:
            CivitAINotFoundError: If model doesn't exist
        """
        data = self._make_request("GET", f"/models/{model_id}")
        return CivitAIModel.from_api_response(data)

    def get_model_version(self, version_id: int) -> CivitAIModelVersion:
        """
        Get information about a specific model version.

        Args:
            version_id: CivitAI model version ID

        Returns:
            CivitAIModelVersion with files and images

        Raises:
            CivitAINotFoundError: If version doesn't exist
        """
        data = self._make_request("GET", f"/model-versions/{version_id}")
        return CivitAIModelVersion.from_api_response(data)

    def get_model_by_hash(self, file_hash: str) -> CivitAIModelVersion:
        """
        Look up a model version by file hash.

        This is the primary method for identifying local models.
        Supports SHA256, AutoV1, AutoV2, AutoV3, CRC32, and BLAKE3.

        Args:
            file_hash: Hash of the model file

        Returns:
            CivitAIModelVersion matching the hash

        Raises:
            CivitAINotFoundError: If no model matches the hash
        """
        data = self._make_request("GET", f"/model-versions/by-hash/{file_hash}")
        return CivitAIModelVersion.from_api_response(data)

    # =========================================================================
    # Search Methods
    # =========================================================================

    def search_models(
        self,
        query: str = None,
        model_type: ModelType = None,
        sort: str = "Most Downloaded",
        period: str = "AllTime",
        limit: int = 20,
        page: int = 1,
        nsfw: bool = None,
        tag: str = None,
        username: str = None,
        base_models: list[str] = None,
        cursor: str = None,
    ) -> CivitAISearchResult:
        """
        Search for models on CivitAI.

        Args:
            query: Search query string
            model_type: Filter by model type (Checkpoint, LORA, etc.)
            sort: Sort order - "Highest Rated", "Most Downloaded",
                  "Most Liked", "Most Discussed", "Most Collected", "Newest"
            period: Time period - "AllTime", "Year", "Month", "Week", "Day"
            limit: Results per page (max 100)
            page: Page number (1-indexed)
            nsfw: Filter NSFW content (None = user preference)
            tag: Filter by tag
            username: Filter by creator username
            base_models: Filter by base model(s)
            cursor: Pagination cursor for subsequent pages

        Returns:
            CivitAISearchResult with models and pagination metadata
        """
        params = {}

        if query:
            params["query"] = query
        if model_type:
            params["types"] = model_type.value
        if sort:
            params["sort"] = sort
        if period:
            params["period"] = period
        if limit:
            params["limit"] = min(limit, 100)  # API max is 100
        if page and not cursor:
            params["page"] = page
        if nsfw is not None:
            params["nsfw"] = str(nsfw).lower()
        if tag:
            params["tag"] = tag
        if username:
            params["username"] = username
        if base_models:
            params["baseModels"] = ",".join(base_models)
        if cursor:
            params["cursor"] = cursor

        data = self._make_request("GET", "/models", params=params)
        return CivitAISearchResult.from_api_response(data)

    def get_trending_models(
        self,
        model_type: ModelType = None,
        limit: int = 20,
    ) -> CivitAISearchResult:
        """
        Get trending models (most downloaded this week).

        Args:
            model_type: Filter by model type
            limit: Number of results

        Returns:
            CivitAISearchResult with trending models
        """
        return self.search_models(
            model_type=model_type,
            sort="Most Downloaded",
            period="Week",
            limit=limit,
        )

    def get_newest_models(
        self,
        model_type: ModelType = None,
        limit: int = 20,
    ) -> CivitAISearchResult:
        """
        Get newest models.

        Args:
            model_type: Filter by model type
            limit: Number of results

        Returns:
            CivitAISearchResult with newest models
        """
        return self.search_models(
            model_type=model_type,
            sort="Newest",
            limit=limit,
        )

    # =========================================================================
    # Download Methods
    # =========================================================================

    def get_download_url(self, version_id: int) -> str:
        """
        Get the download URL for a model version.

        Note: Downloads may require authentication (API key).

        Args:
            version_id: Model version ID

        Returns:
            Download URL string
        """
        base_url = f"{self.DOWNLOAD_URL}/{version_id}"
        if self.api_key:
            return f"{base_url}?token={self.api_key}"
        return base_url

    def get_download_info(self, version_id: int) -> dict:
        """
        Get download information including file details.

        Args:
            version_id: Model version ID

        Returns:
            Dict with download URL, file size, and hash info
        """
        version = self.get_model_version(version_id)
        primary_file = version.primary_file

        return {
            "version_id": version_id,
            "version_name": version.name,
            "download_url": self.get_download_url(version_id),
            "file_name": primary_file.name if primary_file else None,
            "file_size_kb": primary_file.size_kb if primary_file else 0,
            "file_size_mb": primary_file.size_mb if primary_file else 0,
            "sha256": primary_file.sha256 if primary_file else None,
            "autov2": primary_file.autov2 if primary_file else None,
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_tags(self, limit: int = 20, page: int = 1) -> list[dict]:
        """
        Get available model tags.

        Args:
            limit: Results per page
            page: Page number

        Returns:
            List of tag dictionaries with name and count
        """
        params = {"limit": limit, "page": page}
        data = self._make_request("GET", "/tags", params=params)
        return data.get("items", [])

    def get_creators(
        self,
        query: str = None,
        limit: int = 20,
        page: int = 1,
    ) -> list[dict]:
        """
        Get model creators.

        Args:
            query: Search query for creator name
            limit: Results per page
            page: Page number

        Returns:
            List of creator dictionaries
        """
        params = {"limit": limit, "page": page}
        if query:
            params["query"] = query
        data = self._make_request("GET", "/creators", params=params)
        return data.get("items", [])

    def test_connection(self) -> bool:
        """
        Test API connectivity and authentication.

        Returns:
            True if connection successful
        """
        try:
            # Try to fetch a single model to test connection
            self.search_models(limit=1)
            return True
        except CivitAIError:
            return False

    def test_api_key(self) -> dict:
        """
        Test if the API key is valid.

        Returns:
            Dict with status and message
        """
        if not self.api_key:
            return {
                "valid": False,
                "message": "No API key configured",
            }

        try:
            # Search with NSFW filter requires auth
            self.search_models(limit=1, nsfw=True)
            return {
                "valid": True,
                "message": "API key is valid",
            }
        except CivitAIAuthError:
            return {
                "valid": False,
                "message": "Invalid API key",
            }
        except CivitAIError as e:
            return {
                "valid": False,
                "message": f"Error testing API key: {e}",
            }


# Global client instance (initialized when API key is set)
_client: Optional[CivitAIClient] = None


def get_client() -> CivitAIClient:
    """
    Get the global CivitAI client instance.

    Returns:
        CivitAIClient instance

    Raises:
        RuntimeError: If client not initialized
    """
    global _client
    if _client is None:
        # Create client without API key for basic functionality
        _client = CivitAIClient()
    return _client


def init_client(api_key: str = None) -> CivitAIClient:
    """
    Initialize or update the global CivitAI client.

    Args:
        api_key: CivitAI API key

    Returns:
        CivitAIClient instance
    """
    global _client
    _client = CivitAIClient(api_key=api_key)
    return _client
