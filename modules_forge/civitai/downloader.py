"""
CivitAI Download Manager

Handles downloading models from CivitAI with:
- Progress tracking
- Resume support
- Hash verification
- Automatic folder placement
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Callable, Optional

import requests

from modules_forge.civitai.api_client import CivitAIClient, get_client
from modules_forge.civitai.hash_utils import calculate_sha256, get_metadata_path
from modules_forge.civitai.models import CivitAIModelVersion, LocalModelMetadata

logger = logging.getLogger(__name__)


class DownloadStatus(str, Enum):
    """Status of a download"""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class DownloadProgress:
    """Progress information for a download"""

    total_bytes: int = 0
    downloaded_bytes: int = 0
    speed_bytes_per_sec: float = 0
    eta_seconds: float = 0
    percent: float = 0

    @property
    def speed_mb_per_sec(self) -> float:
        return self.speed_bytes_per_sec / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def downloaded_mb(self) -> float:
        return self.downloaded_bytes / (1024 * 1024)

    def format_speed(self) -> str:
        if self.speed_bytes_per_sec < 1024:
            return f"{self.speed_bytes_per_sec:.0f} B/s"
        elif self.speed_bytes_per_sec < 1024 * 1024:
            return f"{self.speed_bytes_per_sec / 1024:.1f} KB/s"
        else:
            return f"{self.speed_mb_per_sec:.1f} MB/s"

    def format_eta(self) -> str:
        if self.eta_seconds < 60:
            return f"{self.eta_seconds:.0f}s"
        elif self.eta_seconds < 3600:
            return f"{self.eta_seconds / 60:.0f}m"
        else:
            return f"{self.eta_seconds / 3600:.1f}h"


@dataclass
class DownloadTask:
    """A single download task"""

    task_id: str
    version_id: int
    download_url: str
    destination_path: str
    expected_sha256: Optional[str] = None
    file_name: str = ""
    model_name: str = ""
    version_name: str = ""
    status: DownloadStatus = DownloadStatus.PENDING
    progress: DownloadProgress = field(default_factory=DownloadProgress)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def is_active(self) -> bool:
        return self.status in (DownloadStatus.DOWNLOADING, DownloadStatus.VERIFYING)

    @property
    def is_finished(self) -> bool:
        return self.status in (
            DownloadStatus.COMPLETED,
            DownloadStatus.FAILED,
            DownloadStatus.CANCELLED,
        )


class CivitAIDownloader:
    """
    Download manager for CivitAI models.

    Supports queued downloads with progress tracking,
    resume capability, and hash verification.
    """

    # Model type to folder mapping
    MODEL_FOLDERS = {
        "Checkpoint": "Stable-diffusion",
        "LORA": "Lora",
        "LoCon": "Lora",
        "VAE": "VAE",
        "Controlnet": "ControlNet",
        "Upscaler": "ESRGAN",
        "TextualInversion": "embeddings",
        "Hypernetwork": "hypernetworks",
        "MotionModule": "motion-modules",
        "Poses": "poses",
        "Wildcards": "wildcards",
        "Other": "other",
    }

    def __init__(
        self,
        client: CivitAIClient = None,
        models_path: str = None,
        max_concurrent: int = 1,
        chunk_size: int = 8192,
    ):
        """
        Initialize the downloader.

        Args:
            client: CivitAI client instance
            models_path: Base path for models directory
            max_concurrent: Maximum concurrent downloads (default 1)
            chunk_size: Download chunk size in bytes
        """
        self.client = client or get_client()
        self.chunk_size = chunk_size
        self.max_concurrent = max_concurrent

        # Get models path
        if models_path:
            self.models_path = models_path
        else:
            try:
                from modules.paths import models_path as mp

                self.models_path = mp
            except ImportError:
                self.models_path = "models"

        # Task management
        self._tasks: dict[str, DownloadTask] = {}
        self._queue: Queue[str] = Queue()
        self._active_downloads: int = 0
        self._lock = threading.Lock()
        self._cancel_flags: dict[str, bool] = {}

        # Worker thread
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def get_model_folder(self, model_type: str) -> str:
        """
        Get the destination folder for a model type.

        Args:
            model_type: CivitAI model type (e.g., "Checkpoint", "LORA")

        Returns:
            Absolute path to the model folder
        """
        folder_name = self.MODEL_FOLDERS.get(model_type, "other")
        return os.path.join(self.models_path, folder_name)

    def generate_task_id(self) -> str:
        """Generate a unique task ID."""
        return f"dl_{int(time.time() * 1000)}_{os.urandom(4).hex()}"

    def queue_download(
        self,
        version_id: int,
        model_type: str = "Checkpoint",
        subfolder: str = None,
        custom_filename: str = None,
        version_info: CivitAIModelVersion = None,
    ) -> DownloadTask:
        """
        Queue a model download.

        Args:
            version_id: CivitAI model version ID
            model_type: Model type for folder placement
            subfolder: Optional subfolder within model type folder
            custom_filename: Custom filename (default uses CivitAI filename)
            version_info: Pre-fetched version info (to avoid extra API call)

        Returns:
            DownloadTask for tracking progress
        """
        # Fetch version info if not provided
        if version_info is None:
            version_info = self.client.get_model_version(version_id)

        # Get primary file info
        primary_file = version_info.primary_file
        if not primary_file:
            raise ValueError(f"No primary file found for version {version_id}")

        # Determine filename
        if custom_filename:
            filename = custom_filename
        else:
            filename = primary_file.name

        # Determine destination path
        base_folder = self.get_model_folder(model_type)
        if subfolder:
            dest_folder = os.path.join(base_folder, subfolder)
        else:
            dest_folder = base_folder

        # Create folder if needed
        os.makedirs(dest_folder, exist_ok=True)

        dest_path = os.path.join(dest_folder, filename)

        # Create task
        task_id = self.generate_task_id()
        task = DownloadTask(
            task_id=task_id,
            version_id=version_id,
            download_url=self.client.get_download_url(version_id),
            destination_path=dest_path,
            expected_sha256=primary_file.sha256,
            file_name=filename,
            model_name=version_info.name,  # Will be version name if model name not available
            version_name=version_info.name,
        )

        with self._lock:
            self._tasks[task_id] = task
            self._cancel_flags[task_id] = False
            self._queue.put(task_id)

        # Start worker if not running
        self._ensure_worker_running()

        logger.info(f"Queued download: {filename} ({task_id})")
        return task

    def _ensure_worker_running(self):
        """Start the worker thread if not running."""
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()

    def _worker(self):
        """Background worker for processing download queue."""
        while self._running:
            try:
                # Get next task from queue
                task_id = self._queue.get(timeout=1.0)

                with self._lock:
                    if self._active_downloads >= self.max_concurrent:
                        # Re-queue if at capacity
                        self._queue.put(task_id)
                        continue

                    task = self._tasks.get(task_id)
                    if not task or task.is_finished:
                        continue

                    self._active_downloads += 1

                # Process the download
                try:
                    self._download_file(task)
                finally:
                    with self._lock:
                        self._active_downloads -= 1

            except Exception:
                # Queue.get timeout, continue loop
                continue

    def _download_file(self, task: DownloadTask):
        """
        Download a file with progress tracking.

        Args:
            task: DownloadTask to process
        """
        task.status = DownloadStatus.DOWNLOADING
        task.started_at = datetime.now()

        temp_path = task.destination_path + ".download"

        try:
            # Check if we can resume
            resume_pos = 0
            headers = {}
            if os.path.exists(temp_path):
                resume_pos = os.path.getsize(temp_path)
                headers["Range"] = f"bytes={resume_pos}-"

            # Start download
            response = requests.get(
                task.download_url,
                headers=headers,
                stream=True,
                timeout=30,
            )

            # Handle response
            if response.status_code == 416:
                # Range not satisfiable - file may be complete
                resume_pos = 0
                response = requests.get(
                    task.download_url,
                    stream=True,
                    timeout=30,
                )

            response.raise_for_status()

            # Get total size
            content_length = response.headers.get("content-length")
            if content_length:
                total_size = int(content_length) + resume_pos
            else:
                total_size = 0

            task.progress.total_bytes = total_size
            task.progress.downloaded_bytes = resume_pos

            # Open file for writing
            mode = "ab" if resume_pos > 0 else "wb"
            start_time = time.time()
            bytes_since_last = 0
            last_time = start_time

            with open(temp_path, mode) as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    # Check for cancellation
                    if self._cancel_flags.get(task.task_id, False):
                        task.status = DownloadStatus.CANCELLED
                        logger.info(f"Download cancelled: {task.file_name}")
                        return

                    if chunk:
                        f.write(chunk)
                        task.progress.downloaded_bytes += len(chunk)
                        bytes_since_last += len(chunk)

                        # Update speed and ETA periodically
                        current_time = time.time()
                        elapsed = current_time - last_time
                        if elapsed >= 0.5:  # Update every 0.5s
                            speed = bytes_since_last / elapsed
                            task.progress.speed_bytes_per_sec = speed

                            remaining = total_size - task.progress.downloaded_bytes
                            if speed > 0:
                                task.progress.eta_seconds = remaining / speed
                            else:
                                task.progress.eta_seconds = 0

                            if total_size > 0:
                                task.progress.percent = (
                                    task.progress.downloaded_bytes / total_size * 100
                                )

                            bytes_since_last = 0
                            last_time = current_time

            # Verify hash if provided
            if task.expected_sha256:
                task.status = DownloadStatus.VERIFYING
                logger.info(f"Verifying download: {task.file_name}")

                actual_hash = calculate_sha256(temp_path)
                if actual_hash.lower() != task.expected_sha256.lower():
                    task.status = DownloadStatus.FAILED
                    task.error = f"Hash mismatch: expected {task.expected_sha256[:16]}..., got {actual_hash[:16]}..."
                    logger.error(f"Hash verification failed for {task.file_name}")
                    os.remove(temp_path)
                    return

            # Move to final location
            if os.path.exists(task.destination_path):
                os.remove(task.destination_path)
            os.rename(temp_path, task.destination_path)

            task.status = DownloadStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress.percent = 100

            logger.info(f"Download completed: {task.file_name}")

        except requests.exceptions.RequestException as e:
            task.status = DownloadStatus.FAILED
            task.error = f"Download error: {e}"
            logger.error(f"Download failed for {task.file_name}: {e}")

        except Exception as e:
            task.status = DownloadStatus.FAILED
            task.error = str(e)
            logger.error(f"Download error for {task.file_name}: {e}")

    def cancel_download(self, task_id: str) -> bool:
        """
        Cancel a download.

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if task was found and cancellation requested
        """
        with self._lock:
            if task_id in self._tasks:
                self._cancel_flags[task_id] = True
                return True
        return False

    def get_task(self, task_id: str) -> Optional[DownloadTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[DownloadTask]:
        """Get all tasks."""
        return list(self._tasks.values())

    def get_active_tasks(self) -> list[DownloadTask]:
        """Get all active (downloading) tasks."""
        return [t for t in self._tasks.values() if t.is_active]

    def get_queue_length(self) -> int:
        """Get number of tasks in queue."""
        return self._queue.qsize()

    def clear_completed(self):
        """Remove completed/failed/cancelled tasks from memory."""
        with self._lock:
            completed_ids = [
                tid for tid, task in self._tasks.items() if task.is_finished
            ]
            for tid in completed_ids:
                del self._tasks[tid]
                self._cancel_flags.pop(tid, None)

    def stop(self):
        """Stop the download worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)


# Global downloader instance
_downloader: Optional[CivitAIDownloader] = None


def get_downloader() -> CivitAIDownloader:
    """Get the global downloader instance."""
    global _downloader
    if _downloader is None:
        _downloader = CivitAIDownloader()
    return _downloader
