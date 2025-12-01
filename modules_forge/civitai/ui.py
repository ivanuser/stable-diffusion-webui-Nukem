"""
CivitAI Browser UI

Gradio-based UI for browsing, searching, and downloading models from CivitAI.
"""

import logging
import os
from typing import Optional

import gradio as gr

from modules import shared
from modules.ui_components import FormRow, ToolButton

from modules_forge.civitai.api_client import (
    CivitAIClient,
    CivitAIError,
    CivitAINotFoundError,
    get_client,
    init_client,
)
from modules_forge.civitai.downloader import (
    DownloadStatus,
    get_downloader,
)
from modules_forge.civitai.metadata_sync import get_sync
from modules_forge.civitai.models import ModelType, CivitAIModel

logger = logging.getLogger(__name__)

# Refresh symbol
refresh_symbol = "\U0001f504"  # 🔄
download_symbol = "\u2B07"  # ⬇
search_symbol = "\U0001f50D"  # 🔍
info_symbol = "\u2139\ufe0f"  # ℹ️
link_symbol = "\U0001f517"  # 🔗


def get_api_key() -> str:
    """Get API key from settings."""
    return getattr(shared.opts, "civitai_api_key", "") or ""


def is_enabled() -> bool:
    """Check if CivitAI integration is enabled."""
    return getattr(shared.opts, "civitai_enabled", True)


def format_number(num: int) -> str:
    """Format large numbers (e.g., 1234 -> 1.2K)."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def format_file_size(size_kb: float) -> str:
    """Format file size in human readable form."""
    if size_kb >= 1_000_000:
        return f"{size_kb / 1_000_000:.1f} GB"
    elif size_kb >= 1_000:
        return f"{size_kb / 1_000:.1f} MB"
    return f"{size_kb:.0f} KB"


def create_model_card_html(model: CivitAIModel) -> str:
    """Create HTML for a model card."""
    # Get preview image
    preview_url = ""
    if model.preview_image:
        preview_url = model.preview_image.url

    # Get stats
    downloads = format_number(model.stats.download_count if model.stats else 0)
    rating = f"{model.stats.rating:.1f}" if model.stats and model.stats.rating else "N/A"

    # Get latest version info
    latest = model.latest_version
    base_model = latest.base_model if latest else "Unknown"

    # File size
    file_size = ""
    if latest and latest.primary_file:
        file_size = format_file_size(latest.primary_file.size_kb)

    nsfw_badge = '<span class="civitai-nsfw-badge">NSFW</span>' if model.nsfw else ""

    return f"""
    <div class="civitai-model-card" data-model-id="{model.id}">
        <div class="civitai-card-image">
            <img src="{preview_url}" alt="{model.name}" loading="lazy" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect fill=%22%23333%22 width=%22100%22 height=%22100%22/><text fill=%22%23666%22 font-size=%2212%22 x=%2250%22 y=%2250%22 text-anchor=%22middle%22>No Image</text></svg>'"/>
            {nsfw_badge}
        </div>
        <div class="civitai-card-info">
            <div class="civitai-card-title" title="{model.name}">{model.name}</div>
            <div class="civitai-card-meta">
                <span class="civitai-card-type">{model.type}</span>
                <span class="civitai-card-base">{base_model}</span>
            </div>
            <div class="civitai-card-stats">
                <span title="Downloads">{download_symbol} {downloads}</span>
                <span title="Rating">★ {rating}</span>
                {f'<span title="Size">{file_size}</span>' if file_size else ''}
            </div>
        </div>
    </div>
    """


def create_model_detail_html(model: CivitAIModel) -> str:
    """Create HTML for model detail view."""
    if not model:
        return "<div class='civitai-no-selection'>Select a model to view details</div>"

    # Creator info
    creator = model.creator.username if model.creator else "Unknown"

    # Stats
    downloads = format_number(model.stats.download_count if model.stats else 0)
    favorites = format_number(model.stats.favorite_count if model.stats else 0)
    rating = f"{model.stats.rating:.1f}" if model.stats and model.stats.rating else "N/A"

    # Tags
    tags_html = " ".join([f'<span class="civitai-tag">{tag}</span>' for tag in model.tags[:10]])

    # Versions list
    versions_html = ""
    for v in model.model_versions[:5]:
        file_size = ""
        if v.primary_file:
            file_size = format_file_size(v.primary_file.size_kb)
        versions_html += f"""
        <div class="civitai-version-item" data-version-id="{v.id}">
            <span class="civitai-version-name">{v.name}</span>
            <span class="civitai-version-base">{v.base_model or 'Unknown'}</span>
            <span class="civitai-version-size">{file_size}</span>
        </div>
        """

    # Description (truncated)
    description = model.description or "No description available."
    if len(description) > 500:
        description = description[:500] + "..."

    # Preview images
    images_html = ""
    if model.latest_version:
        for img in model.latest_version.images[:4]:
            images_html += f'<img src="{img.url}" alt="Preview" loading="lazy" class="civitai-detail-preview"/>'

    return f"""
    <div class="civitai-model-detail">
        <div class="civitai-detail-header">
            <h2>{model.name}</h2>
            <a href="https://civitai.com/models/{model.id}" target="_blank" class="civitai-external-link">{link_symbol} View on CivitAI</a>
        </div>

        <div class="civitai-detail-images">
            {images_html}
        </div>

        <div class="civitai-detail-meta">
            <div class="civitai-detail-row">
                <span class="civitai-detail-label">Creator:</span>
                <span>{creator}</span>
            </div>
            <div class="civitai-detail-row">
                <span class="civitai-detail-label">Type:</span>
                <span>{model.type}</span>
            </div>
            <div class="civitai-detail-row">
                <span class="civitai-detail-label">Downloads:</span>
                <span>{downloads}</span>
            </div>
            <div class="civitai-detail-row">
                <span class="civitai-detail-label">Favorites:</span>
                <span>{favorites}</span>
            </div>
            <div class="civitai-detail-row">
                <span class="civitai-detail-label">Rating:</span>
                <span>★ {rating}</span>
            </div>
        </div>

        <div class="civitai-detail-tags">
            {tags_html}
        </div>

        <div class="civitai-detail-description">
            <h3>Description</h3>
            <p>{description}</p>
        </div>

        <div class="civitai-detail-versions">
            <h3>Versions</h3>
            {versions_html}
        </div>
    </div>
    """


class CivitAIBrowser:
    """CivitAI Browser UI component."""

    def __init__(self):
        self.current_results: list[CivitAIModel] = []
        self.selected_model: Optional[CivitAIModel] = None
        self.current_page = 1
        self.total_pages = 1

    def search(
        self,
        query: str,
        model_type: str,
        sort: str,
        period: str,
        base_model: str,
        page: int = 1,
    ) -> tuple[str, str, str]:
        """
        Search CivitAI for models.

        Returns:
            Tuple of (results_html, status_text, page_info)
        """
        if not is_enabled():
            return (
                "<div class='civitai-disabled'>CivitAI integration is disabled in settings.</div>",
                "CivitAI disabled",
                "Page 0 of 0",
            )

        try:
            # Initialize client with current API key
            api_key = get_api_key()
            client = init_client(api_key) if api_key else get_client()

            # Build search params
            type_filter = None
            if model_type and model_type != "All":
                try:
                    type_filter = ModelType(model_type)
                except ValueError:
                    pass

            base_models = None
            if base_model and base_model != "All":
                base_models = [base_model]

            # Get results per page from settings
            limit = getattr(shared.opts, "civitai_results_per_page", 20)

            # Perform search
            results = client.search_models(
                query=query if query else None,
                model_type=type_filter,
                sort=sort,
                period=period,
                base_models=base_models,
                limit=limit,
                page=page,
            )

            self.current_results = results.items
            self.current_page = results.current_page
            self.total_pages = results.total_pages

            # Generate HTML for results
            if not results.items:
                return (
                    "<div class='civitai-no-results'>No models found matching your search.</div>",
                    f"No results for '{query}'",
                    f"Page {self.current_page} of {self.total_pages}",
                )

            cards_html = '<div class="civitai-results-grid">'
            for model in results.items:
                cards_html += create_model_card_html(model)
            cards_html += "</div>"

            return (
                cards_html,
                f"Found {results.total_items} models",
                f"Page {self.current_page} of {self.total_pages}",
            )

        except CivitAIError as e:
            logger.error(f"CivitAI search error: {e}")
            return (
                f"<div class='civitai-error'>Error: {e}</div>",
                f"Error: {e}",
                "Page 0 of 0",
            )
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return (
                f"<div class='civitai-error'>Unexpected error: {e}</div>",
                f"Error: {e}",
                "Page 0 of 0",
            )

    def select_model(self, model_id: int) -> str:
        """Select a model and return its detail HTML."""
        for model in self.current_results:
            if model.id == model_id:
                self.selected_model = model
                return create_model_detail_html(model)

        # If not in current results, fetch from API
        try:
            api_key = get_api_key()
            client = init_client(api_key) if api_key else get_client()
            model = client.get_model(model_id)
            self.selected_model = model
            return create_model_detail_html(model)
        except Exception as e:
            return f"<div class='civitai-error'>Error loading model: {e}</div>"

    def download_model(self, version_id: int, model_type: str) -> str:
        """Queue a model download."""
        if not version_id:
            return "No version selected"

        api_key = get_api_key()
        if not api_key:
            return "API key required for downloads. Set it in Settings > CivitAI."

        try:
            downloader = get_downloader()
            task = downloader.queue_download(
                version_id=version_id,
                model_type=model_type,
            )
            return f"Download queued: {task.file_name}"
        except Exception as e:
            return f"Download error: {e}"

    def get_download_status(self) -> str:
        """Get current download status."""
        downloader = get_downloader()
        tasks = downloader.get_all_tasks()

        if not tasks:
            return "No active downloads"

        status_lines = []
        for task in tasks[-5:]:  # Show last 5
            if task.status == DownloadStatus.DOWNLOADING:
                status_lines.append(
                    f"{task.file_name}: {task.progress.percent:.0f}% "
                    f"({task.progress.format_speed()}, ETA: {task.progress.format_eta()})"
                )
            elif task.status == DownloadStatus.COMPLETED:
                status_lines.append(f"{task.file_name}: Complete ✓")
            elif task.status == DownloadStatus.FAILED:
                status_lines.append(f"{task.file_name}: Failed - {task.error}")
            else:
                status_lines.append(f"{task.file_name}: {task.status.value}")

        return "\n".join(status_lines)

    def sync_local_models(self, progress=gr.Progress()) -> str:
        """Sync local models with CivitAI metadata."""
        try:
            sync = get_sync()

            def update_progress(p):
                progress(p.processed / max(p.total_models, 1),
                        f"Syncing: {p.current_model}")

            results = sync.sync_all(
                download_preview=getattr(shared.opts, "civitai_download_preview", True),
                skip_existing=True,
                progress_callback=update_progress,
            )

            found = sum(1 for r in results if r.found_on_civitai)
            not_found = sum(1 for r in results if not r.found_on_civitai and r.success)
            errors = sum(1 for r in results if not r.success)

            return f"Sync complete: {found} matched, {not_found} not on CivitAI, {errors} errors"
        except Exception as e:
            return f"Sync error: {e}"

    def test_api_key(self) -> str:
        """Test if the API key is valid."""
        api_key = get_api_key()
        if not api_key:
            return "No API key configured"

        client = CivitAIClient(api_key=api_key)
        result = client.test_api_key()
        return result["message"]


# Global browser instance
_browser: Optional[CivitAIBrowser] = None


def get_browser() -> CivitAIBrowser:
    """Get the global browser instance."""
    global _browser
    if _browser is None:
        _browser = CivitAIBrowser()
    return _browser


def create_ui():
    """Create the CivitAI browser UI tab."""
    browser = get_browser()

    with gr.Blocks(analytics_enabled=False) as civitai_interface:
        # CSS styles
        gr.HTML("""
        <style>
        .civitai-results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 16px;
            padding: 16px;
        }
        .civitai-model-card {
            background: var(--block-background-fill);
            border: 1px solid var(--border-color-primary);
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .civitai-model-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .civitai-card-image {
            position: relative;
            aspect-ratio: 1;
            background: var(--background-fill-secondary);
        }
        .civitai-card-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .civitai-nsfw-badge {
            position: absolute;
            top: 4px;
            right: 4px;
            background: #e53935;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
        }
        .civitai-card-info {
            padding: 8px;
        }
        .civitai-card-title {
            font-weight: 600;
            font-size: 14px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 4px;
        }
        .civitai-card-meta {
            display: flex;
            gap: 8px;
            font-size: 11px;
            color: var(--body-text-color-subdued);
            margin-bottom: 4px;
        }
        .civitai-card-type {
            background: var(--primary-500);
            color: white;
            padding: 1px 4px;
            border-radius: 3px;
        }
        .civitai-card-stats {
            display: flex;
            gap: 8px;
            font-size: 11px;
            color: var(--body-text-color-subdued);
        }
        .civitai-model-detail {
            padding: 16px;
        }
        .civitai-detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .civitai-detail-header h2 {
            margin: 0;
            font-size: 20px;
        }
        .civitai-external-link {
            color: var(--link-text-color);
            text-decoration: none;
        }
        .civitai-detail-images {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            overflow-x: auto;
        }
        .civitai-detail-preview {
            height: 150px;
            border-radius: 4px;
        }
        .civitai-detail-meta {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-bottom: 16px;
        }
        .civitai-detail-row {
            display: flex;
            gap: 8px;
        }
        .civitai-detail-label {
            font-weight: 600;
            color: var(--body-text-color-subdued);
        }
        .civitai-detail-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-bottom: 16px;
        }
        .civitai-tag {
            background: var(--background-fill-secondary);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
        }
        .civitai-version-item {
            display: flex;
            justify-content: space-between;
            padding: 8px;
            border: 1px solid var(--border-color-primary);
            border-radius: 4px;
            margin-bottom: 4px;
            cursor: pointer;
        }
        .civitai-version-item:hover {
            background: var(--background-fill-secondary);
        }
        .civitai-no-results, .civitai-error, .civitai-disabled, .civitai-no-selection {
            padding: 40px;
            text-align: center;
            color: var(--body-text-color-subdued);
        }
        .civitai-error {
            color: #e53935;
        }
        </style>
        """)

        with gr.Row():
            # Left panel - Search and Results
            with gr.Column(scale=2):
                gr.Markdown("## CivitAI Model Browser")

                # Search controls
                with FormRow():
                    search_query = gr.Textbox(
                        label="Search",
                        placeholder="Search models...",
                        scale=3,
                    )
                    search_btn = ToolButton(
                        value=search_symbol,
                        elem_id="civitai_search_btn",
                    )

                with FormRow():
                    model_type = gr.Dropdown(
                        label="Type",
                        choices=["All", "Checkpoint", "LORA", "LoCon", "VAE",
                                "Controlnet", "Upscaler", "TextualInversion", "MotionModule"],
                        value="All",
                        scale=1,
                    )
                    sort_order = gr.Dropdown(
                        label="Sort",
                        choices=["Most Downloaded", "Highest Rated", "Most Liked",
                                "Most Discussed", "Most Collected", "Newest"],
                        value="Most Downloaded",
                        scale=1,
                    )
                    time_period = gr.Dropdown(
                        label="Period",
                        choices=["AllTime", "Year", "Month", "Week", "Day"],
                        value="AllTime",
                        scale=1,
                    )
                    base_model_filter = gr.Dropdown(
                        label="Base Model",
                        choices=["All", "SDXL 1.0", "SD 1.5", "Flux.1 D", "Flux.1 S",
                                "Pony", "SD 3.5", "Wan", "Illustrious"],
                        value="All",
                        scale=1,
                    )

                # Status bar
                with FormRow():
                    status_text = gr.Textbox(
                        label="Status",
                        value="Ready to search",
                        interactive=False,
                        scale=2,
                    )
                    page_info = gr.Textbox(
                        label="Page",
                        value="Page 0 of 0",
                        interactive=False,
                        scale=1,
                    )

                # Results display
                results_html = gr.HTML(
                    value="<div class='civitai-no-results'>Enter a search term or browse by type</div>",
                    elem_id="civitai_results",
                )

                # Pagination
                with FormRow():
                    prev_btn = gr.Button("← Previous", scale=1)
                    current_page = gr.Number(value=1, label="Page", minimum=1, scale=1)
                    next_btn = gr.Button("Next →", scale=1)

            # Right panel - Model Details & Downloads
            with gr.Column(scale=1):
                gr.Markdown("## Model Details")

                # Model ID input (for manual lookup)
                with FormRow():
                    model_id_input = gr.Number(
                        label="Model ID",
                        value=0,
                        precision=0,
                    )
                    load_model_btn = ToolButton(
                        value=info_symbol,
                        elem_id="civitai_load_model_btn",
                    )

                # Detail display
                detail_html = gr.HTML(
                    value="<div class='civitai-no-selection'>Select a model to view details</div>",
                    elem_id="civitai_detail",
                )

                # Download section
                gr.Markdown("### Download")
                with FormRow():
                    version_id_input = gr.Number(
                        label="Version ID",
                        value=0,
                        precision=0,
                    )
                    download_type = gr.Dropdown(
                        label="Save as",
                        choices=["Checkpoint", "LORA", "VAE", "Controlnet",
                                "Upscaler", "TextualInversion"],
                        value="Checkpoint",
                    )

                download_btn = gr.Button(f"{download_symbol} Download Model", variant="primary")
                download_status = gr.Textbox(
                    label="Download Status",
                    value="No active downloads",
                    interactive=False,
                    lines=3,
                )
                refresh_downloads_btn = gr.Button(f"{refresh_symbol} Refresh Status")

                # Sync section
                gr.Markdown("### Local Model Sync")
                sync_btn = gr.Button(f"{refresh_symbol} Sync Local Models with CivitAI")
                sync_status = gr.Textbox(
                    label="Sync Status",
                    value="",
                    interactive=False,
                )

                # API Test
                gr.Markdown("### API Status")
                test_api_btn = gr.Button("Test API Key")
                api_status = gr.Textbox(
                    label="API Status",
                    value="",
                    interactive=False,
                )

        # Event handlers
        def do_search(query, mtype, sort, period, base, page):
            return browser.search(query, mtype, sort, period, base, int(page))

        search_btn.click(
            fn=do_search,
            inputs=[search_query, model_type, sort_order, time_period,
                   base_model_filter, current_page],
            outputs=[results_html, status_text, page_info],
        )

        search_query.submit(
            fn=do_search,
            inputs=[search_query, model_type, sort_order, time_period,
                   base_model_filter, current_page],
            outputs=[results_html, status_text, page_info],
        )

        def go_prev(query, mtype, sort, period, base, page):
            new_page = max(1, int(page) - 1)
            return (new_page,) + browser.search(query, mtype, sort, period, base, new_page)

        def go_next(query, mtype, sort, period, base, page):
            new_page = int(page) + 1
            return (new_page,) + browser.search(query, mtype, sort, period, base, new_page)

        prev_btn.click(
            fn=go_prev,
            inputs=[search_query, model_type, sort_order, time_period,
                   base_model_filter, current_page],
            outputs=[current_page, results_html, status_text, page_info],
        )

        next_btn.click(
            fn=go_next,
            inputs=[search_query, model_type, sort_order, time_period,
                   base_model_filter, current_page],
            outputs=[current_page, results_html, status_text, page_info],
        )

        load_model_btn.click(
            fn=lambda mid: browser.select_model(int(mid)),
            inputs=[model_id_input],
            outputs=[detail_html],
        )

        download_btn.click(
            fn=lambda vid, dtype: browser.download_model(int(vid), dtype),
            inputs=[version_id_input, download_type],
            outputs=[download_status],
        )

        refresh_downloads_btn.click(
            fn=browser.get_download_status,
            outputs=[download_status],
        )

        sync_btn.click(
            fn=browser.sync_local_models,
            outputs=[sync_status],
        )

        test_api_btn.click(
            fn=browser.test_api_key,
            outputs=[api_status],
        )

    return civitai_interface


def on_ui_tabs():
    """Register the CivitAI tab with the WebUI."""
    if not is_enabled():
        return []

    return [(create_ui(), "CivitAI", "civitai")]
