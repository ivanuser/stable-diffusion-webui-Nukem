"""
CivitAI Browser UI

Gradio-based UI for browsing, searching, and downloading models from CivitAI.
"""

import logging
from typing import Optional

import gradio as gr

from modules import shared
from modules.ui_components import FormRow

from modules_forge.civitai.api_client import (
    CivitAIClient,
    CivitAIError,
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


def get_api_key() -> str:
    """Get API key from settings."""
    return getattr(shared.opts, "civitai_api_key", "") or ""


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
    ):
        """Search CivitAI for models. Returns gallery data and status."""
        try:
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

            limit = getattr(shared.opts, "civitai_results_per_page", 20)

            results = client.search_models(
                query=query if query else None,
                model_type=type_filter,
                sort=sort,
                period=period,
                base_models=base_models,
                limit=limit,
                page=int(page),
            )

            self.current_results = results.items
            self.current_page = results.current_page
            self.total_pages = results.total_pages

            if not results.items:
                return (
                    [],  # gallery
                    "No models found",
                    f"Page {self.current_page} of {self.total_pages}",
                    gr.update(choices=[]),  # model selector
                )

            # Build gallery images and model choices
            gallery_items = []
            model_choices = []

            for model in results.items:
                # Get preview image
                preview_url = None
                if model.preview_image and model.preview_image.url:
                    preview_url = model.preview_image.url

                # Get info for label
                downloads = format_number(model.stats.download_count if model.stats else 0)
                model_label = f"{model.name} ({model.type}, {downloads} DLs)"

                if preview_url:
                    gallery_items.append((preview_url, model_label))

                model_choices.append(f"{model.id}: {model.name}")

            return (
                gallery_items,
                f"Found {results.total_items} models",
                f"Page {self.current_page} of {self.total_pages}",
                gr.update(choices=model_choices, value=model_choices[0] if model_choices else None),
            )

        except CivitAIError as e:
            logger.error(f"CivitAI search error: {e}")
            return [], f"Error: {e}", "Page 0 of 0", gr.update(choices=[])
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return [], f"Error: {e}", "Page 0 of 0", gr.update(choices=[])

    def load_model_details(self, model_selection: str):
        """Load details for selected model."""
        if not model_selection:
            return "Select a model to view details", gr.update(choices=[]), "", ""

        try:
            # Parse model ID from selection
            model_id = int(model_selection.split(":")[0])

            # Find in current results or fetch
            model = None
            for m in self.current_results:
                if m.id == model_id:
                    model = m
                    break

            if not model:
                api_key = get_api_key()
                client = init_client(api_key) if api_key else get_client()
                model = client.get_model(model_id)

            self.selected_model = model

            # Build details text
            creator = model.creator.username if model.creator else "Unknown"
            downloads = format_number(model.stats.download_count if model.stats else 0)
            rating = f"{model.stats.rating:.1f}" if model.stats and model.stats.rating else "N/A"

            details = f"""**{model.name}**

**Creator:** {creator}
**Type:** {model.type}
**Downloads:** {downloads}
**Rating:** ★ {rating}

**Tags:** {', '.join(model.tags[:10]) if model.tags else 'None'}

**Description:**
{(model.description or 'No description')[:500]}{'...' if model.description and len(model.description) > 500 else ''}

[View on CivitAI](https://civitai.com/models/{model.id})
"""

            # Build version choices
            version_choices = []
            for v in model.model_versions[:10]:
                size = ""
                if v.primary_file:
                    size = f" ({format_file_size(v.primary_file.size_kb)})"
                version_choices.append(f"{v.id}: {v.name}{size}")

            # Get first version ID for auto-fill
            first_version_id = str(model.model_versions[0].id) if model.model_versions else ""

            return (
                details,
                gr.update(choices=version_choices, value=version_choices[0] if version_choices else None),
                first_version_id,
                model.type,
            )

        except Exception as e:
            logger.error(f"Error loading model details: {e}")
            return f"Error: {e}", gr.update(choices=[]), "", ""

    def get_version_id(self, version_selection: str):
        """Extract version ID from selection."""
        if not version_selection:
            return ""
        try:
            return version_selection.split(":")[0]
        except:
            return ""

    def download_model(self, version_id_str: str, model_type: str) -> str:
        """Queue a model download."""
        if not version_id_str:
            return "No version selected"

        try:
            version_id = int(version_id_str)
        except:
            return "Invalid version ID"

        api_key = get_api_key()
        if not api_key:
            return "API key required for downloads. Set it in Settings > CivitAI."

        try:
            downloader = get_downloader()
            task = downloader.queue_download(
                version_id=version_id,
                model_type=model_type,
            )
            return f"Download started: {task.file_name}\nSaving to: {task.destination_path}"
        except Exception as e:
            return f"Download error: {e}"

    def get_download_status(self) -> str:
        """Get current download status."""
        downloader = get_downloader()
        tasks = downloader.get_all_tasks()

        if not tasks:
            return "No downloads"

        status_lines = []
        for task in tasks[-5:]:
            if task.status == DownloadStatus.DOWNLOADING:
                status_lines.append(
                    f"⬇ {task.file_name}: {task.progress.percent:.0f}% "
                    f"({task.progress.format_speed()})"
                )
            elif task.status == DownloadStatus.COMPLETED:
                status_lines.append(f"✓ {task.file_name}: Complete")
            elif task.status == DownloadStatus.FAILED:
                status_lines.append(f"✗ {task.file_name}: Failed - {task.error}")
            elif task.status == DownloadStatus.VERIFYING:
                status_lines.append(f"🔍 {task.file_name}: Verifying...")
            else:
                status_lines.append(f"⏳ {task.file_name}: {task.status.value}")

        return "\n".join(status_lines) if status_lines else "No downloads"

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

            return f"Sync complete!\n✓ {found} matched on CivitAI\n○ {not_found} not found\n✗ {errors} errors"
        except Exception as e:
            return f"Sync error: {e}"

    def test_api_key(self) -> str:
        """Test if the API key is valid."""
        api_key = get_api_key()
        if not api_key:
            return "⚠ No API key configured.\nSet it in Settings > CivitAI"

        try:
            client = CivitAIClient(api_key=api_key)
            result = client.test_api_key()
            if result["valid"]:
                return "✓ API key is valid!"
            else:
                return f"✗ {result['message']}"
        except Exception as e:
            return f"✗ Error: {e}"


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
        gr.Markdown("# CivitAI Model Browser")
        gr.Markdown("Search and download models directly from CivitAI")

        with gr.Row():
            # Left column - Search
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### Search")
                    with FormRow():
                        search_query = gr.Textbox(
                            label="Search",
                            placeholder="Enter model name or keywords...",
                            scale=3,
                        )
                        search_btn = gr.Button("🔍 Search", scale=1, variant="primary")

                    with gr.Row():
                        model_type = gr.Dropdown(
                            label="Type",
                            choices=["All", "Checkpoint", "LORA", "LoCon", "VAE",
                                    "Controlnet", "Upscaler", "TextualInversion"],
                            value="All",
                        )
                        sort_order = gr.Dropdown(
                            label="Sort",
                            choices=["Most Downloaded", "Highest Rated", "Newest"],
                            value="Most Downloaded",
                        )
                        time_period = gr.Dropdown(
                            label="Period",
                            choices=["AllTime", "Year", "Month", "Week", "Day"],
                            value="AllTime",
                        )
                        base_model_filter = gr.Dropdown(
                            label="Base Model",
                            choices=["All", "SDXL 1.0", "SD 1.5", "Flux.1 D", "Pony", "Illustrious"],
                            value="All",
                        )

                with gr.Group():
                    gr.Markdown("### Results")
                    with gr.Row():
                        status_text = gr.Textbox(label="Status", value="Ready", interactive=False, scale=2)
                        page_info = gr.Textbox(label="Page", value="Page 0 of 0", interactive=False, scale=1)

                    results_gallery = gr.Gallery(
                        label="Models",
                        columns=4,
                        height=400,
                        object_fit="cover",
                        show_label=False,
                    )

                    model_selector = gr.Dropdown(
                        label="Select a model to view details",
                        choices=[],
                        interactive=True,
                    )

                    with gr.Row():
                        prev_btn = gr.Button("← Previous")
                        current_page = gr.Number(value=1, label="Page", minimum=1, precision=0)
                        next_btn = gr.Button("Next →")

            # Right column - Details & Download
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Model Details")
                    model_details = gr.Markdown(value="*Select a model to view details*")

                    version_selector = gr.Dropdown(
                        label="Select Version",
                        choices=[],
                        interactive=True,
                    )

                with gr.Group():
                    gr.Markdown("### Download")
                    with gr.Row():
                        version_id_input = gr.Textbox(label="Version ID", interactive=True)
                        download_type = gr.Dropdown(
                            label="Save as Type",
                            choices=["Checkpoint", "LORA", "VAE", "Controlnet", "Upscaler", "TextualInversion"],
                            value="Checkpoint",
                        )

                    download_btn = gr.Button("⬇ Download Model", variant="primary")
                    download_status = gr.Textbox(label="Download Status", value="No downloads", lines=3, interactive=False)
                    refresh_status_btn = gr.Button("🔄 Refresh Status")

                with gr.Group():
                    gr.Markdown("### Tools")
                    test_api_btn = gr.Button("Test API Key")
                    api_status = gr.Textbox(label="API Status", interactive=False)

                    sync_btn = gr.Button("🔄 Sync Local Models")
                    sync_status = gr.Textbox(label="Sync Status", interactive=False, lines=3)

        # Event handlers
        def do_search(query, mtype, sort, period, base, page):
            return browser.search(query, mtype, sort, period, base, page)

        search_btn.click(
            fn=do_search,
            inputs=[search_query, model_type, sort_order, time_period, base_model_filter, current_page],
            outputs=[results_gallery, status_text, page_info, model_selector],
        )

        search_query.submit(
            fn=do_search,
            inputs=[search_query, model_type, sort_order, time_period, base_model_filter, current_page],
            outputs=[results_gallery, status_text, page_info, model_selector],
        )

        def go_prev(query, mtype, sort, period, base, page):
            new_page = max(1, int(page) - 1)
            result = browser.search(query, mtype, sort, period, base, new_page)
            return (new_page,) + result

        def go_next(query, mtype, sort, period, base, page):
            new_page = int(page) + 1
            result = browser.search(query, mtype, sort, period, base, new_page)
            return (new_page,) + result

        prev_btn.click(
            fn=go_prev,
            inputs=[search_query, model_type, sort_order, time_period, base_model_filter, current_page],
            outputs=[current_page, results_gallery, status_text, page_info, model_selector],
        )

        next_btn.click(
            fn=go_next,
            inputs=[search_query, model_type, sort_order, time_period, base_model_filter, current_page],
            outputs=[current_page, results_gallery, status_text, page_info, model_selector],
        )

        # Model selection
        model_selector.change(
            fn=browser.load_model_details,
            inputs=[model_selector],
            outputs=[model_details, version_selector, version_id_input, download_type],
        )

        # Version selection updates version ID
        version_selector.change(
            fn=browser.get_version_id,
            inputs=[version_selector],
            outputs=[version_id_input],
        )

        # Download
        download_btn.click(
            fn=browser.download_model,
            inputs=[version_id_input, download_type],
            outputs=[download_status],
        )

        refresh_status_btn.click(
            fn=browser.get_download_status,
            outputs=[download_status],
        )

        # Tools
        test_api_btn.click(
            fn=browser.test_api_key,
            outputs=[api_status],
        )

        sync_btn.click(
            fn=browser.sync_local_models,
            outputs=[sync_status],
        )

    return civitai_interface


def on_ui_tabs():
    """Register the CivitAI tab with the WebUI."""
    try:
        print("CivitAI: Creating UI tab...")
        ui = create_ui()
        print("CivitAI: UI tab created successfully")
        return [(ui, "CivitAI", "civitai")]
    except Exception as e:
        print(f"CivitAI: ERROR creating UI tab: {e}")
        import traceback
        traceback.print_exc()
        return []
