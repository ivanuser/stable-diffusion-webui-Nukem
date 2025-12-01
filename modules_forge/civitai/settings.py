"""
CivitAI Settings Integration

Adds CivitAI settings to the WebUI settings tab.
"""

import gradio as gr

from modules.options import OptionDiv, OptionHTML, OptionInfo, categories, options_section


def register_settings(options_templates: dict):
    """
    Register CivitAI settings with the WebUI settings system.

    Args:
        options_templates: The global options_templates dict to update
    """
    # Register the CivitAI category
    categories.register_category("civitai", "CivitAI")

    options_templates.update(
        options_section(
            ("civitai-general", "CivitAI Integration", "civitai"),
            {
                "civitai_explanation": OptionHTML(
                    """
<b>CivitAI Integration</b> allows you to browse, download, and sync models directly from
<a href="https://civitai.com" target="_blank">CivitAI</a>.<br><br>
An API key is recommended for full functionality including:
<ul>
<li>Downloading models</li>
<li>Accessing NSFW content (if enabled)</li>
<li>Higher rate limits</li>
</ul>
Get your API key from <a href="https://civitai.com/user/account" target="_blank">CivitAI Account Settings</a>.
                    """
                ),
                "civitai_api_key": OptionInfo(
                    "",
                    "CivitAI API Key",
                    gr.Textbox,
                    {"type": "password", "placeholder": "Enter your CivitAI API key"},
                ).info("Required for downloads and some features"),
                "civitai_enabled": OptionInfo(
                    True,
                    "Enable CivitAI Integration",
                ).info("Disable to hide CivitAI features from the UI"),
            },
        )
    )

    options_templates.update(
        options_section(
            ("civitai-sync", "Model Sync Settings", "civitai"),
            {
                "civitai_auto_sync": OptionInfo(
                    False,
                    "Auto-sync model metadata on startup",
                ).info("Automatically fetch CivitAI info for local models when WebUI starts"),
                "civitai_download_preview": OptionInfo(
                    True,
                    "Download preview images for synced models",
                ).info("Save preview images alongside model files"),
                "civitai_sync_description": OptionInfo(
                    True,
                    "Sync model descriptions",
                ).info("Store model descriptions in metadata sidecar files"),
                "civitai_sync_trained_words": OptionInfo(
                    True,
                    "Sync trained words/trigger words",
                ).info("Store trigger words for LoRAs and other trained models"),
            },
        )
    )

    options_templates.update(
        options_section(
            ("civitai-content", "Content Filtering", "civitai"),
            {
                "civitai_nsfw_level": OptionInfo(
                    "None",
                    "NSFW Content Level",
                    gr.Radio,
                    {"choices": ("None", "Soft", "Mature", "X")},
                ).info("Maximum NSFW level to show in browser and downloads"),
                "civitai_filter_poi": OptionInfo(
                    True,
                    "Filter Person of Interest (POI) content",
                ).info("Hide models depicting real people"),
            },
        )
    )

    options_templates.update(
        options_section(
            ("civitai-browser", "Browser Settings", "civitai"),
            {
                "civitai_default_sort": OptionInfo(
                    "Most Downloaded",
                    "Default Sort Order",
                    gr.Dropdown,
                    {
                        "choices": [
                            "Most Downloaded",
                            "Highest Rated",
                            "Most Liked",
                            "Most Discussed",
                            "Most Collected",
                            "Newest",
                        ]
                    },
                ),
                "civitai_default_period": OptionInfo(
                    "AllTime",
                    "Default Time Period",
                    gr.Dropdown,
                    {"choices": ["AllTime", "Year", "Month", "Week", "Day"]},
                ),
                "civitai_results_per_page": OptionInfo(
                    20,
                    "Results Per Page",
                    gr.Slider,
                    {"minimum": 10, "maximum": 100, "step": 10},
                ),
                "civitai_div_cards": OptionDiv(),
                "civitai_card_width": OptionInfo(
                    200,
                    "Card Width",
                    gr.Slider,
                    {"minimum": 100, "maximum": 400, "step": 20},
                ).info("in pixels"),
                "civitai_card_height": OptionInfo(
                    250,
                    "Card Height",
                    gr.Slider,
                    {"minimum": 150, "maximum": 500, "step": 20},
                ).info("in pixels"),
            },
        )
    )

    options_templates.update(
        options_section(
            ("civitai-cache", "Cache Settings", "civitai"),
            {
                "civitai_cache_hours": OptionInfo(
                    24,
                    "API Response Cache Duration",
                    gr.Slider,
                    {"minimum": 1, "maximum": 168, "step": 1},
                ).info("in hours; how long to cache search results and model info"),
                "civitai_cache_preview_images": OptionInfo(
                    True,
                    "Cache preview images locally",
                ).info("Save downloaded preview images to speed up browsing"),
            },
        )
    )
