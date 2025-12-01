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
            ("civitai-general", "CivitAI", "civitai"),
            {
                "civitai_explanation": OptionHTML(
                    """
<b>CivitAI Integration</b> allows you to browse, download, and sync models directly from
<a href="https://civitai.com" target="_blank">CivitAI</a>.<br><br>
An API key is recommended for full functionality including downloading models, accessing NSFW content (if enabled), and higher rate limits.<br>
Get your API key from <a href="https://civitai.com/user/account" target="_blank">CivitAI Account Settings</a>.
                    """
                ),
                "civitai_api_key": OptionInfo(
                    "",
                    "CivitAI API Key",
                    gr.Textbox,
                ).info("Required for downloads and some features"),
                "civitai_enabled": OptionInfo(
                    True,
                    "Enable CivitAI Integration",
                ).info("Disable to hide CivitAI features from the UI"),
                "civitai_div1": OptionDiv(),
                "civitai_auto_sync": OptionInfo(
                    False,
                    "Auto-sync model metadata on startup",
                ).info("Automatically fetch CivitAI info for local models when WebUI starts"),
                "civitai_download_preview": OptionInfo(
                    True,
                    "Download preview images for synced models",
                ).info("Save preview images alongside model files"),
                "civitai_sync_trained_words": OptionInfo(
                    True,
                    "Sync trained words/trigger words",
                ).info("Store trigger words for LoRAs and other trained models"),
                "civitai_div2": OptionDiv(),
                "civitai_nsfw_level": OptionInfo(
                    "None",
                    "NSFW Content Level",
                    gr.Radio,
                    {"choices": ["None", "Soft", "Mature", "X"]},
                ).info("Maximum NSFW level to show in browser and downloads"),
                "civitai_div3": OptionDiv(),
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
                "civitai_cache_hours": OptionInfo(
                    24,
                    "API Response Cache Duration (hours)",
                    gr.Slider,
                    {"minimum": 1, "maximum": 168, "step": 1},
                ).info("how long to cache search results and model info"),
            },
        )
    )
