"""
CivitAI Integration Startup

Registers the CivitAI UI tab and initializes the integration.
This module should be imported during WebUI startup.
"""

import logging

logger = logging.getLogger(__name__)

_ui_registered = False


def register_civitai_ui():
    """Register the CivitAI browser UI tab."""
    global _ui_registered
    if _ui_registered:
        return

    try:
        from modules import script_callbacks
        from modules_forge.civitai.ui import on_ui_tabs

        script_callbacks.on_ui_tabs(on_ui_tabs, name="CivitAI")
        _ui_registered = True
        print("CivitAI: UI tab registered")
    except Exception as e:
        print(f"CivitAI: Failed to register UI tab: {e}")


def initialize_civitai():
    """Initialize CivitAI integration on startup."""
    # Register UI tab here - this is called after script_callbacks is ready
    register_civitai_ui()

    try:
        from modules import shared
        from modules_forge.civitai.api_client import init_client

        # Initialize client with API key from settings
        api_key = getattr(shared.opts, "civitai_api_key", "") or ""
        if api_key:
            init_client(api_key)
            print("CivitAI: Client initialized with API key")
        else:
            init_client()
            print("CivitAI: Client initialized (no API key)")

    except Exception as e:
        print(f"CivitAI: Failed to initialize: {e}")
