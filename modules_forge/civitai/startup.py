"""
CivitAI Integration Startup

Registers the CivitAI UI tab and initializes the integration.
This module should be imported during WebUI startup.
"""

import logging

logger = logging.getLogger(__name__)


def register_civitai_ui():
    """Register the CivitAI browser UI tab."""
    try:
        from modules import script_callbacks
        from modules_forge.civitai.ui import on_ui_tabs

        script_callbacks.on_ui_tabs(on_ui_tabs, name="CivitAI")
        logger.info("CivitAI UI tab registered")
    except Exception as e:
        logger.error(f"Failed to register CivitAI UI: {e}")


def initialize_civitai():
    """Initialize CivitAI integration on startup."""
    try:
        from modules import shared
        from modules_forge.civitai.api_client import init_client

        # Initialize client with API key from settings
        api_key = getattr(shared.opts, "civitai_api_key", "") or ""
        if api_key:
            init_client(api_key)
            logger.info("CivitAI client initialized with API key")
        else:
            init_client()
            logger.info("CivitAI client initialized (no API key)")

        # Auto-sync if enabled
        auto_sync = getattr(shared.opts, "civitai_auto_sync", False)
        if auto_sync:
            logger.info("CivitAI auto-sync enabled, will sync on first model load")

    except Exception as e:
        logger.error(f"Failed to initialize CivitAI: {e}")


# Auto-register when this module is imported
register_civitai_ui()
