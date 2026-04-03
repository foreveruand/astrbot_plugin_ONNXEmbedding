"""Plugin-level configuration store shared between submodules.

Avoids circular imports: rerank_provider.py / chat_provider.py import from
here instead of from main.py.
"""

_PLUGIN_CONFIG: dict = {}


def get_plugin_config() -> dict:
    """Return a snapshot of the current plugin-level configuration."""
    return dict(_PLUGIN_CONFIG)


def update_plugin_config(config: dict) -> None:
    """Replace the plugin-level configuration with a fresh copy of *config*."""
    global _PLUGIN_CONFIG
    _PLUGIN_CONFIG = dict(config)
