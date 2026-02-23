"""Plugin discovery and management."""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path

from app.transcription.base import TranscriberPlugin
from app.config import PLUGINS_DIR

logger = logging.getLogger(__name__)


class PluginManager:
    """Discovers, loads, and manages transcription plugins."""

    def __init__(self):
        self._plugins: dict[str, TranscriberPlugin] = {}

    def discover(self):
        """Discover plugins from built-in and external plugin directories."""
        self._plugins.clear()

        # Load built-in plugins
        self._load_builtin_plugins()

        # Load external plugins
        self._load_external_plugins()

        logger.info("Discovered %d plugin(s): %s",
                     len(self._plugins), list(self._plugins.keys()))

    def _load_builtin_plugins(self):
        """Load plugins from app.transcription.plugins package."""
        import app.transcription.plugins as plugins_pkg
        for importer, modname, ispkg in pkgutil.iter_modules(plugins_pkg.__path__):
            try:
                module = importlib.import_module(f"app.transcription.plugins.{modname}")
                self._register_from_module(module)
            except Exception:
                logger.exception("Failed to load built-in plugin: %s", modname)

    def _load_external_plugins(self):
        """Load plugins from the external plugins directory."""
        if not PLUGINS_DIR.exists():
            return
        for path in PLUGINS_DIR.glob("*.py"):
            if path.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(path.stem, str(path))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._register_from_module(module)
            except Exception:
                logger.exception("Failed to load external plugin: %s", path)

    def _register_from_module(self, module):
        """Find and register TranscriberPlugin subclasses in a module."""
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, TranscriberPlugin)
                    and obj is not TranscriberPlugin
                    and not inspect.isabstract(obj)):
                try:
                    instance = obj()
                    plugin_name = instance.name()
                    self._plugins[plugin_name] = instance
                    logger.info("Registered plugin: %s", plugin_name)
                except Exception:
                    logger.exception("Failed to instantiate plugin: %s", name)

    def get_plugin(self, name: str) -> TranscriberPlugin | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[dict]:
        return [p.info() for p in self._plugins.values()]

    @property
    def plugins(self) -> dict[str, TranscriberPlugin]:
        return self._plugins

    @property
    def default_plugin(self) -> TranscriberPlugin | None:
        if self._plugins:
            return next(iter(self._plugins.values()))
        return None
