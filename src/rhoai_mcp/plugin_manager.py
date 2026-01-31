"""Plugin manager using pluggy for RHOAI MCP.

This module provides the PluginManager class that handles plugin
discovery, registration, and lifecycle management using pluggy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pluggy

from rhoai_mcp.hooks import PROJECT_NAME, RHOAIMCPHookSpec

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.plugin import PluginMetadata
    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)

# Entry point group name for external plugin discovery
PLUGIN_ENTRY_POINT_GROUP = "rhoai_mcp.plugins"


class PluginManager:
    """Manages plugin discovery, registration, and lifecycle.

    Uses pluggy for hook-based plugin architecture, providing a unified
    interface for both core domain plugins and external plugins.
    """

    def __init__(self) -> None:
        """Initialize the plugin manager with a pluggy PluginManager."""
        self._pm = pluggy.PluginManager(PROJECT_NAME)
        self._pm.add_hookspecs(RHOAIMCPHookSpec)
        self._registered_plugins: dict[str, Any] = {}
        self._healthy_plugins: dict[str, Any] = {}

    @property
    def hook(self) -> Any:
        """Get the pluggy hook caller for invoking hooks."""
        return self._pm.hook

    @property
    def registered_plugins(self) -> dict[str, Any]:
        """Get all registered plugins by name."""
        return self._registered_plugins

    @property
    def healthy_plugins(self) -> dict[str, Any]:
        """Get plugins that passed health checks."""
        return self._healthy_plugins

    def register_plugin(self, plugin: Any, name: str | None = None) -> str:
        """Register a plugin instance.

        Args:
            plugin: Plugin instance implementing hook methods.
            name: Optional name for the plugin. If not provided,
                  will try to get from plugin metadata.

        Returns:
            The name used to register the plugin.
        """
        if name is None:
            # Try to get name from plugin metadata
            if hasattr(plugin, "rhoai_get_plugin_metadata"):
                meta = plugin.rhoai_get_plugin_metadata()
                name = meta.name
            else:
                name = type(plugin).__name__

        self._pm.register(plugin, name=name)
        self._registered_plugins[name] = plugin
        logger.debug(f"Registered plugin: {name}")
        return name

    def unregister_plugin(self, name: str) -> None:
        """Unregister a plugin by name.

        Args:
            name: Name of the plugin to unregister.
        """
        if name in self._registered_plugins:
            plugin = self._registered_plugins.pop(name)
            self._pm.unregister(plugin)
            self._healthy_plugins.pop(name, None)
            logger.debug(f"Unregistered plugin: {name}")

    def load_entrypoint_plugins(self) -> int:
        """Discover and load external plugins from entry points.

        Uses pluggy's load_setuptools_entrypoints to discover plugins
        registered via the rhoai_mcp.plugins entry point group.

        Returns:
            Number of plugins loaded.
        """
        count = self._pm.load_setuptools_entrypoints(PLUGIN_ENTRY_POINT_GROUP)

        # Track loaded plugins
        for plugin in self._pm.get_plugins():
            name = self._pm.get_name(plugin)
            if name and name not in self._registered_plugins:
                self._registered_plugins[name] = plugin
                logger.info(f"Loaded external plugin from entry point: {name}")

        logger.info(f"Loaded {count} external plugins from entry points")
        return count

    def load_core_plugins(self) -> int:
        """Load core domain plugins and composite plugins.

        Imports and registers all core domain plugins from the domain registry
        and all composite plugins from the composites registry.

        Returns:
            Total number of plugins loaded.
        """
        from rhoai_mcp.composites.registry import get_composite_plugins
        from rhoai_mcp.domains.registry import get_core_plugins

        # Load core domain plugins
        domain_plugins = get_core_plugins()
        for plugin in domain_plugins:
            self.register_plugin(plugin)

        logger.info(f"Loaded {len(domain_plugins)} core domain plugins")

        # Load composite plugins
        composite_plugins = get_composite_plugins()
        for plugin in composite_plugins:
            self.register_plugin(plugin)

        logger.info(f"Loaded {len(composite_plugins)} composite plugins")

        total = len(domain_plugins) + len(composite_plugins)
        return total

    def get_all_metadata(self) -> list[PluginMetadata]:
        """Collect metadata from all registered plugins.

        Returns:
            List of PluginMetadata from all plugins.
        """
        results = self.hook.rhoai_get_plugin_metadata()
        return [meta for meta in results if meta is not None]

    def register_all_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        """Call tool registration hooks on all plugins.

        Args:
            mcp: The FastMCP server instance to register tools with.
            server: The RHOAI server instance.
        """
        self.hook.rhoai_register_tools(mcp=mcp, server=server)
        logger.info(f"Registered tools from {len(self._registered_plugins)} plugins")

    def register_all_resources(self, mcp: FastMCP, server: RHOAIServer) -> None:
        """Call resource registration hooks on all plugins.

        Args:
            mcp: The FastMCP server instance to register resources with.
            server: The RHOAI server instance.
        """
        self.hook.rhoai_register_resources(mcp=mcp, server=server)
        logger.info(f"Registered resources from {len(self._registered_plugins)} plugins")

    def register_all_prompts(self, mcp: FastMCP, server: RHOAIServer) -> None:
        """Call prompt registration hooks on all plugins.

        Args:
            mcp: The FastMCP server instance to register prompts with.
            server: The RHOAI server instance.
        """
        self.hook.rhoai_register_prompts(mcp=mcp, server=server)
        logger.info(f"Registered prompts from {len(self._registered_plugins)} plugins")

    def run_health_checks(self, server: RHOAIServer) -> dict[str, tuple[bool, str]]:
        """Run health checks on all registered plugins.

        Updates the healthy_plugins dict with plugins that pass.

        Args:
            server: The RHOAI server instance for health checks.

        Returns:
            Dictionary mapping plugin names to (healthy, message) tuples.
        """
        results: dict[str, tuple[bool, str]] = {}
        self._healthy_plugins.clear()

        for name, plugin in self._registered_plugins.items():
            try:
                if hasattr(plugin, "rhoai_health_check"):
                    is_healthy, message = plugin.rhoai_health_check(server=server)
                else:
                    # Plugins without health check are assumed healthy
                    is_healthy, message = True, "No health check defined"

                results[name] = (is_healthy, message)

                if is_healthy:
                    self._healthy_plugins[name] = plugin
                    logger.info(f"Plugin {name} health check passed: {message}")
                else:
                    logger.warning(f"Plugin {name} unavailable: {message}")
            except Exception as e:
                results[name] = (False, f"Health check error: {e}")
                logger.warning(f"Plugin {name} health check failed with error: {e}")

        return results

    def get_all_crd_definitions(self) -> list[Any]:
        """Collect CRD definitions from all plugins.

        Returns:
            Flattened list of all CRD definitions.
        """
        all_crds = []
        results = self.hook.rhoai_get_crd_definitions()
        for crd_list in results:
            if crd_list:
                all_crds.extend(crd_list)
        return all_crds
