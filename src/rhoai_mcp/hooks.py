"""Pluggy hook specifications for RHOAI MCP plugins.

This module defines the hook interface that all plugins must implement
to integrate with the RHOAI MCP server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pluggy

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.clients.base import CRDDefinition
    from rhoai_mcp.plugin import PluginMetadata
    from rhoai_mcp.server import RHOAIServer

# Project name used for pluggy hook registration
PROJECT_NAME = "rhoai_mcp"

# Create hook specification marker
hookspec = pluggy.HookspecMarker(PROJECT_NAME)

# Create hook implementation marker (exported for plugins to use)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class RHOAIMCPHookSpec:
    """Hook specifications for RHOAI MCP plugins.

    All plugins implement these hooks to integrate with the server.
    Hooks are called in plugin registration order.
    """

    @hookspec
    def rhoai_get_plugin_metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        This hook must return a PluginMetadata instance describing
        the plugin's name, version, description, and requirements.

        Returns:
            PluginMetadata instance for this plugin.
        """
        raise NotImplementedError

    @hookspec
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        """Register MCP tools provided by this plugin.

        This hook is called during server startup to register all
        tools (functions) that this plugin provides.

        Args:
            mcp: The FastMCP server instance to register tools with.
            server: The RHOAI server instance for accessing K8s client and config.
        """

    @hookspec
    def rhoai_register_resources(self, mcp: FastMCP, server: RHOAIServer) -> None:
        """Register MCP resources provided by this plugin.

        This hook is called during server startup to register all
        resources (data endpoints) that this plugin provides.

        Args:
            mcp: The FastMCP server instance to register resources with.
            server: The RHOAI server instance for accessing K8s client and config.
        """

    @hookspec
    def rhoai_get_crd_definitions(self) -> list[CRDDefinition]:
        """Return CRD definitions used by this plugin.

        This allows the core server to know about all CRDs without
        having to import component-specific code.

        Returns:
            List of CRDDefinition objects for CRDs this plugin uses.
        """
        raise NotImplementedError

    @hookspec
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:
        """Check if this plugin can operate correctly.

        This hook is called during startup to verify that all
        required CRDs are available and the plugin can function.
        Plugins that fail health checks are skipped, allowing the
        server to gracefully degrade when some components are unavailable.

        Args:
            server: The RHOAI server instance for accessing K8s client.

        Returns:
            Tuple of (healthy, message) where healthy is True if the
            plugin can operate, and message provides details.
        """
        raise NotImplementedError
