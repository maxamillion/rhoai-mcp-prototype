"""Plugin interface for RHOAI MCP components.

This module defines the plugin base class and metadata that all RHOAI MCP
plugins use to integrate with the server via pluggy hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rhoai_mcp.hooks import hookimpl

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.clients.base import CRDDefinition
    from rhoai_mcp.server import RHOAIServer


@dataclass
class PluginMetadata:
    """Metadata describing an RHOAI MCP plugin.

    Each plugin must provide this metadata to identify itself and
    declare its requirements.
    """

    name: str
    """Unique plugin name, e.g., 'notebooks', 'inference'."""

    version: str
    """Plugin version following semver, e.g., '1.0.0'."""

    description: str
    """Human-readable description of what this plugin provides."""

    maintainer: str
    """Maintainer email or team, e.g., 'kubeflow-team@redhat.com'."""

    requires_crds: list[str] = field(default_factory=list)
    """List of CRD kinds this plugin requires to function.

    If any of these CRDs are not available in the cluster,
    the plugin will be marked as unavailable but the server
    will continue to run with other plugins.
    """


class BasePlugin:
    """Base implementation of an RHOAI MCP plugin with common functionality.

    Component plugins can extend this class to get default implementations
    of hook methods. All hook methods are decorated with @hookimpl to
    register them with pluggy.

    Example entry point in pyproject.toml for external plugins:
        [project.entry-points."rhoai_mcp.plugins"]
        my_plugin = "my_package.plugin:MyPlugin"
    """

    def __init__(self, metadata: PluginMetadata) -> None:
        """Initialize the plugin with metadata.

        Args:
            metadata: Plugin metadata.
        """
        self._metadata = metadata

    @hookimpl
    def rhoai_get_plugin_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return self._metadata

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        """Register MCP tools. Override in subclass."""
        pass

    @hookimpl
    def rhoai_register_resources(self, mcp: FastMCP, server: RHOAIServer) -> None:
        """Register MCP resources. Override in subclass."""
        pass

    @hookimpl
    def rhoai_register_prompts(self, mcp: FastMCP, server: RHOAIServer) -> None:
        """Register MCP prompts. Override in subclass."""
        pass

    @hookimpl
    def rhoai_get_crd_definitions(self) -> list[CRDDefinition]:
        """Return CRD definitions. Override in subclass."""
        return []

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:
        """Check plugin health by verifying required CRDs are available.

        Default implementation checks that all CRDs listed in
        metadata.requires_crds are accessible in the cluster.
        """
        if not self._metadata.requires_crds:
            return True, "No CRD requirements"

        crd_defs = self.rhoai_get_crd_definitions()
        crd_map = {crd.kind: crd for crd in crd_defs}

        missing_crds = []
        for crd_kind in self._metadata.requires_crds:
            if crd_kind not in crd_map:
                missing_crds.append(crd_kind)
                continue

            crd = crd_map[crd_kind]
            try:
                # Try to get the resource to verify CRD exists
                server.k8s.get_resource(crd)
            except Exception:
                missing_crds.append(crd_kind)

        if missing_crds:
            return False, f"Missing CRDs: {', '.join(missing_crds)}"

        return True, "All required CRDs available"
