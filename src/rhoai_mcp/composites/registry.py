"""Plugin registry for composite tools.

This module provides plugin classes for composite tools that combine
multiple domain operations. All plugins use pluggy hooks for integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rhoai_mcp.hooks import hookimpl
from rhoai_mcp.plugin import BasePlugin, PluginMetadata

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer


class ClusterCompositesPlugin(BasePlugin):
    """Plugin for cluster-wide composite tools.

    Provides context-efficient cluster and project summaries optimized
    for AI agent context windows, offering compact overviews that reduce
    token usage by 70-90% compared to listing individual resources.
    """

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="cluster-composites",
                version="1.0.0",
                description="Context-efficient summary tools for AI agents",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=[],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.composites.cluster.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:  # noqa: ARG002
        return True, "Cluster composites use core domain clients"


class TrainingCompositesPlugin(BasePlugin):
    """Plugin for training composite tools.

    Provides training workflow orchestration including pre-flight
    preparation, unified training operations, and storage management.
    """

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="training-composites",
                version="1.0.0",
                description="Training workflow orchestration tools",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=[],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.composites.training.planning import register_tools as reg_planning
        from rhoai_mcp.composites.training.storage import register_tools as reg_storage
        from rhoai_mcp.composites.training.unified import register_tools as reg_unified

        reg_planning(mcp, server)
        reg_storage(mcp, server)
        reg_unified(mcp, server)

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:  # noqa: ARG002
        return True, "Training composites use training domain client"


class MetaCompositesPlugin(BasePlugin):
    """Plugin for tool discovery and workflow guidance.

    Provides meta-tools that help AI agents discover the right tools
    for their tasks and understand typical workflows.
    """

    def __init__(self) -> None:
        super().__init__(
            PluginMetadata(
                name="meta-composites",
                version="1.0.0",
                description="Tool discovery and workflow guidance",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=[],
            )
        )

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.composites.meta.tools import register_tools

        register_tools(mcp, server)

    @hookimpl
    def rhoai_register_resources(self, mcp: FastMCP, server: RHOAIServer) -> None:
        from rhoai_mcp.composites.meta.resources import register_resources

        register_resources(mcp, server)

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:  # noqa: ARG002
        return True, "Meta composites require no external dependencies"


def get_composite_plugins() -> list[BasePlugin]:
    """Return all composite plugin instances.

    Returns:
        List of plugin instances for all composite tools.
    """
    return [
        ClusterCompositesPlugin(),
        TrainingCompositesPlugin(),
        MetaCompositesPlugin(),
    ]
