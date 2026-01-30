"""Training tools for RHOAI MCP.

This module provides a single register_tools function that delegates to all
training tool submodules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: RHOAIServer) -> None:
    """Register all training tools with the MCP server.

    This function delegates to the individual tool submodules to register
    their specific tools.

    Args:
        mcp: The FastMCP server instance.
        server: The RHOAIServer instance providing configuration and K8s access.
    """
    from rhoai_mcp.domains.training.tools import (
        discovery,
        lifecycle,
        monitoring,
        planning,
        runtimes,
        storage,
        training,
        unified,
    )

    # Register tools from each submodule
    discovery.register_tools(mcp, server)
    lifecycle.register_tools(mcp, server)
    monitoring.register_tools(mcp, server)
    planning.register_tools(mcp, server)
    runtimes.register_tools(mcp, server)
    storage.register_tools(mcp, server)
    training.register_tools(mcp, server)
    unified.register_tools(mcp, server)
