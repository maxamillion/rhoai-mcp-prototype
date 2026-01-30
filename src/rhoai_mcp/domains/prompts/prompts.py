"""Main prompt registration for RHOAI MCP.

This module coordinates the registration of all MCP prompts
from the various prompt submodules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer


def register_prompts(mcp: FastMCP, server: RHOAIServer) -> None:
    """Register all prompts with the MCP server.

    Args:
        mcp: The FastMCP server instance to register prompts with.
        server: The RHOAI server instance for accessing configuration.
    """
    from rhoai_mcp.domains.prompts import (
        deployment_prompts,
        exploration_prompts,
        project_prompts,
        training_prompts,
        troubleshooting_prompts,
    )

    training_prompts.register_prompts(mcp, server)
    exploration_prompts.register_prompts(mcp, server)
    troubleshooting_prompts.register_prompts(mcp, server)
    project_prompts.register_prompts(mcp, server)
    deployment_prompts.register_prompts(mcp, server)
