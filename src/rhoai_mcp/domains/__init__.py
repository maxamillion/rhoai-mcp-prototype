"""Core domain modules for RHOAI MCP.

This package contains the core domain plugins that are registered
directly with the server using pluggy hooks.

External plugins can be discovered via entry points.
"""

from rhoai_mcp.domains.registry import (
    ConnectionsPlugin,
    InferencePlugin,
    NotebooksPlugin,
    PipelinesPlugin,
    ProjectsPlugin,
    StoragePlugin,
    TrainingPlugin,
    get_core_plugins,
)

__all__ = [
    "ConnectionsPlugin",
    "InferencePlugin",
    "NotebooksPlugin",
    "PipelinesPlugin",
    "ProjectsPlugin",
    "StoragePlugin",
    "TrainingPlugin",
    "get_core_plugins",
]
