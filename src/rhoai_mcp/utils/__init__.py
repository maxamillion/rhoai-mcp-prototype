"""Utility functions and helpers for RHOAI MCP server."""

from rhoai_mcp.utils.annotations import RHOAIAnnotations
from rhoai_mcp.utils.cache import (
    cache_stats,
    cached,
    clear_cache,
    clear_expired,
    invalidate,
)
from rhoai_mcp.utils.errors import (
    AuthenticationError,
    ConfigurationError,
    NotFoundError,
    OperationNotAllowedError,
    ResourceExistsError,
    RHOAIError,
    ValidationError,
)
from rhoai_mcp.utils.labels import RHOAILabels
from rhoai_mcp.utils.response import (
    PaginatedResponse,
    ResponseBuilder,
    Verbosity,
    paginate,
)

__all__ = [
    # Errors
    "RHOAIError",
    "NotFoundError",
    "AuthenticationError",
    "ConfigurationError",
    "ValidationError",
    "OperationNotAllowedError",
    "ResourceExistsError",
    # Labels and annotations
    "RHOAIAnnotations",
    "RHOAILabels",
    # Response formatting
    "Verbosity",
    "ResponseBuilder",
    "PaginatedResponse",
    "paginate",
    # Caching
    "cached",
    "clear_cache",
    "clear_expired",
    "cache_stats",
    "invalidate",
]
