"""Configuration management for RHOAI MCP server."""

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthMode(str, Enum):
    """Authentication mode for Kubernetes API."""

    AUTO = "auto"
    KUBECONFIG = "kubeconfig"
    TOKEN = "token"


class TransportMode(str, Enum):
    """MCP transport mode."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"


class LogLevel(str, Enum):
    """Logging level."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class RHOAIConfig(BaseSettings):
    """Configuration for RHOAI MCP server.

    Configuration is loaded from environment variables with RHOAI_MCP_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="RHOAI_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Authentication settings
    auth_mode: AuthMode = Field(
        default=AuthMode.AUTO,
        description="Authentication mode: auto, kubeconfig, or token",
    )
    kubeconfig_path: Path | None = Field(
        default=None,
        description="Path to kubeconfig file (defaults to ~/.kube/config)",
    )
    kubeconfig_context: str | None = Field(
        default=None,
        description="Kubeconfig context to use",
    )
    api_server: str | None = Field(
        default=None,
        description="Kubernetes API server URL (for token auth)",
    )
    api_token: str | None = Field(
        default=None,
        description="Kubernetes API token (for token auth)",
    )

    # Namespace settings
    default_namespace: str | None = Field(
        default=None,
        description="Default namespace for operations",
    )

    # Transport settings
    transport: TransportMode = Field(
        default=TransportMode.STDIO,
        description="MCP transport mode: stdio, sse, or streamable-http",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Host to bind HTTP server to",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port to bind HTTP server to",
    )

    # Safety settings
    enable_dangerous_operations: bool = Field(
        default=False,
        description="Enable dangerous operations like delete",
    )
    read_only_mode: bool = Field(
        default=False,
        description="Disable all write operations",
    )

    # Logging
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level",
    )

    # Context window optimization settings
    default_verbosity: str = Field(
        default="standard",
        description="Default verbosity level for responses: minimal, standard, or full",
    )
    default_list_limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Default limit for list operations (None for all items)",
    )
    max_list_limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum allowed limit for list operations",
    )
    enable_response_caching: bool = Field(
        default=False,
        description="Enable caching of list responses to reduce API calls",
    )
    cache_ttl_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Cache TTL in seconds when caching is enabled",
    )

    @field_validator("kubeconfig_path", mode="before")
    @classmethod
    def resolve_kubeconfig_path(cls, v: str | Path | None) -> Path | None:
        """Resolve kubeconfig path, defaulting to standard location."""
        if v is None:
            return None
        path = Path(v).expanduser().resolve()
        return path

    @property
    def effective_kubeconfig_path(self) -> Path:
        """Get the effective kubeconfig path, with default."""
        if self.kubeconfig_path:
            return self.kubeconfig_path
        # Check KUBECONFIG env var
        env_path = os.environ.get("KUBECONFIG")
        if env_path:
            return Path(env_path).expanduser().resolve()
        # Default to ~/.kube/config
        return Path.home() / ".kube" / "config"

    def validate_auth_config(self) -> list[str]:
        """Validate authentication configuration and return any warnings."""
        warnings = []

        if self.auth_mode == AuthMode.TOKEN:
            if not self.api_server:
                raise ValueError("api_server is required when auth_mode is 'token'")
            if not self.api_token:
                raise ValueError("api_token is required when auth_mode is 'token'")

        if self.auth_mode == AuthMode.KUBECONFIG and not self.effective_kubeconfig_path.exists():
            raise ValueError(f"Kubeconfig file not found: {self.effective_kubeconfig_path}")

        if self.auth_mode == AuthMode.AUTO:
            # Check if running in-cluster
            if Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists():
                warnings.append("Running in-cluster, will use service account")
            elif not self.effective_kubeconfig_path.exists():
                warnings.append(
                    f"No kubeconfig found at {self.effective_kubeconfig_path}, "
                    "will attempt in-cluster auth"
                )

        return warnings

    def is_operation_allowed(self, operation: str) -> tuple[bool, str | None]:
        """Check if an operation is allowed based on safety settings.

        Returns:
            Tuple of (allowed, reason_if_not_allowed)
        """
        if self.read_only_mode and operation in ("create", "update", "delete", "patch"):
            return False, "Read-only mode is enabled"

        if not self.enable_dangerous_operations and operation == "delete":
            return False, "Dangerous operations are disabled"

        return True, None


# Global configuration instance
_config: RHOAIConfig | None = None


def get_config() -> RHOAIConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = RHOAIConfig()
    return _config


def configure(**kwargs: Any) -> RHOAIConfig:
    """Configure the global settings.

    This should be called before get_config() if you want to override defaults.
    """
    global _config
    _config = RHOAIConfig(**kwargs)
    return _config
