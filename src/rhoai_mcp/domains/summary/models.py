"""Pydantic models for the summary domain.

These models provide compact, token-efficient representations of cluster
and project status for AI agent context windows.
"""

from pydantic import BaseModel, Field


class ClusterSummary(BaseModel):
    """Compact cluster overview optimized for context windows."""

    projects: int = Field(..., description="Total number of projects")
    project_names: list[str] = Field(..., description="List of project names")
    workbenches: str = Field(..., description="Workbench status (e.g. '3/5 running')")
    models: str = Field(..., description="Model status (e.g. '2/3 ready')")
    pipelines: int = Field(..., description="Total number of pipelines")
    storage: int = Field(..., description="Total number of storage volumes")
    data_connections: int = Field(..., description="Total number of data connections")


class ProjectSummary(BaseModel):
    """Compact project status optimized for context windows."""

    name: str = Field(..., description="Project name")
    display_name: str | None = Field(None, description="Human-readable display name")
    status: str = Field(..., description="Project status")
    workbenches: str | None = Field(None, description="Workbench status (e.g. '2/3 running')")
    models: str | None = Field(None, description="Model status (e.g. '1/2 ready')")
    pipelines: int | None = Field(None, description="Number of pipelines")
    storage: int | None = Field(None, description="Number of storage volumes")
    data_connections: int | None = Field(None, description="Number of data connections")


class ResourceStatus(BaseModel):
    """Status of a single resource."""

    name: str = Field(..., description="Resource name")
    type: str = Field(..., description="Resource type")
    status: str = Field(default="Unknown", description="Current status")
    url: str | None = Field(None, description="Resource URL if applicable")
    bucket: str | None = Field(None, description="S3 bucket for connections")
    size: str | None = Field(None, description="Storage size")
    progress: str | None = Field(None, description="Progress percentage for training jobs")
    message: str | None = Field(None, description="Status message")
    error: str | None = Field(None, description="Error message if any")
    valid_types: list[str] | None = Field(
        None, description="List of valid resource types (on error)"
    )


class ResourceNameList(BaseModel):
    """List of resource names."""

    type: str = Field(..., description="Resource type")
    count: int = Field(..., description="Number of resources")
    names: list[str] = Field(..., description="List of resource names")
    namespace: str | None = Field(None, description="Namespace (if applicable)")
    error: str | None = Field(None, description="Error message if any")
    valid_types: list[str] | None = Field(
        None, description="List of valid resource types (on error)"
    )


class MultiResourceStatusResult(BaseModel):
    """Status for multiple resources."""

    namespace: str = Field(..., description="Namespace of the resources")
    results: list[ResourceStatus] = Field(..., description="Status for each resource")
    success_count: int = Field(..., description="Number of successful status checks")
    error_count: int = Field(..., description="Number of failed status checks")
