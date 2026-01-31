"""Pydantic models for Data Science Projects (namespaces)."""

from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.models.common import ResourceMetadata, ResourceStatus, ResourceSummary
from rhoai_mcp.utils.labels import RHOAILabels


class DataScienceProject(BaseModel):
    """Data Science Project representation."""

    metadata: ResourceMetadata
    display_name: str | None = Field(None, description="Display name from annotations")
    description: str | None = Field(None, description="Description from annotations")
    requester: str | None = Field(None, description="User who created the project")
    is_modelmesh_enabled: bool = Field(
        False, description="Whether ModelMesh (multi-model) serving is enabled"
    )
    status: ResourceStatus = Field(ResourceStatus.READY, description="Project status")
    resource_summary: ResourceSummary | None = Field(
        None, description="Summary of resources in the project"
    )

    @classmethod
    def from_namespace(
        cls,
        namespace: Any,
        resource_summary: ResourceSummary | None = None,
    ) -> "DataScienceProject":
        """Create from Kubernetes namespace or OpenShift Project object."""
        metadata = namespace.metadata
        # Convert to plain dicts if they're ResourceField objects (from dynamic client)
        labels = metadata.labels
        if labels is not None and not isinstance(labels, dict):
            labels = dict(labels)
        labels = labels or {}
        annotations = metadata.annotations
        if annotations is not None and not isinstance(annotations, dict):
            annotations = dict(annotations)
        annotations = annotations or {}

        # Determine status based on namespace phase
        status = ResourceStatus.READY
        if hasattr(namespace, "status") and namespace.status:
            phase = getattr(namespace.status, "phase", "Active")
            if phase == "Terminating":
                status = ResourceStatus.DELETING
            elif phase != "Active":
                status = ResourceStatus.UNKNOWN

        return cls(
            metadata=ResourceMetadata.from_k8s_metadata(
                metadata,
                kind="Project",
                api_version="project.openshift.io/v1",
            ),
            display_name=annotations.get("openshift.io/display-name"),
            description=annotations.get("openshift.io/description"),
            requester=annotations.get("openshift.io/requester"),
            is_modelmesh_enabled=RHOAILabels.is_modelmesh_enabled(labels),
            status=status,
            resource_summary=resource_summary,
        )

    @classmethod
    def from_project(
        cls,
        project: Any,
        resource_summary: ResourceSummary | None = None,
    ) -> "DataScienceProject":
        """Create from OpenShift Project object.

        OpenShift Projects have the same structure as Kubernetes namespaces,
        so this delegates to from_namespace.
        """
        return cls.from_namespace(project, resource_summary)


class ProjectCreate(BaseModel):
    """Request model for creating a Data Science Project."""

    name: str = Field(..., description="Project name (will be namespace name)")
    display_name: str | None = Field(None, description="Human-readable display name")
    description: str | None = Field(None, description="Project description")
    enable_modelmesh: bool = Field(False, description="Enable ModelMesh (multi-model) serving")


class ProjectUpdate(BaseModel):
    """Request model for updating a Data Science Project."""

    display_name: str | None = Field(None, description="New display name")
    description: str | None = Field(None, description="New description")
    enable_modelmesh: bool | None = Field(None, description="Enable/disable ModelMesh serving")


class ProjectStatus(BaseModel):
    """Comprehensive status of a Data Science Project."""

    project: DataScienceProject
    workbenches: list[dict[str, Any]] = Field(
        default_factory=list, description="Workbench statuses"
    )
    models: list[dict[str, Any]] = Field(
        default_factory=list, description="Deployed model statuses"
    )
    pipeline_server: dict[str, Any] | None = Field(None, description="Pipeline server status")
    data_connections: list[dict[str, Any]] = Field(
        default_factory=list, description="Data connection info"
    )
    storage: list[dict[str, Any]] = Field(default_factory=list, description="Storage (PVC) info")
