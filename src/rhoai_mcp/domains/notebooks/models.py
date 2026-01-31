"""Pydantic models for Notebooks (Workbenches)."""

import contextlib
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.models.common import (
    Condition,
    ContainerResources,
    ResourceMetadata,
)
from rhoai_mcp.utils.annotations import RHOAIAnnotations


class WorkbenchStatus(str, Enum):
    """Workbench-specific status values."""

    RUNNING = "Running"
    STOPPED = "Stopped"
    STARTING = "Starting"
    STOPPING = "Stopping"
    ERROR = "Error"
    UNKNOWN = "Unknown"


class NotebookImage(BaseModel):
    """Notebook image information."""

    name: str = Field(..., description="Image name/tag")
    display_name: str | None = Field(None, description="Human-readable name")
    description: str | None = Field(None, description="Image description")
    recommended: bool = Field(False, description="Whether this is a recommended image")
    order: int = Field(0, description="Display order")


class WorkbenchSize(BaseModel):
    """Workbench size configuration."""

    name: str = Field(..., description="Size name (e.g., 'Small', 'Medium')")
    resources: ContainerResources = Field(..., description="Resource configuration")


class Workbench(BaseModel):
    """Workbench (Notebook) representation."""

    metadata: ResourceMetadata
    display_name: str | None = Field(None, description="Display name")
    image: str = Field(..., description="Container image")
    image_display_name: str | None = Field(None, description="Image display name")
    status: WorkbenchStatus = Field(WorkbenchStatus.UNKNOWN, description="Workbench status")
    stopped_time: datetime | None = Field(None, description="When the workbench was stopped")
    size: str | None = Field(None, description="Size selection name")
    resources: ContainerResources | None = Field(None, description="Resource configuration")
    url: str | None = Field(None, description="Workbench URL")
    conditions: list[Condition] = Field(default_factory=list, description="Status conditions")
    volumes: list[str] = Field(default_factory=list, description="Mounted volume names")
    env_from: list[str] = Field(
        default_factory=list, description="Environment sources (secrets/configmaps)"
    )

    @classmethod
    def from_notebook_cr(cls, notebook: Any, url: str | None = None) -> "Workbench":
        """Create from Kubeflow Notebook CR."""
        metadata = notebook.metadata
        annotations = metadata.annotations or {}
        spec = notebook.spec
        status_obj = getattr(notebook, "status", None)

        # Determine status
        wb_status = cls._determine_status(annotations, status_obj)

        # Get stopped time if present
        stopped_time = None
        stopped_str = RHOAIAnnotations.get_notebook_stopped_time(annotations)
        if stopped_str:
            with contextlib.suppress(ValueError):
                stopped_time = datetime.fromisoformat(stopped_str.replace("Z", "+00:00"))

        # Extract container info
        containers = spec.get("template", {}).get("spec", {}).get("containers", [])
        main_container = containers[0] if containers else {}

        image = main_container.get("image", "unknown")
        resources = None
        if "resources" in main_container:
            res = main_container["resources"]
            resources = ContainerResources.from_k8s_resources(
                res.get("requests"), res.get("limits")
            )

        # Extract volumes
        volumes = []
        for vol in spec.get("template", {}).get("spec", {}).get("volumes", []):
            if "persistentVolumeClaim" in vol:
                volumes.append(vol["persistentVolumeClaim"]["claimName"])

        # Extract env sources
        env_from = []
        for ef in main_container.get("envFrom", []):
            if "secretRef" in ef:
                env_from.append(f"secret:{ef['secretRef']['name']}")
            elif "configMapRef" in ef:
                env_from.append(f"configmap:{ef['configMapRef']['name']}")

        # Parse conditions
        conditions = []
        if status_obj and hasattr(status_obj, "conditions"):
            for cond in status_obj.conditions or []:
                conditions.append(Condition.from_k8s_condition(cond))

        return cls(
            metadata=ResourceMetadata.from_k8s_metadata(
                metadata,
                kind="Notebook",
                api_version="kubeflow.org/v1",
            ),
            display_name=annotations.get("openshift.io/display-name"),
            image=image,
            image_display_name=annotations.get(RHOAIAnnotations.IMAGE_DISPLAY_NAME),
            status=wb_status,
            stopped_time=stopped_time,
            size=annotations.get(RHOAIAnnotations.LAST_SIZE_SELECTION),
            resources=resources,
            url=url,
            conditions=conditions,
            volumes=volumes,
            env_from=env_from,
        )

    @staticmethod
    def _determine_status(annotations: dict[str, Any], status_obj: Any) -> WorkbenchStatus:
        """Determine workbench status from annotations and status object."""
        # Check if stopped via annotation
        if RHOAIAnnotations.is_notebook_stopped(annotations):
            return WorkbenchStatus.STOPPED

        if not status_obj:
            return WorkbenchStatus.UNKNOWN

        # Check conditions
        conditions = getattr(status_obj, "conditions", []) or []
        for cond in conditions:
            if cond.type == "Ready":
                if cond.status == "True":
                    return WorkbenchStatus.RUNNING
                elif cond.reason == "Waiting":
                    return WorkbenchStatus.STARTING
                else:
                    return WorkbenchStatus.ERROR

        # Check ready replicas
        ready_replicas = getattr(status_obj, "readyReplicas", 0) or 0
        if ready_replicas > 0:
            return WorkbenchStatus.RUNNING

        return WorkbenchStatus.STARTING


class WorkbenchCreate(BaseModel):
    """Request model for creating a workbench."""

    name: str = Field(..., description="Workbench name")
    namespace: str = Field(..., description="Project namespace")
    display_name: str | None = Field(None, description="Display name")
    image: str = Field(..., description="Container image to use")
    size: str = Field("Small", description="Size selection (Small, Medium, Large, etc.)")
    cpu_request: str = Field("500m", description="CPU request")
    cpu_limit: str = Field("2", description="CPU limit")
    memory_request: str = Field("1Gi", description="Memory request")
    memory_limit: str = Field("4Gi", description="Memory limit")
    gpu_count: int = Field(0, description="Number of GPUs to request")
    storage_size: str = Field("10Gi", description="Size of the workbench PVC")
    data_connections: list[str] = Field(
        default_factory=list, description="Secret names to mount as env vars"
    )
    additional_pvcs: list[str] = Field(
        default_factory=list, description="Additional PVC names to mount"
    )
    inject_oauth: bool = Field(True, description="Inject OAuth proxy for authentication")
