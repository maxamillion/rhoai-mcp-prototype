"""Common Pydantic models shared across RHOAI resources."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResourceStatus(str, Enum):
    """Common status values for RHOAI resources."""

    UNKNOWN = "Unknown"
    PENDING = "Pending"
    RUNNING = "Running"
    READY = "Ready"
    STOPPED = "Stopped"
    ERROR = "Error"
    FAILED = "Failed"
    CREATING = "Creating"
    DELETING = "Deleting"


class ResourceMetadata(BaseModel):
    """Common metadata for Kubernetes resources."""

    name: str = Field(..., description="Resource name")
    namespace: str | None = Field(None, description="Resource namespace")
    uid: str | None = Field(None, description="Kubernetes UID")
    kind: str | None = Field(None, description="Resource kind")
    api_version: str | None = Field(None, description="API version")
    creation_timestamp: datetime | None = Field(None, description="When the resource was created")
    labels: dict[str, str] = Field(default_factory=dict, description="Resource labels")
    annotations: dict[str, str] = Field(default_factory=dict, description="Resource annotations")

    def to_source_dict(self) -> dict[str, Any]:
        """Return _source metadata for grounding responses to K8s resources."""
        return {
            "kind": self.kind,
            "api_version": self.api_version,
            "name": self.name,
            "namespace": self.namespace,
            "uid": self.uid,
        }

    @classmethod
    def from_k8s_metadata(
        cls,
        metadata: Any,
        kind: str | None = None,
        api_version: str | None = None,
    ) -> "ResourceMetadata":
        """Create from Kubernetes metadata object.

        Handles both core API objects (which return plain dicts) and
        dynamic client objects (which return ResourceField objects).

        Args:
            metadata: Kubernetes metadata object.
            kind: Resource kind (e.g., "Notebook", "InferenceService").
            api_version: API version (e.g., "kubeflow.org/v1").
        """
        # Convert labels/annotations to plain dicts if they're ResourceField objects
        labels = metadata.labels
        if labels is not None and not isinstance(labels, dict):
            labels = dict(labels)
        annotations = metadata.annotations
        if annotations is not None and not isinstance(annotations, dict):
            annotations = dict(annotations)

        return cls(
            name=metadata.name,
            namespace=getattr(metadata, "namespace", None),
            uid=getattr(metadata, "uid", None),
            kind=kind,
            api_version=api_version,
            creation_timestamp=getattr(metadata, "creation_timestamp", None),
            labels=labels or {},
            annotations=annotations or {},
        )


class ResourceReference(BaseModel):
    """Reference to a Kubernetes resource."""

    name: str = Field(..., description="Resource name")
    namespace: str | None = Field(None, description="Resource namespace")
    kind: str | None = Field(None, description="Resource kind")
    api_version: str | None = Field(None, description="API version")


class OwnerReference(BaseModel):
    """Kubernetes owner reference."""

    api_version: str
    kind: str
    name: str
    uid: str
    controller: bool = False
    block_owner_deletion: bool = False


class Condition(BaseModel):
    """Kubernetes-style condition."""

    type: str = Field(..., description="Condition type")
    status: str = Field(..., description="Condition status (True, False, Unknown)")
    reason: str | None = Field(None, description="Machine-readable reason")
    message: str | None = Field(None, description="Human-readable message")
    last_transition_time: datetime | None = Field(None, description="Last transition time")

    @property
    def is_true(self) -> bool:
        """Check if condition status is True."""
        return self.status == "True"

    @classmethod
    def from_k8s_condition(cls, condition: Any) -> "Condition":
        """Create from Kubernetes condition object."""
        return cls(
            type=condition.type,
            status=condition.status,
            reason=getattr(condition, "reason", None),
            message=getattr(condition, "message", None),
            last_transition_time=getattr(condition, "last_transition_time", None),
        )


class ResourceSummary(BaseModel):
    """Summary of resource counts in a namespace."""

    workbenches: int = Field(0, description="Number of workbenches")
    workbenches_running: int = Field(0, description="Number of running workbenches")
    models: int = Field(0, description="Number of deployed models")
    models_ready: int = Field(0, description="Number of ready models")
    pipelines: int = Field(0, description="Number of pipelines")
    data_connections: int = Field(0, description="Number of data connections")
    storage: int = Field(0, description="Number of PVCs")


class ContainerResources(BaseModel):
    """Container resource requests and limits."""

    cpu_request: str | None = Field(None, description="CPU request (e.g., '500m')")
    cpu_limit: str | None = Field(None, description="CPU limit (e.g., '2')")
    memory_request: str | None = Field(None, description="Memory request (e.g., '1Gi')")
    memory_limit: str | None = Field(None, description="Memory limit (e.g., '4Gi')")
    gpu_request: int | None = Field(None, description="Number of GPUs requested")
    gpu_limit: int | None = Field(None, description="Number of GPUs limit")

    @classmethod
    def from_k8s_resources(
        cls, requests: dict[str, Any] | None, limits: dict[str, Any] | None
    ) -> "ContainerResources":
        """Create from Kubernetes resource requirements."""
        requests = requests or {}
        limits = limits or {}
        return cls(
            cpu_request=requests.get("cpu"),
            cpu_limit=limits.get("cpu"),
            memory_request=requests.get("memory"),
            memory_limit=limits.get("memory"),
            gpu_request=_parse_gpu(requests.get("nvidia.com/gpu")),
            gpu_limit=_parse_gpu(limits.get("nvidia.com/gpu")),
        )


def _parse_gpu(value: Any) -> int | None:
    """Parse GPU value to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
