"""Pydantic models for Storage (PVCs)."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.models.common import ResourceMetadata


class StorageStatus(str, Enum):
    """PVC status values."""

    BOUND = "Bound"
    PENDING = "Pending"
    LOST = "Lost"
    UNKNOWN = "Unknown"


class StorageAccessMode(str, Enum):
    """PVC access modes."""

    READ_WRITE_ONCE = "ReadWriteOnce"
    READ_ONLY_MANY = "ReadOnlyMany"
    READ_WRITE_MANY = "ReadWriteMany"


class Storage(BaseModel):
    """Storage (PVC) representation."""

    metadata: ResourceMetadata
    display_name: str | None = Field(None, description="Display name")
    size: str = Field(..., description="Storage size (e.g., '10Gi')")
    access_modes: list[str] = Field(default_factory=list, description="Access modes")
    storage_class: str | None = Field(None, description="Storage class name")
    status: StorageStatus = Field(StorageStatus.UNKNOWN, description="PVC status")
    volume_name: str | None = Field(None, description="Bound volume name")

    @classmethod
    def from_pvc(cls, pvc: Any) -> "Storage":
        """Create from Kubernetes PVC."""
        metadata = pvc.metadata
        annotations = metadata.annotations or {}
        spec = pvc.spec
        status = pvc.status

        # Determine status
        phase = getattr(status, "phase", None) if status else None
        pvc_status = StorageStatus.UNKNOWN
        if phase == "Bound":
            pvc_status = StorageStatus.BOUND
        elif phase == "Pending":
            pvc_status = StorageStatus.PENDING
        elif phase == "Lost":
            pvc_status = StorageStatus.LOST

        # Get size
        size = "unknown"
        if spec and spec.resources and spec.resources.requests:
            size = spec.resources.requests.get("storage", "unknown")

        return cls(
            metadata=ResourceMetadata.from_k8s_metadata(
                metadata,
                kind="PersistentVolumeClaim",
                api_version="v1",
            ),
            display_name=annotations.get("openshift.io/display-name"),
            size=size,
            access_modes=list(spec.access_modes) if spec and spec.access_modes else [],
            storage_class=spec.storage_class_name if spec else None,
            status=pvc_status,
            volume_name=spec.volume_name if spec else None,
        )


class StorageCreate(BaseModel):
    """Request model for creating storage (PVC)."""

    name: str = Field(..., description="PVC name")
    namespace: str = Field(..., description="Project namespace")
    display_name: str | None = Field(None, description="Display name")
    size: str = Field("10Gi", description="Storage size")
    access_mode: str = Field("ReadWriteOnce", description="Access mode")
    storage_class: str | None = Field(None, description="Storage class name")
