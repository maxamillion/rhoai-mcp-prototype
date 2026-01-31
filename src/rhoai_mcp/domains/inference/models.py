"""Pydantic models for InferenceService (Model Serving)."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.models.common import Condition, ContainerResources, ResourceMetadata


class InferenceServiceStatus(str, Enum):
    """InferenceService status values."""

    READY = "Ready"
    PENDING = "Pending"
    LOADING = "Loading"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


class ModelFormat(str, Enum):
    """Supported model formats."""

    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    OPENVINO = "openvino"
    CAIKIT = "caikit"
    VLLM = "vllm"


class ServingRuntime(BaseModel):
    """Serving Runtime representation."""

    metadata: ResourceMetadata
    display_name: str | None = Field(None, description="Display name")
    supported_formats: list[str] = Field(
        default_factory=list, description="Supported model formats"
    )
    multi_model: bool = Field(False, description="Whether this supports multi-model")
    replicas: int = Field(1, description="Number of replicas")


class InferenceService(BaseModel):
    """InferenceService representation."""

    metadata: ResourceMetadata
    display_name: str | None = Field(None, description="Display name")
    runtime: str | None = Field(None, description="Serving runtime name")
    model_format: str | None = Field(None, description="Model format")
    storage_uri: str | None = Field(None, description="Model storage URI (S3, PVC, etc.)")
    status: InferenceServiceStatus = Field(
        InferenceServiceStatus.UNKNOWN, description="Service status"
    )
    url: str | None = Field(None, description="Inference endpoint URL")
    internal_url: str | None = Field(None, description="Internal cluster URL")
    conditions: list[Condition] = Field(default_factory=list, description="Status conditions")
    resources: ContainerResources | None = Field(None, description="Resource configuration")

    @classmethod
    def from_inference_service_cr(cls, isvc: Any, url: str | None = None) -> "InferenceService":
        """Create from KServe InferenceService CR."""
        metadata = isvc.metadata
        annotations = metadata.annotations or {}
        spec = isvc.spec or {}
        status_obj = getattr(isvc, "status", None)

        # Determine status
        svc_status = cls._determine_status(status_obj)

        # Get predictor spec
        predictor = spec.get("predictor", {})
        model = predictor.get("model", {})

        # Get storage URI
        storage_uri = model.get("storageUri")

        # Get model format and runtime
        model_format = model.get("modelFormat", {}).get("name")
        runtime = model.get("runtime")

        # Get resources
        resources = None
        container_resources = model.get("resources")
        if container_resources:
            resources = ContainerResources.from_k8s_resources(
                container_resources.get("requests"),
                container_resources.get("limits"),
            )

        # Parse conditions
        conditions = []
        if status_obj:
            for cond in getattr(status_obj, "conditions", []) or []:
                conditions.append(Condition.from_k8s_condition(cond))

        # Get URLs from status
        internal_url = None
        if status_obj:
            address = getattr(status_obj, "address", None)
            if address:
                internal_url = getattr(address, "url", None)

        return cls(
            metadata=ResourceMetadata.from_k8s_metadata(
                metadata,
                kind="InferenceService",
                api_version="serving.kserve.io/v1beta1",
            ),
            display_name=annotations.get("openshift.io/display-name"),
            runtime=runtime,
            model_format=model_format,
            storage_uri=storage_uri,
            status=svc_status,
            url=url or internal_url,
            internal_url=internal_url,
            conditions=conditions,
            resources=resources,
        )

    @staticmethod
    def _determine_status(status_obj: Any) -> InferenceServiceStatus:
        """Determine service status from status object."""
        if not status_obj:
            return InferenceServiceStatus.UNKNOWN

        conditions = getattr(status_obj, "conditions", []) or []
        for cond in conditions:
            cond_type = getattr(cond, "type", None)
            cond_status = getattr(cond, "status", None)
            cond_reason = getattr(cond, "reason", None)

            if cond_type == "Ready":
                if cond_status == "True":
                    return InferenceServiceStatus.READY
                elif cond_reason == "RevisionMissing":
                    return InferenceServiceStatus.PENDING
                elif cond_reason in ("RevisionFailed", "ContainerCreating"):
                    return InferenceServiceStatus.LOADING
                else:
                    return InferenceServiceStatus.FAILED

        return InferenceServiceStatus.PENDING


class InferenceServiceCreate(BaseModel):
    """Request model for deploying a model."""

    name: str = Field(..., description="Model deployment name")
    namespace: str = Field(..., description="Project namespace")
    display_name: str | None = Field(None, description="Display name")
    runtime: str = Field(..., description="Serving runtime to use")
    model_format: str = Field(..., description="Model format (onnx, pytorch, etc.)")
    storage_uri: str = Field(
        ..., description="Model storage URI (s3://bucket/path, pvc://pvc-name/path)"
    )
    min_replicas: int = Field(1, ge=0, description="Minimum replicas")
    max_replicas: int = Field(1, ge=1, description="Maximum replicas")
    cpu_request: str = Field("1", description="CPU request")
    cpu_limit: str = Field("2", description="CPU limit")
    memory_request: str = Field("4Gi", description="Memory request")
    memory_limit: str = Field("8Gi", description="Memory limit")
    gpu_count: int = Field(0, ge=0, description="Number of GPUs")
