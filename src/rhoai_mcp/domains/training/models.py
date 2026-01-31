"""Pydantic models for Kubeflow Training resources."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TrainingState(str, Enum):
    """Training progress state values."""

    INITIALIZING = "Initializing"
    TRAINING = "Training"
    COMPLETED = "Completed"
    FAILED = "Failed"
    SUSPENDED = "Suspended"


class PeftMethod(str, Enum):
    """Parameter-efficient fine-tuning methods."""

    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"
    DORA = "dora"


class TrainJobStatus(str, Enum):
    """TrainJob status values."""

    CREATED = "Created"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    SUSPENDED = "Suspended"


# Trainer status annotation key
TRAINER_STATUS_ANNOTATION = "trainer.opendatahub.io/trainerStatus"


class TrainingProgress(BaseModel):
    """Training progress parsed from trainer status annotation."""

    state: TrainingState = Field(TrainingState.INITIALIZING, description="Current training state")
    current_epoch: int = Field(0, description="Current epoch number")
    total_epochs: int = Field(0, description="Total number of epochs")
    current_step: int = Field(0, description="Current training step")
    total_steps: int = Field(0, description="Total training steps")
    loss: float | None = Field(None, description="Current loss value")
    learning_rate: float | None = Field(None, description="Current learning rate")
    throughput: float | None = Field(None, description="Training throughput (samples/sec)")
    gradient_norm: float | None = Field(None, description="Gradient norm")
    eta_seconds: int | None = Field(None, description="Estimated time remaining in seconds")

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100.0

    def progress_bar(self, width: int = 20) -> str:
        """Render a text-based progress bar."""
        if self.total_steps == 0:
            return "-" * width
        filled = int(width * self.current_step / self.total_steps)
        return "=" * filled + "-" * (width - filled)

    @classmethod
    def from_annotation(cls, annotation: str) -> TrainingProgress:
        """Parse training progress from trainer status annotation JSON."""
        if not annotation:
            return cls()

        try:
            data = json.loads(annotation)
        except json.JSONDecodeError:
            return cls()

        state_str = data.get("trainingState", "Initializing")
        try:
            state = TrainingState(state_str)
        except ValueError:
            state = TrainingState.INITIALIZING

        return cls(
            state=state,
            current_epoch=data.get("currentEpoch", 0),
            total_epochs=data.get("totalEpochs", 0),
            current_step=data.get("currentStep", 0),
            total_steps=data.get("totalSteps", 0),
            loss=data.get("loss"),
            learning_rate=data.get("learningRate"),
            throughput=data.get("throughput"),
            gradient_norm=data.get("gradientNorm"),
            eta_seconds=data.get("estimatedTimeRemaining"),
        )


class TrainJob(BaseModel):
    """TrainJob resource representation."""

    name: str = Field(..., description="Job name")
    namespace: str = Field(..., description="Job namespace")
    uid: str | None = Field(None, description="Kubernetes UID")
    kind: str = Field("TrainJob", description="Resource kind")
    api_version: str = Field("trainer.kubeflow.org/v1", description="API version")
    status: TrainJobStatus = Field(TrainJobStatus.CREATED, description="Job status")
    model_id: str | None = Field(None, description="Model identifier (e.g., org/model)")
    dataset_id: str | None = Field(None, description="Dataset identifier")
    method: PeftMethod | None = Field(None, description="Fine-tuning method")
    num_nodes: int = Field(1, description="Number of training nodes")
    gpus_per_node: int = Field(0, description="GPUs per node")
    progress: TrainingProgress | None = Field(None, description="Training progress")
    runtime_ref: str | None = Field(None, description="Reference to training runtime")
    checkpoint_dir: str | None = Field(None, description="Checkpoint directory path")
    creation_timestamp: str | None = Field(None, description="When the job was created")

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
    def from_resource(cls, resource: Any) -> TrainJob:
        """Create TrainJob from Kubernetes resource."""
        metadata = resource.metadata
        spec = getattr(resource, "spec", {}) or {}
        status_obj = getattr(resource, "status", {}) or {}

        # Parse model config
        model_config = spec.get("modelConfig", {})
        model_id = model_config.get("name")

        # Parse dataset config
        dataset_config = spec.get("datasetConfig", {})
        dataset_id = dataset_config.get("name")

        # Parse trainer config
        trainer_config = spec.get("trainer", {})
        num_nodes = trainer_config.get("numNodes", 1)

        # Parse runtime reference
        runtime_ref = None
        if "runtimeRef" in spec:
            runtime_ref = spec["runtimeRef"].get("name")

        # Parse status from conditions
        job_status = TrainJobStatus.CREATED
        conditions = status_obj.get("conditions", [])
        for condition in reversed(conditions):
            if condition.get("status") == "True":
                condition_type = condition.get("type", "")
                if condition_type == "Running":
                    job_status = TrainJobStatus.RUNNING
                    break
                elif condition_type == "Completed":
                    job_status = TrainJobStatus.COMPLETED
                    break
                elif condition_type == "Failed":
                    job_status = TrainJobStatus.FAILED
                    break
                elif condition_type == "Suspended":
                    job_status = TrainJobStatus.SUSPENDED
                    break

        # Parse training progress from annotation
        annotations = metadata.annotations or {}
        progress_annotation = annotations.get(TRAINER_STATUS_ANNOTATION, "")
        progress = (
            TrainingProgress.from_annotation(progress_annotation) if progress_annotation else None
        )

        # Parse creation timestamp
        creation_ts = getattr(metadata, "creation_timestamp", None)
        creation_str = str(creation_ts) if creation_ts else None

        return cls(
            name=metadata.name,
            namespace=metadata.namespace,
            uid=getattr(metadata, "uid", None),
            status=job_status,
            model_id=model_id,
            dataset_id=dataset_id,
            num_nodes=num_nodes,
            progress=progress,
            runtime_ref=runtime_ref,
            creation_timestamp=creation_str,
        )


class NodeResources(BaseModel):
    """Node-level resource information."""

    name: str = Field(..., description="Node name")
    cpu_total: int = Field(0, description="Total CPU cores")
    cpu_allocatable: int = Field(0, description="Allocatable CPU cores")
    memory_total_gb: float = Field(0.0, description="Total memory in GB")
    memory_allocatable_gb: float = Field(0.0, description="Allocatable memory in GB")
    gpu_count: int = Field(0, description="Number of GPUs")
    gpu_type: str | None = Field(None, description="GPU resource type label")


class GPUInfo(BaseModel):
    """GPU availability information."""

    type: str = Field(..., description="GPU resource type (e.g., nvidia.com/gpu)")
    total: int = Field(0, description="Total GPUs in cluster")
    available: int = Field(0, description="Available GPUs")
    nodes_with_gpu: int = Field(0, description="Number of nodes with GPUs")


class ClusterResources(BaseModel):
    """Cluster-wide resource information."""

    cpu_total: int = Field(0, description="Total CPU cores")
    cpu_allocatable: int = Field(0, description="Allocatable CPU cores")
    memory_total_gb: float = Field(0.0, description="Total memory in GB")
    memory_allocatable_gb: float = Field(0.0, description="Allocatable memory in GB")
    gpu_info: GPUInfo | None = Field(None, description="GPU information")
    node_count: int = Field(0, description="Number of nodes")
    nodes: list[NodeResources] = Field(default_factory=list, description="Per-node resources")

    @property
    def has_gpus(self) -> bool:
        """Check if cluster has GPUs available."""
        return self.gpu_info is not None and self.gpu_info.total > 0


class TrainingRuntime(BaseModel):
    """ClusterTrainingRuntime or TrainingRuntime representation."""

    name: str = Field(..., description="Runtime name")
    namespace: str | None = Field(None, description="Namespace (None for cluster-scoped)")
    framework: str | None = Field(None, description="Training framework")
    trainer_image: str | None = Field(None, description="Trainer container image")
    has_model_initializer: bool = Field(False, description="Has model initializer")
    has_dataset_initializer: bool = Field(False, description="Has dataset initializer")

    @classmethod
    def from_resource(cls, resource: Any, is_cluster_scoped: bool = True) -> TrainingRuntime:
        """Create TrainingRuntime from Kubernetes resource."""
        metadata = resource.metadata
        spec = getattr(resource, "spec", {}) or {}

        # Parse template to detect initializers and framework
        template = spec.get("template", {})
        spec_template = template.get("spec", {})

        # Check for initializers
        initializers = spec_template.get("initializers", [])
        has_model = any(i.get("type") == "model" for i in initializers)
        has_dataset = any(i.get("type") == "dataset" for i in initializers)

        # Try to detect framework from image or labels
        framework = None
        labels = metadata.labels or {}
        framework = labels.get("training.kubeflow.org/framework")

        return cls(
            name=metadata.name,
            namespace=None if is_cluster_scoped else metadata.namespace,
            framework=framework,
            has_model_initializer=has_model,
            has_dataset_initializer=has_dataset,
        )


class ResourceEstimate(BaseModel):
    """Estimated resource requirements for training."""

    model_memory_gb: float = Field(..., description="Memory for model weights")
    optimizer_state_gb: float = Field(..., description="Memory for optimizer state")
    activation_memory_gb: float = Field(..., description="Memory for activations")
    total_required_gb: float = Field(..., description="Total GPU memory required")
    recommended_gpus: int = Field(..., description="Recommended number of GPUs")
    recommended_gpu_type: str = Field(..., description="Recommended GPU type")
    storage_gb: int = Field(..., description="Recommended storage in GB")
