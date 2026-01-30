"""Unified training tool that consolidates all training operations.

This module provides a single `training` tool that uses an `action` parameter
to route to different operations, reducing the number of tools AI agents
need to discover and remember.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.training.client import TrainingClient
from rhoai_mcp.domains.training.models import PeftMethod, TrainJobStatus

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


TrainingAction = Literal[
    # Discovery
    "list",
    "get",
    "status",
    "progress",
    # Lifecycle
    "create",
    "suspend",
    "resume",
    "delete",
    # Monitoring
    "logs",
    "events",
    "checkpoints",
    # Planning
    "estimate",
    "validate",
    "prerequisites",
]


def register_tools(mcp: FastMCP, server: RHOAIServer) -> None:
    """Register the unified training tool with the MCP server."""

    @mcp.tool()
    def training(
        action: str,
        namespace: str | None = None,
        name: str | None = None,
        # Create action parameters
        model_id: str | None = None,
        dataset_id: str | None = None,
        runtime_name: str | None = None,
        method: str = "lora",
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_nodes: int = 1,
        gpus_per_node: int = 1,
        checkpoint_dir: str | None = None,
        confirmed: bool = False,
        # Logs parameters
        tail_lines: int = 100,
        container: str = "trainer",
        previous: bool = False,
        # Delete parameter
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Unified training operations using action parameter.

        This is a consolidated interface for all training operations.
        Use the 'action' parameter to specify what operation to perform.

        Actions:
            Discovery (namespace required):
                - list: List training jobs in namespace
                - get: Get job details (name required)
                - status: Quick status check (name required)
                - progress: Training metrics (name required)

            Lifecycle (namespace + name required):
                - create: Create job (+ model_id, dataset_id, runtime_name, ...)
                - suspend: Pause training
                - resume: Continue training
                - delete: Remove job (confirm=True required)

            Monitoring (namespace + name required):
                - logs: Get training logs (tail_lines optional)
                - events: Get Kubernetes events
                - checkpoints: List saved checkpoints

            Planning (namespace required):
                - estimate: Resource estimation (model_id required)
                - validate: Config validation (model_id, dataset_id, runtime required)
                - prerequisites: Pre-flight checks (model_id, dataset_id required)

        Args:
            action: The operation to perform.
            namespace: The namespace for the operation.
            name: Job name (required for most actions).
            model_id: Model identifier for create/estimate/validate.
            dataset_id: Dataset identifier for create/validate.
            runtime_name: Training runtime for create/validate.
            method: Fine-tuning method (lora, qlora, dora, full).
            epochs: Number of training epochs.
            batch_size: Per-device batch size.
            learning_rate: Training learning rate.
            num_nodes: Number of training nodes.
            gpus_per_node: GPUs per node.
            checkpoint_dir: Directory for checkpoints.
            confirmed: Confirm job creation.
            tail_lines: Number of log lines to return.
            container: Container name for logs.
            previous: Get logs from previous container instance.
            confirm: Confirm deletion.

        Returns:
            Result of the requested action.
        """
        action = action.lower()

        # Validate common requirements
        if action not in _VALID_ACTIONS:
            return {
                "error": f"Invalid action: {action}",
                "valid_actions": list(_VALID_ACTIONS),
            }

        # Some actions don't require namespace
        no_namespace_actions = {"list", "estimate"}
        if action not in no_namespace_actions and namespace is None:
            return {"error": f"namespace is required for action '{action}'"}

        # Route to appropriate handler
        if action == "list":
            return _action_list(server, namespace)
        elif action == "get":
            return _action_get(server, namespace, name)
        elif action == "status":
            return _action_status(server, namespace, name)
        elif action == "progress":
            return _action_progress(server, namespace, name)
        elif action == "create":
            return _action_create(
                server,
                namespace,
                name,
                model_id,
                dataset_id,
                runtime_name,
                method,
                epochs,
                batch_size,
                learning_rate,
                num_nodes,
                gpus_per_node,
                checkpoint_dir,
                confirmed,
            )
        elif action == "suspend":
            return _action_suspend(server, namespace, name)
        elif action == "resume":
            return _action_resume(server, namespace, name)
        elif action == "delete":
            return _action_delete(server, namespace, name, confirm)
        elif action == "logs":
            return _action_logs(server, namespace, name, container, tail_lines, previous)
        elif action == "events":
            return _action_events(server, namespace, name)
        elif action == "checkpoints":
            return _action_checkpoints(server, namespace, name)
        elif action == "estimate":
            return _action_estimate(server, model_id, method)
        elif action == "validate":
            return _action_validate(server, namespace, model_id, dataset_id, runtime_name)
        elif action == "prerequisites":
            return _action_prerequisites(server, namespace, model_id, dataset_id, checkpoint_dir)

        return {"error": f"Action '{action}' not implemented"}


_VALID_ACTIONS = {
    "list",
    "get",
    "status",
    "progress",
    "create",
    "suspend",
    "resume",
    "delete",
    "logs",
    "events",
    "checkpoints",
    "estimate",
    "validate",
    "prerequisites",
}


def _action_list(server: RHOAIServer, namespace: str | None) -> dict[str, Any]:
    """List training jobs."""
    if namespace is None:
        return {"error": "namespace is required for list action"}

    client = TrainingClient(server.k8s)
    jobs = client.list_training_jobs(namespace)

    return {
        "action": "list",
        "namespace": namespace,
        "count": len(jobs),
        "jobs": [
            {
                "name": job.name,
                "status": job.status.value,
                "model_id": job.model_id,
                "progress": f"{round(job.progress.progress_percent, 1)}%"
                if job.progress
                else None,
            }
            for job in jobs
        ],
    }


def _action_get(
    server: RHOAIServer, namespace: str | None, name: str | None
) -> dict[str, Any]:
    """Get training job details."""
    if namespace is None or name is None:
        return {"error": "namespace and name are required for get action"}

    client = TrainingClient(server.k8s)
    job = client.get_training_job(namespace, name)

    result: dict[str, Any] = {
        "action": "get",
        "name": job.name,
        "namespace": job.namespace,
        "status": job.status.value,
        "model_id": job.model_id,
        "dataset_id": job.dataset_id,
        "runtime_ref": job.runtime_ref,
        "num_nodes": job.num_nodes,
        "gpus_per_node": job.gpus_per_node,
        "suspended": job.status == TrainJobStatus.SUSPENDED,
        "created_at": job.creation_timestamp,
    }

    if job.progress:
        result["progress"] = {
            "state": job.progress.state.value,
            "current_epoch": job.progress.current_epoch,
            "total_epochs": job.progress.total_epochs,
            "current_step": job.progress.current_step,
            "total_steps": job.progress.total_steps,
            "progress_percent": round(job.progress.progress_percent, 1),
            "loss": job.progress.loss,
        }

    return result


def _action_status(
    server: RHOAIServer, namespace: str | None, name: str | None
) -> dict[str, Any]:
    """Quick status check."""
    if namespace is None or name is None:
        return {"error": "namespace and name are required for status action"}

    client = TrainingClient(server.k8s)
    job = client.get_training_job(namespace, name)

    return {
        "action": "status",
        "name": job.name,
        "status": job.status.value,
        "suspended": job.status == TrainJobStatus.SUSPENDED,
        "progress": f"{round(job.progress.progress_percent, 1)}%" if job.progress else None,
    }


def _action_progress(
    server: RHOAIServer, namespace: str | None, name: str | None
) -> dict[str, Any]:
    """Get training progress."""
    if namespace is None or name is None:
        return {"error": "namespace and name are required for progress action"}

    client = TrainingClient(server.k8s)
    job = client.get_training_job(namespace, name)

    if not job.progress:
        return {
            "action": "progress",
            "name": name,
            "message": "No progress information available. Job may not have started.",
        }

    return {
        "action": "progress",
        "name": name,
        "state": job.progress.state.value,
        "current_epoch": job.progress.current_epoch,
        "total_epochs": job.progress.total_epochs,
        "current_step": job.progress.current_step,
        "total_steps": job.progress.total_steps,
        "progress_percent": round(job.progress.progress_percent, 1),
        "progress_bar": job.progress.progress_bar(),
        "loss": job.progress.loss,
        "learning_rate": job.progress.learning_rate,
        "throughput": job.progress.throughput,
        "eta_seconds": job.progress.eta_seconds,
    }


def _action_create(
    server: RHOAIServer,
    namespace: str | None,
    name: str | None,
    model_id: str | None,
    dataset_id: str | None,
    runtime_name: str | None,
    method: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_nodes: int,
    gpus_per_node: int,
    checkpoint_dir: str | None,
    confirmed: bool,
) -> dict[str, Any]:
    """Create a training job."""
    if namespace is None:
        return {"error": "namespace is required for create action"}
    if model_id is None:
        return {"error": "model_id is required for create action"}
    if dataset_id is None:
        return {"error": "dataset_id is required for create action"}
    if runtime_name is None:
        return {"error": "runtime_name is required for create action"}

    # Check if operation is allowed
    allowed, reason = server.config.is_operation_allowed("create")
    if not allowed:
        return {"error": reason}

    # Validate method
    try:
        peft_method = PeftMethod(method.lower())
    except ValueError:
        return {
            "error": f"Invalid method: {method}",
            "valid_methods": [m.value for m in PeftMethod],
        }

    # Generate job name if not provided
    job_name = name
    if not job_name:
        import hashlib
        import time

        suffix = hashlib.md5(f"{model_id}-{time.time()}".encode()).hexdigest()[:8]
        job_name = f"train-{suffix}"

    # Build preview
    preview = {
        "job_name": job_name,
        "namespace": namespace,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "runtime_name": runtime_name,
        "method": peft_method.value,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "checkpoint_dir": checkpoint_dir,
    }

    if not confirmed:
        return {
            "action": "create",
            "preview": preview,
            "message": "Review the configuration. Call training(action='create', confirmed=True, ...) to create.",
        }

    # Create the job
    client = TrainingClient(server.k8s)
    job = client.create_training_job(
        namespace=namespace,
        name=job_name,
        model_id=model_id,
        dataset_id=dataset_id,
        runtime_ref=runtime_name,
        method=peft_method,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
    )

    return {
        "action": "create",
        "success": True,
        "name": job.name,
        "namespace": job.namespace,
        "status": job.status.value,
        "message": f"Training job '{job.name}' created.",
    }


def _action_suspend(
    server: RHOAIServer, namespace: str | None, name: str | None
) -> dict[str, Any]:
    """Suspend a training job."""
    if namespace is None or name is None:
        return {"error": "namespace and name are required for suspend action"}

    client = TrainingClient(server.k8s)
    client.suspend_training_job(namespace, name)

    return {
        "action": "suspend",
        "success": True,
        "name": name,
        "namespace": namespace,
        "message": f"Training job '{name}' suspended.",
    }


def _action_resume(
    server: RHOAIServer, namespace: str | None, name: str | None
) -> dict[str, Any]:
    """Resume a training job."""
    if namespace is None or name is None:
        return {"error": "namespace and name are required for resume action"}

    client = TrainingClient(server.k8s)
    client.resume_training_job(namespace, name)

    return {
        "action": "resume",
        "success": True,
        "name": name,
        "namespace": namespace,
        "message": f"Training job '{name}' resumed.",
    }


def _action_delete(
    server: RHOAIServer, namespace: str | None, name: str | None, confirm: bool
) -> dict[str, Any]:
    """Delete a training job."""
    if namespace is None or name is None:
        return {"error": "namespace and name are required for delete action"}

    allowed, reason = server.config.is_operation_allowed("delete")
    if not allowed:
        return {"error": reason}

    if not confirm:
        return {
            "action": "delete",
            "error": "Deletion not confirmed",
            "message": f"To delete '{name}', call training(action='delete', confirm=True, ...).",
        }

    client = TrainingClient(server.k8s)
    client.delete_training_job(namespace, name)

    return {
        "action": "delete",
        "success": True,
        "name": name,
        "namespace": namespace,
        "message": f"Training job '{name}' deleted.",
    }


def _action_logs(
    server: RHOAIServer,
    namespace: str | None,
    name: str | None,
    container: str,
    tail_lines: int,
    previous: bool,
) -> dict[str, Any]:
    """Get training logs."""
    if namespace is None or name is None:
        return {"error": "namespace and name are required for logs action"}

    client = TrainingClient(server.k8s)
    logs = client.get_training_logs(namespace, name, container=container, tail_lines=tail_lines, previous=previous)

    return {
        "action": "logs",
        "name": name,
        "container": container,
        "logs": logs,
        "lines_returned": len(logs.split("\n")) if logs else 0,
    }


def _action_events(
    server: RHOAIServer, namespace: str | None, name: str | None
) -> dict[str, Any]:
    """Get job events."""
    if namespace is None or name is None:
        return {"error": "namespace and name are required for events action"}

    client = TrainingClient(server.k8s)
    events = client.get_job_events(namespace, name)

    warnings = [e for e in events if e.get("type") == "Warning"]

    return {
        "action": "events",
        "name": name,
        "total_events": len(events),
        "events": events,
        "has_warnings": len(warnings) > 0,
        "warning_count": len(warnings),
    }


def _action_checkpoints(
    server: RHOAIServer, namespace: str | None, name: str | None
) -> dict[str, Any]:
    """Get checkpoint information."""
    if namespace is None or name is None:
        return {"error": "namespace and name are required for checkpoints action"}

    import json

    from rhoai_mcp.domains.training.crds import TrainingCRDs

    resource = server.k8s.get(TrainingCRDs.TRAIN_JOB, name, namespace=namespace)
    annotations = getattr(resource.metadata, "annotations", {}) or {}
    checkpoint_annotation = annotations.get("trainer.opendatahub.io/checkpoint", "")

    if not checkpoint_annotation:
        return {
            "action": "checkpoints",
            "name": name,
            "message": "No checkpoint information available.",
            "latest": None,
            "checkpoints": [],
        }

    try:
        checkpoint_data = json.loads(checkpoint_annotation)
    except json.JSONDecodeError:
        return {
            "action": "checkpoints",
            "name": name,
            "error": "Failed to parse checkpoint annotation",
        }

    return {
        "action": "checkpoints",
        "name": name,
        "latest": checkpoint_data.get("latest"),
        "checkpoints": checkpoint_data.get("checkpoints", []),
    }


def _action_estimate(
    server: RHOAIServer,  # noqa: ARG001
    model_id: str | None,
    method: str,
) -> dict[str, Any]:
    """Estimate resources."""
    if model_id is None:
        return {"error": "model_id is required for estimate action"}

    from rhoai_mcp.domains.training.tools.planning import (
        GPU_MEMORY_ESTIMATES,
        PEFT_MULTIPLIERS,
        _extract_param_count,
    )

    param_count = _extract_param_count(model_id)

    base_memory = 16
    for (min_p, max_p), mem in GPU_MEMORY_ESTIMATES.items():
        if min_p <= param_count < max_p:
            base_memory = mem
            break

    try:
        peft_method = PeftMethod(method.lower())
    except ValueError:
        peft_method = PeftMethod.LORA

    multiplier = PEFT_MULTIPLIERS.get(peft_method, 1.8)
    total_memory = base_memory * multiplier

    recommended_gpus = 1
    if total_memory > 80:
        recommended_gpus = int((total_memory / 80) + 0.5)

    return {
        "action": "estimate",
        "model_id": model_id,
        "estimated_params_billion": param_count,
        "method": peft_method.value,
        "total_required_gb": round(total_memory, 1),
        "recommended_gpus": recommended_gpus,
        "storage_gb": int(param_count * 4) + 50,
    }


def _action_validate(
    server: RHOAIServer,
    namespace: str | None,
    model_id: str | None,
    dataset_id: str | None,
    runtime_name: str | None,
) -> dict[str, Any]:
    """Validate training config."""
    import re

    if namespace is None:
        return {"error": "namespace is required for validate action"}
    if model_id is None:
        return {"error": "model_id is required for validate action"}
    if dataset_id is None:
        return {"error": "dataset_id is required for validate action"}
    if runtime_name is None:
        return {"error": "runtime_name is required for validate action"}

    errors = []

    # Validate runtime exists
    try:
        from rhoai_mcp.domains.training.crds import TrainingCRDs

        server.k8s.get(TrainingCRDs.CLUSTER_TRAINING_RUNTIME, runtime_name)
    except Exception:
        errors.append(f"Runtime '{runtime_name}' not found")

    # Validate model ID format
    if not re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$", model_id):
        errors.append(f"Invalid model ID format: '{model_id}'")

    # Validate dataset ID format
    if not re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$", dataset_id):
        errors.append(f"Invalid dataset ID format: '{dataset_id}'")

    return {
        "action": "validate",
        "valid": len(errors) == 0,
        "errors": errors if errors else None,
        "configuration": {
            "namespace": namespace,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "runtime_name": runtime_name,
        },
    }


def _action_prerequisites(
    server: RHOAIServer,
    namespace: str | None,
    model_id: str | None,
    dataset_id: str | None,
    checkpoint_storage: str | None,
) -> dict[str, Any]:
    """Check prerequisites."""
    if namespace is None:
        return {"error": "namespace is required for prerequisites action"}
    if model_id is None:
        return {"error": "model_id is required for prerequisites action"}
    if dataset_id is None:
        return {"error": "dataset_id is required for prerequisites action"}

    checks = []
    actions_needed = []
    all_passed = True

    client = TrainingClient(server.k8s)

    # Check cluster connectivity and GPUs
    try:
        resources = client.get_cluster_resources()
        checks.append({
            "name": "Cluster connectivity",
            "passed": True,
            "message": f"Connected to cluster with {resources.node_count} nodes",
        })

        if resources.has_gpus and resources.gpu_info:
            checks.append({
                "name": "GPU availability",
                "passed": True,
                "message": f"{resources.gpu_info.total} GPUs available",
            })
        else:
            checks.append({
                "name": "GPU availability",
                "passed": False,
                "message": "No GPUs detected",
            })
            all_passed = False
            actions_needed.append("Ensure GPU nodes are available")
    except Exception as e:
        checks.append({
            "name": "Cluster connectivity",
            "passed": False,
            "message": str(e),
        })
        all_passed = False

    # Check runtime availability
    try:
        runtimes = client.list_cluster_training_runtimes()
        if runtimes:
            checks.append({
                "name": "Training runtimes",
                "passed": True,
                "message": f"{len(runtimes)} runtimes available",
            })
        else:
            checks.append({
                "name": "Training runtimes",
                "passed": False,
                "message": "No runtimes configured",
            })
            all_passed = False
            actions_needed.append("Create a training runtime")
    except Exception:
        checks.append({
            "name": "Training runtimes",
            "passed": False,
            "message": "Failed to list runtimes",
        })
        all_passed = False

    # Check model/dataset format
    if "/" in model_id:
        checks.append({"name": "Model ID format", "passed": True, "message": "Valid format"})
    else:
        checks.append({"name": "Model ID format", "passed": False, "message": "Should be org/model"})
        all_passed = False

    if "/" in dataset_id:
        checks.append({"name": "Dataset ID format", "passed": True, "message": "Valid format"})
    else:
        checks.append({"name": "Dataset ID format", "passed": False, "message": "Should be org/dataset"})
        all_passed = False

    # Check storage if specified
    if checkpoint_storage:
        try:
            pvc = server.k8s.get_pvc(checkpoint_storage, namespace)
            if pvc.status.phase == "Bound":
                checks.append({
                    "name": "Checkpoint storage",
                    "passed": True,
                    "message": f"PVC '{checkpoint_storage}' is bound",
                })
            else:
                checks.append({
                    "name": "Checkpoint storage",
                    "passed": False,
                    "message": f"PVC state: {pvc.status.phase}",
                })
                all_passed = False
        except Exception:
            checks.append({
                "name": "Checkpoint storage",
                "passed": False,
                "message": f"PVC '{checkpoint_storage}' not found",
            })
            all_passed = False
            actions_needed.append(f"Create PVC '{checkpoint_storage}'")

    return {
        "action": "prerequisites",
        "all_passed": all_passed,
        "checks": checks,
        "actions_needed": actions_needed if actions_needed else None,
        "ready_to_train": all_passed,
    }
