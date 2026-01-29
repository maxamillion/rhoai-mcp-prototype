"""Response formatting utilities for context window optimization.

This module provides verbosity levels and response builders to reduce
token usage when AI agents interact with the MCP server.
"""

from enum import Enum
from typing import Any


class Verbosity(str, Enum):
    """Response verbosity levels for context window optimization.

    - MINIMAL: Only essential fields (name, status) - ~85% token savings
    - STANDARD: Default list behavior with key fields - backward compatible
    - FULL: All fields including labels, annotations, conditions
    """

    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"

    @classmethod
    def from_str(cls, value: str | None) -> "Verbosity":
        """Parse verbosity from string, defaulting to STANDARD."""
        if value is None:
            return cls.STANDARD
        try:
            return cls(value.lower())
        except ValueError:
            return cls.STANDARD


class PaginatedResponse:
    """Builder for paginated list responses."""

    @staticmethod
    def build(
        items: list[dict[str, Any]],
        total: int,
        offset: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Build a paginated response with metadata.

        Args:
            items: The paginated items to return.
            total: Total count of items before pagination.
            offset: Starting offset used.
            limit: Limit used (None means all items).

        Returns:
            Response dict with items and pagination metadata.
        """
        return {
            "items": items,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(items) < total,
        }


def paginate(
    items: list[Any],
    offset: int = 0,
    limit: int | None = None,
) -> tuple[list[Any], int]:
    """Apply pagination to a list of items.

    Args:
        items: Full list of items.
        offset: Starting offset (0-indexed).
        limit: Maximum items to return (None for all).

    Returns:
        Tuple of (paginated items, total count).
    """
    total = len(items)
    result = items[offset:]
    if limit is not None and limit > 0:
        result = result[:limit]
    return result, total


class ResponseBuilder:
    """Builds formatted responses at different verbosity levels.

    Each method returns a dict formatted according to the verbosity level.
    MINIMAL returns only essential fields for status checks.
    STANDARD matches current behavior (backward compatible).
    FULL includes all metadata including labels, annotations, conditions.
    """

    @staticmethod
    def workbench_list_item(wb: Any, verbosity: Verbosity = Verbosity.STANDARD) -> dict[str, Any]:
        """Format a workbench for list responses.

        Args:
            wb: Workbench model instance.
            verbosity: Response verbosity level.

        Returns:
            Formatted workbench dict.
        """
        if verbosity == Verbosity.MINIMAL:
            return {
                "name": wb.metadata.name,
                "status": wb.status.value,
            }

        # STANDARD - current list behavior
        result: dict[str, Any] = {
            "name": wb.metadata.name,
            "display_name": wb.display_name,
            "status": wb.status.value,
            "image": wb.image,
            "image_display_name": wb.image_display_name,
            "size": wb.size,
            "url": wb.url,
            "stopped_time": wb.stopped_time.isoformat() if wb.stopped_time else None,
            "volumes": wb.volumes,
            "created": (
                wb.metadata.creation_timestamp.isoformat()
                if wb.metadata.creation_timestamp
                else None
            ),
        }

        if verbosity == Verbosity.FULL:
            result["labels"] = wb.metadata.labels
            result["annotations"] = wb.metadata.annotations
            result["env_from"] = wb.env_from
            if wb.resources:
                result["resources"] = {
                    "cpu_request": wb.resources.cpu_request,
                    "cpu_limit": wb.resources.cpu_limit,
                    "memory_request": wb.resources.memory_request,
                    "memory_limit": wb.resources.memory_limit,
                    "gpu_request": wb.resources.gpu_request,
                    "gpu_limit": wb.resources.gpu_limit,
                }
            if wb.conditions:
                result["conditions"] = [
                    {
                        "type": c.type,
                        "status": c.status,
                        "reason": c.reason,
                        "message": c.message,
                    }
                    for c in wb.conditions
                ]

        return result

    @staticmethod
    def workbench_detail(wb: Any, verbosity: Verbosity = Verbosity.FULL) -> dict[str, Any]:
        """Format a workbench for detail responses.

        Args:
            wb: Workbench model instance.
            verbosity: Response verbosity level.

        Returns:
            Formatted workbench dict.
        """
        if verbosity == Verbosity.MINIMAL:
            return {
                "name": wb.metadata.name,
                "namespace": wb.metadata.namespace,
                "status": wb.status.value,
                "url": wb.url,
            }

        # STANDARD and FULL share the same base
        result: dict[str, Any] = {
            "name": wb.metadata.name,
            "namespace": wb.metadata.namespace,
            "display_name": wb.display_name,
            "status": wb.status.value,
            "image": wb.image,
            "image_display_name": wb.image_display_name,
            "size": wb.size,
            "url": wb.url,
            "stopped_time": wb.stopped_time.isoformat() if wb.stopped_time else None,
            "volumes": wb.volumes,
            "env_from": wb.env_from,
            "created": (
                wb.metadata.creation_timestamp.isoformat()
                if wb.metadata.creation_timestamp
                else None
            ),
        }

        if wb.resources:
            result["resources"] = {
                "cpu_request": wb.resources.cpu_request,
                "cpu_limit": wb.resources.cpu_limit,
                "memory_request": wb.resources.memory_request,
                "memory_limit": wb.resources.memory_limit,
                "gpu_request": wb.resources.gpu_request,
                "gpu_limit": wb.resources.gpu_limit,
            }

        if verbosity == Verbosity.FULL:
            result["labels"] = wb.metadata.labels
            result["annotations"] = wb.metadata.annotations
            if wb.conditions:
                result["conditions"] = [
                    {
                        "type": c.type,
                        "status": c.status,
                        "reason": c.reason,
                        "message": c.message,
                    }
                    for c in wb.conditions
                ]

        return result

    @staticmethod
    def project_list_item(p: Any, verbosity: Verbosity = Verbosity.STANDARD) -> dict[str, Any]:
        """Format a project for list responses.

        Args:
            p: Project model instance.
            verbosity: Response verbosity level.

        Returns:
            Formatted project dict.
        """
        if verbosity == Verbosity.MINIMAL:
            return {
                "name": p.metadata.name,
                "status": p.status.value,
            }

        result: dict[str, Any] = {
            "name": p.metadata.name,
            "display_name": p.display_name,
            "description": p.description,
            "requester": p.requester,
            "is_modelmesh_enabled": p.is_modelmesh_enabled,
            "status": p.status.value,
            "created": (
                p.metadata.creation_timestamp.isoformat() if p.metadata.creation_timestamp else None
            ),
        }

        if verbosity == Verbosity.FULL:
            result["labels"] = p.metadata.labels
            result["annotations"] = p.metadata.annotations

        return result

    @staticmethod
    def project_detail(
        project: Any,
        verbosity: Verbosity = Verbosity.FULL,
    ) -> dict[str, Any]:
        """Format a project for detail responses.

        Args:
            project: Project model instance.
            verbosity: Response verbosity level.

        Returns:
            Formatted project dict.
        """
        if verbosity == Verbosity.MINIMAL:
            result: dict[str, Any] = {
                "name": project.metadata.name,
                "status": project.status.value,
            }
            if project.resource_summary:
                result["resources"] = {
                    "workbenches": project.resource_summary.workbenches,
                    "models": project.resource_summary.models,
                }
            return result

        result = {
            "name": project.metadata.name,
            "display_name": project.display_name,
            "description": project.description,
            "requester": project.requester,
            "is_modelmesh_enabled": project.is_modelmesh_enabled,
            "status": project.status.value,
            "created": (
                project.metadata.creation_timestamp.isoformat()
                if project.metadata.creation_timestamp
                else None
            ),
        }

        if project.resource_summary:
            result["resources"] = {
                "workbenches": project.resource_summary.workbenches,
                "workbenches_running": project.resource_summary.workbenches_running,
                "models": project.resource_summary.models,
                "models_ready": project.resource_summary.models_ready,
                "pipelines": project.resource_summary.pipelines,
                "data_connections": project.resource_summary.data_connections,
                "storage": project.resource_summary.storage,
            }

        if verbosity == Verbosity.FULL:
            result["labels"] = project.metadata.labels
            result["annotations"] = project.metadata.annotations

        return result

    @staticmethod
    def inference_service_list_item(
        isvc: dict[str, Any],
        verbosity: Verbosity = Verbosity.STANDARD,
    ) -> dict[str, Any]:
        """Format an inference service for list responses.

        Note: InferenceClient.list_inference_services returns dicts, not models.

        Args:
            isvc: InferenceService dict from client.
            verbosity: Response verbosity level.

        Returns:
            Formatted inference service dict.
        """
        if verbosity == Verbosity.MINIMAL:
            return {
                "name": isvc.get("name"),
                "status": isvc.get("status"),
            }

        # STANDARD returns what client already returns
        if verbosity == Verbosity.STANDARD:
            return isvc

        # FULL - same as STANDARD since list already returns full data
        return isvc

    @staticmethod
    def inference_service_detail(
        isvc: Any,
        verbosity: Verbosity = Verbosity.FULL,
    ) -> dict[str, Any]:
        """Format an inference service for detail responses.

        Args:
            isvc: InferenceService model instance.
            verbosity: Response verbosity level.

        Returns:
            Formatted inference service dict.
        """
        if verbosity == Verbosity.MINIMAL:
            return {
                "name": isvc.metadata.name,
                "namespace": isvc.metadata.namespace,
                "status": isvc.status.value,
                "url": isvc.url,
            }

        result: dict[str, Any] = {
            "name": isvc.metadata.name,
            "namespace": isvc.metadata.namespace,
            "display_name": isvc.display_name,
            "runtime": isvc.runtime,
            "model_format": isvc.model_format,
            "storage_uri": isvc.storage_uri,
            "status": isvc.status.value,
            "url": isvc.url,
            "internal_url": isvc.internal_url,
            "created": (
                isvc.metadata.creation_timestamp.isoformat()
                if isvc.metadata.creation_timestamp
                else None
            ),
        }

        if isvc.resources:
            result["resources"] = {
                "cpu_request": isvc.resources.cpu_request,
                "cpu_limit": isvc.resources.cpu_limit,
                "memory_request": isvc.resources.memory_request,
                "memory_limit": isvc.resources.memory_limit,
                "gpu_request": isvc.resources.gpu_request,
                "gpu_limit": isvc.resources.gpu_limit,
            }

        if verbosity == Verbosity.FULL:
            result["labels"] = isvc.metadata.labels
            result["annotations"] = isvc.metadata.annotations
            if isvc.conditions:
                result["conditions"] = [
                    {
                        "type": c.type,
                        "status": c.status,
                        "reason": c.reason,
                        "message": c.message,
                    }
                    for c in isvc.conditions
                ]

        return result

    @staticmethod
    def storage_list_item(
        storage: dict[str, Any],
        verbosity: Verbosity = Verbosity.STANDARD,
    ) -> dict[str, Any]:
        """Format a storage PVC for list responses.

        Args:
            storage: Storage dict from client.
            verbosity: Response verbosity level.

        Returns:
            Formatted storage dict.
        """
        if verbosity == Verbosity.MINIMAL:
            return {
                "name": storage.get("name"),
                "status": storage.get("status"),
            }

        # STANDARD and FULL return what client returns
        return storage

    @staticmethod
    def data_connection_list_item(
        conn: dict[str, Any],
        verbosity: Verbosity = Verbosity.STANDARD,
    ) -> dict[str, Any]:
        """Format a data connection for list responses.

        Args:
            conn: Data connection dict from client.
            verbosity: Response verbosity level.

        Returns:
            Formatted data connection dict.
        """
        if verbosity == Verbosity.MINIMAL:
            return {
                "name": conn.get("name"),
                "type": conn.get("type"),
            }

        return conn

    @staticmethod
    def training_job_list_item(
        job: Any,
        verbosity: Verbosity = Verbosity.STANDARD,
    ) -> dict[str, Any]:
        """Format a training job for list responses.

        Args:
            job: TrainJob model instance.
            verbosity: Response verbosity level.

        Returns:
            Formatted training job dict.
        """
        if verbosity == Verbosity.MINIMAL:
            result: dict[str, Any] = {
                "name": job.name,
                "status": job.status.value,
            }
            if job.progress:
                result["progress_percent"] = round(job.progress.progress_percent, 1)
            return result

        # STANDARD
        job_info: dict[str, Any] = {
            "name": job.name,
            "status": job.status.value,
            "model_id": job.model_id,
            "dataset_id": job.dataset_id,
            "num_nodes": job.num_nodes,
            "created": job.creation_timestamp,
        }

        if job.progress:
            job_info["progress"] = {
                "state": job.progress.state.value,
                "current_epoch": job.progress.current_epoch,
                "total_epochs": job.progress.total_epochs,
                "progress_percent": round(job.progress.progress_percent, 1),
            }

        if verbosity == Verbosity.FULL:
            job_info["gpus_per_node"] = job.gpus_per_node
            job_info["runtime_ref"] = job.runtime_ref
            job_info["checkpoint_dir"] = job.checkpoint_dir
            if job.progress:
                job_info["progress"]["current_step"] = job.progress.current_step
                job_info["progress"]["total_steps"] = job.progress.total_steps
                job_info["progress"]["loss"] = job.progress.loss
                job_info["progress"]["learning_rate"] = job.progress.learning_rate

        return job_info

    @staticmethod
    def training_job_detail(
        job: Any,
        verbosity: Verbosity = Verbosity.FULL,
    ) -> dict[str, Any]:
        """Format a training job for detail responses.

        Args:
            job: TrainJob model instance.
            verbosity: Response verbosity level.

        Returns:
            Formatted training job dict.
        """
        if verbosity == Verbosity.MINIMAL:
            result: dict[str, Any] = {
                "name": job.name,
                "namespace": job.namespace,
                "status": job.status.value,
            }
            if job.progress:
                result["progress_percent"] = round(job.progress.progress_percent, 1)
            return result

        result = {
            "name": job.name,
            "namespace": job.namespace,
            "status": job.status.value,
            "model_id": job.model_id,
            "dataset_id": job.dataset_id,
            "num_nodes": job.num_nodes,
            "gpus_per_node": job.gpus_per_node,
            "runtime_ref": job.runtime_ref,
            "checkpoint_dir": job.checkpoint_dir,
            "created": job.creation_timestamp,
        }

        if job.progress:
            result["progress"] = {
                "state": job.progress.state.value,
                "current_epoch": job.progress.current_epoch,
                "total_epochs": job.progress.total_epochs,
                "current_step": job.progress.current_step,
                "total_steps": job.progress.total_steps,
                "loss": job.progress.loss,
                "learning_rate": job.progress.learning_rate,
                "throughput": job.progress.throughput,
                "progress_percent": round(job.progress.progress_percent, 1),
                "progress_bar": job.progress.progress_bar(),
                "eta_seconds": job.progress.eta_seconds,
            }

            if verbosity == Verbosity.FULL and job.progress:
                result["progress"]["gradient_norm"] = job.progress.gradient_norm

        return result
