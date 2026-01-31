"""Pydantic models for Data Science Pipelines."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.models.common import Condition, ResourceMetadata


class PipelineServerStatus(str, Enum):
    """DSPA status values."""

    READY = "Ready"
    CREATING = "Creating"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


class PipelineServer(BaseModel):
    """Data Science Pipelines Application (DSPA) representation."""

    metadata: ResourceMetadata
    status: PipelineServerStatus = Field(PipelineServerStatus.UNKNOWN, description="Server status")
    api_server_ready: bool = Field(False, description="API server readiness")
    persistence_agent_ready: bool = Field(False, description="Persistence agent readiness")
    scheduled_workflow_ready: bool = Field(
        False, description="Scheduled workflow controller readiness"
    )
    database_available: bool = Field(False, description="Database availability")
    object_store_available: bool = Field(False, description="Object store availability")
    conditions: list[Condition] = Field(default_factory=list, description="Status conditions")
    api_server_url: str | None = Field(None, description="Pipeline API server URL")

    @classmethod
    def from_dspa_cr(cls, dspa: Any) -> "PipelineServer":
        """Create from DSPA CR."""
        metadata = dspa.metadata
        status_obj = getattr(dspa, "status", None)

        # Determine overall status
        server_status = cls._determine_status(status_obj)

        # Parse conditions
        conditions = []
        api_server_ready = False
        persistence_agent_ready = False
        scheduled_workflow_ready = False
        database_available = False
        object_store_available = False

        if status_obj:
            for cond in getattr(status_obj, "conditions", []) or []:
                conditions.append(Condition.from_k8s_condition(cond))
                cond_type = getattr(cond, "type", "")
                cond_status = getattr(cond, "status", "") == "True"

                if cond_type == "APIServerReady":
                    api_server_ready = cond_status
                elif cond_type == "PersistenceAgentReady":
                    persistence_agent_ready = cond_status
                elif cond_type == "ScheduledWorkflowReady":
                    scheduled_workflow_ready = cond_status
                elif cond_type == "DatabaseAvailable":
                    database_available = cond_status
                elif cond_type == "ObjectStoreAvailable":
                    object_store_available = cond_status

        return cls(
            metadata=ResourceMetadata.from_k8s_metadata(
                metadata,
                kind="DataSciencePipelinesApplication",
                api_version="datasciencepipelinesapplications.opendatahub.io/v1alpha1",
            ),
            status=server_status,
            api_server_ready=api_server_ready,
            persistence_agent_ready=persistence_agent_ready,
            scheduled_workflow_ready=scheduled_workflow_ready,
            database_available=database_available,
            object_store_available=object_store_available,
            conditions=conditions,
            api_server_url=None,  # URL would be derived from route
        )

    @staticmethod
    def _determine_status(status_obj: Any) -> PipelineServerStatus:
        """Determine server status from status object."""
        if not status_obj:
            return PipelineServerStatus.UNKNOWN

        conditions = getattr(status_obj, "conditions", []) or []
        ready_count = 0
        has_failed = False

        for cond in conditions:
            if getattr(cond, "status", "") == "True":
                ready_count += 1
            if getattr(cond, "reason", "") == "Failed":
                has_failed = True

        if has_failed:
            return PipelineServerStatus.FAILED
        if ready_count >= 5:  # All major components ready
            return PipelineServerStatus.READY
        if ready_count > 0:
            return PipelineServerStatus.CREATING

        return PipelineServerStatus.UNKNOWN


class PipelineServerCreate(BaseModel):
    """Request model for creating a pipeline server (DSPA)."""

    namespace: str = Field(..., description="Project namespace")
    object_storage_secret: str = Field(..., description="Name of secret with S3 credentials")
    object_storage_bucket: str = Field(..., description="S3 bucket for pipeline data")
    object_storage_endpoint: str = Field(..., description="S3 endpoint URL")
    object_storage_region: str = Field("us-east-1", description="S3 region")
