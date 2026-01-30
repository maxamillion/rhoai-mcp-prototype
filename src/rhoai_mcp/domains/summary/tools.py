"""MCP Tools for context-efficient cluster and project summaries.

These tools are optimized for AI agent context windows, providing compact
overviews that reduce token usage significantly compared to full resource listings.
"""

from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.summary.models import (
    ClusterSummary,
    MultiResourceStatusResult,
    ProjectSummary,
    ResourceNameList,
    ResourceStatus,
)

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register summary tools with the MCP server."""

    @mcp.tool()
    def cluster_summary() -> ClusterSummary:
        """Get a compact cluster overview optimized for context windows.

        Returns a minimal summary of the entire cluster including project count,
        workbench status, model deployment status, and resource availability.
        This is the most efficient way to get an overview before drilling down.

        Returns:
            Compact cluster summary with counts and status indicators.
        """
        from rhoai_mcp.domains.projects.client import ProjectClient

        k8s = server.k8s
        project_client = ProjectClient(k8s)

        # Get all projects
        projects = project_client.list_projects()
        project_names = [p.metadata.name for p in projects]

        # Aggregate workbench and model counts across projects
        total_workbenches = 0
        running_workbenches = 0
        total_models = 0
        ready_models = 0
        total_pipelines = 0
        total_storage = 0
        total_connections = 0

        for project in projects:
            if project.resource_summary:
                summary = project.resource_summary
                total_workbenches += summary.workbenches
                running_workbenches += summary.workbenches_running
                total_models += summary.models
                ready_models += summary.models_ready
                total_pipelines += summary.pipelines
                total_storage += summary.storage
                total_connections += summary.data_connections

        return ClusterSummary(
            projects=len(projects),
            project_names=project_names,
            workbenches=f"{running_workbenches}/{total_workbenches} running",
            models=f"{ready_models}/{total_models} ready",
            pipelines=total_pipelines,
            storage=total_storage,
            data_connections=total_connections,
        )

    @mcp.tool()
    def project_summary(namespace: str) -> ProjectSummary:
        """Get a compact project status optimized for context windows.

        Returns a minimal summary of a specific project including workbench,
        model, and pipeline status without full resource details.

        Args:
            namespace: The project (namespace) name.

        Returns:
            Compact project summary with resource counts.
        """
        from rhoai_mcp.domains.projects.client import ProjectClient

        k8s = server.k8s
        project_client = ProjectClient(k8s)

        # Get project with resource summary
        project = project_client.get_project(namespace, include_summary=True)

        result = ProjectSummary(
            name=project.metadata.name,
            display_name=project.display_name,
            status=project.status.value,
        )

        if project.resource_summary:
            summary = project.resource_summary
            result = ProjectSummary(
                name=project.metadata.name,
                display_name=project.display_name,
                status=project.status.value,
                workbenches=f"{summary.workbenches_running}/{summary.workbenches}",
                models=f"{summary.models_ready}/{summary.models}",
                pipelines=summary.pipelines,
                storage=summary.storage,
                data_connections=summary.data_connections,
            )

        return result

    @mcp.tool()
    def resource_status(
        resource_type: str,
        name: str,
        namespace: str,
    ) -> ResourceStatus:
        """Get quick status check for any resource.

        Provides minimal status information for a specific resource without
        full details. Use this for quick health checks.

        Args:
            resource_type: Type of resource - "workbench", "model", "pipeline",
                "storage", or "connection".
            name: The resource name.
            namespace: The project (namespace) name.

        Returns:
            Minimal status information for the resource.
        """
        return _get_resource_status(server, resource_type, name, namespace)

    @mcp.tool()
    def list_resource_names(
        resource_type: str,
        namespace: str | None = None,
    ) -> ResourceNameList:
        """List just resource names without full metadata.

        Returns only the names of resources, which is the most token-efficient
        way to get a list of available resources.

        Args:
            resource_type: Type of resource - "projects", "workbenches", "models",
                "pipelines", "storage", "connections", or "training_jobs".
            namespace: The project (namespace) name. Required for all types
                except "projects".

        Returns:
            List of resource names.
        """
        return _list_resource_names(server, resource_type, namespace)

    @mcp.tool()
    def multi_resource_status(
        namespace: str,
        resources: list[dict[str, str]],
    ) -> MultiResourceStatusResult:
        """Get status for multiple resources in a single call.

        Efficiently retrieves status for multiple resources, reducing the
        number of tool calls needed.

        Args:
            namespace: The project (namespace) name.
            resources: List of resources to check, each with "type" and "name".
                Example: [{"type": "workbench", "name": "my-wb"},
                         {"type": "model", "name": "my-model"}]

        Returns:
            Status for each requested resource.
        """
        results: list[ResourceStatus] = []

        for resource in resources:
            res_type = resource.get("type", "")
            res_name = resource.get("name", "")

            if not res_type or not res_name:
                results.append(
                    ResourceStatus(
                        name=res_name or "unknown",
                        type=res_type or "unknown",
                        error="Both type and name are required",
                    )
                )
                continue

            try:
                status = _get_resource_status(server, res_type, res_name, namespace)
                results.append(status)
            except Exception as e:
                results.append(
                    ResourceStatus(
                        name=res_name,
                        type=res_type,
                        error=str(e),
                    )
                )

        return MultiResourceStatusResult(
            namespace=namespace,
            results=results,
            success_count=sum(1 for r in results if r.error is None),
            error_count=sum(1 for r in results if r.error is not None),
        )

    @mcp.tool()
    def explore_cluster(
        include_resources: bool = True,
        include_health: bool = True,
    ) -> dict:
        """Complete cluster exploration in one call.

        Returns comprehensive overview of the entire RHOAI cluster including
        all projects, their resources, and any issues detected. Use this as
        the first tool when exploring an unfamiliar cluster.

        Args:
            include_resources: Include resource counts per project.
            include_health: Include health checks and issue detection.

        Returns:
            Complete cluster exploration with:
            - cluster: Overall cluster info (GPU availability, node count)
            - projects: List of projects with resource summaries
            - recommendations: Suggested next actions
        """
        from rhoai_mcp.domains.projects.client import ProjectClient

        k8s = server.k8s
        project_client = ProjectClient(k8s)

        # Get cluster resources
        cluster_info: dict = {"nodes": 0, "gpu_available": 0, "gpu_type": None}
        try:
            from rhoai_mcp.domains.training.client import TrainingClient

            training_client = TrainingClient(k8s)
            resources = training_client.get_cluster_resources()
            cluster_info["nodes"] = resources.node_count
            if resources.gpu_info:
                cluster_info["gpu_available"] = resources.gpu_info.available
                cluster_info["gpu_type"] = resources.gpu_info.type
        except Exception:
            pass

        # Get all projects with their resources
        projects = project_client.list_projects()
        project_summaries = []
        all_issues: list[str] = []

        for project in projects:
            proj_info: dict = {
                "name": project.metadata.name,
                "display_name": project.display_name,
                "status": project.status.value,
            }

            if include_resources and project.resource_summary:
                summary = project.resource_summary
                proj_info["workbenches"] = {
                    "running": summary.workbenches_running,
                    "total": summary.workbenches,
                }
                proj_info["models"] = {
                    "ready": summary.models_ready,
                    "total": summary.models,
                }
                proj_info["training_jobs"] = {
                    "active": 0,  # Would need to query training client
                    "total": 0,
                }
                proj_info["storage"] = summary.storage
                proj_info["connections"] = summary.data_connections

            if include_health:
                proj_issues = []
                if project.resource_summary:
                    s = project.resource_summary
                    if s.workbenches > 0 and s.workbenches_running == 0:
                        proj_issues.append("All workbenches stopped")
                    if s.models > 0 and s.models_ready == 0:
                        proj_issues.append("No models ready")
                if proj_issues:
                    proj_info["issues"] = proj_issues
                    all_issues.extend(
                        [f"{project.metadata.name}: {i}" for i in proj_issues]
                    )

            project_summaries.append(proj_info)

        # Generate recommendations
        recommendations = []
        if not cluster_info.get("gpu_available"):
            recommendations.append("No GPUs detected - training jobs may not schedule")
        if not projects:
            recommendations.append("No Data Science projects found - create one to get started")
        if all_issues:
            recommendations.append(f"{len(all_issues)} issue(s) detected across projects")

        return {
            "cluster": cluster_info,
            "projects": project_summaries,
            "project_count": len(projects),
            "issues": all_issues if all_issues else None,
            "recommendations": recommendations if recommendations else None,
        }

    @mcp.tool()
    def diagnose_resource(
        resource_type: str,
        name: str,
        namespace: str,
    ) -> dict:
        """Comprehensive diagnostic for any resource in one call.

        Gathers all relevant information about a resource including status,
        events, logs (if applicable), related resources, and detected issues.
        Use this when troubleshooting a problematic resource.

        Args:
            resource_type: Type of resource - "workbench", "model",
                "training_job", or "pipeline".
            name: The resource name.
            namespace: The project (namespace) name.

        Returns:
            Complete diagnostic with:
            - resource: Resource details
            - status_summary: Human-readable status
            - events: Kubernetes events
            - logs: Container logs (if applicable)
            - issues_detected: Identified problems
            - suggested_fixes: Recommended actions
            - related_resources: Other relevant resources
        """
        resource_type = resource_type.lower()
        result: dict = {
            "resource_type": resource_type,
            "name": name,
            "namespace": namespace,
            "resource": None,
            "status_summary": "Unknown",
            "events": [],
            "logs": None,
            "issues_detected": [],
            "suggested_fixes": [],
            "related_resources": [],
        }

        if resource_type in ("workbench", "notebook"):
            result.update(_diagnose_workbench(server, name, namespace))
        elif resource_type in ("model", "inference", "inferenceservice"):
            result.update(_diagnose_model(server, name, namespace))
        elif resource_type in ("training", "training_job", "trainjob"):
            result.update(_diagnose_training_job(server, name, namespace))
        elif resource_type in ("pipeline", "dspa"):
            result.update(_diagnose_pipeline(server, name, namespace))
        else:
            result["issues_detected"].append(f"Unknown resource type: {resource_type}")
            result["suggested_fixes"].append(
                "Use one of: workbench, model, training_job, pipeline"
            )

        return result

    @mcp.tool()
    def get_resource(
        resource_type: str,
        name: str,
        namespace: str,
        verbosity: str = "standard",
    ) -> dict:
        """Get any RHOAI resource by type.

        Generic resource getter that dispatches to the appropriate domain.
        Use this instead of domain-specific get tools for simpler workflows.

        Args:
            resource_type: Type of resource - "workbench", "model",
                "training_job", "connection", "storage", or "pipeline".
            name: The resource name.
            namespace: The project (namespace) name.
            verbosity: Response detail level - "minimal", "standard", or "full".

        Returns:
            Resource details at the requested verbosity level.
        """
        resource_type = resource_type.lower()

        if resource_type in ("workbench", "notebook"):
            from rhoai_mcp.domains.notebooks.client import NotebookClient
            from rhoai_mcp.utils.response import ResponseBuilder, Verbosity

            nb_client = NotebookClient(server.k8s)
            wb = nb_client.get_workbench(name, namespace)
            v = Verbosity.from_str(verbosity)
            return ResponseBuilder.workbench_detail(wb, v)

        if resource_type in ("model", "inference", "inferenceservice"):
            from rhoai_mcp.domains.inference.client import InferenceClient
            from rhoai_mcp.utils.response import ResponseBuilder, Verbosity

            inf_client = InferenceClient(server.k8s)
            isvc = inf_client.get_inference_service(name, namespace)
            v = Verbosity.from_str(verbosity)
            return ResponseBuilder.inference_service_detail(isvc, v)

        if resource_type in ("training", "training_job", "trainjob"):
            from rhoai_mcp.domains.training.client import TrainingClient
            from rhoai_mcp.utils.response import ResponseBuilder, Verbosity

            tr_client = TrainingClient(server.k8s)
            job = tr_client.get_training_job(namespace, name)
            v = Verbosity.from_str(verbosity)
            return ResponseBuilder.training_job_detail(job, v)

        if resource_type in ("connection", "data_connection"):
            from rhoai_mcp.domains.connections.client import ConnectionClient

            conn_client = ConnectionClient(server.k8s)
            conn = conn_client.get_data_connection(name, namespace, mask_secrets=True)
            return dict(conn.model_dump())

        if resource_type in ("storage", "pvc"):
            from rhoai_mcp.domains.storage.client import StorageClient

            st_client = StorageClient(server.k8s)
            storage_list = st_client.list_storage(namespace)
            for s in storage_list:
                if s.get("name") == name:
                    return dict(s)
            return {"error": f"Storage '{name}' not found in namespace '{namespace}'"}

        return {
            "error": f"Unknown resource type: {resource_type}",
            "valid_types": [
                "workbench",
                "model",
                "training_job",
                "connection",
                "storage",
            ],
        }

    @mcp.tool()
    def list_resources(
        resource_type: str,
        namespace: str | None = None,
        limit: int | None = None,
        verbosity: str = "minimal",  # noqa: ARG001
    ) -> dict:
        """List any RHOAI resource type.

        Generic resource lister that dispatches to the appropriate domain.
        Use this instead of domain-specific list tools for simpler workflows.

        Args:
            resource_type: Type of resource - "projects", "workbenches", "models",
                "training_jobs", "connections", "storage", or "pipelines".
            namespace: The project (namespace) name. Required except for "projects".
            limit: Maximum number of items to return.
            verbosity: Response detail level - "minimal", "standard", or "full".

        Returns:
            Paginated list of resources.
        """
        names_result = _list_resource_names(server, resource_type, namespace)

        if names_result.error:
            return {"error": names_result.error, "valid_types": names_result.valid_types}

        # Apply limit if specified
        names = names_result.names
        if limit and limit < len(names):
            names = names[:limit]

        return {
            "type": names_result.type,
            "namespace": namespace,
            "count": len(names),
            "total": names_result.count,
            "names": names,
        }

    @mcp.tool()
    def manage_resource(
        action: str,
        resource_type: str,
        name: str,
        namespace: str,
        confirm: bool = False,
    ) -> dict:
        """Perform lifecycle actions on any resource type.

        Generic resource management that dispatches to the appropriate domain.
        Supports start, stop, suspend, resume, and delete operations.

        Args:
            action: Action to perform - "start", "stop", "suspend", "resume", or "delete".
            resource_type: Type of resource - "workbench", "model", or "training_job".
            name: The resource name.
            namespace: The project (namespace) name.
            confirm: Required for delete operations.

        Returns:
            Action result.
        """
        action = action.lower()
        resource_type = resource_type.lower()

        valid_actions = ["start", "stop", "suspend", "resume", "delete"]
        if action not in valid_actions:
            return {"error": f"Invalid action: {action}", "valid_actions": valid_actions}

        # Check delete confirmation
        if action == "delete":
            allowed, reason = server.config.is_operation_allowed("delete")
            if not allowed:
                return {"error": reason}
            if not confirm:
                return {
                    "error": "Deletion not confirmed",
                    "message": f"To delete {resource_type} '{name}', set confirm=True.",
                }

        # Dispatch to appropriate handler
        if resource_type in ("workbench", "notebook"):
            return _manage_workbench(server, action, name, namespace)
        elif resource_type in ("model", "inference", "inferenceservice"):
            return _manage_model(server, action, name, namespace)
        elif resource_type in ("training", "training_job", "trainjob"):
            return _manage_training_job(server, action, name, namespace)

        return {
            "error": f"Resource type '{resource_type}' does not support lifecycle management",
            "supported_types": ["workbench", "model", "training_job"],
        }


def _diagnose_workbench(
    server: "RHOAIServer", name: str, namespace: str
) -> dict:
    """Diagnose a workbench."""
    from rhoai_mcp.domains.notebooks.client import NotebookClient

    result: dict = {
        "resource": None,
        "status_summary": "Unknown",
        "events": [],
        "logs": None,
        "issues_detected": [],
        "suggested_fixes": [],
        "related_resources": [],
    }

    try:
        client = NotebookClient(server.k8s)
        wb = client.get_workbench(name, namespace)
        result["resource"] = {
            "name": wb.metadata.name,
            "status": wb.status.value,
            "image": wb.image,
            "url": wb.url,
        }
        result["status_summary"] = f"Workbench is {wb.status.value}"

        # Check for issues
        if wb.status.value == "Stopped":
            result["issues_detected"].append("Workbench is stopped")
            result["suggested_fixes"].append("Use start_workbench() to start it")
        elif wb.status.value == "Error":
            result["issues_detected"].append("Workbench is in error state")
            result["suggested_fixes"].append("Check pod events for details")

        # Get related storage
        result["related_resources"].append(f"PVC: {name}-pvc")

    except Exception as e:
        result["issues_detected"].append(f"Failed to get workbench: {e}")

    return result


def _diagnose_model(
    server: "RHOAIServer", name: str, namespace: str
) -> dict:
    """Diagnose a model deployment."""
    from rhoai_mcp.domains.inference.client import InferenceClient

    result: dict = {
        "resource": None,
        "status_summary": "Unknown",
        "events": [],
        "logs": None,
        "issues_detected": [],
        "suggested_fixes": [],
        "related_resources": [],
    }

    try:
        client = InferenceClient(server.k8s)
        isvc = client.get_inference_service(name, namespace)
        result["resource"] = {
            "name": isvc.metadata.name,
            "status": isvc.status.value,
            "runtime": isvc.runtime,
            "url": isvc.url,
        }
        result["status_summary"] = f"Model is {isvc.status.value}"

        if isvc.status.value != "Ready":
            result["issues_detected"].append(f"Model not ready: {isvc.status.value}")
            result["suggested_fixes"].append("Wait for model to become ready")
            result["suggested_fixes"].append("Check pod events for scheduling issues")

    except Exception as e:
        result["issues_detected"].append(f"Failed to get model: {e}")

    return result


def _diagnose_training_job(
    server: "RHOAIServer", name: str, namespace: str
) -> dict:
    """Diagnose a training job."""
    from rhoai_mcp.domains.training.client import TrainingClient

    result: dict = {
        "resource": None,
        "status_summary": "Unknown",
        "events": [],
        "logs": None,
        "issues_detected": [],
        "suggested_fixes": [],
        "related_resources": [],
    }

    try:
        client = TrainingClient(server.k8s)
        job = client.get_training_job(namespace, name)

        result["resource"] = {
            "name": job.name,
            "status": job.status.value,
            "model_id": job.model_id,
            "dataset_id": job.dataset_id,
        }
        result["status_summary"] = f"Training job is {job.status.value}"

        if job.progress:
            result["resource"]["progress"] = f"{round(job.progress.progress_percent, 1)}%"

        # Get events
        events = client.get_job_events(namespace, name)
        result["events"] = events

        # Get logs
        try:
            logs = client.get_training_logs(namespace, name, tail_lines=50)
            result["logs"] = logs
        except Exception:
            pass

        # Analyze for issues
        if job.status.value == "Failed":
            result["issues_detected"].append("Training job failed")
            for event in events:
                if event.get("type") == "Warning":
                    reason = event.get("reason", "")
                    if "OOMKilled" in reason:
                        result["issues_detected"].append("Out of memory")
                        result["suggested_fixes"].append("Reduce batch size or add memory")
                    if "FailedScheduling" in reason:
                        result["issues_detected"].append("Pod scheduling failed")
                        result["suggested_fixes"].append("Check GPU availability")

    except Exception as e:
        result["issues_detected"].append(f"Failed to get training job: {e}")

    return result


def _diagnose_pipeline(
    server: "RHOAIServer", name: str, namespace: str  # noqa: ARG001
) -> dict:
    """Diagnose a pipeline server."""
    from rhoai_mcp.domains.pipelines.client import PipelineClient

    result: dict = {
        "resource": None,
        "status_summary": "Unknown",
        "events": [],
        "logs": None,
        "issues_detected": [],
        "suggested_fixes": [],
        "related_resources": [],
    }

    try:
        client = PipelineClient(server.k8s)
        pipeline = client.get_pipeline_server(namespace)

        if pipeline is None:
            result["issues_detected"].append("No pipeline server configured")
            result["suggested_fixes"].append("Use create_pipeline_server() to set one up")
        else:
            result["resource"] = pipeline
            result["status_summary"] = f"Pipeline server is {pipeline.get('status', 'Unknown')}"

    except Exception as e:
        result["issues_detected"].append(f"Failed to get pipeline: {e}")

    return result


def _manage_workbench(
    server: "RHOAIServer", action: str, name: str, namespace: str
) -> dict:
    """Manage workbench lifecycle."""
    from rhoai_mcp.domains.notebooks.client import NotebookClient

    client = NotebookClient(server.k8s)

    if action == "start":
        wb = client.start_workbench(name, namespace)
        return {"success": True, "action": "started", "name": name, "status": wb.status.value}
    elif action == "stop":
        wb = client.stop_workbench(name, namespace)
        return {"success": True, "action": "stopped", "name": name, "status": wb.status.value}
    elif action == "delete":
        client.delete_workbench(name, namespace)
        return {"success": True, "action": "deleted", "name": name}
    else:
        return {"error": f"Action '{action}' not supported for workbenches"}


def _manage_model(
    server: "RHOAIServer", action: str, name: str, namespace: str
) -> dict:
    """Manage model lifecycle."""
    from rhoai_mcp.domains.inference.client import InferenceClient

    client = InferenceClient(server.k8s)

    if action == "delete":
        client.delete_inference_service(name, namespace)
        return {"success": True, "action": "deleted", "name": name}
    else:
        return {"error": f"Action '{action}' not supported for models (only delete)"}


def _manage_training_job(
    server: "RHOAIServer", action: str, name: str, namespace: str
) -> dict:
    """Manage training job lifecycle."""
    from rhoai_mcp.domains.training.client import TrainingClient

    client = TrainingClient(server.k8s)

    if action == "suspend":
        client.suspend_training_job(namespace, name)
        return {"success": True, "action": "suspended", "name": name}
    elif action == "resume":
        client.resume_training_job(namespace, name)
        return {"success": True, "action": "resumed", "name": name}
    elif action == "delete":
        client.delete_training_job(namespace, name)
        return {"success": True, "action": "deleted", "name": name}
    else:
        return {"error": f"Action '{action}' not supported for training jobs"}


def _get_resource_status(
    server: "RHOAIServer",
    resource_type: str,
    name: str,
    namespace: str,
) -> ResourceStatus:
    """Get status for a single resource (internal helper)."""
    resource_type = resource_type.lower()

    if resource_type in ("workbench", "notebook"):
        from rhoai_mcp.domains.notebooks.client import NotebookClient

        notebook_client = NotebookClient(server.k8s)
        wb = notebook_client.get_workbench(name, namespace)
        return ResourceStatus(
            name=wb.metadata.name,
            type="workbench",
            status=wb.status.value,
            url=wb.url,
        )

    if resource_type in ("model", "inferenceservice", "inference"):
        from rhoai_mcp.domains.inference.client import InferenceClient

        inference_client = InferenceClient(server.k8s)
        isvc = inference_client.get_inference_service(name, namespace)
        return ResourceStatus(
            name=isvc.metadata.name,
            type="model",
            status=isvc.status.value,
            url=isvc.url,
        )

    if resource_type in ("pipeline", "dspa"):
        from rhoai_mcp.domains.pipelines.client import PipelineClient

        pipeline_client = PipelineClient(server.k8s)
        result = pipeline_client.get_pipeline_server(namespace)
        if result is None:
            return ResourceStatus(
                name=name,
                type="pipeline",
                status="NotFound",
                message="No pipeline server configured",
            )
        return ResourceStatus(
            name=result.get("name", name),
            type="pipeline",
            status=result.get("status", "Unknown"),
        )

    if resource_type in ("storage", "pvc"):
        from rhoai_mcp.domains.storage.client import StorageClient

        storage_client = StorageClient(server.k8s)
        all_storage = storage_client.list_storage(namespace)
        for s in all_storage:
            if s.get("name") == name:
                return ResourceStatus(
                    name=name,
                    type="storage",
                    status=s.get("status", "Unknown"),
                    size=s.get("size"),
                )
        return ResourceStatus(
            name=name,
            type="storage",
            status="NotFound",
        )

    if resource_type in ("connection", "data_connection", "secret"):
        from rhoai_mcp.domains.connections.client import ConnectionClient

        connection_client = ConnectionClient(server.k8s)
        conn = connection_client.get_data_connection(name, namespace, mask_secrets=True)
        return ResourceStatus(
            name=conn.metadata.name,
            type="connection",
            status="Active",
            bucket=conn.aws_s3_bucket,
        )

    if resource_type in ("training", "trainjob", "train_job"):
        from rhoai_mcp.domains.training.client import TrainingClient

        training_client = TrainingClient(server.k8s)
        job = training_client.get_training_job(namespace, name)
        progress_str = None
        if job.progress:
            progress_str = f"{round(job.progress.progress_percent, 1)}%"
        return ResourceStatus(
            name=job.name,
            type="training",
            status=job.status.value,
            progress=progress_str,
        )

    return ResourceStatus(
        name=name,
        type=resource_type,
        status="Unknown",
        error=f"Unknown resource type: {resource_type}",
        valid_types=[
            "workbench",
            "model",
            "pipeline",
            "storage",
            "connection",
            "training",
        ],
    )


def _list_resource_names(
    server: "RHOAIServer",
    resource_type: str,
    namespace: str | None,
) -> ResourceNameList:
    """List resource names (internal helper)."""
    resource_type = resource_type.lower()

    if resource_type in ("project", "projects"):
        from rhoai_mcp.domains.projects.client import ProjectClient

        project_client = ProjectClient(server.k8s)
        projects = project_client.list_projects()
        return ResourceNameList(
            type="projects",
            count=len(projects),
            names=[p.metadata.name for p in projects],
        )

    if namespace is None:
        return ResourceNameList(
            type=resource_type,
            count=0,
            names=[],
            error=f"namespace is required for resource type '{resource_type}'",
        )

    if resource_type in ("workbench", "workbenches", "notebook", "notebooks"):
        from rhoai_mcp.domains.notebooks.client import NotebookClient

        notebook_client = NotebookClient(server.k8s)
        workbenches = notebook_client.list_workbenches(namespace)
        return ResourceNameList(
            type="workbenches",
            namespace=namespace,
            count=len(workbenches),
            names=[wb.metadata.name for wb in workbenches],
        )

    if resource_type in ("model", "models", "inferenceservice", "inferenceservices"):
        from rhoai_mcp.domains.inference.client import InferenceClient

        inference_client = InferenceClient(server.k8s)
        models = inference_client.list_inference_services(namespace)
        return ResourceNameList(
            type="models",
            namespace=namespace,
            count=len(models),
            names=[m.get("name") for m in models],
        )

    if resource_type in ("storage", "pvc", "pvcs"):
        from rhoai_mcp.domains.storage.client import StorageClient

        storage_client = StorageClient(server.k8s)
        storage = storage_client.list_storage(namespace)
        return ResourceNameList(
            type="storage",
            namespace=namespace,
            count=len(storage),
            names=[s.get("name") for s in storage],
        )

    if resource_type in ("connection", "connections", "data_connection", "data_connections"):
        from rhoai_mcp.domains.connections.client import ConnectionClient

        connection_client = ConnectionClient(server.k8s)
        connections = connection_client.list_data_connections(namespace)
        return ResourceNameList(
            type="connections",
            namespace=namespace,
            count=len(connections),
            names=[c.get("name") for c in connections],
        )

    if resource_type in (
        "training",
        "training_job",
        "training_jobs",
        "trainjob",
        "trainjobs",
    ):
        from rhoai_mcp.domains.training.client import TrainingClient

        training_client = TrainingClient(server.k8s)
        jobs = training_client.list_training_jobs(namespace)
        return ResourceNameList(
            type="training_jobs",
            namespace=namespace,
            count=len(jobs),
            names=[j.name for j in jobs],
        )

    if resource_type in ("runtime", "runtimes", "training_runtime", "training_runtimes"):
        from rhoai_mcp.domains.training.client import TrainingClient

        training_client = TrainingClient(server.k8s)
        runtimes = training_client.list_cluster_training_runtimes()
        if namespace:
            runtimes.extend(training_client.list_training_runtimes(namespace))
        return ResourceNameList(
            type="training_runtimes",
            count=len(runtimes),
            names=[r.name for r in runtimes],
        )

    return ResourceNameList(
        type=resource_type,
        count=0,
        names=[],
        error=f"Unknown resource type: {resource_type}",
        valid_types=[
            "projects",
            "workbenches",
            "models",
            "storage",
            "connections",
            "training_jobs",
            "runtimes",
        ],
    )
