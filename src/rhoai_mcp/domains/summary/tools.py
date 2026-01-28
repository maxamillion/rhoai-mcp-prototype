"""MCP Tools for context-efficient cluster and project summaries.

These tools are optimized for AI agent context windows, providing compact
overviews that reduce token usage significantly compared to full resource listings.
"""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register summary tools with the MCP server."""

    @mcp.tool()
    def cluster_summary() -> dict[str, Any]:
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

        return {
            "projects": len(projects),
            "project_names": project_names,
            "workbenches": f"{running_workbenches}/{total_workbenches} running",
            "models": f"{ready_models}/{total_models} ready",
            "pipelines": total_pipelines,
            "storage": total_storage,
            "data_connections": total_connections,
        }

    @mcp.tool()
    def project_summary(namespace: str) -> dict[str, Any]:
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

        result: dict[str, Any] = {
            "name": project.metadata.name,
            "display_name": project.display_name,
            "status": project.status.value,
        }

        if project.resource_summary:
            summary = project.resource_summary
            result["workbenches"] = f"{summary.workbenches_running}/{summary.workbenches}"
            result["models"] = f"{summary.models_ready}/{summary.models}"
            result["pipelines"] = summary.pipelines
            result["storage"] = summary.storage
            result["data_connections"] = summary.data_connections

        return result

    @mcp.tool()
    def resource_status(
        resource_type: str,
        name: str,
        namespace: str,
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
        results: list[dict[str, Any]] = []

        for resource in resources:
            res_type = resource.get("type", "")
            res_name = resource.get("name", "")

            if not res_type or not res_name:
                results.append(
                    {
                        "name": res_name or "unknown",
                        "type": res_type or "unknown",
                        "error": "Both type and name are required",
                    }
                )
                continue

            try:
                status = _get_resource_status(server, res_type, res_name, namespace)
                results.append(status)
            except Exception as e:
                results.append(
                    {
                        "name": res_name,
                        "type": res_type,
                        "error": str(e),
                    }
                )

        return {
            "namespace": namespace,
            "results": results,
            "success_count": sum(1 for r in results if "error" not in r),
            "error_count": sum(1 for r in results if "error" in r),
        }


def _get_resource_status(
    server: "RHOAIServer",
    resource_type: str,
    name: str,
    namespace: str,
) -> dict[str, Any]:
    """Get status for a single resource (internal helper)."""
    resource_type = resource_type.lower()

    if resource_type in ("workbench", "notebook"):
        from rhoai_mcp.domains.notebooks.client import NotebookClient

        notebook_client = NotebookClient(server.k8s)
        wb = notebook_client.get_workbench(name, namespace)
        return {
            "name": wb.metadata.name,
            "type": "workbench",
            "status": wb.status.value,
            "url": wb.url,
        }

    if resource_type in ("model", "inferenceservice", "inference"):
        from rhoai_mcp.domains.inference.client import InferenceClient

        inference_client = InferenceClient(server.k8s)
        isvc = inference_client.get_inference_service(name, namespace)
        return {
            "name": isvc.metadata.name,
            "type": "model",
            "status": isvc.status.value,
            "url": isvc.url,
        }

    if resource_type in ("pipeline", "dspa"):
        from rhoai_mcp.domains.pipelines.client import PipelineClient

        pipeline_client = PipelineClient(server.k8s)
        result = pipeline_client.get_pipeline_server(namespace)
        if result is None:
            return {
                "name": name,
                "type": "pipeline",
                "status": "NotFound",
                "message": "No pipeline server configured",
            }
        return {
            "name": result.get("name", name),
            "type": "pipeline",
            "status": result.get("status", "Unknown"),
        }

    if resource_type in ("storage", "pvc"):
        from rhoai_mcp.domains.storage.client import StorageClient

        storage_client = StorageClient(server.k8s)
        all_storage = storage_client.list_storage(namespace)
        for s in all_storage:
            if s.get("name") == name:
                return {
                    "name": name,
                    "type": "storage",
                    "status": s.get("status", "Unknown"),
                    "size": s.get("size"),
                }
        return {
            "name": name,
            "type": "storage",
            "status": "NotFound",
        }

    if resource_type in ("connection", "data_connection", "secret"):
        from rhoai_mcp.domains.connections.client import ConnectionClient

        connection_client = ConnectionClient(server.k8s)
        conn = connection_client.get_data_connection(name, namespace, mask_secrets=True)
        return {
            "name": conn.metadata.name,
            "type": "connection",
            "bucket": conn.aws_s3_bucket,
        }

    if resource_type in ("training", "trainjob", "train_job"):
        from rhoai_mcp.domains.training.client import TrainingClient

        training_client = TrainingClient(server.k8s)
        job = training_client.get_training_job(namespace, name)
        training_result: dict[str, Any] = {
            "name": job.name,
            "type": "training",
            "status": job.status.value,
        }
        if job.progress:
            training_result["progress"] = f"{round(job.progress.progress_percent, 1)}%"
        return training_result

    return {
        "name": name,
        "type": resource_type,
        "status": "Unknown",
        "error": f"Unknown resource type: {resource_type}",
        "valid_types": [
            "workbench",
            "model",
            "pipeline",
            "storage",
            "connection",
            "training",
        ],
    }


def _list_resource_names(
    server: "RHOAIServer",
    resource_type: str,
    namespace: str | None,
) -> dict[str, Any]:
    """List resource names (internal helper)."""
    resource_type = resource_type.lower()

    if resource_type in ("project", "projects"):
        from rhoai_mcp.domains.projects.client import ProjectClient

        project_client = ProjectClient(server.k8s)
        projects = project_client.list_projects()
        return {
            "type": "projects",
            "count": len(projects),
            "names": [p.metadata.name for p in projects],
        }

    if namespace is None:
        return {"error": f"namespace is required for resource type '{resource_type}'"}

    if resource_type in ("workbench", "workbenches", "notebook", "notebooks"):
        from rhoai_mcp.domains.notebooks.client import NotebookClient

        notebook_client = NotebookClient(server.k8s)
        workbenches = notebook_client.list_workbenches(namespace)
        return {
            "type": "workbenches",
            "namespace": namespace,
            "count": len(workbenches),
            "names": [wb.metadata.name for wb in workbenches],
        }

    if resource_type in ("model", "models", "inferenceservice", "inferenceservices"):
        from rhoai_mcp.domains.inference.client import InferenceClient

        inference_client = InferenceClient(server.k8s)
        models = inference_client.list_inference_services(namespace)
        return {
            "type": "models",
            "namespace": namespace,
            "count": len(models),
            "names": [m.get("name") for m in models],
        }

    if resource_type in ("storage", "pvc", "pvcs"):
        from rhoai_mcp.domains.storage.client import StorageClient

        storage_client = StorageClient(server.k8s)
        storage = storage_client.list_storage(namespace)
        return {
            "type": "storage",
            "namespace": namespace,
            "count": len(storage),
            "names": [s.get("name") for s in storage],
        }

    if resource_type in ("connection", "connections", "data_connection", "data_connections"):
        from rhoai_mcp.domains.connections.client import ConnectionClient

        connection_client = ConnectionClient(server.k8s)
        connections = connection_client.list_data_connections(namespace)
        return {
            "type": "connections",
            "namespace": namespace,
            "count": len(connections),
            "names": [c.get("name") for c in connections],
        }

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
        return {
            "type": "training_jobs",
            "namespace": namespace,
            "count": len(jobs),
            "names": [j.name for j in jobs],
        }

    if resource_type in ("runtime", "runtimes", "training_runtime", "training_runtimes"):
        from rhoai_mcp.domains.training.client import TrainingClient

        training_client = TrainingClient(server.k8s)
        runtimes = training_client.list_cluster_training_runtimes()
        if namespace:
            runtimes.extend(training_client.list_training_runtimes(namespace))
        return {
            "type": "training_runtimes",
            "count": len(runtimes),
            "names": [r.name for r in runtimes],
        }

    return {
        "error": f"Unknown resource type: {resource_type}",
        "valid_types": [
            "projects",
            "workbenches",
            "models",
            "storage",
            "connections",
            "training_jobs",
            "runtimes",
        ],
    }
