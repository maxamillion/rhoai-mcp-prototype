"""MCP Tools for Notebook (Workbench) operations."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.notebooks.client import NotebookClient
from rhoai_mcp.domains.notebooks.models import WorkbenchCreate
from rhoai_mcp.utils.response import (
    PaginatedResponse,
    ResponseBuilder,
    Verbosity,
    paginate,
)

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register workbench management tools with the MCP server."""

    @mcp.tool()
    def list_workbenches(
        namespace: str,
        limit: int | None = None,
        offset: int = 0,
        verbosity: str = "standard",
    ) -> dict[str, Any]:
        """List workbenches in a Data Science Project with pagination.

        Workbenches are Jupyter notebook environments or other IDE environments
        running in the project.

        Args:
            namespace: The project (namespace) name.
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).
            verbosity: Response detail level - "minimal", "standard", or "full".
                Use "minimal" for quick status checks (~85% token savings).

        Returns:
            Paginated list of workbenches with metadata.
        """
        client = NotebookClient(server.k8s)
        workbenches = client.list_workbenches(namespace)

        # Apply config limits
        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        # Paginate
        paginated, total = paginate(workbenches, offset, effective_limit)

        # Format with verbosity
        v = Verbosity.from_str(verbosity)
        items = [ResponseBuilder.workbench_list_item(wb, v) for wb in paginated]

        return PaginatedResponse.build(items, total, offset, effective_limit)

    @mcp.tool()
    def get_workbench(
        name: str,
        namespace: str,
        verbosity: str = "full",
    ) -> dict[str, Any]:
        """Get detailed information about a workbench.

        Args:
            name: The workbench name.
            namespace: The project (namespace) name.
            verbosity: Response detail level - "minimal", "standard", or "full".
                Use "minimal" for quick status checks.

        Returns:
            Workbench information at the requested verbosity level.
        """
        client = NotebookClient(server.k8s)
        wb = client.get_workbench(name, namespace)

        v = Verbosity.from_str(verbosity)
        return ResponseBuilder.workbench_detail(wb, v)

    @mcp.tool()
    def create_workbench(
        name: str,
        namespace: str,
        image: str,
        display_name: str | None = None,
        size: str = "Small",
        cpu_request: str = "500m",
        cpu_limit: str = "2",
        memory_request: str = "1Gi",
        memory_limit: str = "4Gi",
        gpu_count: int = 0,
        storage_size: str = "10Gi",
        data_connections: list[str] | None = None,
        additional_pvcs: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new workbench (notebook environment).

        Creates a Kubeflow Notebook CR with the specified configuration,
        including a PVC for persistent storage.

        Args:
            name: Workbench name (must be DNS-compatible).
            namespace: Project (namespace) name.
            image: Container image to use (use list_notebook_images to see available options).
            display_name: Human-readable display name.
            size: Size selection name (Small, Medium, Large, X-Large).
            cpu_request: CPU request (e.g., '500m', '1').
            cpu_limit: CPU limit (e.g., '2', '4').
            memory_request: Memory request (e.g., '1Gi', '2Gi').
            memory_limit: Memory limit (e.g., '4Gi', '8Gi').
            gpu_count: Number of NVIDIA GPUs to request.
            storage_size: Size of the workbench PVC (e.g., '10Gi', '50Gi').
            data_connections: List of secret names to mount as environment variables.
            additional_pvcs: List of additional PVC names to mount.

        Returns:
            Created workbench information.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        client = NotebookClient(server.k8s)
        request = WorkbenchCreate(
            name=name,
            namespace=namespace,
            display_name=display_name,
            image=image,
            size=size,
            cpu_request=cpu_request,
            cpu_limit=cpu_limit,
            memory_request=memory_request,
            memory_limit=memory_limit,
            gpu_count=gpu_count,
            storage_size=storage_size,
            data_connections=data_connections or [],
            additional_pvcs=additional_pvcs or [],
        )
        wb = client.create_workbench(request)

        return {
            "name": wb.metadata.name,
            "namespace": wb.metadata.namespace,
            "status": wb.status.value,
            "url": wb.url,
            "message": f"Workbench '{name}' created successfully. It will start automatically.",
            "_source": wb.metadata.to_source_dict(),
        }

    @mcp.tool()
    def start_workbench(name: str, namespace: str) -> dict[str, Any]:
        """Start a stopped workbench.

        Removes the kubeflow-resource-stopped annotation to allow the
        workbench pod to be scheduled.

        Args:
            name: The workbench name.
            namespace: The project (namespace) name.

        Returns:
            Updated workbench status.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("update")
        if not allowed:
            return {"error": reason}

        client = NotebookClient(server.k8s)
        wb = client.start_workbench(name, namespace)

        return {
            "name": wb.metadata.name,
            "status": wb.status.value,
            "url": wb.url,
            "message": f"Workbench '{name}' is starting",
            "_source": wb.metadata.to_source_dict(),
        }

    @mcp.tool()
    def stop_workbench(name: str, namespace: str) -> dict[str, Any]:
        """Stop a running workbench.

        Adds the kubeflow-resource-stopped annotation which causes the
        workbench pod to be terminated. The workbench can be restarted later.

        Args:
            name: The workbench name.
            namespace: The project (namespace) name.

        Returns:
            Updated workbench status.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("update")
        if not allowed:
            return {"error": reason}

        client = NotebookClient(server.k8s)
        wb = client.stop_workbench(name, namespace)

        return {
            "name": wb.metadata.name,
            "status": wb.status.value,
            "stopped_time": wb.stopped_time.isoformat() if wb.stopped_time else None,
            "message": f"Workbench '{name}' is stopping",
            "_source": wb.metadata.to_source_dict(),
        }

    @mcp.tool()
    def delete_workbench(
        name: str,
        namespace: str,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Delete a workbench.

        WARNING: This permanently deletes the workbench. The associated PVC
        is NOT automatically deleted to preserve data.

        Args:
            name: The workbench name.
            namespace: The project (namespace) name.
            confirm: Must be True to actually delete.

        Returns:
            Confirmation of deletion.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("delete")
        if not allowed:
            return {"error": reason}

        if not confirm:
            return {
                "error": "Deletion not confirmed",
                "message": (
                    f"To delete workbench '{name}', set confirm=True. "
                    "Note: The workbench PVC will be preserved."
                ),
            }

        client = NotebookClient(server.k8s)
        client.delete_workbench(name, namespace)

        return {
            "name": name,
            "namespace": namespace,
            "deleted": True,
            "message": f"Workbench '{name}' deleted. PVC '{name}-pvc' was preserved.",
            "_source": {
                "kind": "Notebook",
                "api_version": "kubeflow.org/v1",
                "name": name,
                "namespace": namespace,
                "uid": None,
            },
        }

    @mcp.tool()
    def list_notebook_images() -> list[dict[str, Any]]:
        """List available notebook images.

        Returns the standard RHOAI notebook images that can be used
        when creating workbenches.

        Returns:
            List of available images with display names and descriptions.
        """
        client = NotebookClient(server.k8s)
        images = client.list_notebook_images()

        return [
            {
                "name": img.name,
                "display_name": img.display_name,
                "description": img.description,
                "recommended": img.recommended,
            }
            for img in images
        ]

    @mcp.tool()
    def get_workbench_url(name: str, namespace: str) -> dict[str, Any]:
        """Get the access URL for a workbench.

        Returns the OAuth-protected route URL for accessing the workbench
        in a browser.

        Args:
            name: The workbench name.
            namespace: The project (namespace) name.

        Returns:
            The workbench URL and current status.
        """
        client = NotebookClient(server.k8s)
        wb = client.get_workbench(name, namespace)

        return {
            "name": wb.metadata.name,
            "namespace": wb.metadata.namespace,
            "url": wb.url,
            "status": wb.status.value,
            "message": (
                "Workbench is accessible at the URL"
                if wb.status.value == "Running"
                else f"Workbench is {wb.status.value} - URL may not be accessible"
            ),
        }
