"""MCP Tools for Data Connection operations."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.connections.client import ConnectionClient
from rhoai_mcp.domains.connections.models import S3DataConnectionCreate
from rhoai_mcp.utils.response import (
    PaginatedResponse,
    ResponseBuilder,
    Verbosity,
    paginate,
)

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register data connection tools with the MCP server."""

    @mcp.tool()
    def list_data_connections(
        namespace: str,
        limit: int | None = None,
        offset: int = 0,
        verbosity: str = "standard",
    ) -> dict[str, Any]:
        """List data connections in a Data Science Project with pagination.

        Data connections are secrets with RHOAI-specific labels that provide
        credentials for accessing external data sources like S3 buckets.

        Args:
            namespace: The project (namespace) name.
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).
            verbosity: Response detail level - "minimal", "standard", or "full".
                Use "minimal" for quick status checks.

        Returns:
            Paginated list of data connections with metadata (credentials masked).
        """
        client = ConnectionClient(server.k8s)
        all_items = client.list_data_connections(namespace)

        # Apply config limits
        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        # Paginate
        paginated, total = paginate(all_items, offset, effective_limit)

        # Format with verbosity
        v = Verbosity.from_str(verbosity)
        items = [ResponseBuilder.data_connection_list_item(conn, v) for conn in paginated]

        return PaginatedResponse.build(items, total, offset, effective_limit)

    @mcp.tool()
    def get_data_connection(name: str, namespace: str) -> dict[str, Any]:
        """Get detailed information about a data connection.

        Sensitive values like secret keys are masked for security.

        Args:
            name: The data connection (secret) name.
            namespace: The project (namespace) name.

        Returns:
            Data connection details with masked credentials.
        """
        client = ConnectionClient(server.k8s)
        conn = client.get_data_connection(name, namespace, mask_secrets=True)

        return {
            "name": conn.metadata.name,
            "namespace": conn.metadata.namespace,
            "display_name": conn.display_name,
            "type": conn.connection_type,
            "aws_access_key_id": conn.aws_access_key_id,
            "aws_s3_endpoint": conn.aws_s3_endpoint,
            "aws_s3_bucket": conn.aws_s3_bucket,
            "aws_default_region": conn.aws_default_region,
            "created": (
                conn.metadata.creation_timestamp.isoformat()
                if conn.metadata.creation_timestamp
                else None
            ),
            "_source": conn.metadata.to_source_dict(),
        }

    @mcp.tool()
    def create_s3_data_connection(
        name: str,
        namespace: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_s3_endpoint: str,
        aws_s3_bucket: str,
        display_name: str | None = None,
        aws_default_region: str = "us-east-1",
    ) -> dict[str, Any]:
        """Create an S3 data connection.

        Creates a secret with the appropriate RHOAI labels and annotations
        that can be used by workbenches and pipelines to access S3 storage.

        Args:
            name: Connection name (will be the secret name).
            namespace: Project (namespace) name.
            aws_access_key_id: AWS Access Key ID.
            aws_secret_access_key: AWS Secret Access Key.
            aws_s3_endpoint: S3 endpoint URL (e.g., https://s3.amazonaws.com).
            aws_s3_bucket: S3 bucket name.
            display_name: Human-readable display name.
            aws_default_region: AWS region (default: us-east-1).

        Returns:
            Created data connection information (credentials masked).
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        client = ConnectionClient(server.k8s)
        request = S3DataConnectionCreate(
            name=name,
            namespace=namespace,
            display_name=display_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_endpoint=aws_s3_endpoint,
            aws_s3_bucket=aws_s3_bucket,
            aws_default_region=aws_default_region,
        )
        conn = client.create_s3_data_connection(request)

        return {
            "name": conn.metadata.name,
            "namespace": conn.metadata.namespace,
            "type": conn.connection_type,
            "bucket": conn.aws_s3_bucket,
            "message": f"Data connection '{name}' created successfully",
            "_source": conn.metadata.to_source_dict(),
        }

    @mcp.tool()
    def delete_data_connection(
        name: str,
        namespace: str,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Delete a data connection.

        WARNING: Workbenches or pipelines using this connection will lose
        access to the associated data source.

        Args:
            name: The data connection (secret) name.
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
                    f"To delete data connection '{name}', set confirm=True. "
                    "WARNING: Resources using this connection will lose access."
                ),
            }

        client = ConnectionClient(server.k8s)
        client.delete_data_connection(name, namespace)

        return {
            "name": name,
            "namespace": namespace,
            "deleted": True,
            "message": f"Data connection '{name}' deleted",
            "_source": {
                "kind": "Secret",
                "api_version": "v1",
                "name": name,
                "namespace": namespace,
                "uid": None,
            },
        }
