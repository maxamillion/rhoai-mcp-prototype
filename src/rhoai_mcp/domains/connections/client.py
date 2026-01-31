"""Data Connection (Secret) client operations."""

from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains.connections.models import DataConnection, S3DataConnectionCreate
from rhoai_mcp.utils.annotations import RHOAIAnnotations
from rhoai_mcp.utils.labels import RHOAILabels

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient


class ConnectionClient:
    """Client for Data Connection operations."""

    def __init__(self, k8s: "K8sClient") -> None:
        self._k8s = k8s

    def list_data_connections(self, namespace: str) -> list[dict[str, Any]]:
        """List all data connections in a namespace."""
        label_selector = RHOAILabels.filter_selector(**{RHOAILabels.DASHBOARD: "true"})
        secrets = self._k8s.list_secrets(namespace=namespace, label_selector=label_selector)

        results = []
        for secret in secrets:
            # Filter to only data connections
            annotations = secret.metadata.annotations or {}
            if RHOAIAnnotations.CONNECTION_TYPE not in annotations:
                continue

            conn = DataConnection.from_secret(secret, mask_secrets=True)
            results.append(
                {
                    "name": conn.metadata.name,
                    "display_name": conn.display_name,
                    "type": conn.connection_type,
                    "endpoint": conn.aws_s3_endpoint,
                    "bucket": conn.aws_s3_bucket,
                    "region": conn.aws_default_region,
                    "_source": conn.metadata.to_source_dict(),
                }
            )
        return results

    def get_data_connection(
        self, name: str, namespace: str, mask_secrets: bool = True
    ) -> DataConnection:
        """Get a data connection by name."""
        secret = self._k8s.get_secret(name, namespace)
        return DataConnection.from_secret(secret, mask_secrets=mask_secrets)

    def create_s3_data_connection(self, request: S3DataConnectionCreate) -> DataConnection:
        """Create an S3 data connection."""
        # Build labels
        labels = RHOAILabels.data_connection_labels()

        # Build annotations
        annotations = RHOAIAnnotations.data_connection_annotations("s3")
        if request.display_name:
            annotations["openshift.io/display-name"] = request.display_name

        # Build secret data
        data = {
            "AWS_ACCESS_KEY_ID": request.aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": request.aws_secret_access_key,
            "AWS_S3_ENDPOINT": request.aws_s3_endpoint,
            "AWS_S3_BUCKET": request.aws_s3_bucket,
            "AWS_DEFAULT_REGION": request.aws_default_region,
        }

        secret = self._k8s.create_secret(
            name=request.name,
            namespace=request.namespace,
            data=data,
            labels=labels,
            annotations=annotations,
            string_data=True,
        )

        return DataConnection.from_secret(secret, mask_secrets=True)

    def delete_data_connection(self, name: str, namespace: str) -> None:
        """Delete a data connection."""
        self._k8s.delete_secret(name, namespace)
