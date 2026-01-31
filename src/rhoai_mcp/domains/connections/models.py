"""Pydantic models for Data Connections (Secrets)."""

import base64
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.models.common import ResourceMetadata


class ConnectionType(str, Enum):
    """Data connection types."""

    S3 = "s3"
    # Future connection types could include:
    # DATABASE = "database"
    # URI = "uri"


class DataConnection(BaseModel):
    """Data Connection representation."""

    metadata: ResourceMetadata
    display_name: str | None = Field(None, description="Display name")
    connection_type: str = Field("s3", description="Connection type")
    # S3-specific fields (masked for security)
    aws_access_key_id: str | None = Field(None, description="AWS Access Key ID (masked)")
    aws_s3_endpoint: str | None = Field(None, description="S3 endpoint URL")
    aws_s3_bucket: str | None = Field(None, description="S3 bucket name")
    aws_default_region: str | None = Field(None, description="AWS region")

    @classmethod
    def from_secret(cls, secret: Any, mask_secrets: bool = True) -> "DataConnection":
        """Create from Kubernetes Secret.

        Args:
            secret: K8s Secret object
            mask_secrets: Whether to mask sensitive values
        """
        metadata = secret.metadata
        annotations = metadata.annotations or {}
        data = secret.data or {}

        # Decode base64 data or use string_data
        def decode_field(field_name: str) -> str | None:
            if field_name in data:
                try:
                    return base64.b64decode(data[field_name]).decode("utf-8")
                except Exception:
                    return str(data[field_name])
            return None

        # Get and optionally mask sensitive values
        access_key = decode_field("AWS_ACCESS_KEY_ID")
        if access_key and mask_secrets:
            access_key = (
                access_key[:4] + "****" + access_key[-4:] if len(access_key) > 8 else "****"
            )

        return cls(
            metadata=ResourceMetadata.from_k8s_metadata(
                metadata,
                kind="Secret",
                api_version="v1",
            ),
            display_name=annotations.get("openshift.io/display-name"),
            connection_type=annotations.get("opendatahub.io/connection-type", "s3"),
            aws_access_key_id=access_key,
            aws_s3_endpoint=decode_field("AWS_S3_ENDPOINT"),
            aws_s3_bucket=decode_field("AWS_S3_BUCKET"),
            aws_default_region=decode_field("AWS_DEFAULT_REGION"),
        )


class S3DataConnectionCreate(BaseModel):
    """Request model for creating an S3 data connection."""

    name: str = Field(..., description="Connection name")
    namespace: str = Field(..., description="Project namespace")
    display_name: str | None = Field(None, description="Display name")
    aws_access_key_id: str = Field(..., description="AWS Access Key ID")
    aws_secret_access_key: str = Field(..., description="AWS Secret Access Key")
    aws_s3_endpoint: str = Field(..., description="S3 endpoint URL")
    aws_s3_bucket: str = Field(..., description="S3 bucket name")
    aws_default_region: str = Field("us-east-1", description="AWS region")
