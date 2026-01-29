"""Pydantic models for Model Catalog."""

from pydantic import BaseModel, Field


class CatalogModelArtifact(BaseModel):
    """Model artifact representation."""

    uri: str = Field(..., description="Artifact URI (e.g., oci://registry.../model:tag)")


class CatalogModel(BaseModel):
    """Model catalog entry representation."""

    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Model provider")
    description: str = Field(..., description="Model description")
    artifacts: list[CatalogModelArtifact] = Field(
        default_factory=list, description="Model artifacts"
    )


class ModelCatalog(BaseModel):
    """Model catalog representation."""

    source: str = Field(..., description="Catalog source")
    models: list[CatalogModel] = Field(default_factory=list, description="Available models")
