"""Tests for catalog models."""

import pytest

from rhoai_mcp.domains.catalog.models import (
    CatalogModel,
    CatalogModelArtifact,
    ModelCatalog,
)


class TestCatalogModelArtifact:
    """Test CatalogModelArtifact model."""

    def test_artifact_creation(self) -> None:
        """Test creating a model artifact."""
        artifact = CatalogModelArtifact(uri="oci://quay.io/models/test:latest")

        assert artifact.uri == "oci://quay.io/models/test:latest"

    def test_artifact_validation(self) -> None:
        """Test artifact validation requires URI."""
        with pytest.raises(ValueError):
            CatalogModelArtifact()  # type: ignore


class TestCatalogModel:
    """Test CatalogModel model."""

    def test_model_creation_with_artifacts(self, mock_catalog_model: dict) -> None:
        """Test creating a catalog model with artifacts."""
        model = CatalogModel(**mock_catalog_model)

        assert model.name == "llama-2-7b"
        assert model.provider == "Meta"
        assert model.description == "Llama 2 7B base model for fine-tuning"
        assert len(model.artifacts) == 2
        assert model.artifacts[0].uri == "oci://quay.io/models/llama-2-7b:latest"
        assert model.artifacts[1].uri == "oci://quay.io/models/llama-2-7b:v1.0"

    def test_model_creation_minimal(self, mock_catalog_model_minimal: dict) -> None:
        """Test creating a catalog model with minimal data."""
        model = CatalogModel(**mock_catalog_model_minimal)

        assert model.name == "test-model"
        assert model.provider == "Test Provider"
        assert model.description == "A test model"
        assert model.artifacts == []

    def test_model_validation_requires_name(self) -> None:
        """Test model validation requires name."""
        with pytest.raises(ValueError):
            CatalogModel(provider="Provider", description="Description")  # type: ignore

    def test_model_validation_requires_provider(self) -> None:
        """Test model validation requires provider."""
        with pytest.raises(ValueError):
            CatalogModel(name="model", description="Description")  # type: ignore

    def test_model_validation_requires_description(self) -> None:
        """Test model validation requires description."""
        with pytest.raises(ValueError):
            CatalogModel(name="model", provider="Provider")  # type: ignore

    def test_model_with_empty_artifacts(self) -> None:
        """Test model can have empty artifacts list."""
        model = CatalogModel(
            name="test-model",
            provider="Test Provider",
            description="A model without artifacts",
            artifacts=[],
        )

        assert model.artifacts == []

    def test_model_artifacts_default_factory(self) -> None:
        """Test artifacts default to empty list."""
        model = CatalogModel(
            name="test-model",
            provider="Test Provider",
            description="A test model",
        )

        assert model.artifacts == []
        assert isinstance(model.artifacts, list)


class TestModelCatalog:
    """Test ModelCatalog model."""

    def test_catalog_creation(self, mock_catalog_data: dict) -> None:
        """Test creating a model catalog."""
        catalog = ModelCatalog(**mock_catalog_data)

        assert catalog.source == "https://example.com/catalog.yaml"
        assert len(catalog.models) == 3

    def test_catalog_models_parsing(self, mock_catalog_data: dict) -> None:
        """Test catalog models are parsed correctly."""
        catalog = ModelCatalog(**mock_catalog_data)

        # Check first model
        llama = catalog.models[0]
        assert llama.name == "llama-2-7b"
        assert llama.provider == "Meta"
        assert len(llama.artifacts) == 1

        # Check second model
        granite = catalog.models[1]
        assert granite.name == "granite-8b-code"
        assert granite.provider == "IBM"
        assert len(granite.artifacts) == 2

        # Check third model
        mistral = catalog.models[2]
        assert mistral.name == "mistral-7b"
        assert mistral.provider == "Mistral AI"
        assert len(mistral.artifacts) == 0

    def test_catalog_validation_requires_source(self) -> None:
        """Test catalog validation requires source."""
        with pytest.raises(ValueError):
            ModelCatalog(models=[])  # type: ignore

    def test_catalog_empty_models(self) -> None:
        """Test catalog can have empty models list."""
        catalog = ModelCatalog(source="https://example.com/empty.yaml", models=[])

        assert catalog.source == "https://example.com/empty.yaml"
        assert catalog.models == []

    def test_catalog_models_default_factory(self) -> None:
        """Test models default to empty list."""
        catalog = ModelCatalog(source="https://example.com/catalog.yaml")

        assert catalog.models == []
        assert isinstance(catalog.models, list)

    def test_catalog_with_single_model(self, mock_catalog_model: dict) -> None:
        """Test catalog with single model."""
        catalog = ModelCatalog(
            source="https://example.com/catalog.yaml",
            models=[mock_catalog_model],
        )

        assert len(catalog.models) == 1
        assert catalog.models[0].name == "llama-2-7b"
        assert catalog.models[0].provider == "Meta"

    def test_catalog_artifact_access(self, mock_catalog_data: dict) -> None:
        """Test accessing artifacts through catalog."""
        catalog = ModelCatalog(**mock_catalog_data)

        granite_artifacts = catalog.models[1].artifacts
        assert len(granite_artifacts) == 2
        assert granite_artifacts[0].uri == "oci://quay.io/models/granite-8b-code:v1"
        assert granite_artifacts[1].uri == "oci://quay.io/models/granite-8b-code:latest"
