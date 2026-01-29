"""Tests for CatalogClient."""

import yaml
from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.catalog.client import CatalogClient
from rhoai_mcp.domains.catalog.models import CatalogModel, ModelCatalog
from rhoai_mcp.utils.errors import NotFoundError, RHOAIError


class TestCatalogClient:
    """Test CatalogClient operations."""

    @pytest.fixture
    def mock_k8s(self) -> MagicMock:
        """Create a mock K8sClient."""
        mock = MagicMock()
        mock.core_v1 = MagicMock()
        return mock

    @pytest.fixture
    def client(self, mock_k8s: MagicMock) -> CatalogClient:
        """Create a CatalogClient with mocked K8sClient."""
        return CatalogClient(mock_k8s)

    def test_read_catalog_success(
        self, client: CatalogClient, mock_k8s: MagicMock, mock_catalog_data: dict
    ) -> None:
        """Test successfully reading catalog from pod."""
        # Mock pod list response
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod-xyz"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list

        # Mock exec_command to return YAML
        catalog_yaml = yaml.dump(mock_catalog_data)
        mock_k8s.exec_command.return_value = catalog_yaml

        catalog = client.read_catalog()

        assert isinstance(catalog, ModelCatalog)
        assert catalog.source == "https://example.com/catalog.yaml"
        assert len(catalog.models) == 3
        assert catalog.models[0].name == "llama-2-7b"
        assert catalog.models[1].name == "granite-8b-code"
        assert catalog.models[2].name == "mistral-7b"

        # Verify K8s calls
        mock_k8s.core_v1.list_namespaced_pod.assert_called_once_with(
            namespace="rhoai-model-registries",
            label_selector="app=model-catalog",
        )
        mock_k8s.exec_command.assert_called_once_with(
            pod_name="model-catalog-pod-xyz",
            namespace="rhoai-model-registries",
            command=["cat", "/shared-data/models-catalog.yaml"],
        )

    def test_read_catalog_pod_not_found(
        self, client: CatalogClient, mock_k8s: MagicMock
    ) -> None:
        """Test error when catalog pod is not found."""
        # Mock empty pod list
        mock_pod_list = MagicMock()
        mock_pod_list.items = []
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list

        with pytest.raises(NotFoundError) as exc_info:
            client.read_catalog()

        assert "model-catalog" in str(exc_info.value)
        assert "rhoai-model-registries" in str(exc_info.value)

    def test_read_catalog_pod_list_error(
        self, client: CatalogClient, mock_k8s: MagicMock
    ) -> None:
        """Test error when listing pods fails."""
        mock_k8s.core_v1.list_namespaced_pod.side_effect = Exception("API error")

        with pytest.raises(RHOAIError) as exc_info:
            client.read_catalog()

        assert "Failed to find catalog pod" in str(exc_info.value)
        assert "API error" in str(exc_info.value)

    def test_read_catalog_exec_command_error(
        self, client: CatalogClient, mock_k8s: MagicMock
    ) -> None:
        """Test error when exec_command fails."""
        # Mock pod list
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod-xyz"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list

        # Mock exec_command failure
        mock_k8s.exec_command.side_effect = Exception("Permission denied")

        with pytest.raises(RHOAIError) as exc_info:
            client.read_catalog()

        assert "Failed to read catalog file" in str(exc_info.value)
        assert "model-catalog-pod-xyz" in str(exc_info.value)

    def test_read_catalog_empty_yaml(
        self, client: CatalogClient, mock_k8s: MagicMock
    ) -> None:
        """Test error when catalog YAML is empty."""
        # Mock pod list
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod-xyz"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list

        # Mock exec_command to return empty string
        mock_k8s.exec_command.return_value = ""

        with pytest.raises(RHOAIError) as exc_info:
            client.read_catalog()

        assert "Catalog file is empty or invalid" in str(exc_info.value)

    def test_read_catalog_invalid_yaml(
        self, client: CatalogClient, mock_k8s: MagicMock
    ) -> None:
        """Test error when catalog YAML is malformed."""
        # Mock pod list
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod-xyz"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list

        # Mock exec_command to return invalid YAML
        mock_k8s.exec_command.return_value = "{{invalid yaml: [}"

        with pytest.raises(RHOAIError) as exc_info:
            client.read_catalog()

        assert "Failed to parse catalog YAML" in str(exc_info.value)

    def test_read_catalog_invalid_data_structure(
        self, client: CatalogClient, mock_k8s: MagicMock
    ) -> None:
        """Test error when catalog data doesn't match schema."""
        # Mock pod list
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod-xyz"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list

        # Mock exec_command to return YAML with missing required fields
        invalid_data = {"models": [{"name": "test"}]}  # Missing source
        mock_k8s.exec_command.return_value = yaml.dump(invalid_data)

        with pytest.raises(RHOAIError) as exc_info:
            client.read_catalog()

        assert "Failed to create ModelCatalog from data" in str(exc_info.value)

    def test_read_catalog_minimal_valid_data(
        self, client: CatalogClient, mock_k8s: MagicMock
    ) -> None:
        """Test reading catalog with minimal valid data."""
        # Mock pod list
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod-xyz"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list

        # Mock exec_command with minimal catalog
        minimal_data = {"source": "test-source", "models": []}
        mock_k8s.exec_command.return_value = yaml.dump(minimal_data)

        catalog = client.read_catalog()

        assert catalog.source == "test-source"
        assert catalog.models == []

    def test_get_model_success(
        self, client: CatalogClient, mock_k8s: MagicMock, mock_catalog_data: dict
    ) -> None:
        """Test successfully getting a model by name."""
        # Mock catalog reading
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod-xyz"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        model = client.get_model("granite-8b-code")

        assert isinstance(model, CatalogModel)
        assert model.name == "granite-8b-code"
        assert model.provider == "IBM"
        assert model.description == "Granite 8B code model"
        assert len(model.artifacts) == 2

    def test_get_model_not_found(
        self, client: CatalogClient, mock_k8s: MagicMock, mock_catalog_data: dict
    ) -> None:
        """Test error when model is not found in catalog."""
        # Mock catalog reading
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod-xyz"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        with pytest.raises(ValueError) as exc_info:
            client.get_model("non-existent-model")

        assert "not found in catalog" in str(exc_info.value)
        assert "non-existent-model" in str(exc_info.value)

    def test_get_model_first_match(
        self, client: CatalogClient, mock_k8s: MagicMock
    ) -> None:
        """Test get_model returns first match when model name appears multiple times."""
        # Mock catalog with duplicate model names
        catalog_data = {
            "source": "test",
            "models": [
                {
                    "name": "test-model",
                    "provider": "Provider1",
                    "description": "First model",
                    "artifacts": [{"uri": "oci://registry/model:v1"}],
                },
                {
                    "name": "test-model",
                    "provider": "Provider2",
                    "description": "Second model",
                    "artifacts": [{"uri": "oci://registry/model:v2"}],
                },
            ],
        }

        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod-xyz"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_k8s.exec_command.return_value = yaml.dump(catalog_data)

        model = client.get_model("test-model")

        # Should return first match
        assert model.provider == "Provider1"
        assert model.description == "First model"
        assert model.artifacts[0].uri == "oci://registry/model:v1"

    def test_get_model_catalog_read_error(
        self, client: CatalogClient, mock_k8s: MagicMock
    ) -> None:
        """Test error propagation when catalog read fails."""
        # Mock pod list failure
        mock_k8s.core_v1.list_namespaced_pod.side_effect = Exception("API error")

        with pytest.raises(RHOAIError):
            client.get_model("any-model")

    def test_read_catalog_with_artifacts(
        self, client: CatalogClient, mock_k8s: MagicMock
    ) -> None:
        """Test reading catalog with models containing various artifact configurations."""
        catalog_data = {
            "source": "test-source",
            "models": [
                {
                    "name": "no-artifacts",
                    "provider": "Provider1",
                    "description": "Model without artifacts",
                    "artifacts": [],
                },
                {
                    "name": "single-artifact",
                    "provider": "Provider2",
                    "description": "Model with one artifact",
                    "artifacts": [{"uri": "oci://registry/model:v1"}],
                },
                {
                    "name": "multiple-artifacts",
                    "provider": "Provider3",
                    "description": "Model with multiple artifacts",
                    "artifacts": [
                        {"uri": "oci://registry/model:v1"},
                        {"uri": "oci://registry/model:v2"},
                        {"uri": "oci://registry/model:latest"},
                    ],
                },
            ],
        }

        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod-xyz"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_k8s.exec_command.return_value = yaml.dump(catalog_data)

        catalog = client.read_catalog()

        assert len(catalog.models[0].artifacts) == 0
        assert len(catalog.models[1].artifacts) == 1
        assert len(catalog.models[2].artifacts) == 3
