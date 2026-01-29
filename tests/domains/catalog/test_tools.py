"""Tests for catalog tools."""

import yaml
from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.domains.catalog.tools import register_tools
from rhoai_mcp.utils.errors import NotFoundError, RHOAIError


class TestListCatalogModels:
    """Test list_catalog_models tool."""

    @pytest.fixture
    def mock_mcp(self) -> MagicMock:
        """Create a mock FastMCP server."""
        mock = MagicMock()
        mock.tool = MagicMock(return_value=lambda f: f)
        return mock

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock RHOAIServer."""
        server = MagicMock()
        server.k8s = MagicMock()
        server.k8s.core_v1 = MagicMock()
        return server

    def test_register_tools_registers_catalog_tools(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Test that catalog tools are registered."""
        register_tools(mock_mcp, mock_server)

        # Check that tool decorator was called for each tool
        assert mock_mcp.tool.call_count >= 2  # list_catalog_models, deploy_from_catalog

    def test_list_catalog_models_success(
        self, mock_mcp: MagicMock, mock_server: MagicMock, mock_catalog_data: dict
    ) -> None:
        """Test listing catalog models successfully."""
        # Setup mock to return catalog data
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["list_catalog_models"]()

        assert result["total"] == 3
        assert result["returned"] == 3
        assert len(result["models"]) == 3

        # Check first model
        llama = result["models"][0]
        assert llama["name"] == "llama-2-7b"
        assert llama["provider"] == "Meta"
        assert llama["description"] == "Llama 2 7B base model"
        assert llama["oci_uri"] == "oci://quay.io/models/llama-2-7b:latest"

        # Check model with multiple artifacts
        granite = result["models"][1]
        assert granite["name"] == "granite-8b-code"
        assert granite["oci_uri"] == "oci://quay.io/models/granite-8b-code:v1"

        # Check model with no artifacts
        mistral = result["models"][2]
        assert mistral["name"] == "mistral-7b"
        assert mistral["oci_uri"] is None

    def test_list_catalog_models_with_limit(
        self, mock_mcp: MagicMock, mock_server: MagicMock, mock_catalog_data: dict
    ) -> None:
        """Test listing catalog models with limit parameter."""
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["list_catalog_models"](limit=2)

        assert result["total"] == 3
        assert result["returned"] == 2
        assert len(result["models"]) == 2
        assert result["models"][0]["name"] == "llama-2-7b"
        assert result["models"][1]["name"] == "granite-8b-code"

    def test_list_catalog_models_limit_zero(
        self, mock_mcp: MagicMock, mock_server: MagicMock, mock_catalog_data: dict
    ) -> None:
        """Test listing catalog models with limit=0 returns empty list."""
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["list_catalog_models"](limit=0)

        assert result["total"] == 3
        assert result["returned"] == 0
        assert len(result["models"]) == 0

    def test_list_catalog_models_catalog_not_found(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Test listing models when catalog pod is not found."""
        mock_pod_list = MagicMock()
        mock_pod_list.items = []
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["list_catalog_models"]()

        assert "error" in result
        assert result["error"] == "Catalog not found"
        assert "model-catalog" in result["message"]

    def test_list_catalog_models_read_error(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Test listing models when catalog read fails."""
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.side_effect = Exception("Permission denied")

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["list_catalog_models"]()

        assert "error" in result
        assert result["error"] == "Failed to read catalog"

    def test_list_catalog_models_unexpected_error(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Test handling of unexpected errors during catalog read.

        Note: RuntimeError is wrapped as RHOAIError during catalog reading,
        so it appears as "Failed to read catalog" rather than "Unexpected error".
        """
        mock_server.k8s.core_v1.list_namespaced_pod.side_effect = RuntimeError(
            "Unexpected error"
        )

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["list_catalog_models"]()

        assert "error" in result
        assert result["error"] == "Failed to read catalog"
        assert "Unexpected error" in result["message"]


class TestDeployFromCatalog:
    """Test deploy_from_catalog tool."""

    @pytest.fixture
    def mock_mcp(self) -> MagicMock:
        """Create a mock FastMCP server."""
        mock = MagicMock()
        mock.tool = MagicMock(return_value=lambda f: f)
        return mock

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock RHOAIServer."""
        server = MagicMock()
        server.k8s = MagicMock()
        server.k8s.core_v1 = MagicMock()
        server.config.is_operation_allowed.return_value = (True, None)
        return server

    @pytest.fixture
    def mock_runtime_template(self) -> dict:
        """Sample vLLM runtime template."""
        return {
            "objects": [
                {
                    "apiVersion": "serving.kserve.io/v1alpha1",
                    "kind": "ServingRuntime",
                    "metadata": {
                        "name": "vllm-runtime",
                        "labels": {"opendatahub.io/dashboard": "true"},
                    },
                    "spec": {
                        "multiModel": False,
                        "supportedModelFormats": [{"name": "vLLM"}],
                        "containers": [
                            {
                                "name": "kserve-container",
                                "image": "quay.io/modh/vllm:latest",
                            }
                        ],
                    },
                }
            ]
        }

    def test_deploy_from_catalog_success(
        self,
        mock_mcp: MagicMock,
        mock_server: MagicMock,
        mock_catalog_data: dict,
        mock_runtime_template: dict,
    ) -> None:
        """Test successful deployment from catalog."""
        # Mock catalog reading
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        # Mock template retrieval
        mock_server.k8s.get_template.return_value = mock_runtime_template

        # Mock InferenceService creation
        mock_isvc = MagicMock()
        mock_isvc.metadata.name = "llama-deployment"
        mock_isvc.metadata.namespace = "test-project"
        mock_isvc.status.value = "Pending"

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        # Patch InferenceClient
        with patch("rhoai_mcp.domains.catalog.tools.InferenceClient") as mock_inf_client_class:
            mock_inf_client = MagicMock()
            mock_inf_client.deploy_model.return_value = mock_isvc
            mock_inf_client_class.return_value = mock_inf_client

            result = tools["deploy_from_catalog"](
                catalog_model_name="llama-2-7b",
                deployment_name="llama-deployment",
                namespace="test-project",
                gpu_count=2,
            )

        assert result["name"] == "llama-deployment"
        assert result["namespace"] == "test-project"
        assert result["status"] == "Pending"
        assert result["catalog_model"] == "llama-2-7b"
        assert result["storage_uri"] == "oci://quay.io/models/llama-2-7b:latest"
        assert "deployed" in result["message"].lower()

        # Verify runtime was created
        mock_server.k8s.create.assert_called_once()

    def test_deploy_from_catalog_read_only_mode(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Test deployment blocked in read-only mode."""
        mock_server.config.is_operation_allowed.return_value = (False, "Read-only mode enabled")

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["deploy_from_catalog"](
            catalog_model_name="llama-2-7b",
            deployment_name="test-deploy",
            namespace="test-project",
        )

        assert "error" in result
        assert result["error"] == "Read-only mode enabled"

    def test_deploy_from_catalog_model_not_found(
        self, mock_mcp: MagicMock, mock_server: MagicMock, mock_catalog_data: dict
    ) -> None:
        """Test deployment when model is not in catalog."""
        # Mock catalog reading
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["deploy_from_catalog"](
            catalog_model_name="non-existent-model",
            deployment_name="test-deploy",
            namespace="test-project",
        )

        assert "error" in result
        assert result["error"] == "Model not found"
        assert "non-existent-model" in result["message"]

    def test_deploy_from_catalog_no_artifacts(
        self, mock_mcp: MagicMock, mock_server: MagicMock, mock_catalog_data: dict
    ) -> None:
        """Test deployment when model has no artifacts."""
        # Mock catalog reading
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["deploy_from_catalog"](
            catalog_model_name="mistral-7b",  # This model has no artifacts in mock data
            deployment_name="test-deploy",
            namespace="test-project",
        )

        assert "error" in result
        assert result["error"] == "Model has no artifacts"
        assert "mistral-7b" in result["message"]

    def test_deploy_from_catalog_template_not_found(
        self, mock_mcp: MagicMock, mock_server: MagicMock, mock_catalog_data: dict
    ) -> None:
        """Test deployment when runtime template is not found."""
        # Mock catalog reading
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        # Mock template not found
        mock_server.k8s.get_template.side_effect = NotFoundError(
            "Template", "vllm-cuda-runtime-template", "redhat-ods-applications"
        )

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["deploy_from_catalog"](
            catalog_model_name="llama-2-7b",
            deployment_name="test-deploy",
            namespace="test-project",
        )

        assert "error" in result
        assert result["error"] == "Runtime template not found"

    def test_deploy_from_catalog_invalid_template(
        self, mock_mcp: MagicMock, mock_server: MagicMock, mock_catalog_data: dict
    ) -> None:
        """Test deployment when template has invalid structure."""
        # Mock catalog reading
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        # Mock template with no objects
        mock_server.k8s.get_template.return_value = {"objects": []}

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["deploy_from_catalog"](
            catalog_model_name="llama-2-7b",
            deployment_name="test-deploy",
            namespace="test-project",
        )

        assert "error" in result
        assert result["error"] == "Invalid template"

    def test_deploy_from_catalog_runtime_already_exists(
        self,
        mock_mcp: MagicMock,
        mock_server: MagicMock,
        mock_catalog_data: dict,
        mock_runtime_template: dict,
    ) -> None:
        """Test deployment when runtime already exists."""
        # Mock catalog reading
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        # Mock template retrieval
        mock_server.k8s.get_template.return_value = mock_runtime_template

        # Mock runtime creation failure (already exists)
        mock_server.k8s.create.side_effect = Exception("Already exists")

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["deploy_from_catalog"](
            catalog_model_name="llama-2-7b",
            deployment_name="existing-runtime",
            namespace="test-project",
        )

        assert "error" in result
        assert result["error"] == "Runtime already exists"
        assert "existing-runtime" in result["message"]

    def test_deploy_from_catalog_custom_gpu_count(
        self,
        mock_mcp: MagicMock,
        mock_server: MagicMock,
        mock_catalog_data: dict,
        mock_runtime_template: dict,
    ) -> None:
        """Test deployment with custom GPU count."""
        # Mock catalog reading
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        # Mock template retrieval
        mock_server.k8s.get_template.return_value = mock_runtime_template

        # Mock InferenceService creation
        mock_isvc = MagicMock()
        mock_isvc.metadata.name = "test-deploy"
        mock_isvc.metadata.namespace = "test-project"
        mock_isvc.status.value = "Pending"

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        # Patch InferenceClient
        with patch("rhoai_mcp.domains.catalog.tools.InferenceClient") as mock_inf_client_class:
            mock_inf_client = MagicMock()
            mock_inf_client.deploy_model.return_value = mock_isvc
            mock_inf_client_class.return_value = mock_inf_client

            result = tools["deploy_from_catalog"](
                catalog_model_name="llama-2-7b",
                deployment_name="test-deploy",
                namespace="test-project",
                gpu_count=4,
            )

            # Verify InferenceServiceCreate was called with correct GPU count
            create_call = mock_inf_client.deploy_model.call_args[0][0]
            assert create_call.gpu_count == 4

        assert "error" not in result

    def test_deploy_from_catalog_default_gpu_count(
        self,
        mock_mcp: MagicMock,
        mock_server: MagicMock,
        mock_catalog_data: dict,
        mock_runtime_template: dict,
    ) -> None:
        """Test deployment uses default GPU count of 1."""
        # Mock catalog reading
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        # Mock template retrieval
        mock_server.k8s.get_template.return_value = mock_runtime_template

        # Mock InferenceService creation
        mock_isvc = MagicMock()
        mock_isvc.metadata.name = "test-deploy"
        mock_isvc.metadata.namespace = "test-project"
        mock_isvc.status.value = "Pending"

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        # Patch InferenceClient
        with patch("rhoai_mcp.domains.catalog.tools.InferenceClient") as mock_inf_client_class:
            mock_inf_client = MagicMock()
            mock_inf_client.deploy_model.return_value = mock_isvc
            mock_inf_client_class.return_value = mock_inf_client

            # Call without gpu_count parameter
            result = tools["deploy_from_catalog"](
                catalog_model_name="llama-2-7b",
                deployment_name="test-deploy",
                namespace="test-project",
            )

            # Verify default GPU count is 1
            create_call = mock_inf_client.deploy_model.call_args[0][0]
            assert create_call.gpu_count == 1

        assert "error" not in result

    def test_deploy_from_catalog_inference_client_error(
        self,
        mock_mcp: MagicMock,
        mock_server: MagicMock,
        mock_catalog_data: dict,
        mock_runtime_template: dict,
    ) -> None:
        """Test deployment when InferenceClient fails."""
        # Mock catalog reading
        mock_pod = MagicMock()
        mock_pod.metadata.name = "model-catalog-pod"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_pod_list
        mock_server.k8s.exec_command.return_value = yaml.dump(mock_catalog_data)

        # Mock template retrieval
        mock_server.k8s.get_template.return_value = mock_runtime_template

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        # Patch InferenceClient to raise error
        with patch("rhoai_mcp.domains.catalog.tools.InferenceClient") as mock_inf_client_class:
            mock_inf_client = MagicMock()
            mock_inf_client.deploy_model.side_effect = RHOAIError("Deployment failed")
            mock_inf_client_class.return_value = mock_inf_client

            result = tools["deploy_from_catalog"](
                catalog_model_name="llama-2-7b",
                deployment_name="test-deploy",
                namespace="test-project",
            )

        assert "error" in result
        assert result["error"] == "Deployment failed"
