"""Tests for unified training tool."""

from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.domains.training.tools.unified import register_tools


@pytest.fixture
def mock_mcp() -> MagicMock:
    """Create a mock FastMCP server that captures tool registrations."""
    mock = MagicMock()
    registered_tools: dict = {}

    def capture_tool():
        def decorator(f):
            registered_tools[f.__name__] = f
            return f

        return decorator

    mock.tool = capture_tool
    mock._registered_tools = registered_tools
    return mock


@pytest.fixture
def mock_server(mock_k8s_client: MagicMock) -> MagicMock:
    """Create a mock RHOAIServer with K8sClient."""
    server = MagicMock()
    server.k8s = mock_k8s_client
    server.config.is_operation_allowed.return_value = (True, None)
    return server


class TestUnifiedTrainingTool:
    """Tests for the unified training() tool."""

    def test_tool_registration(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """training tool is registered."""
        register_tools(mock_mcp, mock_server)
        assert "training" in mock_mcp._registered_tools

    def test_invalid_action(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Invalid action returns error."""
        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(action="invalid_action", namespace="test")

        assert "error" in result
        assert "valid_actions" in result

    def test_list_requires_namespace(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """List action requires namespace."""
        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(action="list", namespace=None)

        assert "error" in result

    @patch("rhoai_mcp.domains.training.tools.unified.TrainingClient")
    def test_list_action(
        self, mock_client_class: MagicMock, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """List action returns jobs."""
        # Setup mock
        mock_job = MagicMock()
        mock_job.name = "test-job"
        mock_job.status.value = "Running"
        mock_job.model_id = "test/model"
        mock_job.progress = None
        mock_client_class.return_value.list_training_jobs.return_value = [mock_job]

        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(action="list", namespace="test")

        assert result["action"] == "list"
        assert result["count"] == 1
        assert result["jobs"][0]["name"] == "test-job"

    @patch("rhoai_mcp.domains.training.tools.unified.TrainingClient")
    def test_get_action(
        self, mock_client_class: MagicMock, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Get action returns job details."""
        # Setup mock
        mock_job = MagicMock()
        mock_job.name = "test-job"
        mock_job.namespace = "test"
        mock_job.status.value = "Running"
        mock_job.model_id = "test/model"
        mock_job.dataset_id = "test/dataset"
        mock_job.runtime_ref = "runtime"
        mock_job.num_nodes = 1
        mock_job.gpus_per_node = 2
        mock_job.creation_timestamp = "2024-01-01T00:00:00Z"
        mock_job.progress = None
        mock_client_class.return_value.get_training_job.return_value = mock_job

        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(action="get", namespace="test", name="test-job")

        assert result["action"] == "get"
        assert result["name"] == "test-job"

    def test_get_requires_name(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Get action requires name."""
        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(action="get", namespace="test", name=None)

        assert "error" in result

    @patch("rhoai_mcp.domains.training.tools.unified.TrainingClient")
    def test_suspend_action(
        self, mock_client_class: MagicMock, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Suspend action calls client."""
        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(action="suspend", namespace="test", name="job-1")

        assert result["success"] is True
        assert result["action"] == "suspend"
        mock_client_class.return_value.suspend_training_job.assert_called_once_with("test", "job-1")

    @patch("rhoai_mcp.domains.training.tools.unified.TrainingClient")
    def test_resume_action(
        self, mock_client_class: MagicMock, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Resume action calls client."""
        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(action="resume", namespace="test", name="job-1")

        assert result["success"] is True
        assert result["action"] == "resume"
        mock_client_class.return_value.resume_training_job.assert_called_once_with("test", "job-1")

    def test_delete_requires_confirm(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Delete action requires confirmation."""
        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(action="delete", namespace="test", name="job-1", confirm=False)

        assert "error" in result
        assert "not confirmed" in result["error"].lower()

    @patch("rhoai_mcp.domains.training.tools.unified.TrainingClient")
    def test_delete_with_confirm(
        self, mock_client_class: MagicMock, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Delete action with confirmation."""
        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(action="delete", namespace="test", name="job-1", confirm=True)

        assert result["success"] is True
        mock_client_class.return_value.delete_training_job.assert_called_once()

    def test_create_requires_model_id(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Create action requires model_id."""
        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(action="create", namespace="test", model_id=None)

        assert "error" in result
        assert "model_id" in result["error"]

    def test_create_preview_without_confirm(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Create without confirmed returns preview."""
        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(
            action="create",
            namespace="test",
            model_id="test/model",
            dataset_id="test/dataset",
            runtime_name="runtime",
            confirmed=False,
        )

        assert "preview" in result
        assert "message" in result
        assert result["preview"]["model_id"] == "test/model"

    @patch("rhoai_mcp.domains.training.tools.unified.TrainingClient")
    def test_logs_action(
        self, mock_client_class: MagicMock, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Logs action returns logs."""
        mock_client_class.return_value.get_training_logs.return_value = "log line 1\nlog line 2"

        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(action="logs", namespace="test", name="job-1")

        assert result["action"] == "logs"
        assert "logs" in result
        assert result["lines_returned"] == 2

    def test_estimate_action(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Estimate action returns resource estimates."""
        register_tools(mock_mcp, mock_server)
        training = mock_mcp._registered_tools["training"]

        result = training(action="estimate", model_id="meta-llama/Llama-2-7b-hf")

        assert result["action"] == "estimate"
        assert "estimated_params_billion" in result
        assert result["estimated_params_billion"] == 7.0
