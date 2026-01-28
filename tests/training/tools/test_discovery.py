"""Tests for training discovery tools."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.training.tools.discovery import register_tools


class TestDiscoveryTools:
    """Test discovery tools registration and execution."""

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
        # Context window optimization settings
        server.config.default_list_limit = None
        server.config.max_list_limit = 100
        server.config.default_verbosity = "standard"
        return server

    def test_register_tools_registers_all_discovery_tools(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Test that all discovery tools are registered."""
        register_tools(mock_mcp, mock_server)

        # Check that tool decorator was called for each tool
        assert mock_mcp.tool.call_count >= 4  # At least 4 discovery tools

    def test_list_training_jobs_empty(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Test listing training jobs when none exist."""
        # Setup mock to return empty list
        mock_server.k8s.list_resources.return_value = []

        # Get the registered tool
        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["list_training_jobs"](namespace="default")

        assert result["items"] == []
        assert result["total"] == 0
        assert result["namespace"] == "default"

    def test_list_training_jobs_with_results(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Test listing training jobs returns job info."""
        mock_server.k8s.list_resources.return_value = [
            _make_mock_resource("job-1", "default"),
            _make_mock_resource("job-2", "default"),
        ]

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["list_training_jobs"](namespace="default")

        assert result["total"] == 2
        assert len(result["items"]) == 2
        assert result["items"][0]["name"] == "job-1"

    def test_get_training_job(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Test getting a specific training job."""
        mock_server.k8s.get.return_value = _make_mock_resource(
            "my-job",
            "training",
            spec={
                "modelConfig": {"name": "meta-llama/Llama-2-7b-hf"},
                "datasetConfig": {"name": "tatsu-lab/alpaca"},
            },
        )

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["get_training_job"](namespace="training", name="my-job")

        assert result["name"] == "my-job"
        assert result["model_id"] == "meta-llama/Llama-2-7b-hf"
        assert result["dataset_id"] == "tatsu-lab/alpaca"

    def test_get_cluster_resources(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Test getting cluster resources."""
        # Mock node list
        mock_node = MagicMock()
        mock_node.metadata.name = "worker-1"
        mock_node.status.capacity = {"cpu": "32", "memory": "128Gi", "nvidia.com/gpu": "4"}
        mock_node.status.allocatable = {"cpu": "30", "memory": "120Gi", "nvidia.com/gpu": "4"}

        mock_node_list = MagicMock()
        mock_node_list.items = [mock_node]
        mock_server.k8s.core_v1.list_node.return_value = mock_node_list

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["get_cluster_resources"]()

        assert result["cpu_total"] == 32
        assert result["node_count"] == 1
        assert result["has_gpus"] is True
        assert result["gpu_info"]["total"] == 4

    def test_list_training_runtimes(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Test listing training runtimes."""
        mock_server.k8s.list_resources.return_value = [
            _make_mock_resource("transformers-runtime", None),
            _make_mock_resource("pytorch-runtime", None),
        ]

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f

            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["list_training_runtimes"]()

        assert result["count"] == 2
        assert len(result["runtimes"]) == 2


def _make_mock_resource(
    name: str,
    namespace: str | None = "default",
    spec: dict | None = None,
    status: dict | None = None,
    annotations: dict | None = None,
    labels: dict | None = None,
) -> MagicMock:
    """Create a mock Kubernetes resource."""
    mock = MagicMock()
    mock.metadata.name = name
    mock.metadata.namespace = namespace
    mock.metadata.uid = f"{name}-uid"
    mock.metadata.creation_timestamp = datetime.now(timezone.utc)
    mock.metadata.labels = labels or {}
    mock.metadata.annotations = annotations or {}
    mock.spec = spec or {}
    mock.status = status or {}
    return mock
