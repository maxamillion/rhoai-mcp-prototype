"""Tests for response formatting utilities."""

from unittest.mock import MagicMock
from datetime import datetime

import pytest

from rhoai_mcp.utils.response import (
    PaginatedResponse,
    ResponseBuilder,
    Verbosity,
    paginate,
)


class TestVerbosity:
    """Tests for Verbosity enum."""

    def test_from_str_valid_values(self) -> None:
        """Test parsing valid verbosity strings."""
        assert Verbosity.from_str("minimal") == Verbosity.MINIMAL
        assert Verbosity.from_str("standard") == Verbosity.STANDARD
        assert Verbosity.from_str("full") == Verbosity.FULL

    def test_from_str_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert Verbosity.from_str("MINIMAL") == Verbosity.MINIMAL
        assert Verbosity.from_str("Standard") == Verbosity.STANDARD
        assert Verbosity.from_str("FULL") == Verbosity.FULL

    def test_from_str_invalid_defaults_to_standard(self) -> None:
        """Test invalid values default to STANDARD."""
        assert Verbosity.from_str("invalid") == Verbosity.STANDARD
        assert Verbosity.from_str("") == Verbosity.STANDARD

    def test_from_str_none_defaults_to_standard(self) -> None:
        """Test None defaults to STANDARD."""
        assert Verbosity.from_str(None) == Verbosity.STANDARD


class TestPaginate:
    """Tests for paginate function."""

    def test_paginate_no_limit(self) -> None:
        """Test pagination without limit returns all items."""
        items = [1, 2, 3, 4, 5]
        result, total = paginate(items)
        assert result == items
        assert total == 5

    def test_paginate_with_limit(self) -> None:
        """Test pagination with limit."""
        items = [1, 2, 3, 4, 5]
        result, total = paginate(items, limit=3)
        assert result == [1, 2, 3]
        assert total == 5

    def test_paginate_with_offset(self) -> None:
        """Test pagination with offset."""
        items = [1, 2, 3, 4, 5]
        result, total = paginate(items, offset=2)
        assert result == [3, 4, 5]
        assert total == 5

    def test_paginate_with_offset_and_limit(self) -> None:
        """Test pagination with both offset and limit."""
        items = [1, 2, 3, 4, 5]
        result, total = paginate(items, offset=1, limit=2)
        assert result == [2, 3]
        assert total == 5

    def test_paginate_offset_beyond_items(self) -> None:
        """Test pagination with offset beyond item count."""
        items = [1, 2, 3]
        result, total = paginate(items, offset=10)
        assert result == []
        assert total == 3

    def test_paginate_empty_list(self) -> None:
        """Test pagination with empty list."""
        result, total = paginate([], limit=10)
        assert result == []
        assert total == 0


class TestPaginatedResponse:
    """Tests for PaginatedResponse builder."""

    def test_build_basic(self) -> None:
        """Test basic response building."""
        items = [{"name": "a"}, {"name": "b"}]
        response = PaginatedResponse.build(items, total=10, offset=0, limit=2)

        assert response["items"] == items
        assert response["total"] == 10
        assert response["offset"] == 0
        assert response["limit"] == 2
        assert response["has_more"] is True

    def test_build_no_more_items(self) -> None:
        """Test has_more is False when all items returned."""
        items = [{"name": "a"}, {"name": "b"}]
        response = PaginatedResponse.build(items, total=2, offset=0, limit=None)

        assert response["has_more"] is False

    def test_build_last_page(self) -> None:
        """Test has_more is False on last page."""
        items = [{"name": "c"}]
        response = PaginatedResponse.build(items, total=3, offset=2, limit=10)

        assert response["has_more"] is False


class TestResponseBuilderWorkbench:
    """Tests for ResponseBuilder workbench methods."""

    @pytest.fixture
    def mock_workbench(self) -> MagicMock:
        """Create a mock workbench model."""
        wb = MagicMock()
        wb.metadata.name = "test-wb"
        wb.metadata.namespace = "test-ns"
        wb.metadata.uid = "test-uid-123"
        wb.metadata.kind = "Notebook"
        wb.metadata.api_version = "kubeflow.org/v1"
        wb.metadata.labels = {"app": "jupyter"}
        wb.metadata.annotations = {"note": "test"}
        wb.metadata.creation_timestamp = datetime(2024, 1, 1, 0, 0, 0)
        wb.metadata.to_source_dict.return_value = {
            "kind": "Notebook",
            "api_version": "kubeflow.org/v1",
            "name": "test-wb",
            "namespace": "test-ns",
            "uid": "test-uid-123",
        }
        wb.display_name = "Test Workbench"
        wb.status.value = "Running"
        wb.image = "jupyter:latest"
        wb.image_display_name = "Jupyter Notebook"
        wb.size = "Small"
        wb.url = "https://test-wb.example.com"
        wb.stopped_time = None
        wb.volumes = ["storage-pvc"]
        wb.env_from = ["secret:aws-creds"]
        wb.resources = MagicMock()
        wb.resources.cpu_request = "500m"
        wb.resources.cpu_limit = "2"
        wb.resources.memory_request = "1Gi"
        wb.resources.memory_limit = "4Gi"
        wb.resources.gpu_request = None
        wb.resources.gpu_limit = None
        wb.conditions = [
            MagicMock(type="Ready", status="True", reason=None, message=None)
        ]
        return wb

    def test_workbench_list_item_minimal(self, mock_workbench: MagicMock) -> None:
        """Test minimal verbosity returns only essential fields."""
        result = ResponseBuilder.workbench_list_item(mock_workbench, Verbosity.MINIMAL)

        assert result["name"] == "test-wb"
        assert result["status"] == "Running"
        assert "_source" in result
        assert result["_source"]["kind"] == "Notebook"
        assert result["_source"]["api_version"] == "kubeflow.org/v1"

    def test_workbench_list_item_standard(self, mock_workbench: MagicMock) -> None:
        """Test standard verbosity returns key fields."""
        result = ResponseBuilder.workbench_list_item(mock_workbench, Verbosity.STANDARD)

        assert "name" in result
        assert "status" in result
        assert "display_name" in result
        assert "image" in result
        assert "url" in result
        assert "_source" in result
        # Should not include labels/annotations in standard
        assert "labels" not in result
        assert "annotations" not in result

    def test_workbench_list_item_full(self, mock_workbench: MagicMock) -> None:
        """Test full verbosity includes all fields."""
        result = ResponseBuilder.workbench_list_item(mock_workbench, Verbosity.FULL)

        assert "labels" in result
        assert "annotations" in result
        assert "conditions" in result
        assert "resources" in result
        assert "_source" in result

    def test_workbench_detail_minimal(self, mock_workbench: MagicMock) -> None:
        """Test minimal verbosity for detail view."""
        result = ResponseBuilder.workbench_detail(mock_workbench, Verbosity.MINIMAL)

        assert result["name"] == "test-wb"
        assert result["namespace"] == "test-ns"
        assert result["status"] == "Running"
        assert result["url"] == "https://test-wb.example.com"
        assert "_source" in result
        assert result["_source"]["kind"] == "Notebook"


class TestResponseBuilderProject:
    """Tests for ResponseBuilder project methods."""

    @pytest.fixture
    def mock_project(self) -> MagicMock:
        """Create a mock project model."""
        project = MagicMock()
        project.metadata.name = "test-project"
        project.metadata.namespace = None
        project.metadata.uid = "project-uid-123"
        project.metadata.kind = "Project"
        project.metadata.api_version = "project.openshift.io/v1"
        project.metadata.labels = {"team": "ml"}
        project.metadata.annotations = {}
        project.metadata.creation_timestamp = datetime(2024, 1, 1, 0, 0, 0)
        project.metadata.to_source_dict.return_value = {
            "kind": "Project",
            "api_version": "project.openshift.io/v1",
            "name": "test-project",
            "namespace": None,
            "uid": "project-uid-123",
        }
        project.display_name = "Test Project"
        project.description = "A test project"
        project.requester = "user@example.com"
        project.is_modelmesh_enabled = False
        project.status.value = "Active"
        project.resource_summary = MagicMock()
        project.resource_summary.workbenches = 3
        project.resource_summary.workbenches_running = 2
        project.resource_summary.models = 1
        project.resource_summary.models_ready = 1
        project.resource_summary.pipelines = 0
        project.resource_summary.data_connections = 2
        project.resource_summary.storage = 3
        return project

    def test_project_list_item_minimal(self, mock_project: MagicMock) -> None:
        """Test minimal verbosity for project list."""
        result = ResponseBuilder.project_list_item(mock_project, Verbosity.MINIMAL)

        assert result["name"] == "test-project"
        assert result["status"] == "Active"
        assert "_source" in result
        assert result["_source"]["kind"] == "Project"

    def test_project_list_item_standard(self, mock_project: MagicMock) -> None:
        """Test standard verbosity for project list."""
        result = ResponseBuilder.project_list_item(mock_project, Verbosity.STANDARD)

        assert "name" in result
        assert "display_name" in result
        assert "description" in result
        assert "is_modelmesh_enabled" in result
        assert "_source" in result
        assert "labels" not in result

    def test_project_detail_minimal_includes_resources(
        self, mock_project: MagicMock
    ) -> None:
        """Test minimal detail includes basic resource counts."""
        result = ResponseBuilder.project_detail(mock_project, Verbosity.MINIMAL)

        assert result["name"] == "test-project"
        assert "resources" in result
        assert result["resources"]["workbenches"] == 3
        assert result["resources"]["models"] == 1
        assert "_source" in result
        assert result["_source"]["kind"] == "Project"


class TestResponseBuilderTrainingJob:
    """Tests for ResponseBuilder training job methods."""

    @pytest.fixture
    def mock_training_job(self) -> MagicMock:
        """Create a mock training job model."""
        job = MagicMock()
        job.name = "train-abc123"
        job.namespace = "test-ns"
        job.uid = "train-uid-123"
        job.kind = "TrainJob"
        job.api_version = "trainer.kubeflow.org/v1"
        job.to_source_dict.return_value = {
            "kind": "TrainJob",
            "api_version": "trainer.kubeflow.org/v1",
            "name": "train-abc123",
            "namespace": "test-ns",
            "uid": "train-uid-123",
        }
        job.status.value = "Running"
        job.model_id = "llama-7b"
        job.dataset_id = "alpaca"
        job.num_nodes = 2
        job.gpus_per_node = 4
        job.runtime_ref = "pytorch-runtime"
        job.checkpoint_dir = "/checkpoints"
        job.creation_timestamp = "2024-01-01T00:00:00Z"
        job.progress = MagicMock()
        job.progress.state.value = "Training"
        job.progress.current_epoch = 2
        job.progress.total_epochs = 3
        job.progress.current_step = 500
        job.progress.total_steps = 1000
        job.progress.loss = 0.5
        job.progress.learning_rate = 1e-4
        job.progress.throughput = 100.0
        job.progress.gradient_norm = 0.1
        job.progress.progress_percent = 50.0
        job.progress.eta_seconds = 3600
        job.progress.progress_bar = lambda: "█████░░░░░"
        return job

    def test_training_job_list_item_minimal(
        self, mock_training_job: MagicMock
    ) -> None:
        """Test minimal verbosity for training job list."""
        result = ResponseBuilder.training_job_list_item(
            mock_training_job, Verbosity.MINIMAL
        )

        assert result["name"] == "train-abc123"
        assert result["status"] == "Running"
        assert result["progress_percent"] == 50.0
        assert "_source" in result
        assert result["_source"]["kind"] == "TrainJob"

    def test_training_job_list_item_standard(
        self, mock_training_job: MagicMock
    ) -> None:
        """Test standard verbosity for training job list."""
        result = ResponseBuilder.training_job_list_item(
            mock_training_job, Verbosity.STANDARD
        )

        assert "name" in result
        assert "model_id" in result
        assert "progress" in result
        assert "_source" in result
        assert "gpus_per_node" not in result  # Not in standard

    def test_training_job_detail_full(self, mock_training_job: MagicMock) -> None:
        """Test full verbosity for training job detail."""
        result = ResponseBuilder.training_job_detail(
            mock_training_job, Verbosity.FULL
        )

        assert "progress" in result
        assert "gradient_norm" in result["progress"]
        assert "_source" in result
        assert result["_source"]["kind"] == "TrainJob"


class TestResourceMetadataToSourceDict:
    """Tests for ResourceMetadata.to_source_dict method."""

    def test_to_source_dict_returns_expected_fields(self) -> None:
        """Test that to_source_dict returns all expected fields."""
        from rhoai_mcp.models.common import ResourceMetadata

        metadata = ResourceMetadata(
            name="test-resource",
            namespace="test-ns",
            uid="uid-123",
            kind="Notebook",
            api_version="kubeflow.org/v1",
        )

        source = metadata.to_source_dict()

        assert source == {
            "kind": "Notebook",
            "api_version": "kubeflow.org/v1",
            "name": "test-resource",
            "namespace": "test-ns",
            "uid": "uid-123",
        }

    def test_to_source_dict_with_none_values(self) -> None:
        """Test that to_source_dict handles None values."""
        from rhoai_mcp.models.common import ResourceMetadata

        metadata = ResourceMetadata(
            name="test-resource",
        )

        source = metadata.to_source_dict()

        assert source == {
            "kind": None,
            "api_version": None,
            "name": "test-resource",
            "namespace": None,
            "uid": None,
        }
