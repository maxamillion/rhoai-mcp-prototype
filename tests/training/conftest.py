"""Pytest fixtures for training package tests."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_train_job() -> MagicMock:
    """Sample TrainJob CR with trainer status annotation."""
    mock = MagicMock()
    mock.metadata.name = "llama-finetune-abc123"
    mock.metadata.namespace = "training"
    mock.metadata.uid = "job-uid-12345"
    mock.metadata.creation_timestamp = datetime.now(timezone.utc)
    mock.metadata.labels = {
        "training.kubeflow.org/job-name": "llama-finetune-abc123",
    }
    mock.metadata.annotations = {
        "trainer.opendatahub.io/trainerStatus": json.dumps(
            {
                "trainingState": "Training",
                "currentEpoch": 3,
                "totalEpochs": 10,
                "currentStep": 1500,
                "totalSteps": 5000,
                "loss": 2.34,
                "learningRate": 0.0001,
                "throughput": 450.5,
                "gradientNorm": 2.1,
                "estimatedTimeRemaining": 3600,
            }
        ),
    }
    mock.spec = {
        "modelConfig": {
            "name": "meta-llama/Llama-2-7b-hf",
        },
        "datasetConfig": {
            "name": "tatsu-lab/alpaca",
        },
        "trainer": {
            "numNodes": 2,
            "resourcesPerNode": {
                "requests": {
                    "nvidia.com/gpu": "4",
                },
            },
        },
        "runtimeRef": {
            "name": "transformers-runtime",
        },
    }
    mock.status = {
        "conditions": [
            {"type": "Created", "status": "True"},
            {"type": "Running", "status": "True"},
        ],
    }
    return mock


@pytest.fixture
def sample_training_progress() -> dict:
    """Parsed training progress from annotation."""
    return {
        "trainingState": "Training",
        "currentEpoch": 3,
        "totalEpochs": 10,
        "currentStep": 1500,
        "totalSteps": 5000,
        "loss": 2.34,
        "learningRate": 0.0001,
        "throughput": 450.5,
        "gradientNorm": 2.1,
        "estimatedTimeRemaining": 3600,
    }


@pytest.fixture
def sample_cluster_training_runtime() -> MagicMock:
    """Sample ClusterTrainingRuntime CR."""
    mock = MagicMock()
    mock.metadata.name = "transformers-runtime"
    mock.metadata.namespace = None  # Cluster-scoped
    mock.metadata.uid = "runtime-uid-12345"
    mock.metadata.creation_timestamp = datetime.now(timezone.utc)
    mock.metadata.labels = {
        "training.kubeflow.org/framework": "transformers",
    }
    mock.metadata.annotations = {}
    mock.spec = {
        "template": {
            "spec": {
                "trainer": {
                    "image": "quay.io/modh/training:latest",
                },
                "initializers": [
                    {
                        "type": "model",
                        "image": "quay.io/modh/model-initializer:latest",
                    },
                    {
                        "type": "dataset",
                        "image": "quay.io/modh/dataset-initializer:latest",
                    },
                ],
            },
        },
    }
    return mock


@pytest.fixture
def mock_k8s_client() -> MagicMock:
    """Create a fully mocked K8sClient."""
    mock = MagicMock()
    mock.core_v1 = MagicMock()
    mock.dynamic = MagicMock()
    mock.is_connected = True
    return mock


@pytest.fixture
def mock_server(mock_k8s_client: MagicMock) -> MagicMock:
    """Create a mock RHOAIServer with K8sClient."""
    server = MagicMock()
    server.k8s = mock_k8s_client
    server.config.is_operation_allowed.return_value = (True, None)
    # Context window optimization settings
    server.config.default_list_limit = None
    server.config.max_list_limit = 100
    server.config.default_verbosity = "standard"
    return server


@pytest.fixture
def mock_mcp() -> MagicMock:
    """Create a mock FastMCP server that captures tool registrations."""
    mock = MagicMock()
    registered_tools = {}

    def capture_tool():
        def decorator(f):
            registered_tools[f.__name__] = f
            return f

        return decorator

    mock.tool = capture_tool
    mock._registered_tools = registered_tools
    return mock


@pytest.fixture
def sample_cluster_resources() -> dict:
    """Sample cluster resources response."""
    return {
        "cpu_total": 128,
        "cpu_allocatable": 120,
        "memory_total_gb": 512.0,
        "memory_allocatable_gb": 480.0,
        "node_count": 4,
        "has_gpus": True,
        "gpu_info": {
            "type": "nvidia.com/gpu",
            "total": 16,
            "available": 8,
            "nodes_with_gpu": 4,
        },
        "nodes": [
            {"name": "worker-1", "cpu": 30, "memory_gb": 120.0, "gpus": 4},
            {"name": "worker-2", "cpu": 30, "memory_gb": 120.0, "gpus": 4},
            {"name": "worker-3", "cpu": 30, "memory_gb": 120.0, "gpus": 4},
            {"name": "worker-4", "cpu": 30, "memory_gb": 120.0, "gpus": 4},
        ],
    }


@pytest.fixture
def sample_training_job_list() -> list[MagicMock]:
    """Sample list of training jobs."""
    jobs = []
    for i, status in enumerate(["Running", "Completed", "Failed"], 1):
        mock = MagicMock()
        mock.metadata.name = f"job-{i}"
        mock.metadata.namespace = "training"
        mock.metadata.uid = f"job-uid-{i}"
        mock.metadata.creation_timestamp = datetime.now(timezone.utc)
        mock.metadata.labels = {}
        mock.metadata.annotations = {}
        mock.spec = {
            "modelConfig": {"name": f"model-{i}"},
            "datasetConfig": {"name": f"dataset-{i}"},
        }
        mock.status = {
            "conditions": [{"type": status, "status": "True"}],
        }
        jobs.append(mock)
    return jobs
