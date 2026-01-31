"""Tests for project setup prompts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.prompts.project_prompts import register_prompts


class TestProjectPrompts:
    """Tests for project setup prompts registration and output."""

    def test_prompts_registered(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Verify all project prompts are registered."""
        register_prompts(mock_mcp, mock_server)

        assert "setup-training-project" in mock_mcp._registered_prompts
        assert "setup-inference-project" in mock_mcp._registered_prompts
        assert "add-data-connection" in mock_mcp._registered_prompts

    def test_setup_training_project_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify setup-training-project prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["setup-training-project"]["function"]
        result = prompt_func(project_name="my-training", display_name="My Training Project")

        assert "my-training" in result
        assert "My Training Project" in result
        assert "create_data_science_project" in result
        assert "setup_training_runtime" in result
        assert "setup_training_storage" in result

    def test_setup_training_project_has_description(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify setup-training-project prompt has a description."""
        register_prompts(mock_mcp, mock_server)

        prompt_info = mock_mcp._registered_prompts["setup-training-project"]
        assert prompt_info["description"] is not None
        assert "training" in prompt_info["description"].lower()

    def test_setup_training_project_default_display_name(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify setup-training-project uses project name as default display name."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["setup-training-project"]["function"]
        result = prompt_func(project_name="my-training")

        # Should use project_name as display name when not provided
        assert "my-training" in result

    def test_setup_inference_project_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify setup-inference-project prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["setup-inference-project"]["function"]
        result = prompt_func(project_name="my-inference", display_name="My Inference Project")

        assert "my-inference" in result
        assert "My Inference Project" in result
        assert "create_data_science_project" in result
        assert "create_s3_data_connection" in result
        assert "list_serving_runtimes" in result

    def test_setup_inference_project_mentions_modelmesh(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify setup-inference-project mentions ModelMesh option."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["setup-inference-project"]["function"]
        result = prompt_func(project_name="test")

        assert "modelmesh" in result.lower() or "ModelMesh" in result

    def test_add_data_connection_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify add-data-connection prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["add-data-connection"]["function"]
        result = prompt_func(namespace="my-project")

        assert "my-project" in result
        assert "list_data_connections" in result
        assert "create_s3_data_connection" in result
        assert "get_data_connection" in result
        # Should mention required credentials
        assert "Access Key" in result or "access_key" in result
