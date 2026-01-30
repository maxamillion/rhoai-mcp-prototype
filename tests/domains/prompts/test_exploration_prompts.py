"""Tests for cluster and project exploration prompts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.prompts.exploration_prompts import register_prompts


class TestExplorationPrompts:
    """Tests for exploration prompts registration and output."""

    def test_prompts_registered(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Verify all exploration prompts are registered."""
        register_prompts(mock_mcp, mock_server)

        assert "explore-cluster" in mock_mcp._registered_prompts
        assert "explore-project" in mock_mcp._registered_prompts
        assert "find-gpus" in mock_mcp._registered_prompts
        assert "whats-running" in mock_mcp._registered_prompts

    def test_explore_cluster_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify explore-cluster prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["explore-cluster"]["function"]
        result = prompt_func()

        # Should reference relevant tools
        assert "cluster_summary" in result
        assert "list_data_science_projects" in result
        assert "get_cluster_resources" in result

    def test_explore_cluster_has_description(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify explore-cluster prompt has a description."""
        register_prompts(mock_mcp, mock_server)

        prompt_info = mock_mcp._registered_prompts["explore-cluster"]
        assert prompt_info["description"] is not None
        assert "cluster" in prompt_info["description"].lower()

    def test_explore_project_output_contains_namespace(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify explore-project prompt includes the namespace."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["explore-project"]["function"]
        result = prompt_func(namespace="my-ml-project")

        assert "my-ml-project" in result
        assert "project_summary" in result
        assert "list_workbenches" in result
        assert "list_inference_services" in result

    def test_find_gpus_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify find-gpus prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["find-gpus"]["function"]
        result = prompt_func()

        assert "get_cluster_resources" in result
        assert "GPU" in result or "gpu" in result
        assert "estimate_resources" in result

    def test_whats_running_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify whats-running prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["whats-running"]["function"]
        result = prompt_func()

        assert "cluster_summary" in result
        assert "list_training_jobs" in result
        assert "list_workbenches" in result
