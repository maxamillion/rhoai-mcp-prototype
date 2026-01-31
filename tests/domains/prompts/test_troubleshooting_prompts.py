"""Tests for troubleshooting prompts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.prompts.troubleshooting_prompts import register_prompts


class TestTroubleshootingPrompts:
    """Tests for troubleshooting prompts registration and output."""

    def test_prompts_registered(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Verify all troubleshooting prompts are registered."""
        register_prompts(mock_mcp, mock_server)

        assert "troubleshoot-training" in mock_mcp._registered_prompts
        assert "troubleshoot-workbench" in mock_mcp._registered_prompts
        assert "troubleshoot-model" in mock_mcp._registered_prompts
        assert "analyze-oom" in mock_mcp._registered_prompts

    def test_troubleshoot_training_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify troubleshoot-training prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["troubleshoot-training"]["function"]
        result = prompt_func(namespace="my-project", job_name="failed-job")

        assert "my-project" in result
        assert "failed-job" in result
        assert "get_training_job" in result
        assert "analyze_training_failure" in result
        assert "get_job_events" in result
        assert "get_training_logs" in result

    def test_troubleshoot_training_has_description(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify troubleshoot-training prompt has a description."""
        register_prompts(mock_mcp, mock_server)

        prompt_info = mock_mcp._registered_prompts["troubleshoot-training"]
        assert prompt_info["description"] is not None
        assert "training" in prompt_info["description"].lower()

    def test_troubleshoot_workbench_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify troubleshoot-workbench prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["troubleshoot-workbench"]["function"]
        result = prompt_func(namespace="my-project", workbench_name="my-notebook")

        assert "my-project" in result
        assert "my-notebook" in result
        assert "get_workbench" in result
        assert "resource_status" in result

    def test_troubleshoot_model_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify troubleshoot-model prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["troubleshoot-model"]["function"]
        result = prompt_func(namespace="my-project", model_name="my-model")

        assert "my-project" in result
        assert "my-model" in result
        assert "get_inference_service" in result
        assert "get_model_endpoint" in result

    def test_analyze_oom_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify analyze-oom prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["analyze-oom"]["function"]
        result = prompt_func(namespace="my-project", job_name="oom-job")

        assert "my-project" in result
        assert "oom-job" in result
        assert "OOMKilled" in result or "OOM" in result
        assert "get_job_events" in result
        assert "estimate_resources" in result
        # Should suggest mitigation strategies
        assert "batch_size" in result or "qlora" in result
