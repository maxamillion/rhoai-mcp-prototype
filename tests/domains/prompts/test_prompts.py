"""Tests for main prompts registration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.prompts.prompts import register_prompts


class TestPromptsRegistration:
    """Tests for main prompts registration function."""

    def test_all_prompt_categories_registered(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify all prompt categories are registered."""
        register_prompts(mock_mcp, mock_server)

        # Training prompts (3)
        assert "train-model" in mock_mcp._registered_prompts
        assert "monitor-training" in mock_mcp._registered_prompts
        assert "resume-training" in mock_mcp._registered_prompts

        # Exploration prompts (4)
        assert "explore-cluster" in mock_mcp._registered_prompts
        assert "explore-project" in mock_mcp._registered_prompts
        assert "find-gpus" in mock_mcp._registered_prompts
        assert "whats-running" in mock_mcp._registered_prompts

        # Troubleshooting prompts (4)
        assert "troubleshoot-training" in mock_mcp._registered_prompts
        assert "troubleshoot-workbench" in mock_mcp._registered_prompts
        assert "troubleshoot-model" in mock_mcp._registered_prompts
        assert "analyze-oom" in mock_mcp._registered_prompts

        # Project setup prompts (3)
        assert "setup-training-project" in mock_mcp._registered_prompts
        assert "setup-inference-project" in mock_mcp._registered_prompts
        assert "add-data-connection" in mock_mcp._registered_prompts

        # Deployment prompts (4)
        assert "deploy-model" in mock_mcp._registered_prompts
        assert "deploy-llm" in mock_mcp._registered_prompts
        assert "test-endpoint" in mock_mcp._registered_prompts
        assert "scale-model" in mock_mcp._registered_prompts

    def test_total_prompt_count(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify total number of prompts registered."""
        register_prompts(mock_mcp, mock_server)

        # 3 + 4 + 4 + 3 + 4 = 18 prompts total
        assert len(mock_mcp._registered_prompts) == 18

    def test_all_prompts_have_descriptions(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify all prompts have descriptions."""
        register_prompts(mock_mcp, mock_server)

        for name, info in mock_mcp._registered_prompts.items():
            assert info["description"] is not None, f"Prompt {name} missing description"
            assert len(info["description"]) > 0, f"Prompt {name} has empty description"

    def test_all_prompts_are_callable(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify all prompt functions are callable."""
        register_prompts(mock_mcp, mock_server)

        for name, info in mock_mcp._registered_prompts.items():
            assert callable(info["function"]), f"Prompt {name} function not callable"

    def test_all_prompts_return_strings(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify all prompt functions return strings."""
        register_prompts(mock_mcp, mock_server)

        # Test prompts that require no arguments
        no_arg_prompts = ["explore-cluster", "find-gpus", "whats-running"]
        for name in no_arg_prompts:
            func = mock_mcp._registered_prompts[name]["function"]
            result = func()
            assert isinstance(result, str), f"Prompt {name} should return string"
            assert len(result) > 0, f"Prompt {name} returned empty string"

    def test_prompts_with_namespace_arg(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify prompts that take namespace argument work correctly."""
        register_prompts(mock_mcp, mock_server)

        namespace_prompts = [
            "explore-project",
            "add-data-connection",
        ]
        for name in namespace_prompts:
            func = mock_mcp._registered_prompts[name]["function"]
            result = func(namespace="test-namespace")
            assert isinstance(result, str)
            assert "test-namespace" in result
