"""Tests for training workflow prompts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.prompts.training_prompts import register_prompts


class TestTrainingPrompts:
    """Tests for training prompts registration and output."""

    def test_prompts_registered(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Verify all training prompts are registered."""
        register_prompts(mock_mcp, mock_server)

        assert "train-model" in mock_mcp._registered_prompts
        assert "monitor-training" in mock_mcp._registered_prompts
        assert "resume-training" in mock_mcp._registered_prompts

    def test_train_model_prompt_has_description(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify train-model prompt has a description."""
        register_prompts(mock_mcp, mock_server)

        prompt_info = mock_mcp._registered_prompts["train-model"]
        assert prompt_info["description"] is not None
        assert "fine-tun" in prompt_info["description"].lower()

    def test_train_model_output_contains_parameters(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify train-model prompt includes the provided parameters."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["train-model"]["function"]
        result = prompt_func(
            model_id="meta-llama/Llama-2-7b-hf",
            dataset_id="tatsu-lab/alpaca",
            namespace="my-project",
            method="qlora",
        )

        assert "meta-llama/Llama-2-7b-hf" in result
        assert "tatsu-lab/alpaca" in result
        assert "my-project" in result
        assert "qlora" in result

    def test_train_model_output_contains_workflow_steps(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify train-model prompt includes workflow guidance."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["train-model"]["function"]
        result = prompt_func(
            model_id="test-model",
            dataset_id="test-dataset",
            namespace="test-ns",
        )

        # Should reference relevant tools
        assert "check_training_prerequisites" in result
        assert "estimate_resources" in result
        assert "train" in result

    def test_monitor_training_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify monitor-training prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["monitor-training"]["function"]
        result = prompt_func(namespace="my-project", job_name="llama-finetune")

        assert "my-project" in result
        assert "llama-finetune" in result
        assert "get_training_job" in result
        assert "get_training_progress" in result
        assert "get_training_logs" in result

    def test_resume_training_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify resume-training prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["resume-training"]["function"]
        result = prompt_func(namespace="my-project", job_name="llama-finetune")

        assert "my-project" in result
        assert "llama-finetune" in result
        assert "resume_training_job" in result
        assert "manage_checkpoints" in result

    def test_train_model_default_method(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify train-model uses lora as default method."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["train-model"]["function"]
        result = prompt_func(
            model_id="test-model",
            dataset_id="test-dataset",
            namespace="test-ns",
        )

        assert "lora" in result
