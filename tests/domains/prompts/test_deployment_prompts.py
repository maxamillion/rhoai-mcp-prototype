"""Tests for model deployment prompts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.prompts.deployment_prompts import register_prompts


class TestDeploymentPrompts:
    """Tests for deployment prompts registration and output."""

    def test_prompts_registered(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Verify all deployment prompts are registered."""
        register_prompts(mock_mcp, mock_server)

        assert "deploy-model" in mock_mcp._registered_prompts
        assert "deploy-llm" in mock_mcp._registered_prompts
        assert "test-endpoint" in mock_mcp._registered_prompts
        assert "scale-model" in mock_mcp._registered_prompts

    def test_deploy_model_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify deploy-model prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["deploy-model"]["function"]
        result = prompt_func(
            namespace="my-project",
            model_name="my-model",
            storage_uri="s3://bucket/model",
            model_format="onnx",
        )

        assert "my-project" in result
        assert "my-model" in result
        assert "s3://bucket/model" in result
        assert "onnx" in result
        assert "deploy_model" in result
        assert "list_serving_runtimes" in result

    def test_deploy_model_has_description(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify deploy-model prompt has a description."""
        register_prompts(mock_mcp, mock_server)

        prompt_info = mock_mcp._registered_prompts["deploy-model"]
        assert prompt_info["description"] is not None
        assert "deploy" in prompt_info["description"].lower()

    def test_deploy_model_default_format(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify deploy-model uses onnx as default format."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["deploy-model"]["function"]
        result = prompt_func(
            namespace="test",
            model_name="test-model",
            storage_uri="pvc://data/model",
        )

        assert "onnx" in result

    def test_deploy_llm_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify deploy-llm prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["deploy-llm"]["function"]
        result = prompt_func(
            namespace="my-project",
            model_name="llama-7b",
            model_id="meta-llama/Llama-2-7b-hf",
        )

        assert "my-project" in result
        assert "llama-7b" in result
        assert "meta-llama/Llama-2-7b-hf" in result
        assert "GPU" in result or "gpu" in result
        assert "vLLM" in result or "TGIS" in result

    def test_deploy_llm_includes_sizing_guide(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify deploy-llm includes model sizing information."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["deploy-llm"]["function"]
        result = prompt_func(
            namespace="test",
            model_name="test",
            model_id="test",
        )

        # Should mention model sizes
        assert "7B" in result or "13B" in result or "70B" in result

    def test_test_endpoint_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify test-endpoint prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["test-endpoint"]["function"]
        result = prompt_func(namespace="my-project", model_name="my-model")

        assert "my-project" in result
        assert "my-model" in result
        assert "get_inference_service" in result
        assert "get_model_endpoint" in result
        # Should include example request formats
        assert "json" in result.lower() or "JSON" in result

    def test_scale_model_output(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Verify scale-model prompt output."""
        register_prompts(mock_mcp, mock_server)

        prompt_func = mock_mcp._registered_prompts["scale-model"]["function"]
        result = prompt_func(namespace="my-project", model_name="my-model")

        assert "my-project" in result
        assert "my-model" in result
        assert "get_inference_service" in result
        assert "replica" in result.lower()
        # Should mention scale-to-zero option
        assert "zero" in result.lower() or "min_replicas=0" in result
