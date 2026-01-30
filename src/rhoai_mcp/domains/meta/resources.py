"""MCP Resources for tool discovery metadata."""

from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.meta.tools import TOOL_CATEGORIES

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_resources(mcp: FastMCP, server: "RHOAIServer") -> None:  # noqa: ARG001
    """Register meta resources with the MCP server."""

    @mcp.resource("rhoai://tools/categories")
    def tool_categories() -> dict:
        """Tool categories organized by use case with usage hints.

        Returns tools organized by category with workflow suggestions,
        helping AI agents navigate the available capabilities efficiently.
        """
        return {
            "categories": TOOL_CATEGORIES,
            "quick_reference": {
                "start_here": [
                    "explore_cluster() - Complete cluster overview",
                    "cluster_summary() - Quick cluster stats",
                    "project_summary(namespace) - Quick project stats",
                ],
                "training_workflow": [
                    "1. prepare_training() - Pre-flight checks",
                    "2. train(confirmed=True) - Start training",
                    "3. get_training_progress() - Monitor",
                ],
                "deployment_workflow": [
                    "1. prepare_model_deployment() - Pre-flight checks",
                    "2. deploy_model() - Deploy model",
                    "3. test_model_endpoint() - Verify",
                ],
                "troubleshooting": [
                    "diagnose_resource() - Comprehensive diagnostics",
                    "analyze_training_failure() - Training failure analysis",
                ],
            },
            "composite_tools": {
                "description": "High-level tools that combine multiple steps",
                "tools": [
                    "prepare_training - Combines estimate_resources + check_prerequisites + validate_config",
                    "prepare_model_deployment - Combines runtime discovery + validation + resource estimation",
                    "explore_cluster - Complete cluster exploration in one call",
                    "diagnose_resource - Comprehensive resource diagnostics",
                ],
            },
        }

    @mcp.resource("rhoai://tools/workflows")
    def tool_workflows() -> dict:
        """Common workflows with step-by-step tool usage.

        Provides detailed workflow guides for common tasks.
        """
        return {
            "train_model": {
                "description": "Fine-tune a model using LoRA/QLoRA",
                "steps": [
                    {
                        "step": 1,
                        "tool": "prepare_training",
                        "purpose": "Verify prerequisites and get suggested parameters",
                        "example": {
                            "namespace": "my-project",
                            "model_id": "meta-llama/Llama-2-7b-hf",
                            "dataset_id": "tatsu-lab/alpaca",
                        },
                    },
                    {
                        "step": 2,
                        "tool": "train",
                        "purpose": "Start the training job",
                        "note": "Use parameters from prepare_training result",
                        "example": {
                            "confirmed": True,
                        },
                    },
                    {
                        "step": 3,
                        "tool": "get_training_progress",
                        "purpose": "Monitor training progress",
                        "repeat": "Call periodically to check progress",
                    },
                ],
            },
            "deploy_model": {
                "description": "Deploy a model for inference",
                "steps": [
                    {
                        "step": 1,
                        "tool": "prepare_model_deployment",
                        "purpose": "Validate model and find compatible runtime",
                    },
                    {
                        "step": 2,
                        "tool": "deploy_model",
                        "purpose": "Create the InferenceService",
                        "note": "Use parameters from prepare_model_deployment result",
                    },
                    {
                        "step": 3,
                        "tool": "test_model_endpoint",
                        "purpose": "Verify the endpoint is accessible",
                    },
                ],
            },
            "debug_training": {
                "description": "Diagnose a failed or stuck training job",
                "steps": [
                    {
                        "step": 1,
                        "tool": "diagnose_resource",
                        "purpose": "Get comprehensive diagnostics",
                        "example": {
                            "resource_type": "training_job",
                            "name": "my-job",
                            "namespace": "my-project",
                        },
                    },
                    {
                        "step": 2,
                        "tool": "analyze_training_failure",
                        "purpose": "Deep analysis if job failed",
                        "conditional": "Only if status is Failed",
                    },
                ],
            },
            "explore_cluster": {
                "description": "Understand what's in the cluster",
                "steps": [
                    {
                        "step": 1,
                        "tool": "explore_cluster",
                        "purpose": "Get complete cluster overview with all projects",
                    },
                    {
                        "step": 2,
                        "tool": "project_summary",
                        "purpose": "Drill into specific project",
                        "conditional": "If you need details about one project",
                    },
                ],
            },
        }
