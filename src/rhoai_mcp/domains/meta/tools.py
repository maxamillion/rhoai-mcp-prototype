"""MCP Tools for tool discovery and workflow guidance."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


# Tool categories with workflow hints
TOOL_CATEGORIES: dict[str, dict[str, Any]] = {
    "discovery": {
        "description": "Start here to understand cluster state",
        "tools": ["cluster_summary", "project_summary", "explore_cluster", "list_resources"],
        "use_first": True,
    },
    "training": {
        "description": "Model fine-tuning operations",
        "tools": [
            "prepare_training",
            "train",
            "get_training_progress",
            "get_training_logs",
            "analyze_training_failure",
        ],
        "typical_workflow": [
            "prepare_training",
            "train (with confirmed=True)",
            "get_training_progress",
        ],
    },
    "inference": {
        "description": "Model deployment and serving",
        "tools": [
            "prepare_model_deployment",
            "deploy_model",
            "get_model_endpoint",
            "test_model_endpoint",
            "recommend_serving_runtime",
        ],
        "typical_workflow": [
            "prepare_model_deployment",
            "deploy_model",
            "test_model_endpoint",
        ],
    },
    "workbenches": {
        "description": "Jupyter notebook environments",
        "tools": [
            "list_workbenches",
            "create_workbench",
            "start_workbench",
            "stop_workbench",
            "get_workbench_url",
        ],
    },
    "diagnostics": {
        "description": "Troubleshooting and debugging",
        "tools": [
            "diagnose_resource",
            "analyze_training_failure",
            "get_job_events",
            "get_training_logs",
        ],
    },
    "resources": {
        "description": "Generic resource operations",
        "tools": [
            "get_resource",
            "list_resources",
            "manage_resource",
            "resource_status",
        ],
    },
    "storage": {
        "description": "Storage and data connections",
        "tools": [
            "list_storage",
            "create_storage",
            "setup_training_storage",
            "list_data_connections",
            "create_s3_data_connection",
        ],
    },
}


# Intent patterns for suggest_tools
INTENT_PATTERNS = [
    {
        "patterns": ["train", "fine-tune", "finetune", "lora", "qlora"],
        "category": "training",
        "workflow": ["prepare_training", "train"],
        "explanation": "Training workflow: First use prepare_training() to check prerequisites, "
        "then train() with confirmed=True to start the job.",
    },
    {
        "patterns": ["deploy", "serve", "inference", "predict"],
        "category": "inference",
        "workflow": ["prepare_model_deployment", "deploy_model"],
        "explanation": "Deployment workflow: First use prepare_model_deployment() to validate, "
        "then deploy_model() to create the InferenceService.",
    },
    {
        "patterns": ["debug", "troubleshoot", "failed", "error", "why", "broken"],
        "category": "diagnostics",
        "workflow": ["diagnose_resource"],
        "explanation": "Use diagnose_resource() to get comprehensive diagnostics including "
        "status, events, logs, and suggested fixes.",
    },
    {
        "patterns": ["explore", "overview", "cluster", "what's running", "status"],
        "category": "discovery",
        "workflow": ["explore_cluster"],
        "explanation": "Use explore_cluster() to get a complete overview of all projects "
        "and resources in the cluster.",
    },
    {
        "patterns": ["notebook", "workbench", "jupyter", "code"],
        "category": "workbenches",
        "workflow": ["list_workbenches", "create_workbench"],
        "explanation": "Use list_workbenches() to see existing notebooks, "
        "create_workbench() to create a new one.",
    },
    {
        "patterns": ["storage", "pvc", "volume", "data connection", "s3"],
        "category": "storage",
        "workflow": ["list_storage", "list_data_connections"],
        "explanation": "Use list_storage() for PVCs, list_data_connections() for S3 connections.",
    },
]


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:  # noqa: ARG001
    """Register meta tools with the MCP server."""

    @mcp.tool()
    def suggest_tools(
        intent: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get recommended tools and workflow for a given intent.

        Given a natural language description of what you want to do,
        returns the recommended tools and typical workflow steps.

        Args:
            intent: What you want to do (e.g., "train a model", "debug failed job").
            context: Optional context like {"namespace": "...", "resource_name": "..."}.

        Returns:
            Recommended tools with:
            - workflow: List of tools to use in order
            - explanation: How to use them
            - example_calls: Example tool invocations
            - category: Tool category this matches

        Example intents:
            - "train llama on my dataset"
            - "check why my training job failed"
            - "deploy a model for inference"
            - "see what's running in my cluster"
        """
        intent_lower = intent.lower()
        context = context or {}

        # Find matching pattern
        best_match = None
        match_score = 0

        for pattern in INTENT_PATTERNS:
            score = sum(1 for p in pattern["patterns"] if p in intent_lower)
            if score > match_score:
                match_score = score
                best_match = pattern

        if not best_match:
            # Default to discovery
            best_match = INTENT_PATTERNS[3]  # explore/overview

        # Build example calls
        example_calls = []
        namespace = context.get("namespace", "my-project")
        resource_name = context.get("resource_name", "my-resource")

        for tool in best_match["workflow"]:
            if tool == "prepare_training":
                example_calls.append({
                    "tool": tool,
                    "args": {
                        "namespace": namespace,
                        "model_id": "meta-llama/Llama-2-7b-hf",
                        "dataset_id": "tatsu-lab/alpaca",
                    },
                })
            elif tool == "train":
                example_calls.append({
                    "tool": tool,
                    "args": {
                        "namespace": namespace,
                        "model_id": "meta-llama/Llama-2-7b-hf",
                        "dataset_id": "tatsu-lab/alpaca",
                        "runtime_name": "mcp-transformers-runtime",
                        "confirmed": True,
                    },
                })
            elif tool == "prepare_model_deployment":
                example_calls.append({
                    "tool": tool,
                    "args": {
                        "namespace": namespace,
                        "model_id": "meta-llama/Llama-2-7b-hf",
                    },
                })
            elif tool == "deploy_model":
                example_calls.append({
                    "tool": tool,
                    "args": {
                        "namespace": namespace,
                        "name": "my-model",
                        "runtime": "vllm-runtime",
                        "model_format": "pytorch",
                        "storage_uri": "pvc://model-storage/model",
                    },
                })
            elif tool == "diagnose_resource":
                example_calls.append({
                    "tool": tool,
                    "args": {
                        "resource_type": "training_job",
                        "name": resource_name,
                        "namespace": namespace,
                    },
                })
            elif tool == "explore_cluster":
                example_calls.append({
                    "tool": tool,
                    "args": {},
                })
            else:
                example_calls.append({
                    "tool": tool,
                    "args": {"namespace": namespace},
                })

        return {
            "intent": intent,
            "category": best_match["category"],
            "workflow": best_match["workflow"],
            "explanation": best_match["explanation"],
            "example_calls": example_calls,
            "all_categories": list(TOOL_CATEGORIES.keys()),
        }

    @mcp.tool()
    def list_tool_categories() -> dict[str, Any]:
        """List all available tool categories with descriptions.

        Returns a summary of tool categories organized by use case,
        helping you find the right tools for your task.

        Returns:
            Tool categories with descriptions and key tools.
        """
        categories = []
        for name, info in TOOL_CATEGORIES.items():
            categories.append({
                "category": name,
                "description": info["description"],
                "key_tools": info["tools"][:3],
                "use_first": info.get("use_first", False),
            })

        return {
            "categories": categories,
            "recommendation": "Start with 'discovery' tools like explore_cluster() "
            "to understand the cluster state before taking actions.",
        }
