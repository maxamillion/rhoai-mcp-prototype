"""MCP Tools for training resource planning and validation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.training.client import TrainingClient
from rhoai_mcp.domains.training.models import PeftMethod

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


# GPU memory estimates by model size (in billions of parameters)
# Values are approximate based on typical requirements
GPU_MEMORY_ESTIMATES = {
    # (min_params, max_params): base_memory_gb
    (0, 1): 2,
    (1, 3): 6,
    (3, 7): 14,
    (7, 13): 26,
    (13, 30): 48,
    (30, 70): 80,
    (70, 200): 160,
}

# PEFT method memory multipliers
PEFT_MULTIPLIERS = {
    PeftMethod.FULL: 4.0,  # Full fine-tuning needs optimizer states
    PeftMethod.LORA: 1.8,  # LoRA adds adapter weights
    PeftMethod.QLORA: 1.2,  # QLoRA uses quantization
    PeftMethod.DORA: 1.8,  # DoRA similar to LoRA
}


def register_tools(mcp: FastMCP, server: RHOAIServer) -> None:
    """Register training planning tools with the MCP server."""

    @mcp.tool()
    def estimate_resources(
        model_id: str,
        method: str = "lora",
        batch_size: int = 32,
        sequence_length: int = 512,
        num_nodes: int = 1,
        gpus_per_node: int = 1,
    ) -> dict[str, Any]:
        """Estimate GPU and memory requirements for training.

        Calculates approximate resource requirements based on model size,
        fine-tuning method, and training configuration. Use this to plan
        resource allocation before creating training jobs.

        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-2-7b-hf").
            method: Fine-tuning method: "lora", "qlora", "dora", or "full".
            batch_size: Per-device batch size (default: 32).
            sequence_length: Maximum sequence length (default: 512).
            num_nodes: Number of training nodes (default: 1).
            gpus_per_node: GPUs per node (default: 1).

        Returns:
            Resource estimates including GPU memory and recommendations.
        """
        # Parse model size from name
        param_count = _extract_param_count(model_id)

        # Get base memory estimate
        base_memory = 16  # Default for unknown models
        for (min_p, max_p), mem in GPU_MEMORY_ESTIMATES.items():
            if min_p <= param_count < max_p:
                base_memory = mem
                break

        # Apply PEFT multiplier
        try:
            peft_method = PeftMethod(method.lower())
        except ValueError:
            peft_method = PeftMethod.LORA

        multiplier = PEFT_MULTIPLIERS.get(peft_method, 1.8)

        # Calculate memory components
        model_memory = base_memory
        optimizer_memory = (
            base_memory * 0.5 if peft_method == PeftMethod.FULL else base_memory * 0.1
        )
        activation_memory = (batch_size * sequence_length * param_count * 2) / (1024 * 1024 * 1024)
        activation_memory = min(activation_memory, base_memory * 0.5)  # Cap activation estimate

        total_memory = (model_memory + optimizer_memory + activation_memory) * multiplier

        # Calculate per-GPU requirement
        total_gpus = num_nodes * gpus_per_node
        per_gpu_memory = total_memory / total_gpus if total_gpus > 0 else total_memory

        # Determine recommended GPU type
        if per_gpu_memory <= 16:
            recommended_gpu = "NVIDIA T4 (16GB)"
        elif per_gpu_memory <= 24:
            recommended_gpu = "NVIDIA A10 (24GB)"
        elif per_gpu_memory <= 40:
            recommended_gpu = "NVIDIA A100-40GB"
        elif per_gpu_memory <= 80:
            recommended_gpu = "NVIDIA A100-80GB"
        else:
            recommended_gpu = "NVIDIA H100 (80GB) or multiple GPUs"

        # Calculate recommended GPU count
        recommended_gpus = 1
        if per_gpu_memory > 80:
            recommended_gpus = int((total_memory / 80) + 0.5)

        # Estimate storage needs
        storage_gb = int(param_count * 4) + 50  # Model checkpoints + buffer

        return {
            "model_id": model_id,
            "estimated_params_billion": param_count,
            "method": peft_method.value,
            "model_memory_gb": round(model_memory, 1),
            "optimizer_state_gb": round(optimizer_memory, 1),
            "activation_memory_gb": round(activation_memory, 1),
            "total_required_gb": round(total_memory, 1),
            "per_gpu_memory_gb": round(per_gpu_memory, 1),
            "recommended_gpus": recommended_gpus,
            "recommended_gpu_type": recommended_gpu,
            "storage_gb": storage_gb,
            "configuration": {
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "num_nodes": num_nodes,
                "gpus_per_node": gpus_per_node,
            },
        }

    @mcp.tool()
    def check_training_prerequisites(
        namespace: str,
        model_id: str,
        dataset_id: str,
        checkpoint_storage: str | None = None,
    ) -> dict[str, Any]:
        """Check prerequisites before starting a training job.

        Performs comprehensive pre-flight checks including cluster
        connectivity, GPU availability, storage validation, and
        runtime availability.

        Args:
            namespace: The namespace for the training job.
            model_id: Model identifier to train.
            dataset_id: Dataset identifier to use.
            checkpoint_storage: Optional PVC name for checkpoints.

        Returns:
            Status report with any actions needed.
        """
        client = TrainingClient(server.k8s)
        checks = []
        actions_needed = []
        all_passed = True

        # Check 1: Cluster connectivity
        try:
            resources = client.get_cluster_resources()
            checks.append(
                {
                    "name": "Cluster connectivity",
                    "passed": True,
                    "message": f"Connected to cluster with {resources.node_count} nodes",
                }
            )
        except Exception as e:
            checks.append(
                {
                    "name": "Cluster connectivity",
                    "passed": False,
                    "message": f"Failed to connect: {e}",
                }
            )
            all_passed = False
            actions_needed.append("Check Kubernetes configuration and credentials")

        # Check 2: GPU availability
        if resources:
            if resources.has_gpus and resources.gpu_info:
                checks.append(
                    {
                        "name": "GPU availability",
                        "passed": True,
                        "message": f"{resources.gpu_info.total} GPUs available ({resources.gpu_info.type})",
                    }
                )
            else:
                checks.append(
                    {
                        "name": "GPU availability",
                        "passed": False,
                        "message": "No GPUs detected in cluster",
                    }
                )
                all_passed = False
                actions_needed.append("Ensure GPU nodes are available and properly labeled")

        # Check 3: Training runtime availability
        try:
            runtimes = client.list_cluster_training_runtimes()
            if runtimes:
                checks.append(
                    {
                        "name": "Training runtimes",
                        "passed": True,
                        "message": f"{len(runtimes)} training runtimes available",
                    }
                )
            else:
                checks.append(
                    {
                        "name": "Training runtimes",
                        "passed": False,
                        "message": "No training runtimes configured",
                    }
                )
                all_passed = False
                actions_needed.append("Use setup_training_runtime() to create a runtime")
        except Exception:
            checks.append(
                {
                    "name": "Training runtimes",
                    "passed": False,
                    "message": "Failed to list training runtimes",
                }
            )
            all_passed = False

        # Check 4: Checkpoint storage (if specified)
        if checkpoint_storage:
            try:
                pvc = server.k8s.get_pvc(checkpoint_storage, namespace)
                phase = pvc.status.phase if pvc.status else "Unknown"
                if phase == "Bound":
                    checks.append(
                        {
                            "name": "Checkpoint storage",
                            "passed": True,
                            "message": f"PVC '{checkpoint_storage}' is bound",
                        }
                    )
                else:
                    checks.append(
                        {
                            "name": "Checkpoint storage",
                            "passed": False,
                            "message": f"PVC '{checkpoint_storage}' is in state: {phase}",
                        }
                    )
                    all_passed = False
            except Exception:
                checks.append(
                    {
                        "name": "Checkpoint storage",
                        "passed": False,
                        "message": f"PVC '{checkpoint_storage}' not found",
                    }
                )
                all_passed = False
                actions_needed.append(
                    f"Create PVC '{checkpoint_storage}' or use setup_training_storage()"
                )

        # Check 5: Model/Dataset ID format validation
        if "/" in model_id:
            checks.append(
                {
                    "name": "Model ID format",
                    "passed": True,
                    "message": f"Model ID '{model_id}' appears valid",
                }
            )
        else:
            checks.append(
                {
                    "name": "Model ID format",
                    "passed": False,
                    "message": "Model ID should be in format 'organization/model-name'",
                }
            )
            all_passed = False

        if "/" in dataset_id:
            checks.append(
                {
                    "name": "Dataset ID format",
                    "passed": True,
                    "message": f"Dataset ID '{dataset_id}' appears valid",
                }
            )
        else:
            checks.append(
                {
                    "name": "Dataset ID format",
                    "passed": False,
                    "message": "Dataset ID should be in format 'organization/dataset-name'",
                }
            )
            all_passed = False

        return {
            "all_passed": all_passed,
            "checks": checks,
            "actions_needed": actions_needed if actions_needed else None,
            "ready_to_train": all_passed,
        }

    @mcp.tool()
    def validate_training_config(
        namespace: str,
        model_id: str,
        dataset_id: str,
        runtime_name: str,
        pvc_name: str | None = None,
    ) -> dict[str, Any]:
        """Validate a training configuration before job creation.

        Checks that all referenced resources exist and are properly
        configured for training.

        Args:
            namespace: The namespace for the training job.
            model_id: Model identifier to validate.
            dataset_id: Dataset identifier to validate.
            runtime_name: Name of the training runtime to use.
            pvc_name: Optional PVC name to validate.

        Returns:
            Validation results with specific errors.
        """
        errors = []
        warnings = []

        # Validate runtime exists
        try:
            from rhoai_mcp.domains.training.crds import TrainingCRDs

            server.k8s.get(TrainingCRDs.CLUSTER_TRAINING_RUNTIME, runtime_name)
        except Exception:
            errors.append(f"Runtime '{runtime_name}' not found")

        # Validate PVC if specified
        if pvc_name:
            try:
                pvc = server.k8s.get_pvc(pvc_name, namespace)
                if pvc.status.phase != "Bound":
                    warnings.append(f"PVC '{pvc_name}' is not bound (current: {pvc.status.phase})")
            except Exception:
                errors.append(f"PVC '{pvc_name}' not found in namespace '{namespace}'")

        # Validate model ID format
        if not re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$", model_id):
            errors.append(f"Invalid model ID format: '{model_id}'")

        # Validate dataset ID format
        if not re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$", dataset_id):
            errors.append(f"Invalid dataset ID format: '{dataset_id}'")

        return {
            "valid": len(errors) == 0,
            "errors": errors if errors else None,
            "warnings": warnings if warnings else None,
            "configuration": {
                "namespace": namespace,
                "model_id": model_id,
                "dataset_id": dataset_id,
                "runtime_name": runtime_name,
                "pvc_name": pvc_name,
            },
        }

    @mcp.tool()
    def setup_hf_credentials(
        namespace: str,
        token: str,
        secret_name: str = "hf-token",
    ) -> dict[str, Any]:
        """Set up HuggingFace credentials for model/dataset access.

        Creates a Kubernetes secret containing the HuggingFace API token.
        This is required for accessing gated models or private datasets.

        Args:
            namespace: The namespace to create the secret in.
            token: HuggingFace API token.
            secret_name: Name for the secret (default: "hf-token").

        Returns:
            Secret creation confirmation with usage instructions.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        # Validate token format (should start with hf_)
        if not token.startswith("hf_"):
            return {
                "error": "Invalid token format",
                "message": "HuggingFace tokens should start with 'hf_'",
            }

        # Create or update secret
        try:
            # Check if secret exists
            try:
                server.k8s.get_secret(secret_name, namespace)
                # Secret exists, delete it first
                server.k8s.delete_secret(secret_name, namespace)
            except Exception:
                pass  # Secret doesn't exist

            # Create the secret
            server.k8s.create_secret(
                name=secret_name,
                namespace=namespace,
                data={"token": token},
                labels={
                    "app.kubernetes.io/managed-by": "rhoai-mcp",
                    "app.kubernetes.io/component": "hf-credentials",
                },
            )

            return {
                "success": True,
                "secret_name": secret_name,
                "namespace": namespace,
                "message": f"HuggingFace credentials stored in secret '{secret_name}'.",
                "usage": (
                    "The token will be automatically used by training jobs "
                    "that reference this secret."
                ),
            }
        except Exception as e:
            return {
                "error": f"Failed to create secret: {e}",
            }

    @mcp.tool()
    def prepare_training(
        namespace: str,
        model_id: str,
        dataset_id: str,
        runtime_name: str | None = None,
        method: str = "lora",
        create_storage: bool = True,
        storage_size_gb: int = 100,
    ) -> dict[str, Any]:
        """Complete pre-flight setup for a training job in a single call.

        This composite tool combines resource estimation, prerequisite checking,
        configuration validation, and optional storage creation. Use this before
        calling train() to ensure everything is ready.

        Args:
            namespace: The namespace for the training job.
            model_id: Model identifier (e.g., "meta-llama/Llama-2-7b-hf").
            dataset_id: Dataset identifier (e.g., "tatsu-lab/alpaca").
            runtime_name: Training runtime to use (auto-selected if None).
            method: Fine-tuning method: "lora", "qlora", "dora", or "full".
            create_storage: Whether to create checkpoint storage if needed.
            storage_size_gb: Size of checkpoint storage in GB.

        Returns:
            Complete preparation result with:
            - ready: Whether training can proceed
            - issues: List of problems found
            - resource_estimate: GPU/memory requirements
            - recommended_runtime: Runtime to use
            - storage_created: Whether storage was created
            - next_action: "train" or "fix_issues"
            - suggested_train_params: Parameters for train() call
        """
        issues: list[str] = []
        warnings: list[str] = []
        storage_created = False
        recommended_runtime = runtime_name

        # Step 1: Estimate resources
        resource_estimate = _estimate_resources_internal(model_id, method)

        # Step 2: Check prerequisites
        client = TrainingClient(server.k8s)
        prereq_passed = True

        # Check cluster connectivity and GPUs
        try:
            resources = client.get_cluster_resources()
            if not resources.has_gpus:
                issues.append("No GPUs available in cluster")
                prereq_passed = False
            elif resources.gpu_info:
                gpu_available = resources.gpu_info.available
                required = resource_estimate.get("recommended_gpus", 1)
                if gpu_available < required:
                    warnings.append(
                        f"Only {gpu_available} GPUs available, {required} recommended"
                    )
        except Exception as e:
            issues.append(f"Failed to check cluster resources: {e}")
            prereq_passed = False

        # Check/select runtime
        try:
            runtimes = client.list_cluster_training_runtimes()
            if not runtimes:
                issues.append("No training runtimes available")
                prereq_passed = False
            elif not runtime_name:
                # Auto-select first available runtime
                recommended_runtime = runtimes[0].name
        except Exception:
            issues.append("Failed to list training runtimes")
            prereq_passed = False

        # Validate runtime if specified
        if runtime_name:
            try:
                from rhoai_mcp.domains.training.crds import TrainingCRDs

                server.k8s.get(TrainingCRDs.CLUSTER_TRAINING_RUNTIME, runtime_name)
            except Exception:
                issues.append(f"Runtime '{runtime_name}' not found")
                prereq_passed = False

        # Validate model/dataset ID format
        if "/" not in model_id:
            issues.append("Model ID should be in format 'organization/model-name'")
            prereq_passed = False
        if "/" not in dataset_id:
            issues.append("Dataset ID should be in format 'organization/dataset-name'")
            prereq_passed = False

        # Step 3: Handle storage
        pvc_name = f"training-checkpoints-{namespace}"
        storage_exists = False

        try:
            pvc = server.k8s.get_pvc(pvc_name, namespace)
            if pvc.status.phase == "Bound":
                storage_exists = True
        except Exception:
            pass

        if not storage_exists and create_storage:
            # Check if we're allowed to create
            allowed, reason = server.config.is_operation_allowed("create")
            if allowed:
                try:
                    server.k8s.create_pvc(
                        name=pvc_name,
                        namespace=namespace,
                        size=f"{storage_size_gb}Gi",
                        access_modes=["ReadWriteMany"],
                        labels={
                            "app.kubernetes.io/managed-by": "rhoai-mcp",
                            "app.kubernetes.io/component": "training-storage",
                        },
                    )
                    storage_created = True
                    storage_exists = True
                except Exception as e:
                    warnings.append(f"Failed to create storage: {e}")
            else:
                warnings.append(f"Cannot create storage: {reason}")

        # Build suggested parameters for train() call
        suggested_params: dict[str, Any] = {
            "namespace": namespace,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "runtime_name": recommended_runtime,
            "method": method,
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_nodes": 1,
            "gpus_per_node": resource_estimate.get("recommended_gpus", 1),
        }

        if storage_exists:
            suggested_params["checkpoint_dir"] = f"/mnt/{pvc_name}"

        ready = prereq_passed and len(issues) == 0

        return {
            "ready": ready,
            "issues": issues if issues else None,
            "warnings": warnings if warnings else None,
            "resource_estimate": resource_estimate,
            "recommended_runtime": recommended_runtime,
            "storage_created": storage_created,
            "storage_pvc": pvc_name if storage_exists else None,
            "next_action": "train" if ready else "fix_issues",
            "suggested_train_params": suggested_params,
        }


def _estimate_resources_internal(model_id: str, method: str) -> dict[str, Any]:
    """Internal helper for resource estimation used by prepare_training."""
    param_count = _extract_param_count(model_id)

    # Get base memory estimate
    base_memory = 16
    for (min_p, max_p), mem in GPU_MEMORY_ESTIMATES.items():
        if min_p <= param_count < max_p:
            base_memory = mem
            break

    try:
        peft_method = PeftMethod(method.lower())
    except ValueError:
        peft_method = PeftMethod.LORA

    multiplier = PEFT_MULTIPLIERS.get(peft_method, 1.8)
    total_memory = base_memory * multiplier

    recommended_gpus = 1
    if total_memory > 80:
        recommended_gpus = int((total_memory / 80) + 0.5)

    return {
        "model_id": model_id,
        "estimated_params_billion": param_count,
        "method": peft_method.value,
        "total_required_gb": round(total_memory, 1),
        "recommended_gpus": recommended_gpus,
        "storage_gb": int(param_count * 4) + 50,
    }


def _extract_param_count(model_id: str) -> float:
    """Extract parameter count from model ID.

    Attempts to parse common patterns like:
    - Llama-2-7b-hf -> 7
    - Qwen2.5-72B-Instruct -> 72
    - mistral-7b -> 7
    """
    model_lower = model_id.lower()

    # Try common patterns
    patterns = [
        r"(\d+(?:\.\d+)?)\s*b(?:illion)?",  # 7b, 70b, 7.1b
        r"-(\d+(?:\.\d+)?)b-",  # -7b-
        r"(\d+(?:\.\d+)?)b$",  # ends with 7b
    ]

    for pattern in patterns:
        match = re.search(pattern, model_lower)
        if match:
            return float(match.group(1))

    # Check for million parameters
    m_match = re.search(r"(\d+)m", model_lower)
    if m_match:
        return float(m_match.group(1)) / 1000

    # Default estimate based on common models
    if "llama" in model_lower:
        return 7.0
    if "mistral" in model_lower:
        return 7.0
    if "qwen" in model_lower:
        return 7.0

    return 7.0  # Default assumption
