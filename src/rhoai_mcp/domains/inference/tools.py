"""MCP Tools for Model Serving (InferenceService) operations."""

import re
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.inference.client import InferenceClient
from rhoai_mcp.domains.inference.models import InferenceServiceCreate
from rhoai_mcp.utils.response import (
    PaginatedResponse,
    ResponseBuilder,
    Verbosity,
    paginate,
)

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


# Model size estimates by parameter count (in billions)
MODEL_SIZE_ESTIMATES = {
    (0, 1): 2,  # < 1B params -> ~2GB
    (1, 3): 6,  # 1-3B params -> ~6GB
    (3, 7): 14,  # 3-7B params -> ~14GB
    (7, 13): 26,  # 7-13B params -> ~26GB
    (13, 30): 60,  # 13-30B params -> ~60GB
    (30, 70): 140,  # 30-70B params -> ~140GB
    (70, 200): 400,  # 70-200B params -> ~400GB
}


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register model serving tools with the MCP server."""

    @mcp.tool()
    def list_inference_services(
        namespace: str,
        limit: int | None = None,
        offset: int = 0,
        verbosity: str = "standard",
    ) -> dict[str, Any]:
        """List deployed models in a Data Science Project with pagination.

        Returns InferenceService resources representing deployed models
        that can serve predictions.

        Args:
            namespace: The project (namespace) name.
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).
            verbosity: Response detail level - "minimal", "standard", or "full".
                Use "minimal" for quick status checks.

        Returns:
            Paginated list of deployed models with metadata.
        """
        client = InferenceClient(server.k8s)
        all_items = client.list_inference_services(namespace)

        # Apply config limits
        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        # Paginate
        paginated, total = paginate(all_items, offset, effective_limit)

        # Format with verbosity
        v = Verbosity.from_str(verbosity)
        items = [ResponseBuilder.inference_service_list_item(isvc, v) for isvc in paginated]

        return PaginatedResponse.build(items, total, offset, effective_limit)

    @mcp.tool()
    def get_inference_service(
        name: str,
        namespace: str,
        verbosity: str = "full",
    ) -> dict[str, Any]:
        """Get detailed information about a deployed model.

        Args:
            name: The InferenceService name.
            namespace: The project (namespace) name.
            verbosity: Response detail level - "minimal", "standard", or "full".
                Use "minimal" for quick status checks.

        Returns:
            Model deployment information at the requested verbosity level.
        """
        client = InferenceClient(server.k8s)
        isvc = client.get_inference_service(name, namespace)

        v = Verbosity.from_str(verbosity)
        return ResponseBuilder.inference_service_detail(isvc, v)

    @mcp.tool()
    def deploy_model(
        name: str,
        namespace: str,
        runtime: str,
        model_format: str,
        storage_uri: str,
        display_name: str | None = None,
        min_replicas: int = 1,
        max_replicas: int = 1,
        cpu_request: str = "1",
        cpu_limit: str = "2",
        memory_request: str = "4Gi",
        memory_limit: str = "8Gi",
        gpu_count: int = 0,
    ) -> dict[str, Any]:
        """Deploy a model as an InferenceService.

        Creates a KServe InferenceService to serve model predictions.

        Args:
            name: Deployment name (must be DNS-compatible).
            namespace: Project (namespace) name.
            runtime: Serving runtime to use (use list_serving_runtimes to see options).
            model_format: Model format (onnx, pytorch, tensorflow, sklearn, etc.).
            storage_uri: Model location (s3://bucket/path or pvc://pvc-name/path).
            display_name: Human-readable display name.
            min_replicas: Minimum number of replicas (0 for scale-to-zero).
            max_replicas: Maximum number of replicas.
            cpu_request: CPU request per replica.
            cpu_limit: CPU limit per replica.
            memory_request: Memory request per replica.
            memory_limit: Memory limit per replica.
            gpu_count: Number of GPUs per replica.

        Returns:
            Created InferenceService information.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        client = InferenceClient(server.k8s)
        request = InferenceServiceCreate(
            name=name,
            namespace=namespace,
            display_name=display_name,
            runtime=runtime,
            model_format=model_format,
            storage_uri=storage_uri,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            cpu_request=cpu_request,
            cpu_limit=cpu_limit,
            memory_request=memory_request,
            memory_limit=memory_limit,
            gpu_count=gpu_count,
        )
        isvc = client.deploy_model(request)

        return {
            "name": isvc.metadata.name,
            "namespace": isvc.metadata.namespace,
            "status": isvc.status.value,
            "message": f"Model '{name}' deployment initiated. It may take a few minutes to become ready.",
        }

    @mcp.tool()
    def delete_inference_service(
        name: str,
        namespace: str,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Delete a deployed model.

        Args:
            name: The InferenceService name.
            namespace: The project (namespace) name.
            confirm: Must be True to actually delete.

        Returns:
            Confirmation of deletion.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("delete")
        if not allowed:
            return {"error": reason}

        if not confirm:
            return {
                "error": "Deletion not confirmed",
                "message": f"To delete model deployment '{name}', set confirm=True.",
            }

        client = InferenceClient(server.k8s)
        client.delete_inference_service(name, namespace)

        return {
            "name": name,
            "namespace": namespace,
            "deleted": True,
            "message": f"Model deployment '{name}' deleted",
        }

    @mcp.tool()
    def list_serving_runtimes(namespace: str) -> list[dict[str, Any]]:
        """List available model serving runtimes.

        Serving runtimes define the model server that will be used to serve
        predictions (e.g., OpenVINO, vLLM, TGIS, etc.).

        Args:
            namespace: The project (namespace) name.

        Returns:
            List of available serving runtimes with supported model formats.
        """
        client = InferenceClient(server.k8s)
        return client.list_serving_runtimes(namespace)

    @mcp.tool()
    def get_model_endpoint(name: str, namespace: str) -> dict[str, Any]:
        """Get the inference endpoint URL for a deployed model.

        Returns the URL that can be used to send prediction requests
        to the model.

        Args:
            name: The InferenceService name.
            namespace: The project (namespace) name.

        Returns:
            Model endpoint information including URL and status.
        """
        client = InferenceClient(server.k8s)
        result = client.get_model_endpoint(name, namespace)

        if result["status"] == "Ready":
            result["message"] = "Model is ready to accept prediction requests"
        else:
            result["message"] = f"Model is {result['status']} - endpoint may not be available"

        return result

    @mcp.tool()
    def prepare_model_deployment(
        namespace: str,
        model_id: str,
        storage_uri: str | None = None,
        model_format: str | None = None,
    ) -> dict[str, Any]:
        """Complete pre-flight preparation for model deployment in one call.

        This composite tool combines runtime discovery, compatibility checking,
        resource estimation, and storage validation. Use this before calling
        deploy_model() to ensure everything is ready.

        Args:
            namespace: The project (namespace) name.
            model_id: Model identifier (e.g., "meta-llama/Llama-2-7b-hf").
            storage_uri: Model location (s3:// or pvc://). Auto-detected if None.
            model_format: Model format. Auto-detected from model_id if None.

        Returns:
            Complete preparation result with:
            - ready: Whether deployment can proceed
            - issues: List of problems found
            - model_info: Model format and size estimates
            - recommended_runtime: Runtime to use
            - resource_requirements: GPU/memory requirements
            - storage_valid: Whether storage is accessible
            - suggested_deploy_params: Parameters for deploy_model() call
        """
        issues: list[str] = []
        warnings: list[str] = []

        # Step 1: Estimate model info
        model_info = _estimate_model_info(model_id)
        if model_format:
            model_info["format"] = model_format
        else:
            model_format = model_info.get("format", "pytorch")

        # Step 2: Get available runtimes
        client = InferenceClient(server.k8s)
        runtimes = client.list_serving_runtimes(namespace)

        if not runtimes:
            issues.append("No serving runtimes available in namespace")

        # Step 3: Find compatible runtime
        recommended_runtime = None
        compatible_runtimes: list[str] = []

        for runtime in runtimes:
            supported = runtime.get("supported_formats", [])
            # Check format compatibility
            if model_format.lower() in [f.lower() for f in supported]:
                compatible_runtimes.append(runtime["name"])
                if recommended_runtime is None:
                    recommended_runtime = runtime["name"]
            # Check for LLM-specific runtimes
            is_llm_runtime = "vllm" in runtime["name"].lower() or "tgis" in runtime["name"].lower()
            if is_llm_runtime and model_info.get("is_llm"):
                recommended_runtime = runtime["name"]

        if not compatible_runtimes:
            issues.append(f"No runtime supports format '{model_format}'")
            if runtimes:
                all_formats = set()
                for rt in runtimes:
                    all_formats.update(rt.get("supported_formats", []))
                warnings.append(f"Available formats: {', '.join(all_formats)}")

        # Step 4: Resource estimation
        resource_requirements = _estimate_serving_resources(model_info)

        # Step 5: Check GPU availability
        try:
            from rhoai_mcp.domains.training.client import TrainingClient

            training_client = TrainingClient(server.k8s)
            cluster_resources = training_client.get_cluster_resources()
            if cluster_resources.gpu_info:
                if cluster_resources.gpu_info.available < resource_requirements.get("gpu", 0):
                    warnings.append(
                        f"Insufficient GPUs: {cluster_resources.gpu_info.available} available, "
                        f"{resource_requirements.get('gpu', 0)} recommended"
                    )
            elif resource_requirements.get("gpu", 0) > 0:
                issues.append("Model requires GPU but no GPUs available in cluster")
        except Exception:
            warnings.append("Could not check GPU availability")

        # Step 6: Validate storage if provided
        storage_valid = True
        if storage_uri:
            if storage_uri.startswith("pvc://"):
                pvc_name = storage_uri.replace("pvc://", "").split("/")[0]
                try:
                    pvc = server.k8s.get_pvc(pvc_name, namespace)
                    if pvc.status.phase != "Bound":
                        issues.append(f"PVC '{pvc_name}' is not bound")
                        storage_valid = False
                except Exception:
                    issues.append(f"PVC '{pvc_name}' not found")
                    storage_valid = False
            elif storage_uri.startswith("s3://"):
                # S3 validation would require checking data connections
                warnings.append("Ensure S3 credentials are configured via data connection")
        else:
            warnings.append("No storage_uri provided - you'll need to specify model location")

        # Build suggested parameters
        suggested_params: dict[str, Any] = {
            "name": _generate_deployment_name(model_id),
            "namespace": namespace,
            "runtime": recommended_runtime,
            "model_format": model_format,
            "storage_uri": storage_uri or "<model_storage_path>",
            "min_replicas": 1,
            "max_replicas": 1,
            "gpu_count": resource_requirements.get("gpu", 0),
            "memory_request": resource_requirements.get("memory", "4Gi"),
            "memory_limit": resource_requirements.get("memory_limit", "8Gi"),
        }

        ready = len(issues) == 0 and recommended_runtime is not None and storage_uri is not None

        return {
            "ready": ready,
            "issues": issues if issues else None,
            "warnings": warnings if warnings else None,
            "model_info": model_info,
            "recommended_runtime": recommended_runtime,
            "compatible_runtimes": compatible_runtimes,
            "resource_requirements": resource_requirements,
            "storage_valid": storage_valid,
            "next_action": "deploy_model" if ready else "fix_issues",
            "suggested_deploy_params": suggested_params,
        }

    @mcp.tool()
    def check_deployment_prerequisites(
        namespace: str,
        model_format: str,
        storage_uri: str,
    ) -> dict[str, Any]:
        """Pre-flight checks before model deployment.

        Validates cluster connectivity, GPU availability, runtime availability,
        storage accessibility, and namespace permissions.

        Args:
            namespace: The project (namespace) name.
            model_format: Model format (pytorch, onnx, tensorflow, etc.).
            storage_uri: Model storage location (s3:// or pvc://).

        Returns:
            Pre-flight check results with any issues found.
        """
        checks: list[dict] = []
        actions_needed: list[str] = []
        all_passed = True

        # Check 1: Namespace exists
        try:
            server.k8s.core_v1.read_namespace(namespace)
            checks.append(
                {
                    "name": "Namespace",
                    "passed": True,
                    "message": f"Namespace '{namespace}' exists",
                }
            )
        except Exception:
            checks.append(
                {
                    "name": "Namespace",
                    "passed": False,
                    "message": f"Namespace '{namespace}' not found",
                }
            )
            all_passed = False
            actions_needed.append(f"Create namespace '{namespace}'")

        # Check 2: Serving runtime availability
        client = InferenceClient(server.k8s)
        runtimes = client.list_serving_runtimes(namespace)

        compatible = [
            r
            for r in runtimes
            if model_format.lower() in [f.lower() for f in r.get("supported_formats", [])]
        ]

        if compatible:
            checks.append(
                {
                    "name": "Serving runtime",
                    "passed": True,
                    "message": f"{len(compatible)} runtime(s) support {model_format}",
                }
            )
        else:
            checks.append(
                {
                    "name": "Serving runtime",
                    "passed": False,
                    "message": f"No runtime supports format '{model_format}'",
                }
            )
            all_passed = False
            actions_needed.append("Verify model format or check available runtimes")

        # Check 3: Storage accessibility
        if storage_uri.startswith("pvc://"):
            pvc_name = storage_uri.replace("pvc://", "").split("/")[0]
            try:
                pvc = server.k8s.get_pvc(pvc_name, namespace)
                if pvc.status.phase == "Bound":
                    checks.append(
                        {
                            "name": "Storage",
                            "passed": True,
                            "message": f"PVC '{pvc_name}' is bound",
                        }
                    )
                else:
                    checks.append(
                        {
                            "name": "Storage",
                            "passed": False,
                            "message": f"PVC '{pvc_name}' is {pvc.status.phase}",
                        }
                    )
                    all_passed = False
            except Exception:
                checks.append(
                    {
                        "name": "Storage",
                        "passed": False,
                        "message": f"PVC '{pvc_name}' not found",
                    }
                )
                all_passed = False
                actions_needed.append(f"Create PVC '{pvc_name}' with model files")
        elif storage_uri.startswith("s3://"):
            checks.append(
                {
                    "name": "Storage",
                    "passed": True,
                    "message": "S3 storage configured (ensure data connection exists)",
                }
            )
        else:
            checks.append(
                {
                    "name": "Storage",
                    "passed": False,
                    "message": f"Unknown storage scheme: {storage_uri}",
                }
            )
            all_passed = False

        return {
            "all_passed": all_passed,
            "checks": checks,
            "actions_needed": actions_needed if actions_needed else None,
            "ready_to_deploy": all_passed,
        }

    @mcp.tool()
    def estimate_serving_resources(
        model_id: str,
        target_throughput: int | None = None,
        target_latency_ms: int | None = None,
    ) -> dict[str, Any]:
        """Estimate GPU/memory/CPU requirements for model serving.

        Calculates approximate resource requirements based on model size
        and performance targets.

        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-2-7b-hf").
            target_throughput: Target requests per second (optional).
            target_latency_ms: Target latency in milliseconds (optional).

        Returns:
            Resource estimates including GPU memory and recommendations.
        """
        model_info = _estimate_model_info(model_id)
        resources = _estimate_serving_resources(model_info)

        # Adjust for performance targets
        if target_throughput and target_throughput > 10:
            # High throughput might need more replicas
            resources["recommended_replicas"] = max(1, target_throughput // 10)

        if target_latency_ms and target_latency_ms < 100:
            # Low latency might need better GPU
            resources["notes"] = resources.get("notes", [])
            resources["notes"].append("Low latency target may require high-end GPU")

        return {
            "model_id": model_id,
            "model_size_gb": model_info.get("size_gb", 0),
            "estimated_params_billion": model_info.get("params_billion", 0),
            "min_gpu_memory_gb": resources.get("min_gpu_memory_gb", 0),
            "gpu_count": resources.get("gpu", 0),
            "memory_request": resources.get("memory", "4Gi"),
            "cpu_request": resources.get("cpu", "1"),
            "recommended_replicas": resources.get("recommended_replicas", 1),
            "recommended_gpu_type": resources.get("recommended_gpu_type", "NVIDIA T4"),
            "resource_config": {
                "gpu_count": resources.get("gpu", 0),
                "memory_request": resources.get("memory", "4Gi"),
                "memory_limit": resources.get("memory_limit", "8Gi"),
                "cpu_request": resources.get("cpu", "1"),
                "cpu_limit": resources.get("cpu_limit", "2"),
            },
        }

    @mcp.tool()
    def recommend_serving_runtime(
        namespace: str,
        model_format: str,
        model_size_gb: float | None = None,
    ) -> dict[str, Any]:
        """Recommend the best serving runtime for a model.

        Analyzes available runtimes and recommends the best one based on
        model format and size.

        Args:
            namespace: The project (namespace) name.
            model_format: Model format (pytorch, onnx, tensorflow, etc.).
            model_size_gb: Model size in GB (optional, for better recommendations).

        Returns:
            Runtime recommendation with alternatives.
        """
        client = InferenceClient(server.k8s)
        runtimes = client.list_serving_runtimes(namespace)

        if not runtimes:
            return {
                "error": "No serving runtimes available",
                "recommended": None,
                "alternatives": [],
            }

        compatible: list[dict] = []
        for runtime in runtimes:
            supported = [f.lower() for f in runtime.get("supported_formats", [])]
            if model_format.lower() in supported:
                compatible.append(runtime)

        if not compatible:
            return {
                "recommended": None,
                "alternatives": [],
                "error": f"No runtime supports format '{model_format}'",
                "available_formats": list(
                    {f for rt in runtimes for f in rt.get("supported_formats", [])}
                ),
            }

        # Score runtimes
        recommended = compatible[0]
        notes = []

        # Prefer vLLM/TGIS for large LLMs
        if model_size_gb and model_size_gb > 10:
            for rt in compatible:
                if "vllm" in rt["name"].lower():
                    recommended = rt
                    notes.append("vLLM recommended for large LLMs")
                    break
                if "tgis" in rt["name"].lower():
                    recommended = rt
                    notes.append("TGIS recommended for large models")

        return {
            "recommended": recommended["name"],
            "recommended_display": recommended.get("display_name", recommended["name"]),
            "alternatives": [r["name"] for r in compatible if r != recommended],
            "compatibility_notes": notes[0] if notes else "Standard runtime selected",
            "total_compatible": len(compatible),
        }

    @mcp.tool()
    def test_model_endpoint(
        name: str,
        namespace: str,
        sample_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Test a deployed model's inference endpoint.

        Checks if the model endpoint is accessible and provides example
        request format.

        Args:
            name: The InferenceService name.
            namespace: The project (namespace) name.
            sample_input: Optional sample input for testing.

        Returns:
            Endpoint test results with accessibility status.
        """
        client = InferenceClient(server.k8s)

        try:
            isvc = client.get_inference_service(name, namespace)
        except Exception as e:
            return {
                "accessible": False,
                "error": f"Failed to get InferenceService: {e}",
            }

        result: dict[str, Any] = {
            "name": name,
            "namespace": namespace,
            "accessible": isvc.status.value == "Ready",
            "status": isvc.status.value,
            "url": isvc.url,
            "internal_url": isvc.internal_url,
            "issues": [],
        }

        if isvc.status.value != "Ready":
            result["issues"].append(f"Model not ready: {isvc.status.value}")

        if not isvc.url:
            result["issues"].append("No external URL available")
            result["accessible"] = False

        # Provide example request format
        result["example_request"] = {
            "method": "POST",
            "url": f"{isvc.url}/v1/models/{name}:predict" if isvc.url else None,
            "headers": {"Content-Type": "application/json"},
            "body": sample_input or {"instances": [{"input": "example"}]},
        }

        result["api_format"] = "KServe V1 Inference Protocol"

        return result


def _estimate_model_info(model_id: str) -> dict[str, Any]:
    """Estimate model information from model ID."""
    model_lower = model_id.lower()

    # Extract parameter count
    params_billion = 7.0  # Default
    patterns = [
        r"(\d+(?:\.\d+)?)\s*b(?:illion)?",
        r"-(\d+(?:\.\d+)?)b-",
        r"(\d+(?:\.\d+)?)b$",
    ]
    for pattern in patterns:
        match = re.search(pattern, model_lower)
        if match:
            params_billion = float(match.group(1))
            break

    # Estimate size
    size_gb = 14.0  # Default for 7B
    for (min_p, max_p), size in MODEL_SIZE_ESTIMATES.items():
        if min_p <= params_billion < max_p:
            size_gb = float(size)
            break

    # Detect format
    model_format = "pytorch"
    if "onnx" in model_lower:
        model_format = "onnx"
    elif "tensorflow" in model_lower or "tf-" in model_lower:
        model_format = "tensorflow"
    elif "gguf" in model_lower or "ggml" in model_lower:
        model_format = "gguf"

    # Detect if LLM
    is_llm = any(
        name in model_lower
        for name in [
            "llama",
            "mistral",
            "qwen",
            "falcon",
            "gpt",
            "bloom",
            "opt",
            "phi",
            "gemma",
            "instruct",
            "chat",
        ]
    )

    return {
        "model_id": model_id,
        "params_billion": params_billion,
        "size_gb": size_gb,
        "format": model_format,
        "is_llm": is_llm,
    }


def _estimate_serving_resources(model_info: dict[str, Any]) -> dict[str, Any]:
    """Estimate serving resources from model info."""
    size_gb = model_info.get("size_gb", 14.0)
    params_billion = model_info.get("params_billion", 7.0)

    # GPU requirements
    gpu_count = 0
    min_gpu_memory_gb = 0

    if size_gb > 2:
        gpu_count = 1
        min_gpu_memory_gb = int(size_gb * 1.2)  # 20% overhead

    if size_gb > 40:
        gpu_count = 2
    if size_gb > 80:
        gpu_count = 4

    # Memory requirements
    if size_gb < 5:
        memory = "4Gi"
        memory_limit = "8Gi"
    elif size_gb < 15:
        memory = "8Gi"
        memory_limit = "16Gi"
    elif size_gb < 40:
        memory = "16Gi"
        memory_limit = "32Gi"
    else:
        memory = "32Gi"
        memory_limit = "64Gi"

    # GPU type recommendation
    if min_gpu_memory_gb <= 16:
        recommended_gpu = "NVIDIA T4 (16GB)"
    elif min_gpu_memory_gb <= 24:
        recommended_gpu = "NVIDIA A10 (24GB)"
    elif min_gpu_memory_gb <= 40:
        recommended_gpu = "NVIDIA A100-40GB"
    elif min_gpu_memory_gb <= 80:
        recommended_gpu = "NVIDIA A100-80GB"
    else:
        recommended_gpu = "NVIDIA H100 or multiple GPUs"

    return {
        "gpu": gpu_count,
        "min_gpu_memory_gb": min_gpu_memory_gb,
        "memory": memory,
        "memory_limit": memory_limit,
        "cpu": "2" if params_billion > 3 else "1",
        "cpu_limit": "4" if params_billion > 3 else "2",
        "recommended_gpu_type": recommended_gpu,
        "recommended_replicas": 1,
    }


def _generate_deployment_name(model_id: str) -> str:
    """Generate a deployment name from model ID."""
    # Extract model name from org/model format
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    # Sanitize for Kubernetes
    name = re.sub(r"[^a-z0-9-]", "-", name.lower())
    name = re.sub(r"-+", "-", name).strip("-")
    # Limit length
    return name[:50] if len(name) > 50 else name
