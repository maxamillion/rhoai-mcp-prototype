"""Model deployment prompts for RHOAI MCP.

Provides prompts that guide AI agents through deploying, testing,
and scaling model serving endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer


def register_prompts(mcp: FastMCP, server: RHOAIServer) -> None:  # noqa: ARG001
    """Register deployment prompts.

    Args:
        mcp: The FastMCP server instance to register prompts with.
        server: The RHOAI server instance (unused but required for interface).
    """

    @mcp.prompt(
        name="deploy-model",
        description="Deploy a model for inference serving",
    )
    def deploy_model(
        namespace: str,
        model_name: str,
        storage_uri: str,
        model_format: str = "onnx",
    ) -> str:
        """Generate guidance for deploying a model.

        Args:
            namespace: Target namespace for deployment.
            model_name: Name for the deployed model.
            storage_uri: Model location (s3:// or pvc://).
            model_format: Model format (onnx, pytorch, tensorflow, etc.).

        Returns:
            Workflow guidance as a string prompt.
        """
        return f"""I want to deploy a model for inference serving.

**Deployment Configuration:**
- Namespace: {namespace}
- Model Name: {model_name}
- Storage URI: {storage_uri}
- Model Format: {model_format}

**Please help me complete these steps:**

1. **Check Prerequisites**
   - Use `project_summary` to verify the project exists
   - Use `list_serving_runtimes` to find a runtime that supports {model_format}

2. **Verify Model Access**
   - If using S3 (s3://), ensure a data connection exists
   - Use `list_data_connections` to check
   - If using PVC (pvc://), verify the PVC exists with `list_storage`

3. **Select Runtime**
   - Choose a runtime that supports {model_format}
   - Common options:
     - OpenVINO: onnx, tensorflow, pytorch
     - vLLM: pytorch (LLMs)
     - sklearn: sklearn, xgboost

4. **Deploy the Model**
   - Use `deploy_model` with:
     - name="{model_name}"
     - namespace="{namespace}"
     - storage_uri="{storage_uri}"
     - model_format="{model_format}"
     - Select appropriate runtime
   - Configure replicas (min_replicas=1 to avoid cold starts)

5. **Verify Deployment**
   - Use `get_inference_service` to check the Ready status
   - Use `get_model_endpoint` to get the inference URL

6. **Test the Endpoint**
   - The endpoint URL can be used for prediction requests
   - Format depends on the serving runtime

Please start by checking the available serving runtimes."""

    @mcp.prompt(
        name="deploy-llm",
        description="Deploy a Large Language Model with vLLM or TGIS",
    )
    def deploy_llm(
        namespace: str,
        model_name: str,
        model_id: str,
    ) -> str:
        """Generate guidance for deploying an LLM.

        Args:
            namespace: Target namespace for deployment.
            model_name: Name for the deployed model.
            model_id: HuggingFace model ID or storage path.

        Returns:
            Workflow guidance as a string prompt.
        """
        return f"""I want to deploy a Large Language Model for inference.

**LLM Configuration:**
- Namespace: {namespace}
- Model Name: {model_name}
- Model ID: {model_id}

**Please help me complete these steps:**

1. **Check GPU Availability**
   - Use `get_cluster_resources` to verify GPU availability
   - LLMs typically require significant GPU memory (16GB+)

2. **Check Serving Runtimes**
   - Use `list_serving_runtimes` to find LLM-capable runtimes
   - Look for: vLLM, TGIS (Text Generation Inference Server)

3. **Prepare Model Storage**
   - If model is on HuggingFace, you may need to download it first
   - Create an S3 data connection for model storage
   - Or use a PVC with the model files

4. **Configure GPU Resources**
   - LLMs need GPU allocation
   - Use gpu_count parameter in deploy_model
   - Adjust memory_limit based on model size

5. **Deploy the Model**
   - Use `deploy_model` with:
     - name="{model_name}"
     - namespace="{namespace}"
     - runtime: vLLM or TGIS
     - model_format: "pytorch" (typical for LLMs)
     - storage_uri pointing to model files
     - gpu_count: 1 or more based on model size

6. **Model Sizing Guide:**
   - 7B models: 1x 24GB GPU or 2x 16GB GPUs
   - 13B models: 2x 24GB GPUs
   - 70B models: 4+ 80GB GPUs or quantized version

7. **Verify Deployment**
   - Use `get_inference_service` to monitor startup
   - LLMs may take several minutes to load
   - Use `get_model_endpoint` once ready

8. **Optimize (Optional)**
   - Set min_replicas=0 for scale-to-zero (cost saving)
   - But first request will have cold start latency

Please start by checking GPU availability."""

    @mcp.prompt(
        name="test-endpoint",
        description="Test a deployed model endpoint",
    )
    def test_endpoint(namespace: str, model_name: str) -> str:
        """Generate guidance for testing a model endpoint.

        Args:
            namespace: Namespace containing the model.
            model_name: Name of the deployed model.

        Returns:
            Workflow guidance as a string prompt.
        """
        return f"""I want to test a deployed model endpoint.

**Model Details:**
- Namespace: {namespace}
- Model Name: {model_name}

**Please help me:**

1. **Get Endpoint Information**
   - Use `get_inference_service` to check the model status
   - Verify the model shows Ready=True
   - Use `get_model_endpoint` to get the inference URL

2. **Understand the Endpoint**
   - The URL format depends on the serving runtime
   - Common patterns:
     - KServe v1: /v1/models/{model_name}:predict
     - KServe v2: /v2/models/{model_name}/infer
     - OpenAI-compatible (vLLM): /v1/completions

3. **Check Serving Runtime**
   - Use `list_serving_runtimes` to understand the API format
   - Each runtime has different request/response schemas

4. **Test Request Examples:**

   **For ONNX/sklearn (KServe v1):**
   ```json
   {{"instances": [[1.0, 2.0, 3.0, 4.0]]}}
   ```

   **For vLLM (OpenAI-compatible):**
   ```json
   {{
     "model": "{model_name}",
     "prompt": "Hello, how are you?",
     "max_tokens": 100
   }}
   ```

5. **Troubleshooting**
   - If endpoint not responding, check replica count
   - If min_replicas=0, first request triggers scale-up
   - Use `get_inference_service` to see current replicas

Please start by getting the endpoint information."""

    @mcp.prompt(
        name="scale-model",
        description="Scale a model deployment up or down",
    )
    def scale_model(namespace: str, model_name: str) -> str:
        """Generate guidance for scaling a model deployment.

        Args:
            namespace: Namespace containing the model.
            model_name: Name of the deployed model.

        Returns:
            Workflow guidance as a string prompt.
        """
        return f"""I want to scale a model deployment.

**Model Details:**
- Namespace: {namespace}
- Model Name: {model_name}

**Please help me:**

1. **Check Current Configuration**
   - Use `get_inference_service` to see current replica settings
   - Note current min_replicas and max_replicas

2. **Scaling Options:**

   **Scale Up (More Capacity):**
   - Increase min_replicas for guaranteed capacity
   - Increase max_replicas for burst handling
   - Ensure sufficient GPU/memory resources exist

   **Scale Down (Save Resources):**
   - Reduce min_replicas (minimum 0 for scale-to-zero)
   - Scale-to-zero saves resources but has cold start latency

   **Scale to Zero:**
   - Set min_replicas=0
   - Model pods terminate when idle
   - First request triggers scale-up (may take 30s-2min)

3. **Apply Scaling**
   - Currently requires redeploying with new settings
   - Use `delete_inference_service` then `deploy_model`
   - Or use kubectl/oc to patch the InferenceService directly

4. **Verify Scaling**
   - Use `get_inference_service` to confirm replica changes
   - Check that pods are running with expected count

5. **Resource Considerations**
   - Check `get_cluster_resources` for available capacity
   - Each replica needs its own GPU allocation
   - Consider cost vs. latency tradeoffs

Please start by checking the current configuration."""
