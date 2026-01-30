"""Troubleshooting prompts for RHOAI MCP.

Provides prompts that guide AI agents through diagnosing and resolving
issues with training jobs, workbenches, and model deployments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer


def register_prompts(mcp: FastMCP, server: RHOAIServer) -> None:  # noqa: ARG001
    """Register troubleshooting prompts.

    Args:
        mcp: The FastMCP server instance to register prompts with.
        server: The RHOAI server instance (unused but required for interface).
    """

    @mcp.prompt(
        name="troubleshoot-training",
        description="Diagnose and fix issues with a training job",
    )
    def troubleshoot_training(namespace: str, job_name: str) -> str:
        """Generate guidance for troubleshooting a training job.

        Args:
            namespace: Namespace containing the training job.
            job_name: Name of the failing training job.

        Returns:
            Workflow guidance as a string prompt.
        """
        return f"""I need to troubleshoot a training job that's having issues.

**Job Details:**
- Namespace: {namespace}
- Job Name: {job_name}

**Please help me diagnose the problem:**

1. **Get Job Status**
   - Use `get_training_job` to see the current status and conditions
   - Check if the job is Failed, Suspended, or stuck in Pending

2. **Analyze Failure**
   - Use `analyze_training_failure` for automated diagnosis
   - This will check logs, events, and provide suggestions

3. **Check Events**
   - Use `get_job_events` to see Kubernetes events
   - Look for: ImagePullBackOff, OOMKilled, FailedScheduling, etc.

4. **Check Logs**
   - Use `get_training_logs` to see trainer container output
   - If container crashed, use `get_training_logs` with previous=True

5. **Common Issues:**
   - **OOM**: Reduce batch_size or use qlora method
   - **ImagePull**: Check container registry access
   - **Pending**: Use `get_cluster_resources` to check GPU availability
   - **Storage**: Use `list_storage` to verify PVC status

6. **Resolution**
   - If fixable, use `suspend_training_job` then `resume_training_job`
   - If job is corrupted, use `delete_training_job` and recreate with `train`

Please start with the job status and failure analysis."""

    @mcp.prompt(
        name="troubleshoot-workbench",
        description="Diagnose and fix issues with a workbench",
    )
    def troubleshoot_workbench(namespace: str, workbench_name: str) -> str:
        """Generate guidance for troubleshooting a workbench.

        Args:
            namespace: Namespace containing the workbench.
            workbench_name: Name of the problematic workbench.

        Returns:
            Workflow guidance as a string prompt.
        """
        return f"""I need to troubleshoot a workbench that's having issues.

**Workbench Details:**
- Namespace: {namespace}
- Workbench: {workbench_name}

**Please help me diagnose the problem:**

1. **Get Workbench Status**
   - Use `get_workbench` to see the current status and configuration
   - Check the pod phase and any error conditions

2. **Check Resource Status**
   - Use `resource_status` with type="workbench" for a quick status check
   - This shows if the workbench is Running, Pending, or Failed

3. **Common Issues:**

   **Stuck Starting:**
   - Check if image is pulling: look for ImagePullBackOff in status
   - Check storage: use `list_storage` to verify PVC is Bound
   - Check resources: GPU or memory may be unavailable

   **Not Accessible:**
   - Use `get_workbench_url` to get the correct URL
   - Verify the route exists and is properly configured

   **Stopped Unexpectedly:**
   - Check if manually stopped (kubeflow-resource-stopped annotation)
   - Use `start_workbench` to restart if stopped

4. **Resolution**
   - If stuck, try `stop_workbench` followed by `start_workbench`
   - If persistent issues, may need to recreate with `delete_workbench`
     and `create_workbench`

Please start by getting the workbench status."""

    @mcp.prompt(
        name="troubleshoot-model",
        description="Diagnose and fix issues with a deployed model",
    )
    def troubleshoot_model(namespace: str, model_name: str) -> str:
        """Generate guidance for troubleshooting a deployed model.

        Args:
            namespace: Namespace containing the model.
            model_name: Name of the InferenceService.

        Returns:
            Workflow guidance as a string prompt.
        """
        return f"""I need to troubleshoot a deployed model that's having issues.

**Model Details:**
- Namespace: {namespace}
- Model Name: {model_name}

**Please help me diagnose the problem:**

1. **Get Model Status**
   - Use `get_inference_service` to see the current status
   - Check the Ready condition and any failure messages

2. **Check Endpoint**
   - Use `get_model_endpoint` to see if the endpoint is available
   - An empty or error URL indicates deployment issues

3. **Common Issues:**

   **Model Not Ready:**
   - Container may be pulling or starting
   - Check if model files are accessible at storage_uri
   - Verify the serving runtime supports this model format

   **Prediction Errors:**
   - Model format may not match runtime expectations
   - Input data format may be incorrect
   - Memory/GPU resources may be insufficient

   **Scale Issues:**
   - If min_replicas=0, first request triggers cold start
   - Check resource limits for the serving runtime

4. **Check Related Resources**
   - Use `list_serving_runtimes` to verify runtime availability
   - Check data connections if model is from S3

5. **Resolution**
   - Redeploy with correct configuration using `deploy_model`
   - If persistent issues, use `delete_inference_service` and redeploy

Please start by getting the inference service status."""

    @mcp.prompt(
        name="analyze-oom",
        description="Analyze and resolve Out-of-Memory issues in training",
    )
    def analyze_oom(namespace: str, job_name: str) -> str:
        """Generate guidance for analyzing OOM issues.

        Args:
            namespace: Namespace containing the training job.
            job_name: Name of the training job with OOM issues.

        Returns:
            Workflow guidance as a string prompt.
        """
        return f"""I need to analyze and fix an Out-of-Memory (OOM) issue in a training job.

**Job Details:**
- Namespace: {namespace}
- Job Name: {job_name}

**Please help me resolve this:**

1. **Confirm OOM Issue**
   - Use `get_job_events` to look for OOMKilled events
   - Use `get_training_logs` with previous=True to see last output before crash

2. **Analyze Current Configuration**
   - Use `get_training_job` to see current batch_size, method, and resources
   - Use `get_job_spec` for the full job specification

3. **Estimate Required Resources**
   - Use `estimate_resources` with the same model_id and method
   - Compare estimated GPU memory vs. what was allocated

4. **Mitigation Strategies (in order of preference):**

   **A. Reduce Memory Usage:**
   - Switch from lora to qlora (4-bit quantization)
   - Reduce batch_size (most impactful)
   - Reduce sequence_length if applicable

   **B. Increase Resources:**
   - Request more GPUs per node
   - Use multi-node training for larger models

   **C. Gradient Checkpointing:**
   - Enable in training config (trades compute for memory)

5. **Create New Job**
   - Use `delete_training_job` to clean up the failed job
   - Use `train` with adjusted parameters
   - Preview first, then confirm

6. **Monitor New Job**
   - Use `get_training_progress` to verify training starts
   - Watch memory usage in early epochs

Please start by confirming the OOM issue in events and logs."""
