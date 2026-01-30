"""Cluster and project exploration prompts for RHOAI MCP.

Provides prompts that guide AI agents through discovering and understanding
RHOAI cluster resources, projects, and GPU availability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer


def register_prompts(mcp: FastMCP, server: RHOAIServer) -> None:  # noqa: ARG001
    """Register exploration prompts.

    Args:
        mcp: The FastMCP server instance to register prompts with.
        server: The RHOAI server instance (unused but required for interface).
    """

    @mcp.prompt(
        name="explore-cluster",
        description="Discover what's available in the RHOAI cluster",
    )
    def explore_cluster() -> str:
        """Generate guidance for exploring the RHOAI cluster.

        Returns:
            Workflow guidance as a string prompt.
        """
        return """I want to explore what's available in this Red Hat OpenShift AI cluster.

**Please help me discover:**

1. **Cluster Overview**
   - Use `cluster_summary` to get a compact overview of the entire cluster
   - This shows project count, workbench status, model deployments, and resources

2. **Available Projects**
   - Use `list_data_science_projects` to see all Data Science Projects
   - For each interesting project, use `project_summary` to get details

3. **GPU and Accelerator Availability**
   - Use `get_cluster_resources` to see CPU, memory, and GPU resources
   - Check which GPU types are available and their current usage

4. **Training Runtimes**
   - Use `list_training_runtimes` to see available training configurations
   - These define what frameworks and images are available for training

5. **Notebook Images**
   - Use `list_notebook_images` to see available workbench images
   - These are the IDE environments users can launch

Please start with the cluster summary to get an overview."""

    @mcp.prompt(
        name="explore-project",
        description="Explore resources within a specific Data Science Project",
    )
    def explore_project(namespace: str) -> str:
        """Generate guidance for exploring a specific project.

        Args:
            namespace: The project namespace to explore.

        Returns:
            Workflow guidance as a string prompt.
        """
        return f"""I want to explore the resources in a Data Science Project.

**Project:** {namespace}

**Please help me discover:**

1. **Project Overview**
   - Use `project_summary` for namespace={namespace} to get a compact summary
   - This shows workbench, model, pipeline, and storage counts

2. **Workbenches**
   - Use `list_workbenches` to see all notebook environments
   - For running workbenches, use `get_workbench_url` to get access URLs

3. **Deployed Models**
   - Use `list_inference_services` to see deployed models
   - Use `get_model_endpoint` to get inference URLs for each model

4. **Training Jobs**
   - Use `list_training_jobs` to see current and past training jobs
   - Use `get_training_progress` for any active jobs

5. **Data Connections**
   - Use `list_data_connections` to see configured S3 connections
   - These provide access to external data sources

6. **Storage**
   - Use `list_storage` to see PersistentVolumeClaims
   - Check storage capacity and what's available

7. **Pipeline Server**
   - Use `get_pipeline_server` to check if pipelines are configured

Please start with the project summary to get an overview."""

    @mcp.prompt(
        name="find-gpus",
        description="Find available GPU resources for training or inference",
    )
    def find_gpus() -> str:
        """Generate guidance for finding GPU resources.

        Returns:
            Workflow guidance as a string prompt.
        """
        return """I need to find available GPU resources for training or inference.

**Please help me discover:**

1. **Cluster GPU Resources**
   - Use `get_cluster_resources` to see total GPU capacity
   - This shows GPU counts per node and availability

2. **Current GPU Usage**
   - Use `cluster_summary` to see what's running across the cluster
   - Check training jobs and inference services that are using GPUs

3. **GPU-capable Projects**
   - Use `list_data_science_projects` to find projects
   - Use `list_workbenches` per project to see GPU allocations

4. **For Training**
   - Use `estimate_resources` with your model to see GPU requirements
   - Compare requirements against available resources

5. **Recommendations**
   - If GPUs are scarce, consider using qlora method (lower memory)
   - Check if any suspended training jobs can be cleaned up
   - Consider scheduling jobs during off-peak hours

Please start by checking the cluster GPU resources."""

    @mcp.prompt(
        name="whats-running",
        description="Quick status check of all active workloads",
    )
    def whats_running() -> str:
        """Generate guidance for checking active workloads.

        Returns:
            Workflow guidance as a string prompt.
        """
        return """I want to see what's currently running in the RHOAI cluster.

**Please help me check:**

1. **Quick Overview**
   - Use `cluster_summary` for a compact view of everything running
   - This shows workbench count, training jobs, and deployed models

2. **Active Training Jobs**
   - For each project with training, use `list_training_jobs`
   - Use `get_training_progress` for jobs in "Running" status
   - Check estimated time remaining

3. **Running Workbenches**
   - Use `list_workbenches` per project with verbosity="minimal"
   - Look for workbenches with status "Running"

4. **Deployed Models**
   - Use `list_inference_services` per project
   - Check which models are ready to serve traffic

5. **Resource Consumption**
   - Use `get_cluster_resources` to see current resource usage
   - Identify any resource bottlenecks

Please start with the cluster summary for a quick overview."""
