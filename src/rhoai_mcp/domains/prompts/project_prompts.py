"""Project setup prompts for RHOAI MCP.

Provides prompts that guide AI agents through setting up Data Science
Projects for training and inference workloads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer


def register_prompts(mcp: FastMCP, server: RHOAIServer) -> None:  # noqa: ARG001
    """Register project setup prompts.

    Args:
        mcp: The FastMCP server instance to register prompts with.
        server: The RHOAI server instance (unused but required for interface).
    """

    @mcp.prompt(
        name="setup-training-project",
        description="Set up a new project for model training",
    )
    def setup_training_project(project_name: str, display_name: str = "") -> str:
        """Generate guidance for setting up a training project.

        Args:
            project_name: Name for the new project (DNS-compatible).
            display_name: Human-readable display name.

        Returns:
            Workflow guidance as a string prompt.
        """
        display = display_name or project_name
        return f"""I want to set up a new Data Science Project for model training.

**Project Configuration:**
- Project Name: {project_name}
- Display Name: {display}

**Please help me complete these steps:**

1. **Create the Project**
   - Use `create_data_science_project` with name="{project_name}"
   - Set display_name="{display}"
   - Add a description explaining the project purpose

2. **Set Up Training Runtime**
   - Use `list_training_runtimes` to check if a runtime exists
   - If not, use `setup_training_runtime` to create one
   - The runtime defines container images and frameworks

3. **Create Storage for Checkpoints**
   - Use `setup_training_storage` to create a PVC
   - Recommended size: 100GB for model checkpoints
   - Use ReadWriteMany access mode for distributed training

4. **Configure Model Access (if needed)**
   - For gated HuggingFace models, use `setup_hf_credentials`
   - This creates a secret with your HF token

5. **Optional: Add Data Connection**
   - Use `create_s3_data_connection` if you need S3 access
   - Useful for storing datasets or final model artifacts

6. **Verify Setup**
   - Use `project_summary` to confirm all resources are created
   - Use `check_training_prerequisites` with your target model

Please start by creating the project."""

    @mcp.prompt(
        name="setup-inference-project",
        description="Set up a new project for model serving",
    )
    def setup_inference_project(project_name: str, display_name: str = "") -> str:
        """Generate guidance for setting up an inference project.

        Args:
            project_name: Name for the new project (DNS-compatible).
            display_name: Human-readable display name.

        Returns:
            Workflow guidance as a string prompt.
        """
        display = display_name or project_name
        return f"""I want to set up a new Data Science Project for model serving.

**Project Configuration:**
- Project Name: {project_name}
- Display Name: {display}

**Please help me complete these steps:**

1. **Create the Project**
   - Use `create_data_science_project` with name="{project_name}"
   - Set display_name="{display}"
   - Set enable_modelmesh=False for single-model serving (KServe)
   - Or enable_modelmesh=True for multi-model serving (ModelMesh)

2. **Add Model Storage Connection**
   - Use `create_s3_data_connection` to configure S3 access
   - This is where your model files are stored
   - Provide: endpoint, bucket, access_key, secret_key

3. **Check Available Serving Runtimes**
   - Use `list_serving_runtimes` to see what's available
   - Common options: OpenVINO, vLLM, TGIS, sklearn

4. **Optional: Create Additional Storage**
   - Use `create_storage` for model caching if needed
   - Useful for large models to avoid re-downloading

5. **Verify Setup**
   - Use `project_summary` to confirm configuration
   - The project is ready for model deployment

6. **Next: Deploy a Model**
   - Use `deploy_model` to deploy your first model
   - Specify: runtime, model_format, storage_uri

Please start by creating the project."""

    @mcp.prompt(
        name="add-data-connection",
        description="Add an S3 data connection to an existing project",
    )
    def add_data_connection(namespace: str) -> str:
        """Generate guidance for adding a data connection.

        Args:
            namespace: Target namespace for the data connection.

        Returns:
            Workflow guidance as a string prompt.
        """
        return f"""I want to add an S3 data connection to a project.

**Target Project:** {namespace}

**Please help me complete these steps:**

1. **Check Existing Connections**
   - Use `list_data_connections` for namespace={namespace}
   - Verify the connection doesn't already exist

2. **Gather S3 Credentials**
   I need the following information:
   - **Endpoint URL**: S3 endpoint (e.g., https://s3.amazonaws.com)
   - **Bucket Name**: The S3 bucket to access
   - **Access Key ID**: AWS access key
   - **Secret Access Key**: AWS secret key
   - **Region**: AWS region (default: us-east-1)

3. **Create the Connection**
   - Use `create_s3_data_connection` with the gathered credentials
   - Choose a meaningful name (e.g., "training-data", "model-artifacts")
   - Optionally set a display_name for the UI

4. **Verify the Connection**
   - Use `get_data_connection` to confirm it was created
   - Credentials will be masked in the output

5. **Usage**
   - Data connections are automatically available to workbenches
   - They can be referenced in model deployments as storage sources
   - Training jobs can use them for dataset access

**Security Note:**
- Credentials are stored as Kubernetes secrets
- Use a dedicated service account with minimal permissions
- Rotate credentials periodically

Please provide the S3 connection details, and I'll help you create it."""
