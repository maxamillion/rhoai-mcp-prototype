# RHOAI MCP Server

[![CI](https://github.com/opendatahub-io/rhoai-mcp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/opendatahub-io/rhoai-mcp/actions/workflows/ci.yml)
[![Container Build](https://github.com/opendatahub-io/rhoai-mcp/actions/workflows/container-build.yml/badge.svg)](https://github.com/opendatahub-io/rhoai-mcp/actions/workflows/container-build.yml)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/opendatahub-io/rhoai-mcp)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/opendatahub-io/rhoai-mcp)
[![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)

An MCP (Model Context Protocol) server that enables AI agents to interact with Red Hat OpenShift AI (RHOAI) environments. This server replicates the capabilities of the OpenShift AI Dashboard through programmatic tools.

## Features

- **Project Management**: Create, list, and manage Data Science Projects
- **Workbench Operations**: Create, start, stop, and delete Jupyter workbenches
- **Model Serving**: Deploy and manage InferenceServices with KServe
- **Data Connections**: Manage S3 credentials for data access
- **Pipelines**: Configure Data Science Pipelines infrastructure
- **Storage**: Create and manage persistent volume claims
- **Training**: Fine-tune models with Kubeflow Training Operator
- **MCP Prompts**: Workflow guidance for multi-step operations (18 prompts)

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Runtime** | Python 3.10+ | Core language |
| **MCP Framework** | FastMCP 1.0+ | Model Context Protocol server |
| **Kubernetes Client** | kubernetes-python 28.1+ | Cluster API interactions |
| **Data Validation** | Pydantic 2.0+ | Type-safe models and settings |
| **HTTP Client** | httpx 0.27+ | Async HTTP requests |
| **Container Base** | Red Hat UBI 9 | Production container image |
| **Package Manager** | uv | Fast Python dependency management |

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/admiller/rhoai-mcp-prototype.git
cd rhoai-mcp-prototype

# Install dependencies
uv sync

# Run the server
uv run rhoai-mcp
```

### Using pip

```bash
pip install -e .
rhoai-mcp
```

### Using Container (Podman/Docker)

```bash
# Build the image
make build

# Run with HTTP transport
make run-http

# Run with STDIO transport (interactive)
make run-stdio

# Run with debug logging
make run-dev
```

Or run directly without Make:

```bash
# Build
podman build -f Containerfile -t rhoai-mcp:latest .

# Run with HTTP transport
podman run -p 8000:8000 \
  -v ~/.kube/config:/opt/app-root/src/kubeconfig/config:ro \
  -e RHOAI_MCP_AUTH_MODE=kubeconfig \
  -e RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config \
  rhoai-mcp:latest --transport sse

# Run with STDIO transport
podman run -it \
  -v ~/.kube/config:/opt/app-root/src/kubeconfig/config:ro \
  -e RHOAI_MCP_AUTH_MODE=kubeconfig \
  -e RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config \
  rhoai-mcp:latest --transport stdio
```

Available Make targets:

| Target | Description |
|--------|-------------|
| `make build` | Build the container image |
| `make run-http` | Run with SSE transport on port 8000 |
| `make run-streamable` | Run with streamable-http transport |
| `make run-stdio` | Run with STDIO transport (interactive) |
| `make run-dev` | Run with debug logging |
| `make run-token` | Run with token auth (requires TOKEN and API_SERVER) |
| `make stop` | Stop the running container |
| `make logs` | View container logs |
| `make clean` | Remove container and image |

### Kubernetes Deployment

For in-cluster deployment, apply the Kubernetes manifests:

```bash
kubectl apply -f deploy/kubernetes/deployment.yaml
```

This creates:
- Namespace `rhoai-mcp`
- ServiceAccount with RBAC for RHOAI resources
- Deployment running the MCP server with SSE transport
- Service exposing port 8000
- Route (OpenShift only) with TLS termination

## Configuration

The server can be configured via environment variables (with `RHOAI_MCP_` prefix) or a `.env` file.

### Authentication

The server supports three authentication modes:

1. **Auto (default)**: Tries in-cluster authentication first, falls back to kubeconfig
2. **Kubeconfig**: Uses a kubeconfig file
3. **Token**: Uses explicit API server URL and token

```bash
# Auto mode (default)
export RHOAI_MCP_AUTH_MODE=auto

# Kubeconfig mode
export RHOAI_MCP_AUTH_MODE=kubeconfig
export RHOAI_MCP_KUBECONFIG_PATH=/path/to/kubeconfig
export RHOAI_MCP_KUBECONFIG_CONTEXT=my-context

# Token mode
export RHOAI_MCP_AUTH_MODE=token
export RHOAI_MCP_API_SERVER=https://api.cluster.example.com:6443
export RHOAI_MCP_API_TOKEN=sha256~xxxxx
```

### Transport

```bash
# stdio (default) - for Claude Desktop and similar tools
export RHOAI_MCP_TRANSPORT=stdio

# HTTP transports
export RHOAI_MCP_TRANSPORT=sse
export RHOAI_MCP_HOST=127.0.0.1
export RHOAI_MCP_PORT=8000
```

### Safety Settings

```bash
# Enable delete operations (disabled by default)
export RHOAI_MCP_ENABLE_DANGEROUS_OPERATIONS=true

# Read-only mode (disable all write operations)
export RHOAI_MCP_READ_ONLY_MODE=true
```

### Safety Features Summary

| Feature | Description | Default |
|---------|-------------|---------|
| **Read-Only Mode** | Disables all create/update/delete operations | Off |
| **Dangerous Operations Gate** | Delete operations require explicit enablement | Disabled |
| **Confirmation Pattern** | Delete tools require `confirm=True` parameter | Required |
| **Credential Masking** | S3 secret keys are masked in all responses | Always |
| **RBAC-Aware** | Uses OpenShift Projects API to respect user permissions | Always |
| **Auth Validation** | Validates authentication configuration at startup | Always |

## Usage with Claude Code

Add to your project's `.mcp.json` file:

```json
{
  "mcpServers": {
    "rhoai": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/opendatahub-io/rhoai-mcp", "rhoai-mcp"],
      "env": {
        "RHOAI_MCP_KUBECONFIG_PATH": "/home/user/.kube/config"
      }
    }
  }
}
```

## Usage with Claude Desktop

Add to your Claude Desktop configuration (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rhoai": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/opendatahub-io/rhoai-mcp", "rhoai-mcp"],
      "env": {
        "RHOAI_MCP_KUBECONFIG_PATH": "/home/user/.kube/config"
      }
    }
  }
}
```

### Local Development

For contributors working with a local clone:

```json
{
  "mcpServers": {
    "rhoai": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/rhoai-mcp", "rhoai-mcp"],
      "env": {
        "RHOAI_MCP_KUBECONFIG_PATH": "/home/user/.kube/config"
      }
    }
  }
}
```

### Using Container Image (Podman/Docker)

First, build the container image:

```bash
make build
```

Then configure Claude Desktop with the container:

**Podman:**

```json
{
  "mcpServers": {
    "rhoai": {
      "command": "podman",
      "args": [
        "run", "-i", "--rm",
        "--userns=keep-id",
        "-v", "${HOME}/.kube/config:/opt/app-root/src/kubeconfig/config:ro",
        "-e", "RHOAI_MCP_AUTH_MODE=kubeconfig",
        "-e", "RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config",
        "rhoai-mcp:latest"
      ]
    }
  }
}
```

**Docker:**

```json
{
  "mcpServers": {
    "rhoai": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "${HOME}/.kube/config:/opt/app-root/src/kubeconfig/config:ro",
        "-e", "RHOAI_MCP_AUTH_MODE=kubeconfig",
        "-e", "RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config",
        "rhoai-mcp:latest"
      ]
    }
  }
}
```

Note: The container uses `stdio` transport by default, which is required for Claude Desktop integration.

## Available Tools

### Project Management (6 tools)

| Tool | Description |
|------|-------------|
| `list_data_science_projects` | List all RHOAI projects |
| `get_project_details` | Get project with resource summary |
| `create_data_science_project` | Create new project |
| `delete_data_science_project` | Delete project (requires confirmation) |
| `get_project_status` | Get comprehensive project status |
| `set_model_serving_mode` | Set single vs multi-model serving |

### Workbench Management (8 tools)

| Tool | Description |
|------|-------------|
| `list_workbenches` | List workbenches in project |
| `get_workbench` | Get workbench details |
| `create_workbench` | Create new workbench |
| `start_workbench` | Start a stopped workbench |
| `stop_workbench` | Stop a running workbench |
| `delete_workbench` | Delete workbench |
| `list_notebook_images` | List available images |
| `get_workbench_url` | Get OAuth-protected URL |

### Model Serving (6 tools)

| Tool | Description |
|------|-------------|
| `list_inference_services` | List deployed models |
| `get_inference_service` | Get model details |
| `deploy_model` | Create InferenceService |
| `delete_inference_service` | Delete deployed model |
| `list_serving_runtimes` | List available runtimes |
| `get_model_endpoint` | Get inference endpoint URL |

### Data Connections (4 tools)

| Tool | Description |
|------|-------------|
| `list_data_connections` | List connections in project |
| `get_data_connection` | Get connection details (masked) |
| `create_s3_data_connection` | Create S3 connection |
| `delete_data_connection` | Delete connection |

### Pipelines (3 tools)

| Tool | Description |
|------|-------------|
| `get_pipeline_server` | Get DSPA status |
| `create_pipeline_server` | Create DSPA |
| `delete_pipeline_server` | Delete DSPA |

### Storage (3 tools)

| Tool | Description |
|------|-------------|
| `list_storage` | List PVCs in project |
| `create_storage` | Create PVC |
| `delete_storage` | Delete PVC (requires confirmation) |

## MCP Resources

The server also exposes read-only resources:

| Resource URI | Description |
|--------------|-------------|
| `rhoai://cluster/status` | Cluster health and RHOAI status |
| `rhoai://cluster/components` | DataScienceCluster component status |
| `rhoai://cluster/accelerators` | Available GPU profiles |
| `rhoai://projects/{name}/status` | Project resource summary |
| `rhoai://projects/{name}/workbenches` | Workbench list with status |
| `rhoai://projects/{name}/models` | Deployed models with status |

## MCP Prompts

The server provides 18 prompts that guide AI agents through multi-step workflows. Prompts are templates that provide step-by-step instructions and reference the appropriate tools for each workflow stage.

### Training Workflow (3 prompts)

| Prompt | Description |
|--------|-------------|
| `train-model` | Guide through fine-tuning a model with LoRA/QLoRA |
| `monitor-training` | Monitor an active training job and diagnose issues |
| `resume-training` | Resume a suspended or failed training job from checkpoint |

### Cluster Exploration (4 prompts)

| Prompt | Description |
|--------|-------------|
| `explore-cluster` | Discover what's available in the RHOAI cluster |
| `explore-project` | Explore resources within a specific Data Science Project |
| `find-gpus` | Find available GPU resources for training or inference |
| `whats-running` | Quick status check of all active workloads |

### Troubleshooting (4 prompts)

| Prompt | Description |
|--------|-------------|
| `troubleshoot-training` | Diagnose and fix issues with a training job |
| `troubleshoot-workbench` | Diagnose and fix issues with a workbench |
| `troubleshoot-model` | Diagnose and fix issues with a deployed model |
| `analyze-oom` | Analyze and resolve Out-of-Memory issues in training |

### Project Setup (3 prompts)

| Prompt | Description |
|--------|-------------|
| `setup-training-project` | Set up a new project for model training |
| `setup-inference-project` | Set up a new project for model serving |
| `add-data-connection` | Add an S3 data connection to an existing project |

### Model Deployment (4 prompts)

| Prompt | Description |
|--------|-------------|
| `deploy-model` | Deploy a model for inference serving |
| `deploy-llm` | Deploy a Large Language Model with vLLM or TGIS |
| `test-endpoint` | Test a deployed model endpoint |
| `scale-model` | Scale a model deployment up or down |

## Example Interactions

### Create a Data Science Project

```
User: Create a new data science project called "fraud-detection" for my ML experiments
Agent: I'll create that project for you with a descriptive display name.
[Calls create_data_science_project with name="fraud-detection", 
 display_name="Fraud Detection ML", description="Machine learning experiments for fraud detection"]

Result: Project 'fraud-detection' created successfully
```

### Deploy a Model

```
User: Deploy my trained fraud model from S3 to the fraud-detection project

Agent: I'll deploy your model. First, let me check available serving runtimes.
[Calls list_serving_runtimes with namespace="fraud-detection"]
[Calls deploy_model with name="fraud-model", namespace="fraud-detection",
 runtime="ovms", model_format="onnx", 
 storage_uri="s3://models/fraud-detection/model.onnx"]

Result: Model 'fraud-model' deployment initiated. It may take a few minutes to become ready.
```

### Check Project Status

```
User: What's the status of my fraud-detection project?

Agent: [Calls get_project_status with namespace="fraud-detection"]

Result:
- Project: fraud-detection (Fraud Detection ML)
- Workbenches: 2 total, 1 running
- Models: 1 deployed, 1 ready
- Pipeline Server: Ready
- Data Connections: 1
- Storage: 3 PVCs
```

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=rhoai_mcp
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint
uv run ruff check

# Type check
uv run mypy src/rhoai_mcp
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                MCP Transport Layer (stdio/SSE/HTTP)             │
├─────────────────────────────────────────────────────────────────┤
│  FastMCP Server (server.py)                                     │
│  - Tool registration     - Resource registration                │
│  - Prompt registration   - Lifecycle management                 │
├───────────────────┬─────────────────────┬───────────────────────┤
│  Tools Layer      │  Resources Layer    │  Prompts Layer        │
│  - projects       │  - cluster.py       │  - training (3)       │
│  - notebooks      │  - projects.py      │  - exploration (4)    │
│  - inference      │                     │  - troubleshooting (4)│
│  - connections    │                     │  - project setup (3)  │
│  - storage        │                     │  - deployment (4)     │
│  - pipelines      │                     │                       │
│  - training       │                     │                       │
├───────────────────┴─────────────────────┴───────────────────────┤
│  Clients Layer (clients/) - Business Logic                      │
│  - base.py (K8sClient)   - projects.py    - notebooks.py        │
│  - inference.py          - connections.py - storage.py          │
│  - pipelines.py          - training.py                          │
├─────────────────────────────────────────────────────────────────┤
│  Models Layer (models/) - Pydantic Data Structures              │
│  - common.py (shared)    - Domain-specific models per resource  │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                           │
│  - K8sClient: Kubernetes API abstraction (Core + CRDs)          │
│  - Configuration: Environment-based settings                    │
│  - Plugin Manager: Pluggy-based plugin system                   │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

| Directory | Purpose |
|-----------|---------|
| **clients/** | Kubernetes client abstractions for each resource type |
| **models/** | Pydantic models for type-safe resource handling |
| **tools/** | MCP tool definitions that wrap client operations |
| **resources/** | MCP resource definitions for read-only data access |
| **utils/** | Helper functions for annotations, labels, and errors |

### Request Flow

```
AI Agent Request → MCP Transport → Tool Handler → Domain Client
                                                       ↓
AI Agent Response ← Pydantic Model ← K8s Response ← K8sClient → Kubernetes API
```

### Key CRDs Supported

| Resource | API Group | Purpose |
|----------|-----------|---------|
| Namespace | core/v1 | Data Science Projects |
| Notebook | kubeflow.org/v1 | Workbenches |
| InferenceService | serving.kserve.io/v1beta1 | Model serving |
| ServingRuntime | serving.kserve.io/v1alpha1 | Model server configs |
| DataSciencePipelinesApplication | datasciencepipelinesapplications.opendatahub.io/v1alpha1 | Pipeline infrastructure |
| AcceleratorProfile | dashboard.opendatahub.io/v1 | GPU profiles |

## License

MIT License - see [LICENSE](LICENSE) for details.
