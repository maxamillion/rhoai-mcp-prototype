# Contributing to RHOAI MCP Server

This document describes the architecture and contribution guidelines for the RHOAI MCP Server.

## Repository Structure

```text
rhoai-mcp/
├── src/
│   └── rhoai_mcp/            # Main package
│       ├── __init__.py
│       ├── __main__.py       # CLI entry point
│       ├── config.py         # Configuration
│       ├── server.py         # FastMCP server
│       ├── plugin.py         # Plugin protocol
│       ├── hooks.py          # Pluggy hook specifications
│       ├── plugin_manager.py # Plugin lifecycle management
│       ├── clients/          # K8s client abstractions
│       ├── models/           # Shared Pydantic models
│       ├── utils/            # Helper functions
│       └── domains/          # Domain modules
│           ├── projects/     # Data Science Project management
│           ├── notebooks/    # Kubeflow Notebook/Workbench
│           ├── inference/    # KServe InferenceService
│           ├── pipelines/    # Data Science Pipelines
│           ├── connections/  # S3 data connections
│           ├── storage/      # PersistentVolumeClaim
│           ├── training/     # Kubeflow Training Operator
│           ├── summary/      # Context-efficient summaries
│           ├── meta/         # Tool discovery and workflows
│           └── prompts/      # MCP workflow prompts
├── tests/                    # Test suite
│   ├── conftest.py
│   ├── domains/              # Domain-specific tests
│   │   ├── prompts/          # Prompts domain tests
│   │   └── ...
│   ├── training/             # Training domain tests
│   └── integration/          # Cross-component tests
├── docs/                     # Documentation
│   └── ARCHITECTURE.md       # Internal architecture guide
├── pyproject.toml            # Project configuration
└── uv.lock                   # Lockfile
```

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- Access to a Kubernetes/OpenShift cluster (for integration testing)

### Getting Started

```bash
# Clone the repository
git clone https://github.com/admiller/rhoai-mcp-prototype.git
cd rhoai-mcp-prototype

# Install in development mode
make dev

# Or using uv directly
uv sync
```

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration
```

### Code Quality

```bash
# Run linter
make lint

# Format code
make format

# Run type checker
make typecheck

# Run all checks
make check
```

### Running the Server Locally

```bash
# Run with SSE transport
make run-local

# Run with stdio transport
make run-local-stdio

# Run with debug logging
make run-local-debug
```

## Domain Module Architecture

### Domain Module Structure

Each domain module in `src/rhoai_mcp/domains/` follows this layout:

```text
domains/<name>/
├── __init__.py          # Exports public API
├── client.py            # K8s resource client
├── models.py            # Pydantic models
├── tools.py             # MCP tool implementations
├── crds.py              # CRD definitions (if applicable)
├── resources.py         # MCP resources (optional)
└── prompts.py           # MCP prompts (optional)
```

The `prompts` domain is special - it only contains prompts (no client or models):

```text
domains/prompts/
├── __init__.py
├── prompts.py                  # Main registration
├── training_prompts.py         # Training workflow prompts
├── exploration_prompts.py      # Cluster exploration prompts
├── troubleshooting_prompts.py  # Troubleshooting prompts
├── project_prompts.py          # Project setup prompts
└── deployment_prompts.py       # Model deployment prompts
```

### Adding a New Domain

1. Create a new directory under `src/rhoai_mcp/domains/`:
   ```
   domains/my-domain/
   ├── __init__.py
   ├── client.py
   ├── models.py
   ├── tools.py
   └── prompts.py  # Optional: if domain has workflow prompts
   ```

2. Implement the domain client:
   ```python
   from rhoai_mcp.clients.base import BaseClient

   class MyDomainClient(BaseClient):
       def list_resources(self, namespace: str) -> list[MyResource]:
           # Implement K8s API calls
           pass
   ```

3. Register the domain in `domains/registry.py`:
   ```python
   from rhoai_mcp.hooks import hookimpl
   from rhoai_mcp.plugin import BasePlugin, PluginMetadata

   class MyDomainPlugin(BasePlugin):
       def __init__(self) -> None:
           super().__init__(
               PluginMetadata(
                   name="my-domain",
                   version="1.0.0",
                   description="My domain description",
                   maintainer="team@example.com",
                   requires_crds=[],
               )
           )

       @hookimpl
       def rhoai_register_tools(self, mcp, server) -> None:
           from rhoai_mcp.domains.my_domain.tools import register_tools
           register_tools(mcp, server)

       # Optional: add prompts
       @hookimpl
       def rhoai_register_prompts(self, mcp, server) -> None:
           from rhoai_mcp.domains.my_domain.prompts import register_prompts
           register_prompts(mcp, server)
   ```

4. Add to `get_core_plugins()` list in `registry.py`

5. Add tests in `tests/domains/my_domain/`

6. Update plugin count in test files:
   - `tests/test_plugin_manager.py`
   - `tests/integration/test_plugin_discovery.py`

## Container Build

```bash
# Build container image
make build

# Test the build
make test-build

# Run container with HTTP transport
make run-http

# Run container with stdio transport
make run-stdio
```

## Pull Request Guidelines

1. Ensure all tests pass: `make test`
2. Ensure code is formatted: `make format`
3. Ensure no lint errors: `make lint`
4. Update relevant documentation
5. Add tests for new functionality
6. Keep changes focused

## Questions?

For questions, reach out to rhoai-mcp@redhat.com.
