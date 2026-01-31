# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RHOAI MCP Server is an MCP (Model Context Protocol) server that enables AI agents to interact with Red Hat OpenShift AI (RHOAI) environments. It provides programmatic access to RHOAI features (projects, workbenches, model serving, pipelines, data connections, storage, training) through domain modules, plus workflow prompts that guide agents through multi-step operations.

## Build and Development Commands

```bash
# Setup development environment
uv sync                          # Install package
make dev                         # Alias for setup

# Run the server locally
uv run rhoai-mcp                 # Default (stdio transport)
uv run rhoai-mcp --transport sse # HTTP transport

# Testing
make test                        # All tests
make test-unit                   # Unit tests only (tests/training)
make test-integration            # Integration tests (tests/integration)

# Code quality
make lint                        # ruff check
make format                      # ruff format + fix
make typecheck                   # mypy
make check                       # lint + typecheck

# Container operations
make build                       # Build container image
make run-http                    # Run with SSE transport
make run-stdio                   # Run with stdio transport
make run-dev                     # Debug logging + dangerous ops enabled
```

## Architecture

### Project Structure

```
rhoai-mcp/
├── src/
│   └── rhoai_mcp/               # Main package
│       ├── __init__.py
│       ├── __main__.py          # CLI entry point
│       ├── config.py            # Configuration (pydantic-settings)
│       ├── server.py            # FastMCP server
│       ├── hooks.py             # Pluggy hook specifications
│       ├── plugin.py            # Plugin protocol and base class
│       ├── plugin_manager.py    # Plugin lifecycle management
│       ├── clients/             # K8s client abstractions
│       ├── models/              # Shared Pydantic models
│       ├── utils/               # Helper functions
│       ├── domains/             # Domain modules (pure CRUD operations)
│       │   ├── projects/        # Data Science Project management
│       │   ├── notebooks/       # Kubeflow Notebook/Workbench
│       │   ├── inference/       # KServe InferenceService
│       │   ├── pipelines/       # Data Science Pipelines (DSPA)
│       │   ├── connections/     # S3 data connections
│       │   ├── storage/         # PersistentVolumeClaim
│       │   ├── training/        # Kubeflow Training Operator
│       │   ├── evaluation/      # Model evaluation jobs
│       │   ├── prompts/         # MCP workflow prompts (18 prompts)
│       │   └── registry.py      # Domain plugin registry (9 plugins)
│       └── composites/          # Cross-cutting composite tools
│           ├── cluster/         # Cluster summaries and exploration
│           ├── training/        # Training workflow orchestration
│           ├── meta/            # Tool discovery and guidance
│           └── registry.py      # Composite plugin registry (3 plugins)
├── tests/                       # Test suite
├── docs/                        # Documentation
├── pyproject.toml               # Project configuration
└── Containerfile                # Container build
```

**Domains vs Composites**: Domain modules provide CRUD operations for specific Kubernetes resource types. Composite modules provide cross-cutting tools that orchestrate multiple domains (e.g., `prepare_training` validates storage, credentials, and runtime before creating a training job).

### Domain Module Structure

Each domain module in `src/rhoai_mcp/domains/` follows this layout:
```
domains/<name>/
├── __init__.py
├── client.py            # K8s resource client
├── models.py            # Pydantic models
├── tools.py             # MCP tool implementations
├── crds.py              # CRD definitions (if applicable)
├── resources.py         # MCP resources (if applicable)
└── prompts.py           # MCP prompts (if applicable)
```

The domain registry (`domains/registry.py`) defines all domains and provides them to the server for registration.

### Plugin Hooks

Plugins can implement these hooks (defined in `hooks.py`):
- `rhoai_register_tools`: Register MCP tools
- `rhoai_register_resources`: Register MCP resources
- `rhoai_register_prompts`: Register MCP prompts
- `rhoai_get_crd_definitions`: Return CRD definitions
- `rhoai_health_check`: Check plugin health

### Configuration

Environment variables use `RHOAI_MCP_` prefix. Key settings:
- `AUTH_MODE`: auto | kubeconfig
- `TRANSPORT`: stdio | sse | streamable-http
- `KUBECONFIG_PATH`, `KUBECONFIG_CONTEXT`: For kubeconfig auth
- `ENABLE_DANGEROUS_OPERATIONS`: Enable delete operations
- `READ_ONLY_MODE`: Disable all writes

### Key Dependencies

- `mcp>=1.0.0`: Model Context Protocol (FastMCP)
- `kubernetes>=28.1.0`: K8s Python client
- `pydantic>=2.0.0`: Data validation and settings

## Development Principles

### Test-Driven Development

Follow TDD for all code changes:

1. **Write tests first**: Before implementing any feature or fix, write failing tests that define the expected behavior
2. **Red-Green-Refactor**: Run tests to see them fail (red), write minimal code to pass (green), then refactor while keeping tests green
3. **Test coverage**: All new code must have corresponding tests; run `make test` before committing
4. **Test types**: Unit tests go in `tests/`, integration tests in `tests/integration/`

### Simplicity and Maintainability

Favor simple, maintainable solutions at all times:

- **KISS**: Choose the simplest solution that works; avoid premature optimization or over-abstraction
- **Single responsibility**: Each function, class, and module should do one thing well
- **Explicit over implicit**: Code should be self-documenting; avoid magic or clever tricks
- **Minimal dependencies**: Only add dependencies when truly necessary
- **Delete dead code**: Remove unused code rather than commenting it out
- **Small functions**: Keep functions short and focused; if a function needs extensive comments, it should be split

### Idiomatic Python

Write Pythonic code that follows community conventions:

- Use list/dict/set comprehensions where they improve readability
- Prefer `pathlib.Path` over `os.path` for file operations
- Use context managers (`with` statements) for resource management
- Leverage dataclasses and Pydantic models for structured data
- Use type hints consistently (required by mypy)
- Follow PEP 8 naming: `snake_case` for functions/variables, `PascalCase` for classes
- Use `typing` module for complex types; prefer `|` union syntax (Python 3.10+)
- Prefer raising specific exceptions over generic ones
- Use f-strings for string formatting

## Code Style

- Python 3.10+, line length 100
- Ruff for linting/formatting (isort included)
- Mypy with `disallow_untyped_defs=true`
- Pytest with `asyncio_mode = "auto"`
