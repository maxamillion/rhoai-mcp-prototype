# RHOAI MCP Server Architecture

This document covers the internal architecture and implementation patterns for maintainers of the RHOAI MCP server.

## Domain-Based Plugin Architecture

Each domain is a self-contained module in `src/rhoai_mcp/domains/` with:

```
domains/<name>/
├── __init__.py
├── client.py      # K8s resource operations
├── tools.py       # MCP tool implementations
├── models.py      # Pydantic models
├── crds.py        # CRD definitions (if applicable)
├── resources.py   # MCP resources (if applicable)
└── prompts.py     # MCP prompts (if applicable)
```

Domains are registered via the plugin system in `domains/registry.py`. Each plugin class:
1. Inherits from `BasePlugin`
2. Provides metadata via `PluginMetadata`
3. Implements `@hookimpl` decorated methods for registration

### Available Hooks

| Hook | Purpose |
|------|---------|
| `rhoai_get_plugin_metadata` | Return plugin metadata |
| `rhoai_register_tools` | Register MCP tools |
| `rhoai_register_resources` | Register MCP resources |
| `rhoai_register_prompts` | Register MCP prompts |
| `rhoai_get_crd_definitions` | Return CRD definitions |
| `rhoai_health_check` | Check plugin health |

## Composite Workflow Tools

These high-level tools combine multiple operations to reduce tool call round-trips for AI agents.

### `prepare_training()` - Training Pre-flight

**Location:** `src/rhoai_mcp/domains/training/tools/planning.py`

**Purpose:** Combines resource estimation, prerequisite checking, config validation, and optional storage creation.

**Implementation notes:**
- Uses `_estimate_resources_internal()` helper for resource estimation
- Runtime auto-selection picks the first available if none specified
- Storage creation uses `create_training_pvc()` from the storage module
- Returns `suggested_train_params` that can be passed directly to `train()`

**Adding new prerequisites:**
Add checks between the runtime validation and storage handling sections. Follow the pattern:
```python
try:
    # Check something
    if check_fails:
        issues.append("Description of issue")
        prereq_passed = False
except Exception as e:
    issues.append(f"Failed to check: {e}")
    prereq_passed = False
```

### `explore_cluster()` - Cluster Overview

**Location:** `src/rhoai_mcp/domains/summary/tools.py`

**Purpose:** Complete cluster exploration with all projects and resource summaries in one call.

**Implementation notes:**
- Uses `TrainingClient.get_cluster_resources()` for GPU info
- Iterates projects and extracts `resource_summary` from Project model
- Health checks are simple heuristics (e.g., "all workbenches stopped")

**Adding new health checks:**
Add them in the `if include_health:` block:
```python
if include_health:
    proj_issues = []
    # Add your check here
    if some_condition:
        proj_issues.append("Issue description")
```

### `diagnose_resource()` - Resource Diagnostics

**Location:** `src/rhoai_mcp/domains/summary/tools.py`

**Purpose:** Comprehensive diagnostics for any resource type.

**Implementation notes:**
- Uses dispatcher pattern to route to type-specific `_diagnose_*` functions
- Type names are normalized to lowercase for flexible matching
- Each helper follows the same structure: get resource, extract info, detect issues

**Adding a new resource type:**
1. Create `_diagnose_<type>(server, name, namespace) -> dict` function
2. Add routing in the main function:
   ```python
   elif resource_type in ("newtype", "alias"):
       result.update(_diagnose_newtype(server, name, namespace))
   ```

### `prepare_model_deployment()` - Deployment Pre-flight

**Location:** `src/rhoai_mcp/domains/inference/tools.py`

**Purpose:** Pre-flight preparation for model deployment with runtime discovery and resource estimation.

**Implementation notes:**
- `_estimate_model_info()` parses model ID to extract parameter count using regex patterns
- Runtime compatibility checks `model_format` against `supported_formats`
- Prefers vLLM/TGIS runtimes for LLMs
- Storage validation for `pvc://` URIs checks if PVC is bound

**Updating model size estimates:**
Modify `MODEL_SIZE_ESTIMATES` dict at the top of the file:
```python
MODEL_SIZE_ESTIMATES = {
    (0, 1): 2,      # < 1B params -> ~2GB
    (1, 3): 6,      # 1-3B params -> ~6GB
    # Add or adjust ranges as needed
}
```

## Generic Resource Tools

**Location:** `src/rhoai_mcp/domains/summary/tools.py`

These provide a unified interface over domain-specific clients: `get_resource()`, `list_resources()`, `manage_resource()`.

**Implementation notes:**
- Accept multiple type aliases (e.g., "workbench", "notebook" both work)
- Use distinct variable names for each client type to satisfy mypy
- Lifecycle actions route to `_manage_*` helper functions

**Adding a new domain:**
Add type aliases and client calls to each function:
```python
if resource_type in ("newtype", "newtypealias"):
    from rhoai_mcp.domains.newdomain.client import NewClient
    new_client = NewClient(server.k8s)
    # ... rest of implementation
```

## Meta Domain - Tool Discovery

**Location:** `src/rhoai_mcp/domains/meta/`

Helps AI agents discover the right tools for their tasks.

### `suggest_tools()`

**Implementation notes:**
- `INTENT_PATTERNS` list defines keyword patterns and their workflows
- Pattern matching counts keyword occurrences and selects best match
- Falls back to "discovery" category if no patterns match

**Improving intent matching:**
Add more keywords to existing patterns or create new ones:
```python
{
    "patterns": ["keyword1", "keyword2", ...],
    "category": "category_name",
    "workflow": ["tool1", "tool2"],
    "explanation": "How to use these tools...",
}
```

### MCP Resources

`resources.py` exposes static metadata via `@mcp.resource()` decorator:
- `rhoai://tools/categories` - Tool organization
- `rhoai://tools/workflows` - Step-by-step workflow guides

## Unified Training Tool

**Location:** `src/rhoai_mcp/domains/training/tools/unified.py`

Single `training()` tool with `action` parameter that consolidates all training operations.

**Implementation notes:**
- `_VALID_ACTIONS` set defines allowed actions
- Some actions don't require namespace (defined in `no_namespace_actions`)
- Each `_action_*` function implements one action
- `TrainJob` model uses `status == TrainJobStatus.SUSPENDED` for suspension check (no direct `suspended` field)

**Adding a new action:**
1. Add to `_VALID_ACTIONS` set
2. Add to `no_namespace_actions` if namespace not required
3. Create `_action_<name>(server, ...) -> dict[str, Any]` function
4. Add routing in the dispatch section:
   ```python
   elif action == "newaction":
       return _action_newaction(server, namespace, name, ...)
   ```

## Inference Planning Tools

**Location:** `src/rhoai_mcp/domains/inference/tools.py`

Mirror the training domain's planning tools pattern.

| Tool | Purpose |
|------|---------|
| `check_deployment_prerequisites()` | Pre-flight checks |
| `estimate_serving_resources()` | GPU/memory estimation |
| `recommend_serving_runtime()` | Runtime selection |
| `test_model_endpoint()` | Endpoint accessibility |

**Helper functions:**
- `_estimate_model_info()` - Parses model ID for params and format
- `_estimate_serving_resources()` - Calculates GPU/memory needs
- `_generate_deployment_name()` - Creates DNS-safe name from model ID

## Prompts Domain

**Location:** `src/rhoai_mcp/domains/prompts/`

MCP Prompts provide workflow guidance templates that help AI agents through multi-step operations.

### Structure

```
domains/prompts/
├── __init__.py
├── prompts.py                  # Main registration coordinator
├── training_prompts.py         # Training workflow (3 prompts)
├── exploration_prompts.py      # Cluster exploration (4 prompts)
├── troubleshooting_prompts.py  # Troubleshooting (4 prompts)
├── project_prompts.py          # Project setup (3 prompts)
└── deployment_prompts.py       # Model deployment (4 prompts)
```

### Prompt Registration Pattern

Prompts are registered using the `@mcp.prompt()` decorator:

```python
@mcp.prompt(
    name="train-model",
    description="Guide through fine-tuning a model with LoRA/QLoRA",
)
def train_model(model_id: str, dataset_id: str, namespace: str, method: str = "lora") -> str:
    return f"""I need to fine-tune a model...

**Training Configuration:**
- Model: {model_id}
- Dataset: {dataset_id}

**Please help me complete these steps:**

1. **Check Prerequisites**
   - Use `check_training_prerequisites` to verify...
...
"""
```

### Adding a New Prompt

1. Choose the appropriate prompt file based on category
2. Add a new function with the `@mcp.prompt()` decorator:
   ```python
   @mcp.prompt(
       name="my-prompt",
       description="Short description of what this prompt does",
   )
   def my_prompt(param1: str, param2: str = "default") -> str:
       return f"""Workflow guidance text...

   **Configuration:**
   - Param1: {param1}
   - Param2: {param2}

   **Steps:**
   1. Use `tool_name` to do something
   2. Use `another_tool` to do something else
   ...
   """
   ```
3. Add tests in `tests/domains/prompts/test_<category>_prompts.py`

### Adding a New Prompt Category

1. Create a new file `domains/prompts/<category>_prompts.py`
2. Implement `register_prompts(mcp, server)` function
3. Import and call it from `prompts.py`:
   ```python
   from rhoai_mcp.domains.prompts import new_category_prompts
   new_category_prompts.register_prompts(mcp, server)
   ```
4. Add corresponding test file

### Prompt Best Practices

- **Reference specific tools**: Use backticks around tool names (e.g., `get_training_job`)
- **Provide clear steps**: Number each step and explain what to do
- **Include parameters**: Reference the prompt parameters in the output
- **Handle edge cases**: Mention what to do if something goes wrong
- **Keep focused**: Each prompt should address one workflow

## Common Patterns

### Lazy Imports
Imports inside functions avoid circular dependencies:
```python
def some_tool():
    from rhoai_mcp.domains.other.client import OtherClient
    # ...
```

### Error Returns
Return error dicts rather than raising exceptions:
```python
if something_wrong:
    return {"error": "Description of what went wrong"}
```

### Confirmation Pattern
Destructive operations require explicit confirmation:
```python
if not confirm:
    return {
        "error": "Deletion not confirmed",
        "message": "Set confirm=True to proceed",
    }
```

### Operation Allowed Check
Check permissions before create/delete:
```python
allowed, reason = server.config.is_operation_allowed("create")
if not allowed:
    return {"error": reason}
```

### Type Hints
Use `dict[str, Any]` for return types when structure varies by code path.

## Testing Patterns

### Mock MCP Fixture
Captures tool registrations for direct testing:
```python
@pytest.fixture
def mock_mcp():
    mock = MagicMock()
    registered_tools = {}

    def capture_tool():
        def decorator(f):
            registered_tools[f.__name__] = f
            return f
        return decorator

    mock.tool = capture_tool
    mock._registered_tools = registered_tools
    return mock
```

### Testing Registered Tools
```python
def test_something(mock_mcp, mock_server):
    register_tools(mock_mcp, mock_server)
    my_tool = mock_mcp._registered_tools["my_tool"]
    result = my_tool(arg1="value")
    assert result["expected_key"] == "expected_value"
```

## Plugin Registration

To add a new domain:

1. Create the domain directory with required files
2. Create a plugin class in `domains/registry.py`:
   ```python
   class NewDomainPlugin(BasePlugin):
       def __init__(self) -> None:
           super().__init__(
               PluginMetadata(
                   name="newdomain",
                   version="1.0.0",
                   description="Description",
                   maintainer="email@example.com",
                   requires_crds=[],
               )
           )

       @hookimpl
       def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
           from rhoai_mcp.domains.newdomain.tools import register_tools
           register_tools(mcp, server)

       @hookimpl
       def rhoai_register_resources(self, mcp: FastMCP, server: RHOAIServer) -> None:
           from rhoai_mcp.domains.newdomain.resources import register_resources
           register_resources(mcp, server)

       @hookimpl
       def rhoai_register_prompts(self, mcp: FastMCP, server: RHOAIServer) -> None:
           from rhoai_mcp.domains.newdomain.prompts import register_prompts
           register_prompts(mcp, server)
   ```
3. Add to `get_core_plugins()` list
4. Update test assertions for plugin count in:
   - `tests/test_plugin_manager.py`
   - `tests/integration/test_plugin_discovery.py`

### Hook Registration Order

During server startup, hooks are called in this order:
1. `rhoai_get_plugin_metadata` - Collect plugin metadata
2. `rhoai_register_tools` - Register all MCP tools
3. `rhoai_register_resources` - Register all MCP resources
4. `rhoai_register_prompts` - Register all MCP prompts
5. `rhoai_health_check` - Run health checks (during lifespan)
