"""Tests for meta domain tools."""

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.meta.tools import (
    INTENT_PATTERNS,
    TOOL_CATEGORIES,
    register_tools,
)


@pytest.fixture
def mock_mcp() -> MagicMock:
    """Create a mock FastMCP server that captures tool registrations."""
    mock = MagicMock()
    registered_tools: dict = {}

    def capture_tool():
        def decorator(f):
            registered_tools[f.__name__] = f
            return f

        return decorator

    mock.tool = capture_tool
    mock._registered_tools = registered_tools
    return mock


@pytest.fixture
def mock_server() -> MagicMock:
    """Create a mock RHOAIServer."""
    server = MagicMock()
    return server


class TestToolCategories:
    """Tests for tool category definitions."""

    def test_categories_defined(self) -> None:
        """All expected categories are defined."""
        expected = ["discovery", "training", "inference", "workbenches", "diagnostics", "resources", "storage"]
        for cat in expected:
            assert cat in TOOL_CATEGORIES

    def test_discovery_has_use_first(self) -> None:
        """Discovery category has use_first flag."""
        assert TOOL_CATEGORIES["discovery"].get("use_first") is True

    def test_all_categories_have_description_and_tools(self) -> None:
        """All categories have description and tools."""
        for name, info in TOOL_CATEGORIES.items():
            assert "description" in info, f"{name} missing description"
            assert "tools" in info, f"{name} missing tools"
            assert len(info["tools"]) > 0, f"{name} has no tools"


class TestIntentPatterns:
    """Tests for intent pattern definitions."""

    def test_patterns_defined(self) -> None:
        """Intent patterns are defined."""
        assert len(INTENT_PATTERNS) > 0

    def test_all_patterns_have_required_fields(self) -> None:
        """All patterns have required fields."""
        for pattern in INTENT_PATTERNS:
            assert "patterns" in pattern
            assert "category" in pattern
            assert "workflow" in pattern
            assert "explanation" in pattern


class TestSuggestTools:
    """Tests for suggest_tools function."""

    def test_tool_registration(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """suggest_tools is registered as a tool."""
        register_tools(mock_mcp, mock_server)
        assert "suggest_tools" in mock_mcp._registered_tools

    def test_suggest_training_intent(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Training intent returns training workflow."""
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("I want to train a model", None)

        assert result["category"] == "training"
        assert "prepare_training" in result["workflow"]
        assert "train" in result["workflow"]

    def test_suggest_deploy_intent(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Deploy intent returns inference workflow."""
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("deploy a model for inference", None)

        assert result["category"] == "inference"
        assert "prepare_model_deployment" in result["workflow"]

    def test_suggest_debug_intent(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Debug intent returns diagnostics workflow."""
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("debug failed job", None)

        assert result["category"] == "diagnostics"
        assert "diagnose_resource" in result["workflow"]

    def test_suggest_explore_intent(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Explore intent returns discovery workflow."""
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("what's running in the cluster", None)

        assert result["category"] == "discovery"
        assert "explore_cluster" in result["workflow"]

    def test_suggest_with_context(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Context is used in example calls."""
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("train a model", {"namespace": "my-project"})

        # Check that namespace from context is used
        assert result["example_calls"][0]["args"]["namespace"] == "my-project"


class TestListToolCategories:
    """Tests for list_tool_categories function."""

    def test_tool_registration(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """list_tool_categories is registered as a tool."""
        register_tools(mock_mcp, mock_server)
        assert "list_tool_categories" in mock_mcp._registered_tools

    def test_returns_all_categories(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Returns all defined categories."""
        register_tools(mock_mcp, mock_server)
        list_categories = mock_mcp._registered_tools["list_tool_categories"]

        result = list_categories()

        assert "categories" in result
        assert len(result["categories"]) == len(TOOL_CATEGORIES)

    def test_returns_recommendation(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Returns a recommendation."""
        register_tools(mock_mcp, mock_server)
        list_categories = mock_mcp._registered_tools["list_tool_categories"]

        result = list_categories()

        assert "recommendation" in result
