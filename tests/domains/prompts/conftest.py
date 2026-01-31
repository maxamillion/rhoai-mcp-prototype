"""Test fixtures for prompts domain tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_mcp() -> MagicMock:
    """Create a mock FastMCP that captures prompt registrations.

    The mock captures all @mcp.prompt() decorated functions so tests
    can verify prompts were registered and call them directly.
    """
    mock = MagicMock()
    registered_prompts: dict[str, dict[str, Any]] = {}

    def capture_prompt(
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Decorator that captures prompt registration."""

        def decorator(f: Any) -> Any:
            prompt_name = name or f.__name__
            registered_prompts[prompt_name] = {
                "function": f,
                "description": description,
            }
            return f

        return decorator

    mock.prompt = capture_prompt
    mock._registered_prompts = registered_prompts
    return mock


@pytest.fixture
def mock_server() -> MagicMock:
    """Create a mock RHOAIServer."""
    return MagicMock()
