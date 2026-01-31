"""MCP Prompts domain for RHOAI workflows.

This module provides MCP prompts that guide AI agents through
multi-step workflows for training, exploration, troubleshooting,
project setup, and model deployment.
"""

from rhoai_mcp.domains.prompts.prompts import register_prompts

__all__ = ["register_prompts"]
