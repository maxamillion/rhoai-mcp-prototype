"""Composite tools that combine multiple domain operations.

Composites are cross-cutting tools that orchestrate operations across
multiple domains. They differ from domain-specific tools in that they:

1. Combine multiple steps into unified workflows
2. Import from multiple domains
3. Are designed for AI agent efficiency (fewer tool calls)

The dependency flow is one-way: composites import from domains,
never the reverse.
"""
