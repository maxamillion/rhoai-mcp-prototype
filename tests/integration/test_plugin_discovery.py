"""Integration tests for pluggy-based plugin discovery."""

from unittest.mock import MagicMock


def test_plugin_manager_loads_core_plugins():
    """Verify PluginManager loads all core domain plugins."""
    from rhoai_mcp.plugin_manager import PluginManager

    pm = PluginManager()
    count = pm.load_core_plugins()

    # Should load all 9 core domain plugins
    assert count == 9
    assert len(pm.registered_plugins) == 9

    expected_plugins = {
        "projects",
        "notebooks",
        "inference",
        "pipelines",
        "connections",
        "storage",
        "training",
        "summary",
        "meta",
    }
    assert set(pm.registered_plugins.keys()) == expected_plugins


def test_core_plugins_have_valid_metadata():
    """Verify all core plugins provide valid metadata."""
    from rhoai_mcp.plugin_manager import PluginManager

    pm = PluginManager()
    pm.load_core_plugins()

    metadata_list = pm.get_all_metadata()
    assert len(metadata_list) == 9

    for meta in metadata_list:
        assert meta.name
        assert meta.version
        assert meta.description
        assert meta.maintainer


def test_plugins_can_register_tools():
    """Verify plugins can register tools without error."""
    from rhoai_mcp.plugin_manager import PluginManager
    from rhoai_mcp.server import RHOAIServer

    pm = PluginManager()
    pm.load_core_plugins()

    mock_mcp = MagicMock()
    server = RHOAIServer()

    # Should not raise
    pm.register_all_tools(mock_mcp, server)


def test_plugins_can_register_resources():
    """Verify plugins can register resources without error."""
    from rhoai_mcp.plugin_manager import PluginManager
    from rhoai_mcp.server import RHOAIServer

    pm = PluginManager()
    pm.load_core_plugins()

    mock_mcp = MagicMock()
    server = RHOAIServer()

    # Should not raise
    pm.register_all_resources(mock_mcp, server)


def test_server_creates_plugin_manager():
    """Verify RHOAIServer creates and uses PluginManager."""
    from rhoai_mcp.server import RHOAIServer

    server = RHOAIServer()
    mcp = server.create_mcp()

    assert server._plugin_manager is not None
    assert len(server.plugins) == 9


def test_external_plugins_discovered():
    """Verify external plugins are discovered via entry points.

    With the pluggy architecture, external plugins are discovered
    via load_setuptools_entrypoints. Currently no external plugins
    are registered, so this tests the mechanism works.
    """
    from rhoai_mcp.plugin_manager import PluginManager

    pm = PluginManager()

    # This should not raise even with no external plugins
    count = pm.load_entrypoint_plugins()

    # No external plugins registered currently
    assert count == 0


def test_get_core_plugins_returns_plugin_instances():
    """Verify get_core_plugins returns proper plugin instances."""
    from rhoai_mcp.domains.registry import get_core_plugins
    from rhoai_mcp.plugin import BasePlugin

    plugins = get_core_plugins()
    assert len(plugins) == 9

    for plugin in plugins:
        assert isinstance(plugin, BasePlugin)
        # All should have hookimpl-decorated methods
        assert hasattr(plugin.rhoai_get_plugin_metadata, "rhoai_mcp_impl")
