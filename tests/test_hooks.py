"""Unit tests for hook specifications."""

import pluggy
import pytest

from rhoai_mcp.hooks import PROJECT_NAME, RHOAIMCPHookSpec, hookimpl


class TestHookSpec:
    """Tests for RHOAIMCPHookSpec."""

    def test_project_name_defined(self) -> None:
        """Verify project name is defined correctly."""
        assert PROJECT_NAME == "rhoai_mcp"

    def test_hookspec_has_required_methods(self) -> None:
        """Verify hookspec defines all required hook methods."""
        spec = RHOAIMCPHookSpec()

        # Check all expected hooks exist
        assert hasattr(spec, "rhoai_get_plugin_metadata")
        assert hasattr(spec, "rhoai_register_tools")
        assert hasattr(spec, "rhoai_register_resources")
        assert hasattr(spec, "rhoai_get_crd_definitions")
        assert hasattr(spec, "rhoai_health_check")

    def test_hookspec_can_be_added_to_pluggy(self) -> None:
        """Verify hookspec can be registered with pluggy."""
        pm = pluggy.PluginManager(PROJECT_NAME)
        pm.add_hookspecs(RHOAIMCPHookSpec)

        # Should have the hook caller
        assert hasattr(pm.hook, "rhoai_get_plugin_metadata")
        assert hasattr(pm.hook, "rhoai_register_tools")
        assert hasattr(pm.hook, "rhoai_register_resources")
        assert hasattr(pm.hook, "rhoai_get_crd_definitions")
        assert hasattr(pm.hook, "rhoai_health_check")


class TestHookImpl:
    """Tests for hook implementations."""

    def test_hookimpl_decorator_works(self) -> None:
        """Verify hookimpl decorator can be used on methods."""

        class TestPlugin:
            @hookimpl
            def rhoai_get_plugin_metadata(self) -> dict:
                return {"name": "test"}

        plugin = TestPlugin()
        assert hasattr(plugin.rhoai_get_plugin_metadata, "rhoai_mcp_impl")

    def test_plugin_with_hookimpl_can_register(self) -> None:
        """Verify plugin with hookimpl decorators can be registered."""
        from rhoai_mcp.plugin import PluginMetadata

        class TestPlugin:
            @hookimpl
            def rhoai_get_plugin_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="test",
                    version="1.0.0",
                    description="Test plugin",
                    maintainer="test@example.com",
                )

        pm = pluggy.PluginManager(PROJECT_NAME)
        pm.add_hookspecs(RHOAIMCPHookSpec)
        pm.register(TestPlugin())

        results = pm.hook.rhoai_get_plugin_metadata()
        assert len(results) == 1
        assert results[0].name == "test"

    def test_multiple_plugins_can_register(self) -> None:
        """Verify multiple plugins can be registered and hooks called."""
        from rhoai_mcp.plugin import PluginMetadata

        class PluginA:
            @hookimpl
            def rhoai_get_plugin_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="plugin_a",
                    version="1.0.0",
                    description="Plugin A",
                    maintainer="test@example.com",
                )

        class PluginB:
            @hookimpl
            def rhoai_get_plugin_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="plugin_b",
                    version="2.0.0",
                    description="Plugin B",
                    maintainer="test@example.com",
                )

        pm = pluggy.PluginManager(PROJECT_NAME)
        pm.add_hookspecs(RHOAIMCPHookSpec)
        pm.register(PluginA())
        pm.register(PluginB())

        results = pm.hook.rhoai_get_plugin_metadata()
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"plugin_a", "plugin_b"}
