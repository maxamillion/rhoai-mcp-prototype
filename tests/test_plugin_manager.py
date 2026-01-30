"""Unit tests for PluginManager."""

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.hooks import hookimpl
from rhoai_mcp.plugin import PluginMetadata
from rhoai_mcp.plugin_manager import PluginManager


class TestPluginManager:
    """Tests for PluginManager class."""

    def test_init_creates_pluggy_manager(self) -> None:
        """Verify PluginManager initializes pluggy correctly."""
        pm = PluginManager()
        assert pm._pm is not None
        assert pm.hook is not None

    def test_register_plugin_with_name(self) -> None:
        """Verify plugin can be registered with explicit name."""

        class TestPlugin:
            pass

        pm = PluginManager()
        name = pm.register_plugin(TestPlugin(), name="test-plugin")

        assert name == "test-plugin"
        assert "test-plugin" in pm.registered_plugins

    def test_register_plugin_uses_metadata_name(self) -> None:
        """Verify plugin name is taken from metadata if not provided."""

        class TestPlugin:
            def rhoai_get_plugin_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="from-metadata",
                    version="1.0.0",
                    description="Test",
                    maintainer="test@example.com",
                )

        pm = PluginManager()
        name = pm.register_plugin(TestPlugin())

        assert name == "from-metadata"
        assert "from-metadata" in pm.registered_plugins

    def test_unregister_plugin(self) -> None:
        """Verify plugin can be unregistered."""

        class TestPlugin:
            pass

        pm = PluginManager()
        pm.register_plugin(TestPlugin(), name="test")
        assert "test" in pm.registered_plugins

        pm.unregister_plugin("test")
        assert "test" not in pm.registered_plugins

    def test_get_all_metadata(self) -> None:
        """Verify metadata can be collected from all plugins."""

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

        pm = PluginManager()
        pm.register_plugin(PluginA(), name="plugin_a")
        pm.register_plugin(PluginB(), name="plugin_b")

        metadata = pm.get_all_metadata()
        assert len(metadata) == 2
        names = {m.name for m in metadata}
        assert names == {"plugin_a", "plugin_b"}

    def test_register_all_tools(self) -> None:
        """Verify tools are registered from all plugins."""
        tool_calls = []

        class TestPlugin:
            @hookimpl
            def rhoai_register_tools(self, mcp: MagicMock, server: MagicMock) -> None:
                tool_calls.append(("register_tools", mcp, server))

        pm = PluginManager()
        pm.register_plugin(TestPlugin(), name="test")

        mock_mcp = MagicMock()
        mock_server = MagicMock()
        pm.register_all_tools(mock_mcp, mock_server)

        assert len(tool_calls) == 1
        assert tool_calls[0][1] is mock_mcp
        assert tool_calls[0][2] is mock_server

    def test_register_all_resources(self) -> None:
        """Verify resources are registered from all plugins."""
        resource_calls = []

        class TestPlugin:
            @hookimpl
            def rhoai_register_resources(
                self, mcp: MagicMock, server: MagicMock
            ) -> None:
                resource_calls.append(("register_resources", mcp, server))

        pm = PluginManager()
        pm.register_plugin(TestPlugin(), name="test")

        mock_mcp = MagicMock()
        mock_server = MagicMock()
        pm.register_all_resources(mock_mcp, mock_server)

        assert len(resource_calls) == 1

    def test_run_health_checks_healthy_plugin(self) -> None:
        """Verify healthy plugins pass health check."""

        class HealthyPlugin:
            @hookimpl
            def rhoai_health_check(self, server: MagicMock) -> tuple[bool, str]:
                return True, "All good"

        pm = PluginManager()
        pm.register_plugin(HealthyPlugin(), name="healthy")

        mock_server = MagicMock()
        results = pm.run_health_checks(mock_server)

        assert results["healthy"] == (True, "All good")
        assert "healthy" in pm.healthy_plugins

    def test_run_health_checks_unhealthy_plugin(self) -> None:
        """Verify unhealthy plugins fail health check."""

        class UnhealthyPlugin:
            @hookimpl
            def rhoai_health_check(self, server: MagicMock) -> tuple[bool, str]:
                return False, "Missing CRD"

        pm = PluginManager()
        pm.register_plugin(UnhealthyPlugin(), name="unhealthy")

        mock_server = MagicMock()
        results = pm.run_health_checks(mock_server)

        assert results["unhealthy"] == (False, "Missing CRD")
        assert "unhealthy" not in pm.healthy_plugins

    def test_run_health_checks_no_health_check(self) -> None:
        """Verify plugins without health check are assumed healthy."""

        class NoHealthCheckPlugin:
            pass

        pm = PluginManager()
        pm.register_plugin(NoHealthCheckPlugin(), name="no-check")

        mock_server = MagicMock()
        results = pm.run_health_checks(mock_server)

        assert results["no-check"][0] is True
        assert "no-check" in pm.healthy_plugins

    def test_run_health_checks_exception(self) -> None:
        """Verify health check exceptions are handled."""

        class FailingPlugin:
            def rhoai_health_check(self, server: MagicMock) -> tuple[bool, str]:
                raise RuntimeError("Connection failed")

        pm = PluginManager()
        pm.register_plugin(FailingPlugin(), name="failing")

        mock_server = MagicMock()
        results = pm.run_health_checks(mock_server)

        assert results["failing"][0] is False
        assert "error" in results["failing"][1].lower()
        assert "failing" not in pm.healthy_plugins

    def test_get_all_crd_definitions(self) -> None:
        """Verify CRD definitions are collected from all plugins."""

        class PluginWithCRDs:
            @hookimpl
            def rhoai_get_crd_definitions(self) -> list:
                return [{"kind": "TestCRD", "group": "test.io"}]

        pm = PluginManager()
        pm.register_plugin(PluginWithCRDs(), name="with-crds")

        crds = pm.get_all_crd_definitions()
        assert len(crds) == 1
        assert crds[0]["kind"] == "TestCRD"

    def test_load_core_plugins(self) -> None:
        """Verify core plugins are loaded from registry."""
        pm = PluginManager()
        count = pm.load_core_plugins()

        # Should load all 9 core domain plugins
        assert count == 9
        assert len(pm.registered_plugins) == 9

        # Verify expected plugins are loaded
        expected = {
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
        assert set(pm.registered_plugins.keys()) == expected
