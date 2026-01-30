"""FastMCP server definition for RHOAI with pluggy-based plugin system."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.clients.base import K8sClient
from rhoai_mcp.config import RHOAIConfig, get_config
from rhoai_mcp.plugin_manager import PluginManager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RHOAIServer:
    """RHOAI MCP Server with pluggy-based plugin system."""

    def __init__(self, config: RHOAIConfig | None = None) -> None:
        self._config = config or get_config()
        self._k8s_client: K8sClient | None = None
        self._mcp: FastMCP | None = None
        self._plugin_manager: PluginManager | None = None
        self._evaluation_plugin: Any = None

    @property
    def config(self) -> RHOAIConfig:
        """Get server configuration."""
        return self._config

    @property
    def k8s(self) -> K8sClient:
        """Get the Kubernetes client.

        Raises:
            RuntimeError: If server is not running.
        """
        if self._k8s_client is None:
            raise RuntimeError("Server not running. K8s client not available.")
        return self._k8s_client

    @property
    def mcp(self) -> FastMCP:
        """Get the MCP server instance.

        Raises:
            RuntimeError: If server is not initialized.
        """
        if self._mcp is None:
            raise RuntimeError("Server not initialized.")
        return self._mcp

    @property
    def plugin_manager(self) -> PluginManager:
        """Get the plugin manager.

        Raises:
            RuntimeError: If server is not initialized.
        """
        if self._plugin_manager is None:
            raise RuntimeError("Server not initialized.")
        return self._plugin_manager

    @property
    def plugins(self) -> dict[str, Any]:
        """Get all registered plugins."""
        if self._plugin_manager is None:
            return {}
        return self._plugin_manager.registered_plugins

    @property
    def healthy_plugins(self) -> dict[str, Any]:
        """Get plugins that passed health checks."""
        if self._plugin_manager is None:
            return {}
        return self._plugin_manager.healthy_plugins

    def _create_lifespan(self) -> Callable[[Any], AbstractAsyncContextManager[None]]:
        """Create the lifespan context manager for the MCP server."""
        server_self = self

        @asynccontextmanager
        async def lifespan(_app: Any) -> AsyncIterator[None]:
            """Manage server lifecycle - connect K8s on startup, disconnect on shutdown."""
            logger.info("Starting RHOAI MCP server...")

            # Connect to Kubernetes
            server_self._k8s_client = K8sClient(server_self._config)
            try:
                server_self._k8s_client.connect()

                # Run health checks on all plugins
                if server_self._plugin_manager:
                    server_self._plugin_manager.run_health_checks(server_self)

                pm = server_self._plugin_manager
                total = len(pm.registered_plugins) if pm else 0
                healthy = len(pm.healthy_plugins) if pm else 0

                logger.info(f"RHOAI MCP server started with {healthy}/{total} plugins active")
                yield
            finally:
                logger.info("Shutting down RHOAI MCP server...")
                if server_self._k8s_client:
                    server_self._k8s_client.disconnect()
                server_self._k8s_client = None
                logger.info("RHOAI MCP server shut down")

        return lifespan

    def create_mcp(self) -> FastMCP:
        """Create and configure the FastMCP server."""
        # Create plugin manager
        self._plugin_manager = PluginManager()

        # Load core domain plugins
        core_count = self._plugin_manager.load_core_plugins()
        logger.info(f"Loaded {core_count} core domain plugins")

        # Load evaluation plugin if enabled
        if self._config.enable_evaluation:
            self._load_evaluation_plugin()

        # Discover and load external plugins
        external_count = self._plugin_manager.load_entrypoint_plugins()
        logger.info(f"Discovered {external_count} external plugins")

        # Create MCP server with lifespan
        mcp = FastMCP(
            name="rhoai-mcp",
            instructions="MCP server for Red Hat OpenShift AI - enables AI agents to "
            "interact with RHOAI environments including workbenches, "
            "model serving, pipelines, and data connections.",
            lifespan=self._create_lifespan(),
            host=self._config.host,
            port=self._config.port,
        )

        # Store reference
        self._mcp = mcp

        # Instrument tools for evaluation if enabled
        if self._config.enable_evaluation and self._evaluation_plugin:
            self._instrument_tools_for_evaluation(mcp)

        # Register tools, resources, and prompts from all plugins
        self._plugin_manager.register_all_tools(mcp, self)
        self._plugin_manager.register_all_resources(mcp, self)
        self._plugin_manager.register_all_prompts(mcp, self)

        # Register core resources (cluster status, etc.)
        self._register_core_resources(mcp)

        return mcp

    def _load_evaluation_plugin(self) -> None:
        """Load the evaluation plugin."""
        from rhoai_mcp.domains.evaluation.plugin import EvaluationPlugin

        if self._plugin_manager is None:
            raise RuntimeError("Plugin manager not initialized")

        self._evaluation_plugin = EvaluationPlugin()
        self._plugin_manager.register_plugin(self._evaluation_plugin, "evaluation")
        logger.info("Loaded evaluation plugin for agent performance tracking")

    def _instrument_tools_for_evaluation(self, mcp: FastMCP) -> None:
        """Instrument MCP tools for evaluation tracking."""
        from rhoai_mcp.evaluation.instrumentation import instrument_mcp_tools

        if self._plugin_manager is None:
            raise RuntimeError("Plugin manager not initialized")

        def session_provider() -> str | None:
            if self._evaluation_plugin:
                session_id: str | None = self._evaluation_plugin.get_active_session_id()
                return session_id
            return None

        instrument_mcp_tools(
            mcp=mcp,
            hook_caller=self._plugin_manager.hook,
            session_provider=session_provider,
        )
        logger.info("Instrumented MCP tools for evaluation tracking")

    def _register_core_resources(self, mcp: FastMCP) -> None:
        """Register core MCP resources for cluster information."""
        from rhoai_mcp.clients.base import CRDs

        @mcp.resource("rhoai://cluster/status")
        def cluster_status() -> dict:
            """Get RHOAI cluster status and health.

            Returns overall cluster status including RHOAI operator status,
            available components, and loaded plugins.
            """
            k8s = self.k8s
            pm = self._plugin_manager

            result: dict = {
                "connected": k8s.is_connected,
                "rhoai_available": False,
                "components": {},
                "plugins": {
                    "total": len(pm.registered_plugins) if pm else 0,
                    "active": list(pm.healthy_plugins.keys()) if pm else [],
                },
                "accelerators": [],
            }

            # Check for DataScienceCluster
            try:
                dsc_list = k8s.list_resources(CRDs.DATA_SCIENCE_CLUSTER)
                if dsc_list:
                    result["rhoai_available"] = True
                    dsc = dsc_list[0]
                    status = getattr(dsc, "status", None)
                    if status:
                        # Extract component status
                        installed = getattr(status, "installedComponents", {}) or {}
                        for component, state in installed.items():
                            result["components"][component] = state
            except Exception:
                pass

            # Check for accelerator profiles
            try:
                accelerators = k8s.list_resources(CRDs.ACCELERATOR_PROFILE)
                result["accelerators"] = [
                    {
                        "name": acc.metadata.name,
                        "display_name": (acc.metadata.annotations or {}).get(
                            "openshift.io/display-name", acc.metadata.name
                        ),
                        "enabled": getattr(acc.spec, "enabled", True)
                        if hasattr(acc, "spec")
                        else True,
                    }
                    for acc in accelerators
                ]
            except Exception:
                pass

            return result

        @mcp.resource("rhoai://cluster/plugins")
        def cluster_plugins() -> dict:
            """Get information about loaded plugins.

            Returns details about all plugins with their health status.
            """
            pm = self._plugin_manager
            if not pm:
                return {"plugins": {}}

            plugin_info = {}
            for name, plugin in pm.registered_plugins.items():
                is_healthy = name in pm.healthy_plugins

                # Get metadata if available
                meta = None
                if hasattr(plugin, "rhoai_get_plugin_metadata"):
                    meta = plugin.rhoai_get_plugin_metadata()

                plugin_info[name] = {
                    "version": meta.version if meta else "unknown",
                    "description": meta.description if meta else "No description",
                    "maintainer": meta.maintainer if meta else "unknown",
                    "requires_crds": meta.requires_crds if meta else [],
                    "healthy": is_healthy,
                }

            return {
                "total": len(pm.registered_plugins),
                "active": len(pm.healthy_plugins),
                "plugins": plugin_info,
            }

        @mcp.resource("rhoai://cluster/accelerators")
        def cluster_accelerators() -> list[dict]:
            """Get available accelerator profiles (GPUs).

            Returns the list of AcceleratorProfile resources that define
            available GPU types and configurations.
            """
            k8s = self.k8s

            try:
                accelerators = k8s.list_resources(CRDs.ACCELERATOR_PROFILE)
                return [
                    {
                        "name": acc.metadata.name,
                        "display_name": (acc.metadata.annotations or {}).get(
                            "openshift.io/display-name", acc.metadata.name
                        ),
                        "description": (acc.metadata.annotations or {}).get(
                            "openshift.io/description", ""
                        ),
                        "enabled": getattr(acc.spec, "enabled", True)
                        if hasattr(acc, "spec")
                        else True,
                        "identifier": getattr(acc.spec, "identifier", "nvidia.com/gpu")
                        if hasattr(acc, "spec")
                        else "nvidia.com/gpu",
                        "tolerations": getattr(acc.spec, "tolerations", [])
                        if hasattr(acc, "spec")
                        else [],
                    }
                    for acc in accelerators
                ]
            except Exception as e:
                return [{"error": str(e)}]

        logger.info("Registered core MCP resources")


# Global server instance
_server: RHOAIServer | None = None


def get_server() -> RHOAIServer:
    """Get the global server instance."""
    global _server
    if _server is None:
        _server = RHOAIServer()
    return _server


def create_server(config: RHOAIConfig | None = None) -> FastMCP:
    """Create and return the MCP server instance.

    This is the main entry point for creating the server.
    """
    global _server
    _server = RHOAIServer(config)
    return _server.create_mcp()
