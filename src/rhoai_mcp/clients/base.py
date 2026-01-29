"""Base Kubernetes client with RHOAI CRD definitions."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from kubernetes import client, config  # type: ignore[import-untyped]
from kubernetes.client import ApiException  # type: ignore[import-untyped]
from kubernetes.dynamic import DynamicClient  # type: ignore[import-untyped]
from kubernetes.dynamic.resource import Resource, ResourceInstance  # type: ignore[import-untyped]

from rhoai_mcp.config import AuthMode, RHOAIConfig, get_config
from rhoai_mcp.utils.errors import AuthenticationError, NotFoundError, RHOAIError

logger = logging.getLogger(__name__)


class CRDDefinition:
    """Definition of a Custom Resource."""

    def __init__(
        self,
        group: str,
        version: str,
        plural: str,
        kind: str,
    ) -> None:
        self.group = group
        self.version = version
        self.plural = plural
        self.kind = kind

    @property
    def api_version(self) -> str:
        """Get the full API version string."""
        if self.group:
            return f"{self.group}/{self.version}"
        return self.version


class CRDs:
    """RHOAI Custom Resource Definitions.

    Core CRDs that are shared across components. Component-specific CRDs
    are defined in their respective packages.
    """

    # OpenShift Projects (for listing user-accessible projects)
    PROJECT = CRDDefinition(
        group="project.openshift.io",
        version="v1",
        plural="projects",
        kind="Project",
    )

    # ODH/RHOAI cluster configuration
    DATA_SCIENCE_CLUSTER = CRDDefinition(
        group="datasciencecluster.opendatahub.io",
        version="v1",
        plural="datascienceclusters",
        kind="DataScienceCluster",
    )

    DSCI = CRDDefinition(
        group="dscinitialization.opendatahub.io",
        version="v1",
        plural="dscinitializations",
        kind="DSCInitialization",
    )

    # Dashboard resources
    ACCELERATOR_PROFILE = CRDDefinition(
        group="dashboard.opendatahub.io",
        version="v1",
        plural="acceleratorprofiles",
        kind="AcceleratorProfile",
    )


class K8sClient:
    """Kubernetes client with RHOAI CRD support.

    Supports multiple authentication modes:
    - auto: Try in-cluster first, fall back to kubeconfig
    - kubeconfig: Use kubeconfig file with optional context
    - token: Use explicit API server URL and token
    """

    def __init__(self, config_obj: RHOAIConfig | None = None) -> None:
        self._config = config_obj or get_config()
        self._api_client: client.ApiClient | None = None
        self._dynamic_client: DynamicClient | None = None
        self._core_v1: client.CoreV1Api | None = None
        self._crd_cache: dict[str, Resource] = {}

    def connect(self) -> None:
        """Establish connection to Kubernetes API."""
        try:
            self._api_client = self._create_api_client()
            self._dynamic_client = DynamicClient(self._api_client)
            self._core_v1 = client.CoreV1Api(self._api_client)
            logger.info("Connected to Kubernetes API")
        except Exception as e:
            raise AuthenticationError(f"Failed to connect to Kubernetes API: {e}")

    def disconnect(self) -> None:
        """Close connection to Kubernetes API."""
        if self._api_client:
            self._api_client.close()
            self._api_client = None
            self._dynamic_client = None
            self._core_v1 = None
            self._crd_cache.clear()
            logger.info("Disconnected from Kubernetes API")

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._api_client is not None

    def _create_api_client(self) -> client.ApiClient:
        """Create API client based on authentication mode."""
        auth_mode = self._config.auth_mode

        if auth_mode == AuthMode.TOKEN:
            return self._create_token_client()
        elif auth_mode == AuthMode.KUBECONFIG:
            return self._create_kubeconfig_client()
        else:  # AUTO
            return self._create_auto_client()

    def _create_token_client(self) -> client.ApiClient:
        """Create client using explicit token authentication."""
        if not self._config.api_server or not self._config.api_token:
            raise AuthenticationError(
                "api_server and api_token are required for token authentication"
            )

        configuration = client.Configuration()
        configuration.host = self._config.api_server
        configuration.api_key = {"authorization": f"Bearer {self._config.api_token}"}
        configuration.verify_ssl = True

        return client.ApiClient(configuration)

    def _create_kubeconfig_client(self) -> client.ApiClient:
        """Create client using kubeconfig file."""
        kubeconfig_path = self._config.effective_kubeconfig_path
        if not kubeconfig_path.exists():
            raise AuthenticationError(f"Kubeconfig not found: {kubeconfig_path}")

        # Use new_client_from_config to get a properly configured ApiClient
        # This avoids issues with global configuration state
        return config.new_client_from_config(
            config_file=str(kubeconfig_path),
            context=self._config.kubeconfig_context,
        )

    def _create_auto_client(self) -> client.ApiClient:
        """Auto-detect authentication mode."""
        # Try in-cluster first
        if Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists():
            logger.info("Using in-cluster authentication")
            # Load in-cluster config and return properly configured client
            config.load_incluster_config()
            # After load_incluster_config, we need to create client with the loaded config
            # The incluster config sets the default, so we create Configuration from it
            configuration = client.Configuration.get_default_copy()
            return client.ApiClient(configuration)

        # Fall back to kubeconfig
        kubeconfig_path = self._config.effective_kubeconfig_path
        if kubeconfig_path.exists():
            logger.info(f"Using kubeconfig: {kubeconfig_path}")
            # Use new_client_from_config to get a properly configured ApiClient
            return config.new_client_from_config(
                config_file=str(kubeconfig_path),
                context=self._config.kubeconfig_context,
            )

        raise AuthenticationError(
            "No valid authentication method found. "
            "Not running in-cluster and no kubeconfig available."
        )

    @property
    def dynamic(self) -> DynamicClient:
        """Get the dynamic client."""
        if not self._dynamic_client:
            raise RHOAIError("Client not connected. Call connect() first.")
        return self._dynamic_client

    @property
    def core_v1(self) -> client.CoreV1Api:
        """Get the CoreV1 API client."""
        if not self._core_v1:
            raise RHOAIError("Client not connected. Call connect() first.")
        return self._core_v1

    def get_resource(self, crd: CRDDefinition) -> Resource:
        """Get a dynamic resource for a CRD.

        Uses caching to avoid repeated API discovery calls.
        """
        cache_key = f"{crd.api_version}/{crd.plural}"
        if cache_key not in self._crd_cache:
            self._crd_cache[cache_key] = self.dynamic.resources.get(
                api_version=crd.api_version,
                kind=crd.kind,
            )
        return self._crd_cache[cache_key]

    def get(
        self,
        crd: CRDDefinition,
        name: str,
        namespace: str | None = None,
    ) -> ResourceInstance:
        """Get a resource by name."""
        resource = self.get_resource(crd)
        try:
            if namespace:
                return resource.get(name=name, namespace=namespace)
            return resource.get(name=name)
        except ApiException as e:
            if e.status == 404:
                raise NotFoundError(crd.kind, name, namespace)
            raise RHOAIError(f"Failed to get {crd.kind} '{name}': {e.reason}")

    def list_resources(
        self,
        crd: CRDDefinition,
        namespace: str | None = None,
        label_selector: str | None = None,
        field_selector: str | None = None,
    ) -> list[Any]:
        """List resources."""
        resource = self.get_resource(crd)
        try:
            kwargs: dict[str, Any] = {}
            if namespace:
                kwargs["namespace"] = namespace
            if label_selector:
                kwargs["label_selector"] = label_selector
            if field_selector:
                kwargs["field_selector"] = field_selector

            result = resource.get(**kwargs)
            return list(result.items) if hasattr(result, "items") else [result]
        except ApiException as e:
            raise RHOAIError(f"Failed to list {crd.kind}: {e.reason}")

    def create(
        self,
        crd: CRDDefinition,
        body: dict[str, Any],
        namespace: str | None = None,
    ) -> ResourceInstance:
        """Create a resource."""
        resource = self.get_resource(crd)
        try:
            if namespace:
                return resource.create(body=body, namespace=namespace)
            return resource.create(body=body)
        except ApiException as e:
            if e.status == 409:
                name = body.get("metadata", {}).get("name", "unknown")
                from rhoai_mcp.utils.errors import ResourceExistsError

                raise ResourceExistsError(crd.kind, name, namespace)
            raise RHOAIError(f"Failed to create {crd.kind}: {e.reason}")

    def delete(
        self,
        crd: CRDDefinition,
        name: str,
        namespace: str | None = None,
    ) -> None:
        """Delete a resource."""
        resource = self.get_resource(crd)
        try:
            if namespace:
                resource.delete(name=name, namespace=namespace)
            else:
                resource.delete(name=name)
        except ApiException as e:
            if e.status == 404:
                raise NotFoundError(crd.kind, name, namespace)
            raise RHOAIError(f"Failed to delete {crd.kind} '{name}': {e.reason}")

    def patch(
        self,
        crd: CRDDefinition,
        name: str,
        body: dict[str, Any],
        namespace: str | None = None,
    ) -> ResourceInstance:
        """Patch a resource."""
        resource = self.get_resource(crd)
        try:
            if namespace:
                return resource.patch(name=name, body=body, namespace=namespace)
            return resource.patch(name=name, body=body)
        except ApiException as e:
            if e.status == 404:
                raise NotFoundError(crd.kind, name, namespace)
            raise RHOAIError(f"Failed to patch {crd.kind} '{name}': {e.reason}")

    # OpenShift Project operations (preferred for listing user-accessible projects)
    def list_projects(
        self,
        label_selector: str | None = None,
    ) -> list[Any]:
        """List OpenShift projects the user has access to.

        This uses the OpenShift Projects API which only returns projects
        the authenticated user has permission to access, unlike listing
        all namespaces which requires cluster-wide permissions.
        """
        return self.list_resources(CRDs.PROJECT, label_selector=label_selector)

    def patch_project(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> ResourceInstance:
        """Patch OpenShift Project labels and/or annotations.

        Uses the OpenShift Projects API (project.openshift.io/v1) which
        regular users have permission to modify, unlike the Kubernetes
        Namespace API which typically requires cluster-admin permissions.
        """
        body: dict[str, Any] = {"metadata": {}}
        if labels is not None:
            body["metadata"]["labels"] = labels
        if annotations is not None:
            body["metadata"]["annotations"] = annotations

        return self.patch(CRDs.PROJECT, name, body)

    # Namespace operations (used for Data Science Projects)
    def get_namespace(self, name: str) -> Any:
        """Get a namespace."""
        try:
            return self.core_v1.read_namespace(name=name)
        except ApiException as e:
            if e.status == 404:
                raise NotFoundError("Namespace", name)
            raise RHOAIError(f"Failed to get namespace '{name}': {e.reason}")

    def list_namespaces(
        self,
        label_selector: str | None = None,
    ) -> list[Any]:
        """List namespaces."""
        try:
            result = self.core_v1.list_namespace(label_selector=label_selector)
            return list(result.items)
        except ApiException as e:
            raise RHOAIError(f"Failed to list namespaces: {e.reason}")

    def create_namespace(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> Any:
        """Create a namespace."""
        body = client.V1Namespace(
            metadata=client.V1ObjectMeta(
                name=name,
                labels=labels,
                annotations=annotations,
            )
        )
        try:
            return self.core_v1.create_namespace(body=body)
        except ApiException as e:
            if e.status == 409:
                from rhoai_mcp.utils.errors import ResourceExistsError

                raise ResourceExistsError("Namespace", name)
            raise RHOAIError(f"Failed to create namespace '{name}': {e.reason}")

    def delete_namespace(self, name: str) -> None:
        """Delete a namespace."""
        try:
            self.core_v1.delete_namespace(name=name)
        except ApiException as e:
            if e.status == 404:
                raise NotFoundError("Namespace", name)
            raise RHOAIError(f"Failed to delete namespace '{name}': {e.reason}")

    def patch_namespace(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> Any:
        """Patch namespace labels and/or annotations."""
        body: dict[str, Any] = {"metadata": {}}
        if labels is not None:
            body["metadata"]["labels"] = labels
        if annotations is not None:
            body["metadata"]["annotations"] = annotations

        try:
            return self.core_v1.patch_namespace(
                name=name,
                body=body,
            )
        except ApiException as e:
            if e.status == 404:
                raise NotFoundError("Namespace", name)
            raise RHOAIError(f"Failed to patch namespace '{name}': {e.reason}")

    # Secret operations (used for Data Connections)
    def get_secret(self, name: str, namespace: str) -> Any:
        """Get a secret."""
        try:
            return self.core_v1.read_namespaced_secret(name=name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                raise NotFoundError("Secret", name, namespace)
            raise RHOAIError(f"Failed to get secret '{name}': {e.reason}")

    def list_secrets(
        self,
        namespace: str,
        label_selector: str | None = None,
    ) -> list[Any]:
        """List secrets in a namespace."""
        try:
            result = self.core_v1.list_namespaced_secret(
                namespace=namespace,
                label_selector=label_selector,
            )
            return list(result.items)
        except ApiException as e:
            raise RHOAIError(f"Failed to list secrets: {e.reason}")

    def create_secret(
        self,
        name: str,
        namespace: str,
        data: dict[str, str],
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        string_data: bool = True,
    ) -> Any:
        """Create a secret."""
        body = client.V1Secret(
            metadata=client.V1ObjectMeta(
                name=name,
                namespace=namespace,
                labels=labels,
                annotations=annotations,
            ),
            string_data=data if string_data else None,
            data=None if string_data else data,
        )
        try:
            return self.core_v1.create_namespaced_secret(namespace=namespace, body=body)
        except ApiException as e:
            if e.status == 409:
                from rhoai_mcp.utils.errors import ResourceExistsError

                raise ResourceExistsError("Secret", name, namespace)
            raise RHOAIError(f"Failed to create secret '{name}': {e.reason}")

    def delete_secret(self, name: str, namespace: str) -> None:
        """Delete a secret."""
        try:
            self.core_v1.delete_namespaced_secret(name=name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                raise NotFoundError("Secret", name, namespace)
            raise RHOAIError(f"Failed to delete secret '{name}': {e.reason}")

    # PVC operations (used for Storage)
    def get_pvc(self, name: str, namespace: str) -> Any:
        """Get a PersistentVolumeClaim."""
        try:
            return self.core_v1.read_namespaced_persistent_volume_claim(
                name=name, namespace=namespace
            )
        except ApiException as e:
            if e.status == 404:
                raise NotFoundError("PersistentVolumeClaim", name, namespace)
            raise RHOAIError(f"Failed to get PVC '{name}': {e.reason}")

    def list_pvcs(
        self,
        namespace: str,
        label_selector: str | None = None,
    ) -> list[Any]:
        """List PVCs in a namespace."""
        try:
            result = self.core_v1.list_namespaced_persistent_volume_claim(
                namespace=namespace,
                label_selector=label_selector,
            )
            return list(result.items)
        except ApiException as e:
            raise RHOAIError(f"Failed to list PVCs: {e.reason}")

    def create_pvc(
        self,
        name: str,
        namespace: str,
        size: str,
        access_modes: list[str] | None = None,
        storage_class: str | None = None,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> Any:
        """Create a PersistentVolumeClaim."""
        body = client.V1PersistentVolumeClaim(
            metadata=client.V1ObjectMeta(
                name=name,
                namespace=namespace,
                labels=labels,
                annotations=annotations,
            ),
            spec=client.V1PersistentVolumeClaimSpec(
                access_modes=access_modes or ["ReadWriteOnce"],
                resources=client.V1VolumeResourceRequirements(requests={"storage": size}),
                storage_class_name=storage_class,
            ),
        )
        try:
            return self.core_v1.create_namespaced_persistent_volume_claim(
                namespace=namespace, body=body
            )
        except ApiException as e:
            if e.status == 409:
                from rhoai_mcp.utils.errors import ResourceExistsError

                raise ResourceExistsError("PersistentVolumeClaim", name, namespace)
            raise RHOAIError(f"Failed to create PVC '{name}': {e.reason}")

    def delete_pvc(self, name: str, namespace: str) -> None:
        """Delete a PersistentVolumeClaim."""
        try:
            self.core_v1.delete_namespaced_persistent_volume_claim(name=name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                raise NotFoundError("PersistentVolumeClaim", name, namespace)
            raise RHOAIError(f"Failed to delete PVC '{name}': {e.reason}")

    # Pod operations
    def exec_command(
        self,
        pod_name: str,
        namespace: str,
        command: list[str],
        container: str | None = None,
    ) -> str:
        """Execute a command in a pod container.

        Args:
            pod_name: Name of the pod
            namespace: Namespace containing the pod
            command: Command to execute as a list of arguments
            container: Optional container name (uses first container if not specified)

        Returns:
            Command output as a string

        Raises:
            NotFoundError: If pod is not found
            RHOAIError: If command execution fails
        """
        # Import at method level to avoid global dependency
        from kubernetes import stream

        try:
            kwargs: dict[str, Any] = {
                "name": pod_name,
                "namespace": namespace,
                "command": command,
                "stderr": True,
                "stdin": False,
                "stdout": True,
                "tty": False,
            }
            if container:
                kwargs["container"] = container

            if not self._api_client:
                raise RHOAIError("Client not connected. Call connect() first.")

            response = stream.stream(
                self.core_v1.connect_get_namespaced_pod_exec,
                **kwargs,
                _preload_content=False,
            )

            # Read all output
            output = ""
            while response.is_open():
                response.update(timeout=1)
                if response.peek_stdout():
                    output += response.read_stdout()
                if response.peek_stderr():
                    output += response.read_stderr()

            # Check exit code
            error = response.read_channel(3)  # Error channel
            if error:
                error_data = eval(error)  # Parse error JSON
                if error_data.get("status") == "Success":
                    return output
                raise RHOAIError(
                    f"Command failed in pod '{pod_name}': {error_data.get('message', error)}"
                )

            return output

        except ApiException as e:
            if e.status == 404:
                raise NotFoundError("Pod", pod_name, namespace)
            raise RHOAIError(f"Failed to exec command in pod '{pod_name}': {e.reason}")
        except Exception as e:
            raise RHOAIError(f"Failed to exec command in pod '{pod_name}': {e}")

    # Template operations
    def get_template(self, name: str, namespace: str) -> dict[str, Any]:
        """Get an OpenShift Template.

        Args:
            name: Name of the template
            namespace: Namespace containing the template

        Returns:
            Template resource as a dictionary (full K8s object)

        Raises:
            NotFoundError: If template is not found
            RHOAIError: If template retrieval fails
        """
        template_crd = CRDDefinition(
            group="template.openshift.io",
            version="v1",
            plural="templates",
            kind="Template",
        )

        try:
            resource = self.get_resource(template_crd)
            result = resource.get(name=name, namespace=namespace)
            # Convert ResourceInstance to dict
            template_dict: dict[str, Any] = result.to_dict()
            return template_dict
        except ApiException as e:
            if e.status == 404:
                raise NotFoundError("Template", name, namespace)
            raise RHOAIError(f"Failed to get template '{name}': {e.reason}")


@contextmanager
def get_k8s_client(
    config_obj: RHOAIConfig | None = None,
) -> Generator[K8sClient, None, None]:
    """Context manager for K8s client with automatic cleanup."""
    k8s_client = K8sClient(config_obj)
    k8s_client.connect()
    try:
        yield k8s_client
    finally:
        k8s_client.disconnect()
