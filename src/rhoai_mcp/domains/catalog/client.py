"""Model Catalog client operations."""

from typing import TYPE_CHECKING

import yaml

from rhoai_mcp.domains.catalog.models import CatalogModel, ModelCatalog
from rhoai_mcp.utils.errors import NotFoundError, RHOAIError

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient


class CatalogClient:
    """Client for Model Catalog operations."""

    def __init__(self, k8s: "K8sClient") -> None:
        self._k8s = k8s

    def read_catalog(self) -> ModelCatalog:
        """Read the model catalog from the catalog pod.

        Returns:
            ModelCatalog instance parsed from the catalog YAML file

        Raises:
            NotFoundError: If catalog pod is not found
            RHOAIError: If catalog file cannot be read or parsed
        """
        namespace = "rhoai-model-registries"
        label_selector = "app=model-catalog"

        # Find the catalog pod
        try:
            from kubernetes.client import V1PodList

            pod_list: V1PodList = self._k8s.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector,
            )

            if not pod_list.items:
                raise NotFoundError("Pod", "model-catalog", namespace)

            pod_name = pod_list.items[0].metadata.name

        except Exception as e:
            if isinstance(e, NotFoundError):
                raise
            raise RHOAIError(f"Failed to find catalog pod: {e}")

        # Execute command to read catalog file
        try:
            output = self._k8s.exec_command(
                pod_name=pod_name,
                namespace=namespace,
                command=["cat", "/shared-data/models-catalog.yaml"],
            )
        except Exception as e:
            raise RHOAIError(f"Failed to read catalog file from pod '{pod_name}': {e}")

        # Parse YAML
        try:
            catalog_data = yaml.safe_load(output)
            if not catalog_data:
                raise RHOAIError("Catalog file is empty or invalid")

            return ModelCatalog(**catalog_data)
        except yaml.YAMLError as e:
            raise RHOAIError(f"Failed to parse catalog YAML: {e}")
        except Exception as e:
            raise RHOAIError(f"Failed to create ModelCatalog from data: {e}")

    def get_model(self, name: str) -> CatalogModel:
        """Get a model from the catalog by name.

        Args:
            name: Name of the model to retrieve

        Returns:
            CatalogModel instance

        Raises:
            ValueError: If model is not found in catalog
            RHOAIError: If catalog cannot be read
        """
        catalog = self.read_catalog()

        for model in catalog.models:
            if model.name == name:
                return model

        raise ValueError(f"Model '{name}' not found in catalog")
