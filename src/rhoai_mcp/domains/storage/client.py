"""Storage (PVC) client operations."""

from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains.storage.models import Storage, StorageCreate
from rhoai_mcp.utils.labels import RHOAILabels

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient


class StorageClient:
    """Client for Storage (PVC) operations."""

    def __init__(self, k8s: "K8sClient") -> None:
        self._k8s = k8s

    def list_storage(self, namespace: str) -> list[dict[str, Any]]:
        """List all PVCs in a namespace."""
        pvcs = self._k8s.list_pvcs(namespace=namespace)

        results = []
        for pvc in pvcs:
            storage = Storage.from_pvc(pvc)
            results.append(
                {
                    "name": storage.metadata.name,
                    "display_name": storage.display_name,
                    "size": storage.size,
                    "access_modes": storage.access_modes,
                    "storage_class": storage.storage_class,
                    "status": storage.status.value,
                    "_source": storage.metadata.to_source_dict(),
                }
            )
        return results

    def get_storage(self, name: str, namespace: str) -> Storage:
        """Get a PVC by name."""
        pvc = self._k8s.get_pvc(name, namespace)
        return Storage.from_pvc(pvc)

    def create_storage(self, request: StorageCreate) -> Storage:
        """Create a new PVC."""
        # Build labels
        labels = RHOAILabels.dashboard_project_labels()

        # Build annotations
        annotations: dict[str, str] = {}
        if request.display_name:
            annotations["openshift.io/display-name"] = request.display_name

        pvc = self._k8s.create_pvc(
            name=request.name,
            namespace=request.namespace,
            size=request.size,
            access_modes=[request.access_mode],
            storage_class=request.storage_class,
            labels=labels,
            annotations=annotations if annotations else None,
        )

        return Storage.from_pvc(pvc)

    def delete_storage(self, name: str, namespace: str) -> None:
        """Delete a PVC."""
        self._k8s.delete_pvc(name, namespace)
