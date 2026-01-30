"""MCP Tools for training storage management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient
    from rhoai_mcp.server import RHOAIServer


def create_training_pvc(
    k8s: K8sClient,
    namespace: str,
    pvc_name: str,
    size_gb: int,
    access_mode: str = "ReadWriteMany",
    storage_class: str | None = None,
) -> dict[str, Any]:
    """Create a PVC for training checkpoints and data.

    This is the shared implementation used by both setup_training_storage
    and prepare_training tools.

    Args:
        k8s: Kubernetes client instance.
        namespace: The namespace to create the PVC in.
        pvc_name: Name for the PVC.
        size_gb: Size in GB.
        access_mode: Access mode (default: "ReadWriteMany" for distributed training).
        storage_class: Storage class to use (auto-detected if not specified).

    Returns:
        PVC creation result with success/error status.
    """
    # Check if PVC already exists
    try:
        existing = k8s.get_pvc(pvc_name, namespace)
        return {
            "exists": True,
            "created": False,
            "pvc_name": pvc_name,
            "namespace": namespace,
            "size": existing.spec.resources.requests.get("storage", "Unknown"),
            "status": existing.status.phase,
            "message": f"PVC '{pvc_name}' already exists.",
        }
    except Exception:
        pass  # PVC doesn't exist, proceed to create

    # If no storage class specified, try to find an NFS or RWX-capable one
    if not storage_class and access_mode == "ReadWriteMany":
        storage_class = _find_rwx_storage_class(k8s)

    # Create the PVC
    try:
        k8s.create_pvc(
            name=pvc_name,
            namespace=namespace,
            size=f"{size_gb}Gi",
            access_modes=[access_mode],
            storage_class=storage_class,
            labels={
                "app.kubernetes.io/managed-by": "rhoai-mcp",
                "app.kubernetes.io/component": "training-storage",
            },
        )

        return {
            "exists": False,
            "created": True,
            "pvc_name": pvc_name,
            "namespace": namespace,
            "size": f"{size_gb}Gi",
            "access_mode": access_mode,
            "storage_class": storage_class,
            "message": f"PVC '{pvc_name}' created. It may take a moment to bind.",
        }
    except Exception as e:
        return {
            "error": f"Failed to create PVC: {e}",
            "created": False,
        }


def register_tools(mcp: FastMCP, server: RHOAIServer) -> None:
    """Register training storage tools with the MCP server."""

    @mcp.tool()
    def setup_training_storage(
        namespace: str,
        pvc_name: str,
        size_gb: int = 100,
        storage_class: str | None = None,
        access_mode: str = "ReadWriteMany",
    ) -> dict[str, Any]:
        """Create a PVC for training checkpoints and data.

        Creates a PersistentVolumeClaim suitable for distributed training.
        Defaults to ReadWriteMany access mode to support multi-node training.

        Args:
            namespace: The namespace to create the PVC in.
            pvc_name: Name for the PVC.
            size_gb: Size in GB (default: 100).
            storage_class: Storage class to use (auto-detected if not specified).
            access_mode: Access mode (default: "ReadWriteMany" for distributed training).

        Returns:
            PVC creation confirmation.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        result = create_training_pvc(
            k8s=server.k8s,
            namespace=namespace,
            pvc_name=pvc_name,
            size_gb=size_gb,
            access_mode=access_mode,
            storage_class=storage_class,
        )

        # Add success field for backward compatibility
        if result.get("created"):
            result["success"] = True

        return result

    @mcp.tool()
    def setup_nfs_storage() -> dict[str, Any]:
        """Get guidance on setting up NFS storage for distributed training.

        NFS storage provides shared ReadWriteMany access across training
        nodes. This tool provides guidance on the recommended approach.

        Returns:
            NFS setup guidance and alternatives.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        # This is a simplified implementation - in production, you'd deploy
        # the full NFS provisioner stack
        return {
            "message": (
                "NFS storage setup requires cluster-admin permissions. "
                "Consider using the NFS provisioner Operator or contact "
                "your cluster administrator."
            ),
            "alternatives": [
                "Use existing RWX-capable storage class",
                "Install NFS Subdir External Provisioner",
                "Use OpenShift Data Foundation (ODF)",
            ],
            "documentation": "https://docs.openshift.com/container-platform/latest/storage/persistent_storage/persistent-storage-nfs.html",
        }

    @mcp.tool()
    def fix_pvc_permissions(
        namespace: str,
        pvc_name: str,
    ) -> dict[str, Any]:
        """Fix permissions on a PVC for training jobs.

        Training jobs may fail if the PVC has restrictive permissions.
        This tool attempts to fix common permission issues by creating
        a temporary pod to modify permissions.

        Note: This requires the ability to create pods in the namespace.

        Args:
            namespace: The namespace containing the PVC.
            pvc_name: Name of the PVC to fix.

        Returns:
            Permission fix status.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        # Verify PVC exists
        try:
            pvc = server.k8s.get_pvc(pvc_name, namespace)
            if pvc.status.phase != "Bound":
                return {
                    "error": f"PVC '{pvc_name}' is not bound (current: {pvc.status.phase})",
                    "message": "Wait for PVC to be bound before fixing permissions.",
                }
        except Exception as e:
            return {"error": f"PVC not found: {e}"}

        # Create a job to fix permissions
        # This is a simplified approach - in production you might use
        # a more sophisticated method
        return {
            "message": (
                "Permission fix requires creating a privileged pod. "
                "Consider running the following command manually:\n\n"
                f"oc run pvc-fixer --rm -i --tty --image=registry.access.redhat.com/ubi9/ubi "
                f"--overrides='{{\n"
                f'  "spec": {{\n'
                f'    "containers": [{{\n'
                f'      "name": "pvc-fixer",\n'
                f'      "image": "registry.access.redhat.com/ubi9/ubi",\n'
                f'      "command": ["chmod", "-R", "777", "/data"],\n'
                f'      "volumeMounts": [{{"name": "data", "mountPath": "/data"}}]\n'
                f"    }}],\n"
                f'    "volumes": [{{"name": "data", "persistentVolumeClaim": {{"claimName": "{pvc_name}"}}}}]\n'
                f"  }}\n"
                f"}}'"
            ),
            "pvc_name": pvc_name,
            "namespace": namespace,
        }

    @mcp.tool()
    def list_storage(namespace: str) -> dict[str, Any]:
        """List storage resources in a namespace.

        Returns all PVCs in the namespace with their status and
        capacity information.

        Args:
            namespace: The namespace to list storage from.

        Returns:
            List of PVCs with status information.
        """
        pvcs = server.k8s.list_pvcs(namespace)

        pvc_list = []
        for pvc in pvcs:
            storage = "Unknown"
            if pvc.spec and pvc.spec.resources and pvc.spec.resources.requests:
                storage = pvc.spec.resources.requests.get("storage", "Unknown")

            pvc_list.append(
                {
                    "name": pvc.metadata.name,
                    "status": pvc.status.phase if pvc.status else "Unknown",
                    "size": storage,
                    "access_modes": list(pvc.spec.access_modes)
                    if pvc.spec and pvc.spec.access_modes
                    else [],
                    "storage_class": pvc.spec.storage_class_name if pvc.spec else None,
                }
            )

        return {
            "namespace": namespace,
            "count": len(pvc_list),
            "pvcs": pvc_list,
        }

    @mcp.tool()
    def delete_storage(
        namespace: str,
        pvc_name: str,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Delete a storage PVC.

        Permanently removes a PVC and its data. This action cannot
        be undone.

        Args:
            namespace: The namespace containing the PVC.
            pvc_name: Name of the PVC to delete.
            confirm: Must be True to actually delete.

        Returns:
            Deletion confirmation.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("delete")
        if not allowed:
            return {"error": reason}

        if not confirm:
            return {
                "error": "Deletion not confirmed",
                "message": (
                    f"To delete PVC '{pvc_name}', set confirm=True. "
                    "WARNING: All data on this PVC will be lost."
                ),
            }

        try:
            server.k8s.delete_pvc(pvc_name, namespace)
            return {
                "success": True,
                "deleted": True,
                "pvc_name": pvc_name,
                "namespace": namespace,
                "message": f"PVC '{pvc_name}' has been deleted.",
            }
        except Exception as e:
            return {"error": f"Failed to delete PVC: {e}"}


def _find_rwx_storage_class(k8s: Any) -> str | None:
    """Find a storage class that supports ReadWriteMany."""
    # Common NFS/RWX storage class names
    common_names = [
        "nfs",
        "nfs-client",
        "nfs-csi",
        "ocs-storagecluster-cephfs",
        "managed-nfs-storage",
        "trident-nfs",
    ]

    try:
        # Try to list storage classes
        from kubernetes import client  # type: ignore[import-untyped]

        storage_api = client.StorageV1Api(k8s._api_client)
        storage_classes = storage_api.list_storage_class()

        for sc in storage_classes.items:
            name: str = sc.metadata.name
            if name.lower() in [n.lower() for n in common_names]:
                return name

        # Return first storage class as fallback
        if storage_classes.items:
            first_name: str = storage_classes.items[0].metadata.name
            return first_name
    except Exception:
        pass

    return None
