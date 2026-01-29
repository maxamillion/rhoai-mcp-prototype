"""InferenceService (Model Serving) client operations."""

from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains.inference.crds import InferenceCRDs
from rhoai_mcp.domains.inference.models import (
    InferenceService,
    InferenceServiceCreate,
)

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient


class InferenceClient:
    """Client for InferenceService (Model Serving) operations."""

    def __init__(self, k8s: "K8sClient") -> None:
        self._k8s = k8s

    def list_inference_services(self, namespace: str) -> list[dict[str, Any]]:
        """List all InferenceServices in a namespace."""
        try:
            isvc_list = self._k8s.list_resources(
                InferenceCRDs.INFERENCE_SERVICE, namespace=namespace
            )
        except Exception:
            return []

        results = []
        for isvc in isvc_list:
            model = InferenceService.from_inference_service_cr(isvc)
            results.append(
                {
                    "name": model.metadata.name,
                    "display_name": model.display_name,
                    "runtime": model.runtime,
                    "model_format": model.model_format,
                    "status": model.status.value,
                    "url": model.url,
                }
            )
        return results

    def get_inference_service(self, name: str, namespace: str) -> InferenceService:
        """Get an InferenceService by name."""
        isvc = self._k8s.get(InferenceCRDs.INFERENCE_SERVICE, name=name, namespace=namespace)
        return InferenceService.from_inference_service_cr(isvc)

    def deploy_model(self, request: InferenceServiceCreate) -> InferenceService:
        """Deploy a model as an InferenceService."""
        body = self._build_inference_service_cr(request)
        isvc = self._k8s.create(
            InferenceCRDs.INFERENCE_SERVICE, body=body, namespace=request.namespace
        )
        return InferenceService.from_inference_service_cr(isvc)

    def delete_inference_service(self, name: str, namespace: str) -> None:
        """Delete an InferenceService."""
        self._k8s.delete(InferenceCRDs.INFERENCE_SERVICE, name=name, namespace=namespace)

    def list_serving_runtimes(self, namespace: str) -> list[dict[str, Any]]:
        """List available ServingRuntimes in a namespace."""
        try:
            runtimes = self._k8s.list_resources(InferenceCRDs.SERVING_RUNTIME, namespace=namespace)
        except Exception:
            return []

        results = []
        for rt in runtimes:
            metadata = rt.metadata
            annotations = metadata.annotations or {}
            spec = rt.spec or {}

            # Get supported formats
            containers = spec.get("containers", [])
            supported_formats = []
            for container in containers:
                for model_format in container.get("supportedModelFormats", []):
                    if isinstance(model_format, dict):
                        supported_formats.append(model_format.get("name", ""))
                    else:
                        supported_formats.append(str(model_format))

            results.append(
                {
                    "name": metadata.name,
                    "display_name": annotations.get("openshift.io/display-name", metadata.name),
                    "supported_formats": supported_formats,
                    "multi_model": spec.get("multiModel", False),
                }
            )
        return results

    def get_model_endpoint(self, name: str, namespace: str) -> dict[str, Any]:
        """Get the inference endpoint URL for a model."""
        isvc = self.get_inference_service(name, namespace)
        return {
            "name": isvc.metadata.name,
            "status": isvc.status.value,
            "url": isvc.url,
            "internal_url": isvc.internal_url,
        }

    def _build_inference_service_cr(self, request: InferenceServiceCreate) -> dict[str, Any]:
        """Build the InferenceService CR body from request."""
        # Build annotations
        annotations: dict[str, str] = {}
        if request.display_name:
            annotations["openshift.io/display-name"] = request.display_name

        # Build resource requirements
        resources: dict[str, dict[str, str]] = {
            "requests": {
                "cpu": request.cpu_request,
                "memory": request.memory_request,
            },
            "limits": {
                "cpu": request.cpu_limit,
                "memory": request.memory_limit,
            },
        }
        if request.gpu_count > 0:
            resources["requests"]["nvidia.com/gpu"] = str(request.gpu_count)
            resources["limits"]["nvidia.com/gpu"] = str(request.gpu_count)

        return {
            "apiVersion": InferenceCRDs.INFERENCE_SERVICE.api_version,
            "kind": InferenceCRDs.INFERENCE_SERVICE.kind,
            "metadata": {
                "name": request.name,
                "namespace": request.namespace,
                "annotations": annotations if annotations else None,
            },
            "spec": {
                "predictor": {
                    "minReplicas": request.min_replicas,
                    "maxReplicas": request.max_replicas,
                    "model": {
                        "modelFormat": {"name": request.model_format},
                        "runtime": request.runtime,
                        "storageUri": request.storage_uri,
                        "resources": resources,
                    },
                },
            },
        }
