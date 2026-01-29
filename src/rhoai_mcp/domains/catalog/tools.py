"""MCP Tools for Model Catalog operations."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.catalog.client import CatalogClient
from rhoai_mcp.domains.inference.client import InferenceClient
from rhoai_mcp.domains.inference.crds import InferenceCRDs
from rhoai_mcp.domains.inference.models import InferenceServiceCreate
from rhoai_mcp.utils.errors import NotFoundError, RHOAIError

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register model catalog tools with the MCP server."""

    @mcp.tool()
    def list_catalog_models(limit: int | None = None) -> dict[str, Any]:
        """List models available in the RHOAI model catalog.

        The catalog contains pre-configured models that can be deployed
        with a single command, including metadata about providers and
        OCI artifact URIs.

        Args:
            limit: Maximum number of models to return (None for all).

        Returns:
            Dictionary with catalog summary and model list.
        """
        try:
            client = CatalogClient(server.k8s)
            catalog = client.read_catalog()

            models_list = []
            for model in catalog.models:
                # Extract OCI URI from first artifact if available
                oci_uri = None
                if model.artifacts:
                    oci_uri = model.artifacts[0].uri

                models_list.append(
                    {
                        "name": model.name,
                        "provider": model.provider,
                        "description": model.description,
                        "oci_uri": oci_uri,
                    }
                )

            # Apply limit if specified
            total = len(models_list)
            if limit is not None:
                models_list = models_list[:limit]

            return {
                "total": total,
                "returned": len(models_list),
                "models": models_list,
            }

        except NotFoundError as e:
            return {
                "error": "Catalog not found",
                "message": str(e),
            }
        except RHOAIError as e:
            return {
                "error": "Failed to read catalog",
                "message": str(e),
            }
        except Exception as e:
            return {
                "error": "Unexpected error",
                "message": str(e),
            }

    @mcp.tool()
    def deploy_from_catalog(
        catalog_model_name: str,
        deployment_name: str,
        namespace: str,
        gpu_count: int = 1,
    ) -> dict[str, Any]:
        """Deploy a model from the catalog as an InferenceService.

        This is a convenience tool that automates the deployment of catalog models
        by fetching model metadata, creating a ServingRuntime, and deploying the
        InferenceService.

        Args:
            catalog_model_name: Name of the model in the catalog.
            deployment_name: Name for the deployment (must be DNS-compatible).
            namespace: Project (namespace) to deploy into.
            gpu_count: Number of GPUs per replica (default: 1).

        Returns:
            Deployment status and information.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        try:
            # Step 1: Get model from catalog
            catalog_client = CatalogClient(server.k8s)
            model = catalog_client.get_model(catalog_model_name)

            # Step 2: Extract storage URI from first artifact
            if not model.artifacts:
                return {
                    "error": "Model has no artifacts",
                    "message": f"Catalog model '{catalog_model_name}' has no artifacts defined",
                }

            storage_uri = model.artifacts[0].uri

            # Step 3: Get vLLM runtime template
            try:
                template = server.k8s.get_template(
                    "vllm-cuda-runtime-template", "redhat-ods-applications"
                )
            except NotFoundError:
                return {
                    "error": "Runtime template not found",
                    "message": "vLLM CUDA runtime template not found. Ensure RHOAI is properly installed.",
                }

            # Step 4: Extract and customize runtime spec
            if "objects" not in template or not template["objects"]:
                return {
                    "error": "Invalid template",
                    "message": "Runtime template does not contain any objects",
                }

            runtime_spec = template["objects"][0]
            runtime_spec["metadata"]["name"] = deployment_name
            runtime_spec["metadata"]["namespace"] = namespace

            # Step 5: Create ServingRuntime
            try:
                server.k8s.create(
                    InferenceCRDs.SERVING_RUNTIME, body=runtime_spec, namespace=namespace
                )
            except Exception as e:
                # Check if runtime already exists
                error_msg = str(e).lower()
                if "already exists" in error_msg:
                    return {
                        "error": "Runtime already exists",
                        "message": f"ServingRuntime '{deployment_name}' already exists in namespace '{namespace}'",
                    }
                raise

            # Step 6: Create InferenceService
            inference_client = InferenceClient(server.k8s)
            request = InferenceServiceCreate(
                name=deployment_name,
                namespace=namespace,
                runtime=deployment_name,
                model_format="vLLM",
                storage_uri=storage_uri,
                min_replicas=1,
                max_replicas=1,
                cpu_request="2",
                cpu_limit="2",
                memory_request="4Gi",
                memory_limit="4Gi",
                gpu_count=gpu_count,
            )

            isvc = inference_client.deploy_model(request)

            return {
                "name": isvc.metadata.name,
                "namespace": isvc.metadata.namespace,
                "status": isvc.status.value,
                "catalog_model": catalog_model_name,
                "storage_uri": storage_uri,
                "message": f"Model '{catalog_model_name}' deployed as '{deployment_name}'. It may take a few minutes to become ready.",
            }

        except ValueError:
            return {
                "error": "Model not found",
                "message": f"Catalog model '{catalog_model_name}' not found in catalog",
            }
        except NotFoundError as e:
            return {
                "error": "Resource not found",
                "message": str(e),
            }
        except RHOAIError as e:
            return {
                "error": "Deployment failed",
                "message": str(e),
            }
        except Exception as e:
            return {
                "error": "Unexpected error",
                "message": str(e),
            }
