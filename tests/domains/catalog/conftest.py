"""Pytest fixtures for catalog package tests."""

import pytest


@pytest.fixture
def mock_catalog_data() -> dict:
    """Sample catalog YAML data."""
    return {
        "source": "https://example.com/catalog.yaml",
        "models": [
            {
                "name": "llama-2-7b",
                "provider": "Meta",
                "description": "Llama 2 7B base model",
                "artifacts": [
                    {"uri": "oci://quay.io/models/llama-2-7b:latest"},
                ],
            },
            {
                "name": "granite-8b-code",
                "provider": "IBM",
                "description": "Granite 8B code model",
                "artifacts": [
                    {"uri": "oci://quay.io/models/granite-8b-code:v1"},
                    {"uri": "oci://quay.io/models/granite-8b-code:latest"},
                ],
            },
            {
                "name": "mistral-7b",
                "provider": "Mistral AI",
                "description": "Mistral 7B instruction-tuned model",
                "artifacts": [],
            },
        ],
    }


@pytest.fixture
def mock_catalog_model() -> dict:
    """Sample CatalogModel object data."""
    return {
        "name": "llama-2-7b",
        "provider": "Meta",
        "description": "Llama 2 7B base model for fine-tuning",
        "artifacts": [
            {"uri": "oci://quay.io/models/llama-2-7b:latest"},
            {"uri": "oci://quay.io/models/llama-2-7b:v1.0"},
        ],
    }


@pytest.fixture
def mock_catalog_model_minimal() -> dict:
    """Sample minimal CatalogModel object data."""
    return {
        "name": "test-model",
        "provider": "Test Provider",
        "description": "A test model",
    }
