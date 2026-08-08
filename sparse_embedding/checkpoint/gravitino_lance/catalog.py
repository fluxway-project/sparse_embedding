from __future__ import annotations

from typing import Any

import requests

from .config import SdkConfig


class GravitinoCatalogAdmin:
    """Idempotent Gravitino catalog configuration with strict drift checks."""

    def __init__(self, config: SdkConfig, session: requests.Session | None = None):
        self._config = config
        self._session = session or requests.Session()
        base = config.require("GRAVITINO_API_BASE_URI")
        metalake = config.require("GRAVITINO_METALAKE")
        self._catalog_name = config.require("GRAVITINO_CATALOG_NAME")
        self._collection_url = f"{base}/metalakes/{metalake}/catalogs"
        self._url = f"{self._collection_url}/{self._catalog_name}"
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **config.authorization_headers(),
        }

    def get_catalog(self) -> dict[str, Any]:
        response = self._session.get(
            self._url,
            headers=self._headers,
            timeout=self._config.get_float("REST_TIMEOUT"),
        )
        if response.status_code == 404:
            raise LookupError("Gravitino Lance catalog does not exist")
        response.raise_for_status()
        payload = response.json()
        catalog = payload.get("catalog")
        if not isinstance(catalog, dict):
            raise TypeError("Gravitino catalog response is malformed")
        return catalog

    def create_catalog(self) -> dict[str, Any]:
        request: dict[str, Any] = {
            "name": self._catalog_name,
            "type": "RELATIONAL",
            "provider": "lakehouse-generic",
            "properties": self._config.catalog_properties(),
        }
        response = self._session.post(
            self._collection_url,
            headers=self._headers,
            json=request,
            timeout=self._config.get_float("REST_TIMEOUT"),
        )
        response.raise_for_status()
        payload = response.json()
        catalog = payload.get("catalog")
        if not isinstance(catalog, dict):
            raise TypeError("Gravitino catalog response is malformed")
        return catalog

    def ensure_catalog(self) -> str:
        try:
            catalog = self.get_catalog()
        except LookupError:
            self.create_catalog()
            return "created"
        properties = catalog.get("properties")
        if not isinstance(properties, dict):
            raise TypeError("Gravitino catalog properties are malformed")
        expected = self._config.catalog_properties()
        mismatches = {
            key: (properties.get(key), value)
            for key, value in expected.items()
            if properties.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Gravitino catalog configuration drift: {mismatches}")
        return "unchanged"
