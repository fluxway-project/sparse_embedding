from __future__ import annotations

from math import isfinite
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlsplit


class SdkConfig:
    """Immutable configuration with separate REST and catalog-admin scopes."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, str]) -> None:
        items = tuple(sorted(dict(values).items()))
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str)
            for key, value in items
        ):
            raise ValueError("SDK configuration must contain string values")
        self._items = items

    @property
    def values(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._items))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SdkConfig):
            return self._items == other._items
        return False

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return f"SdkConfig({dict(self._items)!r})"

    @classmethod
    def for_rest(
        cls,
        *,
        lance_rest_api_base_uri: str,
        bearer_token: str | None = None,
        timeout: float = 60.0,
    ) -> "SdkConfig":
        cls._require_api_base_uri("lance_rest_api_base_uri", lance_rest_api_base_uri)
        cls._require_positive_timeout(timeout)
        values = {
            "LANCE_REST_API_BASE_URI": lance_rest_api_base_uri.rstrip("/"),
            "REST_TIMEOUT": str(timeout),
        }
        if bearer_token is not None:
            cls._require_text("bearer_token", bearer_token)
            values["GRAVITINO_BEARER_TOKEN"] = bearer_token
        return cls(values)

    @classmethod
    def for_catalog_admin(
        cls,
        *,
        metalake: str,
        catalog_name: str,
        catalog_location: str,
        s3_endpoint: str,
        s3_region: str,
        s3_access_key: str,
        s3_secret_key: str,
        s3_allow_http: bool = True,
        schema_refresh_mode: str = "VERSION_CHECK",
        gravitino_api_base_uri: str,
        bearer_token: str | None = None,
        timeout: float = 60.0,
    ) -> "SdkConfig":
        required = {
            "metalake": metalake,
            "catalog_name": catalog_name,
            "catalog_location": catalog_location,
            "s3_endpoint": s3_endpoint,
            "s3_region": s3_region,
            "s3_access_key": s3_access_key,
            "s3_secret_key": s3_secret_key,
            "schema_refresh_mode": schema_refresh_mode,
        }
        for name, value in required.items():
            cls._require_text(name, value)
        cls._require_positive_timeout(timeout)
        values = {
            "GRAVITINO_API_BASE_URI": cls._normalized_api_base_uri(
                "gravitino_api_base_uri", gravitino_api_base_uri
            ),
            "LANCE_CATALOG_LOCATION": catalog_location,
            "GRAVITINO_METALAKE": metalake,
            "GRAVITINO_CATALOG_NAME": catalog_name,
            "LANCE_SCHEMA_REFRESH_MODE": schema_refresh_mode,
            "REST_TIMEOUT": str(timeout),
            "S3_ENDPOINT": s3_endpoint,
            "S3_REGION": s3_region,
            "S3_ACCESS_KEY": s3_access_key,
            "S3_SECRET_KEY": s3_secret_key,
            "S3_ALLOW_HTTP": str(s3_allow_http).lower(),
        }
        if bearer_token is not None:
            cls._require_text("bearer_token", bearer_token)
            values["GRAVITINO_BEARER_TOKEN"] = bearer_token
        return cls(values)

    def require(self, key: str) -> str:
        value = dict(self._items).get(key)
        if value is None or not value.strip():
            raise ValueError(f"SDK configuration is missing required value: {key}")
        return value

    def get(self, key: str, default: str | None = None) -> str | None:
        return dict(self._items).get(key, default)

    def get_float(self, key: str) -> float:
        value = self.require(key)
        try:
            parsed = float(value)
        except ValueError as error:
            raise ValueError(
                f"SDK configuration value {key} must be numeric"
            ) from error
        if not isfinite(parsed) or parsed <= 0:
            raise ValueError(
                f"SDK configuration value {key} must be a positive finite number"
            )
        return parsed

    def authorization_headers(self) -> dict[str, str]:
        token = self.get("GRAVITINO_BEARER_TOKEN")
        return {} if token is None else {"Authorization": f"Bearer {token}"}

    def catalog_properties(self) -> dict[str, str]:
        return {
            "location": self.require("LANCE_CATALOG_LOCATION"),
            "lance.schema-refresh-mode": self.require("LANCE_SCHEMA_REFRESH_MODE"),
            "lance.storage.endpoint": self.require("S3_ENDPOINT"),
            "lance.storage.region": self.require("S3_REGION"),
            "lance.storage.access_key_id": self.require("S3_ACCESS_KEY"),
            "lance.storage.secret_access_key": self.require("S3_SECRET_KEY"),
            "lance.storage.allow_http": self.require("S3_ALLOW_HTTP"),
        }

    @staticmethod
    def _require_text(name: str, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must not be blank")

    @classmethod
    def _require_api_base_uri(cls, name: str, value: str) -> None:
        cls._normalized_api_base_uri(name, value)

    @classmethod
    def _normalized_api_base_uri(cls, name: str, value: str) -> str:
        cls._require_text(name, value)
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{name} must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise ValueError(f"{name} must not contain a query or fragment")
        return value.rstrip("/")

    @staticmethod
    def _require_positive_timeout(value: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
