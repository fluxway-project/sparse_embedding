from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

from ..catalog import TableHandle
from .config import SdkConfig

DELIMITER = "$"


@dataclass(frozen=True)
class LanceRestError(RuntimeError):
    status_code: int
    message: str
    detail: str | None = None

    def __str__(self) -> str:
        suffix = "" if self.detail is None else f": {self.detail}"
        return f"Lance REST HTTP {self.status_code}: {self.message}{suffix}"


class LanceRestCatalog:
    """Metadata-through-REST and data-through-Lance adapter."""

    def __init__(self, config: SdkConfig, session: requests.Session | None = None):
        self._config = config
        self._api_base_uri = config.require("LANCE_REST_API_BASE_URI")
        self._timeout = config.get_float("REST_TIMEOUT")
        self._session = session or requests.Session()
        self._headers = {"Accept": "application/json", **config.authorization_headers()}

    def close(self) -> None:
        self._session.close()

    def list_namespaces(
        self, parent: Sequence[str] | str | None = None
    ) -> dict[str, Any]:
        parts = self._parts(parent, allow_empty=True)
        path = (
            "namespace/list" if not parts else f"namespace/{self._path_id(parts)}/list"
        )
        return self._json_request("GET", path, params={"delimiter": DELIMITER})

    def create_namespace(
        self,
        identifier: Sequence[str] | str,
        *,
        mode: str = "create",
        properties: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        parts = self._parts(identifier)
        if len(parts) not in (1, 2):
            raise ValueError("namespace identifier must have one or two levels")
        return self._json_request(
            "POST",
            f"namespace/{self._path_id(parts)}/create",
            params={"delimiter": DELIMITER},
            json={"id": parts, "mode": mode, "properties": dict(properties or {})},
        )

    def describe_table(self, identifier: Sequence[str] | str) -> TableHandle:
        parts = self._table_parts(identifier)
        payload = self._json_request(
            "POST",
            f"table/{self._path_id(parts)}/describe",
            params={"delimiter": DELIMITER},
            json={
                "id": parts,
                "load_detailed_metadata": False,
                "check_declared": False,
            },
        )
        storage_options = payload.get("storage_options")
        if not isinstance(storage_options, Mapping):
            raise ValueError("Lance REST response is missing storage_options")
        return TableHandle(
            table_id=tuple(parts),
            location=self._required_string(payload, "location"),
            storage_options={
                str(key): str(value) for key, value in storage_options.items()
            },
        )

    def register_table(
        self,
        identifier: Sequence[str] | str,
        *,
        location: str,
        properties: Mapping[str, str] | None = None,
    ) -> TableHandle:
        if not isinstance(location, str) or not location.strip():
            raise ValueError("location must not be blank")
        parts = self._table_parts(identifier)
        self._json_request(
            "POST",
            f"table/{self._path_id(parts)}/register",
            params={"delimiter": DELIMITER},
            headers={"x-lance-table-properties": "{}"},
            json={
                "id": parts,
                "location": location,
                "properties": dict(properties or {}),
            },
        )
        return self.describe_table(parts)

    def table_exists(self, identifier: Sequence[str] | str) -> bool:
        parts = self._table_parts(identifier)
        self._json_request(
            "POST",
            f"table/{self._path_id(parts)}/exists",
            params={"delimiter": DELIMITER},
            json={"id": parts},
        )
        return True

    def list_tables(self, namespace: Sequence[str] | str) -> tuple[str, ...]:
        parts = self._parts(namespace)
        if len(parts) != 2:
            raise ValueError("table namespace must have catalog and schema levels")
        payload = self._json_request(
            "GET",
            f"namespace/{self._path_id(parts)}/table/list",
            params={"delimiter": DELIMITER},
        )
        tables = payload.get("tables")
        if not isinstance(tables, list) or not all(
            isinstance(item, str) and item for item in tables
        ):
            raise TypeError("Lance REST table list is malformed")
        return tuple(tables)

    def drop_table(self, identifier: Sequence[str] | str) -> None:
        parts = self._table_parts(identifier)
        self._json_request(
            "POST",
            f"table/{self._path_id(parts)}/drop",
            params={"delimiter": DELIMITER},
            json={"id": parts},
        )

    def _json_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        request_kwargs = dict(kwargs)
        request_kwargs["timeout"] = self._timeout
        headers = dict(self._headers)
        headers.update(request_kwargs.pop("headers", {}))
        response = self._session.request(
            method, self._url(path), headers=headers, **request_kwargs
        )
        if not response.ok:
            detail = response.text[:500]
            raise LanceRestError(response.status_code, "request failed", detail)
        if not response.content:
            return {}
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("Lance REST response must be a JSON object")
        return payload

    def _url(self, resource_path: str) -> str:
        if resource_path.startswith("/"):
            raise ValueError("REST resource path must be relative")
        return f"{self._api_base_uri}/{resource_path}"

    @staticmethod
    def _required_string(payload: Mapping[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Lance REST response is missing required field: {key}")
        return value

    @staticmethod
    def _parts(
        identifier: Sequence[str] | str | None, *, allow_empty: bool = False
    ) -> list[str]:
        parts = (
            []
            if identifier is None
            else identifier.split(".")
            if isinstance(identifier, str)
            else list(identifier)
        )
        if not allow_empty and not parts:
            raise ValueError("identifier must not be empty")
        if any(not isinstance(part, str) or not part.strip() for part in parts):
            raise ValueError("identifier levels must be non-empty strings")
        if any(DELIMITER in part for part in parts):
            raise ValueError("identifier levels must not contain '$'")
        return parts

    @classmethod
    def _table_parts(cls, identifier: Sequence[str] | str) -> list[str]:
        parts = cls._parts(identifier)
        if len(parts) != 3:
            raise ValueError(
                "table identifier must have catalog, schema, and table levels"
            )
        return parts

    @staticmethod
    def _path_id(parts: Sequence[str]) -> str:
        return quote(DELIMITER.join(parts), safe="")
