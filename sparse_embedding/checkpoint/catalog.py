"""Catalog control-plane values used by Lance storage adapters."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol


class TableHandle:
    __slots__ = ("_table_id", "_location", "_storage_options")

    def __init__(
        self,
        table_id: tuple[str, ...],
        location: str,
        storage_options: Mapping[str, str],
    ) -> None:
        if not table_id or any(not part for part in table_id):
            raise ValueError("table_id must contain non-empty levels")
        if not location:
            raise ValueError("table location must be non-empty")
        options = dict(storage_options)
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in options.items()
        ):
            raise ValueError("storage options must contain non-empty strings")
        self._table_id = tuple(table_id)
        self._location = location
        self._storage_options = MappingProxyType(options)

    @property
    def table_id(self) -> tuple[str, ...]:
        return self._table_id

    @property
    def location(self) -> str:
        return self._location

    @property
    def storage_options(self) -> Mapping[str, str]:
        return self._storage_options

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TableHandle):
            return (
                self._table_id == other._table_id
                and self._location == other._location
                and self._storage_options == other._storage_options
            )
        return False

    def __hash__(self) -> int:
        return hash(
            (
                self._table_id,
                self._location,
                tuple(sorted(self._storage_options.items())),
            )
        )

    def __repr__(self) -> str:
        return (
            f"TableHandle(table_id={self._table_id!r}, "
            f"location={self._location!r}, "
            f"storage_options={dict(self._storage_options)!r})"
        )


class LanceCatalog(Protocol):
    def describe_table(self, table_id: tuple[str, ...]) -> TableHandle: ...

    def close(self) -> None: ...
