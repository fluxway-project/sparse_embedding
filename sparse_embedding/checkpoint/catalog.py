"""Catalog control-plane values used by Lance storage adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TableHandle:
    table_id: tuple[str, ...]
    location: str
    storage_options: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.table_id or any(not part for part in self.table_id):
            raise ValueError("table_id must contain non-empty levels")
        if not self.location:
            raise ValueError("table location must be non-empty")
        options = dict(self.storage_options)
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in options.items()
        ):
            raise ValueError("storage options must contain non-empty strings")
        object.__setattr__(self, "storage_options", MappingProxyType(options))


class LanceCatalog(Protocol):
    def describe_table(self, table_id: tuple[str, ...]) -> TableHandle: ...

    def close(self) -> None: ...
