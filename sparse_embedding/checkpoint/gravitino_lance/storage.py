from __future__ import annotations

from collections.abc import Mapping

from .._lance_workspace import S3LanceWorkspace
from ..catalog import TableHandle
from ..lance_backend import _LanceCheckpointStorage
from ..schema import SNAPSHOT_MANIFEST_DATASET
from .config import SdkConfig
from .rest import LanceRestCatalog, LanceRestError

_DATASETS = (("snapshot_manifest", SNAPSHOT_MANIFEST_DATASET),)


class GravitinoLanceStorage(_LanceCheckpointStorage):
    def __init__(
        self,
        config: SdkConfig,
        *,
        namespace: tuple[str, str],
        component_id: str,
        location: str,
        max_storage_bytes: int | None = None,
        lock_timeout_seconds: float = 60.0,
        lock_ttl_seconds: float = 300.0,
        provision: bool = True,
        catalog: LanceRestCatalog | None = None,
    ) -> None:
        if len(namespace) != 2 or any(
            not isinstance(part, str) or not part.strip() for part in namespace
        ):
            raise ValueError("namespace must contain a catalog and schema")
        if not isinstance(component_id, str) or not component_id.strip():
            raise ValueError("component_id must be non-empty")
        if not isinstance(location, str) or not location.startswith("s3://"):
            raise ValueError("checkpoint location must be an s3:// URI")
        self._catalog = catalog or LanceRestCatalog(config)
        self._owns_catalog = catalog is None
        created_tables: list[tuple[str, str, str]] = []
        try:
            handles = []
            for dataset, physical_name in _DATASETS:
                table_id = (*namespace, f"{component_id}_{dataset}")
                handle, created = self._ensure_table(
                    table_id,
                    f"{location.rstrip('/')}/{physical_name}",
                    provision=provision,
                )
                handles.append(handle)
                if created:
                    created_tables.append(table_id)
            storage_options = self._validate_handles(tuple(handles), location)
            workspace = S3LanceWorkspace(
                location,
                storage_options,
                lock_timeout_seconds=lock_timeout_seconds,
                lock_ttl_seconds=lock_ttl_seconds,
            )
        except BaseException as error:
            cleanup_errors = []
            for table_id in reversed(created_tables):
                try:
                    self._catalog.drop_table(table_id)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if self._owns_catalog:
                try:
                    self._catalog.close()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "Gravitino checkpoint initialization and rollback failed",
                    [error, *cleanup_errors],
                )
            raise
        super().__init__(workspace, max_storage_bytes=max_storage_bytes)

    def close(self) -> None:
        workspace_error: BaseException | None = None
        try:
            super().close()
        except BaseException as error:
            workspace_error = error
        if self._owns_catalog:
            try:
                self._catalog.close()
            except BaseException as error:
                if workspace_error is not None:
                    raise BaseExceptionGroup(
                        "Gravitino checkpoint storage close failed",
                        [workspace_error, error],
                    )
                raise
        if workspace_error is not None:
            raise workspace_error

    def _ensure_table(
        self,
        table_id: tuple[str, str, str],
        location: str,
        *,
        provision: bool,
    ) -> tuple[TableHandle, bool]:
        created = False
        try:
            handle = self._catalog.describe_table(table_id)
        except LanceRestError as error:
            if error.status_code != 404:
                raise
            if not provision:
                raise RuntimeError(
                    f"Gravitino table {table_id!r} is not provisioned by control"
                ) from error
            try:
                handle = self._catalog.register_table(table_id, location=location)
                created = True
            except LanceRestError as register_error:
                if register_error.status_code not in {400, 409}:
                    raise
                handle = self._catalog.describe_table(table_id)
        if handle.table_id != table_id:
            raise ValueError("Gravitino returned a different table identity")
        if handle.location.rstrip("/") != location.rstrip("/"):
            raise ValueError(f"Gravitino table {table_id!r} points to another location")
        return handle, created

    @staticmethod
    def _validate_handles(
        handles: tuple[TableHandle, ...],
        location: str,
    ) -> Mapping[str, str]:
        if len(handles) != len(_DATASETS):
            raise ValueError("Gravitino checkpoint requires exactly one manifest table")
        (first_handle, *rest_handles) = handles
        first = dict(first_handle.storage_options)
        if not first:
            raise ValueError("Gravitino checkpoint storage options are empty")
        for handle in rest_handles:
            if dict(handle.storage_options) != first:
                raise ValueError(
                    f"Gravitino tables under {location!r} have different storage options"
                )
        return first
