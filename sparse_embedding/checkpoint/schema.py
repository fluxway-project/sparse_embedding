"""Canonical Lance Checkpoint Schemas, imported from C++ Single Source of Truth."""

import pyarrow as pa

from sparse_embedding import _core

_checkpoint = _core.checkpoint

ENTRY_SCHEMA: pa.Schema = pa.Schema._import_from_c_capsule(
    _checkpoint.lance_entries_schema_capsule()
)
PAGE_IMAGE_SCHEMA: pa.Schema = pa.Schema._import_from_c_capsule(
    _checkpoint.lance_page_images_schema_capsule()
)
MANIFEST_SCHEMA: pa.Schema = pa.Schema._import_from_c_capsule(
    _checkpoint.lance_manifest_schema_capsule()
)

__all__ = [
    "ENTRY_SCHEMA",
    "PAGE_IMAGE_SCHEMA",
    "MANIFEST_SCHEMA",
]
