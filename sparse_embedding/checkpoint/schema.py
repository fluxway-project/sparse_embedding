"""Canonical Lance Checkpoint Schemas, imported from C++ Single Source of Truth."""

import pyarrow as pa

from sparse_embedding import _core

_checkpoint = _core.checkpoint
ENTRIES_DATASET: str = _checkpoint.ENTRIES_DATASET
PAGE_IMAGES_DATASET: str = _checkpoint.PAGE_IMAGES_DATASET
SNAPSHOT_MANIFEST_DATASET: str = _checkpoint.SNAPSHOT_MANIFEST_DATASET


def entry_schema(key_dtype: str = "uint64") -> pa.Schema:
    return pa.Schema._import_from_c_capsule(
        _checkpoint.lance_entries_schema_capsule(key_dtype)
    )


ENTRY_SCHEMA: pa.Schema = entry_schema("uint64")
PAGE_IMAGE_SCHEMA: pa.Schema = pa.Schema._import_from_c_capsule(
    _checkpoint.lance_page_images_schema_capsule()
)
MANIFEST_SCHEMA: pa.Schema = pa.Schema._import_from_c_capsule(
    _checkpoint.lance_manifest_schema_capsule()
)

__all__ = [
    "ENTRIES_DATASET",
    "PAGE_IMAGES_DATASET",
    "SNAPSHOT_MANIFEST_DATASET",
    "ENTRY_SCHEMA",
    "MANIFEST_SCHEMA",
    "PAGE_IMAGE_SCHEMA",
    "entry_schema",
]
