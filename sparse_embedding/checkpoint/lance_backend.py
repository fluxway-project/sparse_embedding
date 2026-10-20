"""Local Lance implementation of the sparse checkpoint protocol."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import posixpath
import struct
import threading
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from urllib.parse import urlparse

import numpy as np  # pyrefly: ignore[missing-import]
import pyarrow as pa  # pyrefly: ignore[missing-import]

from sparse_embedding import _core as _native

from ._lance_workspace import (
    LanceWorkspace,
    LocalLanceWorkspace,
    S3LanceWorkspace,
    _validate_relative_path,
)
from .base import (
    PUBLISHED_WEIGHT_DESCRIPTOR_FILENAME,
    CheckpointableStore,
    CheckpointCommitAuthority,
    CheckpointConflictError,
    CheckpointFragment,
    CheckpointHandle,
    CheckpointLease,
    CheckpointLeaseRequest,
    CheckpointParticipant,
    CheckpointSaveContext,
    CheckpointSaveIdentity,
    CheckpointSaveMode,
    CleanupReservation,
    CommittedCheckpoint,
    IngestionEngine,
    InvalidCheckpointFragment,
    PartitionTarget,
    RestorePlan,
    SparseCheckpointLocator,
    StorageBudgetExceeded,
    TableSpecification,
    execute_cleanup_actions,
    run_cleanup_preserving_primary,
)
from .fragment_codec import decode_checkpoint_fragment, encode_checkpoint_fragment
from .schema import (
    ENTRIES_DATASET,
    ENTRY_SCHEMA,
    MANIFEST_SCHEMA,
    PAGE_IMAGE_SCHEMA,
    PAGE_IMAGES_DATASET,
    SNAPSHOT_MANIFEST_DATASET,
    entry_schema,
)

_checkpoint_native = _native.checkpoint
_ACTIVE = "active"
_DELETED = "deleted"
_PREPARED = "PREPARED"
_COMMITTED = "COMMITTED"
_FRAGMENT_STORAGE_MODE = "fragment_manifest"
_PAGE_DATASET_ROOT = "_page_images"
_PAGE_ATTEMPT_TAG_PREFIX = "sepageattempt"
_PAGE_PROTECTION_TAG_PREFIX = "sepage"
_PAGE_WRITE_INTENT_TAG_PREFIX = "sepagewrite"
_CHECKPOINT_HANDLE_MAGIC = b"sparse_embedding.checkpoint_handle\0"
_LOCAL_VALIDATION_DESCRIPTOR = "sparse_embedding.local_validation_descriptor"
_UINT32 = struct.Struct("!I")
_UINT64 = struct.Struct("!Q")


def _strict_integer(
    value: Any,
    field: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{field} is outside its valid range")
    return value


def _strict_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _serialized_integer(
    value: Any,
    field: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, str) or not value.isdecimal():
        raise ValueError(f"{field} must be a serialized integer")
    return _strict_integer(int(value), field, minimum=minimum, maximum=maximum)


def _strict_sequence(value: Any, field: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    return tuple(value)


def _freeze_descriptor_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_descriptor_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_descriptor_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_descriptor_value(item) for item in value)
    return value


def _strict_compaction_descriptor(metadata: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema_digest",
        "dimensions",
        "embedding_dtype",
        "optimizer_dtype",
        "score_policy",
        "key_dtype",
        "entry_count",
        "page_count",
        "content_digest",
        "participant_count",
        "participant_indices",
    )
    missing = tuple(field for field in required if field not in metadata)
    if missing:
        raise InvalidCheckpointFragment(
            "fragment validation descriptor lacks required fields: "
            + ", ".join(missing)
        )
    participant_count = _strict_integer(
        metadata["participant_count"], "participant_count", minimum=1
    )
    page_count = _strict_integer(metadata["page_count"], "page_count")
    entry_count = _strict_integer(metadata["entry_count"], "entry_count")
    raw_indices = _strict_sequence(
        metadata["participant_indices"], "participant_indices"
    )
    if not raw_indices:
        raise InvalidCheckpointFragment("participant_indices must not be empty")
    indices = tuple(
        _strict_integer(index, "participant index", minimum=0) for index in raw_indices
    )
    if len(set(indices)) != len(indices):
        raise InvalidCheckpointFragment("participant_indices must be unique")
    if tuple(sorted(indices)) != indices:
        raise InvalidCheckpointFragment("participant_indices must be sorted")
    if any(index >= participant_count for index in indices):
        raise InvalidCheckpointFragment(
            "participant index is outside participant_count"
        )
    dimensions = _strict_sequence(metadata["dimensions"], "dimensions")
    if not dimensions:
        raise InvalidCheckpointFragment("dimensions must not be empty")
    dimensions = tuple(
        _strict_integer(value, "dimension", minimum=1) for value in dimensions
    )
    if tuple(sorted(set(dimensions))) != dimensions:
        raise InvalidCheckpointFragment("dimensions must be sorted and unique")
    for field in required:
        if field in {
            "entry_count",
            "page_count",
            "participant_count",
            "participant_indices",
            "dimensions",
        }:
            continue
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise InvalidCheckpointFragment(f"{field} has an invalid type")
    return {
        **dict(metadata),
        "dimensions": dimensions,
        "participant_count": participant_count,
        "entry_count": entry_count,
        "page_count": page_count,
        "participant_indices": indices,
    }


def _strict_equal_canonical_descriptors(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    left_descriptor = _strict_compaction_descriptor(left)
    right_descriptor = _strict_compaction_descriptor(right)
    fields = (
        "schema_digest",
        "dimensions",
        "embedding_dtype",
        "optimizer_dtype",
        "score_policy",
        "key_dtype",
        "entry_count",
        "page_count",
        "content_digest",
        "participant_count",
        "participant_indices",
    )
    return all(left_descriptor[field] == right_descriptor[field] for field in fields)


class _ManifestFragmentDescriptorNotFound(LookupError):
    """The source root is not referenced by a committed manifest."""


class PublishedWeightHandleAsset:
    @staticmethod
    def _root(path: str | Path) -> Path:
        asset_root = Path(path).expanduser()
        if asset_root.is_symlink() or not asset_root.is_dir():
            raise FileNotFoundError(
                f"sparse published weight directory does not exist: {asset_root}"
            )
        return asset_root

    @staticmethod
    def _descriptor(asset_root: Path) -> Path:
        descriptor = asset_root / PUBLISHED_WEIGHT_DESCRIPTOR_FILENAME
        if descriptor.is_symlink():
            raise ValueError("sparse published weight descriptor must not be a symlink")
        return descriptor

    @classmethod
    def publish(cls, path: str | Path, handle: CheckpointHandle) -> None:
        asset_root = cls._root(path)
        if any(
            (asset_root / name).exists()
            for name in (
                SNAPSHOT_MANIFEST_DATASET,
                ENTRIES_DATASET,
                PAGE_IMAGES_DATASET,
            )
        ):
            raise ValueError(
                "sparse weight path must contain a published handle asset, "
                "not a checkpoint root"
            )
        descriptor = cls._descriptor(asset_root)
        if any(
            item.name != PUBLISHED_WEIGHT_DESCRIPTOR_FILENAME
            for item in asset_root.iterdir()
        ):
            raise ValueError("sparse published weight directory has an unknown layout")
        if descriptor.exists():
            if not descriptor.is_file() or descriptor.read_bytes() != handle:
                raise CheckpointConflictError(
                    "sparse published weight handle already exists with another identity"
                )
            return
        temporary = (
            asset_root
            / f".{PUBLISHED_WEIGHT_DESCRIPTOR_FILENAME}.tmp-{uuid.uuid4().hex}"
        )
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            descriptor_fd = os.open(temporary, flags, 0o600)
            try:
                with os.fdopen(descriptor_fd, "wb") as stream:
                    descriptor_fd = -1
                    stream.write(handle)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                if descriptor_fd >= 0:
                    os.close(descriptor_fd)
            os.replace(temporary, descriptor)
            directory_descriptor = os.open(asset_root, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            if descriptor.read_bytes() != handle:
                raise RuntimeError("sparse published weight verification failed")
        except BaseException as primary:
            run_cleanup_preserving_primary(
                primary,
                [
                    (
                        "temporary published weight descriptor",
                        lambda: temporary.unlink() if temporary.exists() else None,
                    )
                ],
                "published weight descriptor",
            )
            raise

    @classmethod
    def read(cls, path: str | Path) -> CheckpointHandle:
        asset_root = cls._root(path)
        descriptor = asset_root / PUBLISHED_WEIGHT_DESCRIPTOR_FILENAME
        if not descriptor.is_file() or descriptor.is_symlink():
            raise FileNotFoundError(
                f"sparse published weight descriptor is missing: {descriptor}"
            )
        if any(
            item.name.startswith(PUBLISHED_WEIGHT_DESCRIPTOR_FILENAME)
            and item.name != PUBLISHED_WEIGHT_DESCRIPTOR_FILENAME
            for item in asset_root.iterdir()
        ):
            raise LookupError("sparse published weight directory is ambiguous")
        if any(
            item.name != PUBLISHED_WEIGHT_DESCRIPTOR_FILENAME
            for item in asset_root.iterdir()
        ):
            raise ValueError("sparse published weight directory has an unknown layout")
        return descriptor.read_bytes()


@dataclass(frozen=True, slots=True)
class _CommitOutcome:
    locator: SparseCheckpointLocator
    reference_id: str
    manifest_before: int


@dataclass(frozen=True, slots=True)
class _LeaseRecord:
    lease: CheckpointLease
    locator: SparseCheckpointLocator


@dataclass(frozen=True, slots=True)
class ManifestParticipantRecord:
    participant_index: int
    fragment_id: str
    capture_id: int
    journal_identity: str
    watermark: int
    page_capacity: int
    page_dataset: str
    page_version: int
    page_layout_generation: int


@dataclass(frozen=True, slots=True)
class _PreparedRestore:
    chain: tuple[dict[str, Any], ...]
    preserves_source_ownership: bool
    page_participant: ManifestParticipantRecord | None


@dataclass(frozen=True, slots=True)
class ValidatedManifestParticipants:
    """One validated named view over a manifest's participant descriptors."""

    descriptors_by_participant: Mapping[int, Mapping[str, Any]]
    records_by_participant: Mapping[int, ManifestParticipantRecord]
    roots_by_fragment_id: Mapping[str, str]
    descriptors_by_root: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        roots_by_fragment_id: Mapping[str, str],
    ) -> ValidatedManifestParticipants:
        descriptor_values: list[dict[str, Any]] = []
        for value in _strict_sequence(
            manifest["fragment_descriptors"], "fragment_descriptors"
        ):
            if not isinstance(value, (str, bytes)):
                raise ValueError("checkpoint fragment descriptor is not JSON text")
            descriptor = json.loads(value)
            if not isinstance(descriptor, dict):
                raise ValueError("checkpoint fragment descriptor is not an object")
            descriptor_values.append(descriptor)
        participant_count = _strict_integer(
            manifest["participant_count"], "participant_count", minimum=1
        )
        columns = {
            "watermark": manifest["participant_watermarks"],
            "page_capacity": manifest["participant_page_capacities"],
            "page_dataset": manifest["participant_page_datasets"],
            "page_version": manifest["participant_page_versions"],
            "page_layout_generation": manifest["participant_page_layout_generations"],
        }
        column_values = {
            name: _strict_sequence(value, f"participant_{name}")
            for name, value in columns.items()
        }
        if len(descriptor_values) != participant_count or any(
            len(column) != participant_count for column in column_values.values()
        ):
            raise ValueError(
                "checkpoint participant descriptor columns are inconsistent"
            )
        raw_fragment_ids = _strict_sequence(manifest["fragment_ids"], "fragment_ids")
        if any(
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in raw_fragment_ids
        ):
            raise ValueError("checkpoint manifest fragment IDs are invalid")
        fragment_ids = set(raw_fragment_ids)
        if (
            len(fragment_ids) != participant_count
            or set(roots_by_fragment_id) != fragment_ids
        ):
            raise ValueError("checkpoint participant fragment coverage is inconsistent")
        if len(set(roots_by_fragment_id.values())) != len(roots_by_fragment_id):
            raise ValueError("checkpoint participant fragment roots are duplicated")
        if any(
            not isinstance(root, str) or not root
            for root in roots_by_fragment_id.values()
        ):
            raise ValueError("checkpoint participant fragment roots are invalid")

        descriptors_by_participant: dict[int, Mapping[str, Any]] = {}
        records_by_participant: dict[int, ManifestParticipantRecord] = {}
        descriptors_by_root: dict[str, Mapping[str, Any]] = {}
        for descriptor in descriptor_values:
            _strict_integer(
                descriptor.get("entry_count"), "descriptor entry_count", minimum=0
            )
            _strict_integer(
                descriptor.get("page_count"), "descriptor page_count", minimum=0
            )
            _strict_integer(
                descriptor.get("page_version"),
                "descriptor page_version",
                maximum=(1 << 64) - 1,
            )
            _strict_integer(
                descriptor.get("page_layout_generation"),
                "descriptor page_layout_generation",
                maximum=(1 << 64) - 1,
            )
            _strict_integer(
                descriptor.get("mutation_watermark"),
                "descriptor mutation_watermark",
                maximum=(1 << 64) - 1,
            )
            entries_per_page = _strict_integer(
                descriptor.get("entries_per_page"),
                "descriptor entries_per_page",
                minimum=1,
            )
            if entries_per_page & (entries_per_page - 1):
                raise ValueError("descriptor entries_per_page must be a power of two")
            participant_indices = descriptor.get("participant_indices")
            if (
                not isinstance(participant_indices, Sequence)
                or isinstance(participant_indices, (str, bytes))
                or len(participant_indices) != 1
            ):
                raise ValueError("checkpoint participant routing is invalid")
            (participant_index,) = participant_indices
            try:
                participant_index = _strict_integer(
                    participant_index,
                    "participant_index",
                    minimum=0,
                )
            except ValueError as error:
                raise ValueError("checkpoint participant routing is invalid") from error
            if (
                participant_index >= participant_count
                or participant_index in descriptors_by_participant
            ):
                raise ValueError("checkpoint participant routing is invalid")
            if (
                _strict_integer(
                    descriptor.get("participant_count"),
                    "descriptor participant_count",
                    minimum=1,
                )
                != participant_count
            ):
                raise ValueError("checkpoint participant topology is invalid")
            fragment_id = descriptor.get("fragment_id")
            expected_fragment_id = hashlib.sha256(
                f"{manifest['checkpoint_name']}\0{manifest['save_attempt']}\0{participant_index}".encode()
            ).hexdigest()
            if (
                not isinstance(fragment_id, str)
                or fragment_id != expected_fragment_id
                or fragment_id not in roots_by_fragment_id
            ):
                raise ValueError("checkpoint participant fragment root is missing")
            immutable_descriptor = _freeze_descriptor_value(descriptor)
            root = roots_by_fragment_id[fragment_id]
            if root in descriptors_by_root:
                raise ValueError("checkpoint participant fragment roots are duplicated")
            capture_id = _strict_integer(
                descriptor.get("capture_id"), "capture_id", maximum=(1 << 64) - 1
            )
            journal_identity = descriptor.get("journal_identity")
            if not isinstance(journal_identity, str):
                raise ValueError("checkpoint journal identity is invalid")
            watermark = _strict_integer(
                column_values["watermark"][participant_index],
                "participant watermark",
                maximum=(1 << 64) - 1,
            )
            page_capacity = _strict_integer(
                column_values["page_capacity"][participant_index],
                "participant page capacity",
                minimum=1,
            )
            page_dataset = column_values["page_dataset"][participant_index]
            if not isinstance(page_dataset, str):
                raise ValueError("participant page dataset is invalid")
            page_version = _strict_integer(
                column_values["page_version"][participant_index],
                "participant page version",
                maximum=(1 << 64) - 1,
            )
            page_layout_generation = _strict_integer(
                column_values["page_layout_generation"][participant_index],
                "participant page layout generation",
                maximum=(1 << 64) - 1,
            )
            descriptors_by_participant[participant_index] = immutable_descriptor
            descriptors_by_root[root] = immutable_descriptor
            records_by_participant[participant_index] = ManifestParticipantRecord(
                participant_index=participant_index,
                fragment_id=fragment_id,
                capture_id=capture_id,
                journal_identity=journal_identity,
                watermark=watermark,
                page_capacity=page_capacity,
                page_dataset=page_dataset,
                page_version=page_version,
                page_layout_generation=page_layout_generation,
            )
        expected_participants = set(range(participant_count))
        if (
            set(descriptors_by_participant) != expected_participants
            or set(records_by_participant) != expected_participants
            or len(descriptors_by_root) != participant_count
        ):
            raise ValueError("checkpoint participant descriptor coverage is incomplete")
        return cls(
            MappingProxyType(descriptors_by_participant),
            MappingProxyType(records_by_participant),
            MappingProxyType(dict(roots_by_fragment_id)),
            MappingProxyType(descriptors_by_root),
        )

    def descriptor_for(self, participant_index: int) -> Mapping[str, Any]:
        descriptor = self.descriptors_by_participant.get(participant_index)
        if descriptor is None:
            raise ValueError("checkpoint participant descriptor is missing")
        return descriptor

    def record_for(self, participant_index: int) -> ManifestParticipantRecord:
        record = self.records_by_participant.get(participant_index)
        if record is None:
            raise ValueError("checkpoint participant descriptor is missing")
        return record

    def root_for(self, participant_index: int) -> str:
        record = self.record_for(participant_index)
        root = self.roots_by_fragment_id.get(record.fragment_id)
        if root is None:
            raise RuntimeError("checkpoint participant fragment root is missing")
        return root

    def descriptors(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            self.descriptor_for(participant_index)
            for participant_index in sorted(self.descriptors_by_participant)
        )


@dataclass(frozen=True, slots=True)
class _LeaseTag:
    lease_id: str
    generation: int
    expires_at_micros: int
    checkpoint_version: int
    checkpoint_name: str
    holder_id: str
    request_id: str

    @classmethod
    def parse(cls, tag: str, decoder: Callable[[str], str]) -> _LeaseTag:
        parts = tag.split(".")
        if len(parts) != 8:
            raise RuntimeError("invalid sparse checkpoint lease tag")
        (
            prefix,
            lease_id,
            generation_str,
            expires_at_micros_str,
            checkpoint_version_str,
            encoded_name,
            encoded_holder,
            encoded_request,
        ) = parts
        if prefix != "selease":
            raise RuntimeError("invalid sparse checkpoint lease tag")
        try:
            return cls(
                lease_id=lease_id,
                generation=_serialized_integer(generation_str, "lease generation"),
                expires_at_micros=_serialized_integer(
                    expires_at_micros_str, "lease expiry"
                ),
                checkpoint_version=_serialized_integer(
                    checkpoint_version_str, "checkpoint version", minimum=1
                ),
                checkpoint_name=decoder(encoded_name),
                holder_id=decoder(encoded_holder),
                request_id=decoder(encoded_request),
            )
        except (TypeError, ValueError, UnicodeDecodeError) as error:
            raise RuntimeError("invalid sparse checkpoint lease tag") from error


@dataclass(frozen=True, slots=True)
class _ReferenceTag:
    reference_token: str
    checkpoint_version: int
    checkpoint_name: str

    @classmethod
    def parse(cls, tag: str, decoder: Callable[[str], str]) -> _ReferenceTag:
        parts = tag.split(".")
        if len(parts) != 4:
            raise RuntimeError("checkpoint reference tag is corrupt")
        prefix, reference_token, version_str, encoded_name = parts
        if prefix != "sereference":
            raise RuntimeError("checkpoint reference tag is corrupt")
        return cls(
            reference_token=reference_token,
            checkpoint_version=_serialized_integer(
                version_str, "checkpoint version", minimum=1
            ),
            checkpoint_name=decoder(encoded_name),
        )


@dataclass(frozen=True, slots=True)
class _ParentHeadReservationTag:
    parent_checkpoint_version: int
    own_attempt_token: str
    attempt_token: str

    @classmethod
    def parse(cls, tag: str) -> _ParentHeadReservationTag:
        parts = tag.split(".")
        if len(parts) != 4:
            raise RuntimeError("parent-head reservation tag is corrupt")
        prefix, version_str, own_attempt_token, attempt_token = parts
        if prefix != "separenthead":
            raise RuntimeError("parent-head reservation tag is corrupt")
        return cls(
            parent_checkpoint_version=_serialized_integer(
                version_str, "parent checkpoint version", minimum=1
            ),
            own_attempt_token=own_attempt_token,
            attempt_token=attempt_token,
        )


@dataclass(frozen=True, slots=True)
class _PageAttemptTag:
    attempt_token: str
    page_version: int
    participant_index: int

    @classmethod
    def parse(cls, tag: str) -> _PageAttemptTag:
        parts = tag.split(".")
        if len(parts) != 4:
            raise RuntimeError("page attempt tag is corrupt")
        prefix, attempt_token, page_version_str, participant_index_str = parts
        if prefix != _PAGE_ATTEMPT_TAG_PREFIX:
            raise RuntimeError("page attempt tag is corrupt")
        return cls(
            attempt_token=attempt_token,
            page_version=_serialized_integer(page_version_str, "page version"),
            participant_index=_serialized_integer(
                participant_index_str, "participant index"
            ),
        )


@dataclass(frozen=True, slots=True)
class _PageWriteIntentTag:
    attempt_token: str
    participant_index: int
    parent_version: int

    @classmethod
    def parse(cls, tag: str) -> _PageWriteIntentTag:
        parts = tag.split(".")
        if len(parts) != 4:
            raise RuntimeError("page write intent tag is corrupt")
        prefix, attempt_token, participant_index_str, parent_version_str = parts
        if prefix != _PAGE_WRITE_INTENT_TAG_PREFIX:
            raise RuntimeError("page write intent tag is corrupt")
        return cls(
            attempt_token=attempt_token,
            participant_index=_serialized_integer(
                participant_index_str, "participant index"
            ),
            parent_version=_serialized_integer(
                parent_version_str, "parent checkpoint version", minimum=1
            ),
        )


@dataclass(frozen=True, slots=True)
class _ReclaimMarkerTag:
    checkpoint_version: int
    checkpoint_name: str

    @classmethod
    def parse(cls, tag: str, decoder: Callable[[str], str]) -> _ReclaimMarkerTag:
        parts = tag.split(".")
        if len(parts) != 3:
            raise RuntimeError("checkpoint reclaim marker is corrupt")
        prefix, version_str, encoded_name = parts
        try:
            name = decoder(encoded_name)
            return cls(
                checkpoint_version=_serialized_integer(
                    version_str, "checkpoint version", minimum=1
                ),
                checkpoint_name=name,
            )
        except (TypeError, ValueError, UnicodeDecodeError) as error:
            raise RuntimeError("checkpoint reclaim marker is corrupt") from error


@dataclass(frozen=True, slots=True)
class _AuthorityTag:
    generation: int
    authority_token: str

    @classmethod
    def parse_suffix(cls, suffix: str) -> _AuthorityTag:
        parts = suffix.split(".")
        if len(parts) != 3:
            raise RuntimeError("checkpoint authority tag is corrupt")
        prefix, generation_str, authority_token = parts
        try:
            generation = _serialized_integer(generation_str, "authority generation")
        except ValueError as error:
            raise RuntimeError("checkpoint authority tag is corrupt") from error
        return cls(generation=generation, authority_token=authority_token)


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.Table.from_arrays(
        [pa.array([], type=field.type) for field in schema], schema=schema
    )


def _schema_digest(specification: TableSpecification | None = None) -> str:
    entry_s = (
        entry_schema(specification.key_dtype)
        if specification is not None
        else ENTRY_SCHEMA
    )
    payload = b"".join(
        schema.serialize().to_pybytes()
        for schema in (entry_s, PAGE_IMAGE_SCHEMA, MANIFEST_SCHEMA)
    )
    if specification is not None:
        payload += _canonical_digest(
            [
                {
                    "dimensions": specification.dimensions,
                    "embedding_dtype": specification.embedding_dtype,
                    "optimizer_dtype": specification.optimizer_dtype,
                    "score_policy": specification.score_policy,
                    "key_dtype": specification.key_dtype,
                }
            ]
        ).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(rows: Sequence[dict[str, Any]]) -> str:
    def normalize(value: Any) -> Any:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {"__bytes__": bytes(value).hex()}
        if isinstance(value, np.ndarray):
            return {"__ndarray__": np.ascontiguousarray(value).tobytes().hex()}
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    payload = json.dumps(normalize(rows), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _encode_checkpoint_handle(
    storage_uri: str,
    locator: SparseCheckpointLocator,
    reference_id: str,
) -> bytes:
    storage_bytes = storage_uri.encode("utf-8")
    name_bytes = locator.checkpoint_name.encode("utf-8")
    reference_bytes = reference_id.encode("utf-8")
    if any(
        len(value) >= 1 << 32 for value in (storage_bytes, name_bytes, reference_bytes)
    ):
        raise ValueError("checkpoint handle field is too large")
    return b"".join(
        (
            _CHECKPOINT_HANDLE_MAGIC,
            _UINT32.pack(len(storage_bytes)),
            storage_bytes,
            _UINT32.pack(len(name_bytes)),
            name_bytes,
            _UINT64.pack(locator.checkpoint_version),
            _UINT32.pack(len(reference_bytes)),
            reference_bytes,
        )
    )


def _decode_checkpoint_handle(
    payload: bytes,
) -> tuple[str, SparseCheckpointLocator, str]:
    if not isinstance(payload, bytes):
        raise TypeError("checkpoint handle must be bytes")
    offset = len(_CHECKPOINT_HANDLE_MAGIC)
    if not payload.startswith(_CHECKPOINT_HANDLE_MAGIC):
        raise ValueError("sparse checkpoint handle has an unsupported format")

    def read_bytes() -> bytes:
        nonlocal offset
        if offset + _UINT32.size > len(payload):
            raise ValueError("sparse checkpoint handle is truncated")
        (length,) = _UINT32.unpack_from(payload, offset)
        offset += _UINT32.size
        end = offset + length
        if end > len(payload):
            raise ValueError("sparse checkpoint handle is truncated")
        value = payload[offset:end]
        offset = end
        return value

    storage_uri = read_bytes().decode("utf-8")
    checkpoint_name = read_bytes().decode("utf-8")
    if offset + _UINT64.size > len(payload):
        raise ValueError("sparse checkpoint handle has trailing data")
    (checkpoint_version,) = _UINT64.unpack_from(payload, offset)
    offset += _UINT64.size
    reference_id = read_bytes().decode("utf-8")
    if offset != len(payload):
        raise ValueError("sparse checkpoint handle has trailing data")
    if not storage_uri or not checkpoint_name:
        raise ValueError("sparse checkpoint handle contains empty identity fields")
    return (
        storage_uri,
        SparseCheckpointLocator(checkpoint_name, checkpoint_version),
        reference_id,
    )


def compact_fragments(
    source_path: str | Path,
    target_path: str | Path | None = None,
    *,
    storage_options: Mapping[str, str] | None = None,
) -> Path | str:
    """Compact one fragment or participant-fragment directory to a new path.

    This export-stage operation is independent of checkpoint manifests and
    storage instances.  It writes a new consolidated fragment through the
    native streaming Lance operator and leaves every source fragment intact.
    Supports both local filesystem paths and remote s3:// URIs.
    """
    source_str = str(source_path)
    if source_str.startswith(("s3://", "s3a://")):
        options = dict(storage_options or {})
        parsed = urlparse(source_str)
        workspace = S3LanceWorkspace(f"{parsed.scheme}://{parsed.netloc}", options)
        engine = _LanceCheckpointStorage(workspace)
        target_str = str(target_path) if target_path is not None else None
        published = engine.compact_fragments(source_str, target_str)
        return published
    source = Path(source_path).expanduser()
    source_text = str(source_path)
    if "\\" in source_text:
        raise ValueError("local fragment paths must use forward slashes")
    raw_parts = source_text.split("/")
    if any(
        part in {"", ".", ".."}
        for index, part in enumerate(raw_parts)
        if part != "" or index != 0 or not source_text.startswith("/")
    ):
        raise ValueError("local fragment paths must use canonical path segments")
    if not source.is_absolute():
        source = Path.cwd() / source
    workspace_root = source.parent
    source_relative = source.name
    target = Path(target_path).expanduser() if target_path is not None else None
    if target is not None:
        if not target.is_absolute():
            target = workspace_root / target
        try:
            target_relative = target.relative_to(workspace_root).as_posix()
        except ValueError as error:
            raise ValueError("target path is outside the source workspace") from error
    else:
        target_relative = None
    workspace = LocalLanceWorkspace(workspace_root)
    engine = _LanceCheckpointStorage(workspace)
    published = engine.compact_fragments(source_relative, target_relative)
    return Path(workspace.uri(published))


async def compact_fragments_async(
    source_path: str | Path,
    target_path: str | Path | None = None,
    *,
    storage_options: Mapping[str, str] | None = None,
) -> Path | str:
    return await asyncio.to_thread(
        compact_fragments, source_path, target_path, storage_options=storage_options
    )


class _LanceCheckpointStorage:
    """Shared sparse checkpoint transaction engine over a Lance workspace."""

    def __init__(
        self,
        workspace: LanceWorkspace,
        *,
        max_storage_bytes: int | None = None,
    ) -> None:
        self._workspace = workspace
        if max_storage_bytes is not None and max_storage_bytes <= 0:
            raise ValueError("max_storage_bytes must be positive")
        self._max_storage_bytes = max_storage_bytes
        self._bound_handle: CheckpointHandle | None = None
        self._manifest_cache_version: int | None = None
        self._manifest_cache_rows: list[dict[str, Any]] | None = None
        self._participant_view_cache: dict[
            tuple[str, int, str], ValidatedManifestParticipants
        ] = {}
        self._capture_cleanup: dict[tuple[str, str, int], tuple[Any, int, str]] = {}
        self._commit_lock_state = threading.local()
        self._exclusive_recovery_owner: int | None = None

    def _invalidate_manifest_cache(self) -> None:
        self._manifest_cache_version = None
        self._manifest_cache_rows = None
        self._participant_view_cache.clear()

    def requires_full_checkpoint(self, store: Any) -> bool:
        if not isinstance(store, CheckpointableStore):
            raise TypeError(
                f"store must implement CheckpointableStore; got {type(store).__name__}"
            )
        return bool(store.checkpoint_requires_full())

    def prepare_checkpoint(self, identity: CheckpointSaveIdentity) -> None:
        self._require_mutation_allowed()
        self._ensure_tables()
        with self._commit_lock():
            self._reconcile_staging()
            self._check_storage_budget()
            attempt = self._attempt_prefix(identity)
            if self._workspace.exists(attempt):
                raise CheckpointConflictError("checkpoint save attempt already exists")
            self._publish_authority(identity)
            self._workspace.create_prefix(attempt)

    def _write_checkpoint_fragment(
        self,
        store: Any,
        identity: CheckpointSaveIdentity,
        participant: CheckpointParticipant,
        include_page_images: bool = False,
        mode: CheckpointSaveMode = CheckpointSaveMode.FULL,
        parent_manifest: Mapping[str, Any] | None = None,
    ) -> CheckpointFragment:
        self._require_mutation_allowed()
        if not isinstance(mode, CheckpointSaveMode):
            raise TypeError("mode must be a CheckpointSaveMode")
        if mode is CheckpointSaveMode.INCREMENTAL and parent_manifest is None:
            raise ValueError("incremental fragment requires a parent manifest")
        if mode is CheckpointSaveMode.INCREMENTAL and self.requires_full_checkpoint(
            store
        ):
            raise ValueError(
                "reshard restore requires the next training checkpoint to be FULL"
            )
        self._require_authority(identity)
        attempt = self._attempt_prefix(identity)
        if not self._workspace.exists(attempt):
            raise CheckpointConflictError("checkpoint must be prepared before writing")
        fragment_id = self._fragment_id(identity, participant)
        fragment_root = f"{attempt}/{fragment_id}"
        if self._workspace.exists(fragment_root):
            raise CheckpointConflictError("checkpoint fragment already exists")
        specification = self._store_specification(store)
        if mode is CheckpointSaveMode.FULL and parent_manifest is not None:
            raise ValueError("FULL fragment cannot specify a parent")
        if (
            parent_manifest is not None
            and _strict_integer(
                parent_manifest["participant_count"], "participant_count"
            )
            != participant.participant_count
        ):
            raise ValueError(
                "incremental checkpoint requires stable participant topology"
            )
        baseline = (
            self._checkpoint_baseline(parent_manifest, participant.participant_index)
            if parent_manifest is not None
            else ""
        )
        if mode is CheckpointSaveMode.INCREMENTAL:
            if store.checkpoint_baseline() != baseline:
                raise RuntimeError("incremental checkpoint requires the store baseline")
        page_dataset = ""
        page_dataset_version = 0
        expected_page_layout_generation = 0
        parent_page_descriptor: dict[str, Any] | None = None
        if include_page_images:
            if parent_manifest is None:
                page_dataset = self._page_dataset_relative(
                    identity.save_attempt, participant.participant_index
                )
            else:
                parent_page_descriptor = self._manifest_fragment_descriptor(
                    parent_manifest, participant.participant_index
                )
                if not bool(parent_manifest["include_page_images"]):
                    raise FileNotFoundError(
                        "incremental page checkpoint requires a parent page snapshot"
                    )
                page_dataset = str(parent_page_descriptor["page_dataset"])
                page_dataset_version = _strict_integer(
                    parent_page_descriptor["page_version"], "page_version"
                )
                expected_page_layout_generation = _strict_integer(
                    parent_page_descriptor["page_layout_generation"],
                    "page_layout_generation",
                )
                if not page_dataset:
                    raise InvalidCheckpointFragment(
                        "incremental page checkpoint parent has no page dataset"
                    )
        parent_locator = (
            SparseCheckpointLocator(
                str(parent_manifest["checkpoint_name"]),
                _strict_integer(
                    parent_manifest["checkpoint_version"],
                    "checkpoint_version",
                    minimum=1,
                ),
            )
            if parent_manifest is not None
            else None
        )
        if parent_locator is not None:
            self._reserve_incremental_parent_head(identity, parent_locator)
        page_write_intent: tuple[str, str] | None = None
        if (
            include_page_images
            and parent_manifest is not None
            and page_dataset
            and page_dataset_version > 0
        ):
            page_write_intent = self._create_page_write_intent(
                identity,
                participant.participant_index,
                page_dataset,
                page_dataset_version,
            )
        artifact: dict[str, Any] | None = None
        page_attempt_tag: tuple[str, str, int] | None = None
        try:
            self._workspace.create_prefix(fragment_root)
            artifact = _checkpoint_native.write_native_fragment(
                store,
                self._workspace.uri(fragment_root),
                participant.participant_index,
                list(specification.dimensions),
                specification.embedding_dtype,
                specification.optimizer_dtype or "",
                specification.score_policy,
                specification.key_dtype,
                _schema_digest(specification),
                include_page_images,
                mode is CheckpointSaveMode.INCREMENTAL,
                baseline,
                self._workspace.uri(page_dataset) if page_dataset else "",
                page_dataset_version,
                expected_page_layout_generation,
            )
            if not isinstance(artifact, dict):
                raise TypeError(
                    "native fragment writer returned a non-mapping artifact"
                )
            required = {
                "capture_id",
                "journal_identity",
                "entries_per_page",
                "mutation_watermark",
                "locator",
                "entries_version",
                "page_images_version",
                "page_layout_generation",
                "entry_count",
                "page_count",
                "physical_bytes",
                "schema_digest",
                "content_digest",
            }
            if set(artifact) != required:
                raise ValueError("native fragment artifact fields are invalid")
            content_digest = str(artifact["content_digest"])
            page_version = _strict_integer(
                artifact["page_images_version"], "page_images_version"
            )
            page_layout_generation = _strict_integer(
                artifact["page_layout_generation"], "page_layout_generation"
            )
            if include_page_images:
                if mode is CheckpointSaveMode.INCREMENTAL and page_version == 0:
                    page_version = page_dataset_version
                if (
                    _strict_integer(artifact["page_count"], "page_count") > 0
                    and page_version <= 0
                ):
                    raise InvalidCheckpointFragment(
                        "page-image checkpoint did not produce a page dataset version"
                    )
            if (
                include_page_images
                and page_dataset
                and page_version > page_dataset_version
                and _strict_integer(artifact["page_count"], "page_count") > 0
            ):
                page_attempt_tag = self._protect_page_attempt(
                    identity,
                    participant.participant_index,
                    page_dataset,
                    page_version,
                )
            context = CheckpointSaveContext(
                mode,
                SparseCheckpointLocator(
                    str(parent_manifest["checkpoint_name"]),
                    _strict_integer(
                        parent_manifest["checkpoint_version"],
                        "checkpoint_version",
                        minimum=1,
                    ),
                )
                if parent_manifest is not None
                else None,
                str(parent_manifest["reference_id"])
                if parent_manifest is not None
                else "",
                _strict_integer(artifact["mutation_watermark"], "mutation_watermark"),
                include_page_images,
                _strict_integer(artifact["entries_per_page"], "entries_per_page"),
                _strict_integer(artifact["capture_id"], "capture_id"),
                str(artifact["journal_identity"]),
            )
            fragment = CheckpointFragment(
                fragment_id=fragment_id,
                checkpoint_name=identity.checkpoint_name,
                save_attempt=identity.save_attempt,
                participant=participant,
                entry_count=_strict_integer(artifact["entry_count"], "entry_count"),
                page_count=_strict_integer(artifact["page_count"], "page_count"),
                schema_digest=str(artifact["schema_digest"]),
                content_digest=content_digest,
                specification=specification,
                save_context=context,
                page_dataset=page_dataset,
                page_version=page_version,
                page_layout_generation=page_layout_generation,
            )
            self._publish_save_descriptor(fragment_root, fragment)
            self._require_authority(identity)
            if page_write_intent is not None:
                self._delete_page_write_intent(*page_write_intent)
            self._capture_cleanup[
                (
                    identity.checkpoint_name,
                    identity.save_attempt,
                    participant.participant_index,
                )
            ] = (store, context.capture_id, context.journal_identity)
            return fragment
        except BaseException as primary:
            actions: list[tuple[str, Any]] = []
            if page_write_intent is not None:
                actions.append(
                    (
                        "page write intent",
                        lambda intent=page_write_intent: self._delete_page_write_intent(
                            *intent
                        ),
                    )
                )
            if page_attempt_tag is not None:
                actions.append(
                    (
                        "page attempt tag",
                        lambda tag=page_attempt_tag: self._delete_page_attempt_tag(
                            *tag
                        ),
                    )
                )
            if (
                isinstance(artifact, dict)
                and "capture_id" in artifact
                and "journal_identity" in artifact
            ):
                actions.append(
                    (
                        "capture cancellation",
                        lambda: store.cancel_checkpoint_capture(
                            _strict_integer(artifact["capture_id"], "capture_id"),
                            str(artifact["journal_identity"]),
                        ),
                    )
                )
            run_cleanup_preserving_primary(primary, actions, "fragment write")
            raise

    def _commit_checkpoint(
        self,
        identity: CheckpointSaveIdentity,
        fragments: Sequence[CheckpointFragment],
        lease_request: CheckpointLeaseRequest | None = None,
    ) -> CommittedCheckpoint:
        reference_id = self._owner_reference_id(identity)
        with self._commit_lock():
            locator = self._commit_checkpoint_locked(
                identity, fragments, lease_request, reference_id
            )
            manifest = self._resolve_manifest(locator)
            return self._committed_checkpoint(locator, str(manifest["reference_id"]))

    def write_checkpoint_fragment_bytes(
        self,
        store: Any,
        identity: CheckpointSaveIdentity,
        participant: CheckpointParticipant,
        include_page_images: bool = False,
        mode: CheckpointSaveMode = CheckpointSaveMode.FULL,
        parent_handle: CheckpointHandle | None = None,
    ) -> bytes:
        self._require_mutation_allowed()
        with self._commit_lock():
            parent_manifest = (
                self._resolve_manifest(self._resolve_handle(parent_handle)[0])
                if parent_handle is not None
                else None
            )
            fragment = self._write_checkpoint_fragment(
                store,
                identity,
                participant,
                include_page_images=include_page_images,
                mode=mode,
                parent_manifest=parent_manifest,
            )
            return encode_checkpoint_fragment(fragment)

    def commit_checkpoint_bytes(
        self,
        identity: CheckpointSaveIdentity,
        fragments: Sequence[bytes],
        lease_request: CheckpointLeaseRequest | None = None,
    ) -> CommittedCheckpoint:
        self._require_mutation_allowed()
        decoded = tuple(decode_checkpoint_fragment(payload) for payload in fragments)
        return self._commit_checkpoint(identity, decoded, lease_request)

    def _committed_checkpoint(
        self, locator: SparseCheckpointLocator, reference_id: str
    ) -> CommittedCheckpoint:
        return CommittedCheckpoint(
            _encode_checkpoint_handle(self._workspace.uri(""), locator, reference_id),
            self._checkpoint_version_id(locator),
        )

    def _resolve_handle_and_manifest(
        self, handle: CheckpointHandle
    ) -> tuple[SparseCheckpointLocator, str, dict[str, Any]]:
        storage_uri, locator, reference_id = _decode_checkpoint_handle(handle)
        self._validate_handle_storage_uri(storage_uri)
        manifest = self._resolve_manifest(locator)
        if str(manifest["reference_id"]) != reference_id:
            raise ValueError("checkpoint handle reference identity does not match")
        return locator, reference_id, manifest

    def _resolve_handle(
        self, handle: CheckpointHandle
    ) -> tuple[SparseCheckpointLocator, str]:
        locator, reference_id, _ = self._resolve_handle_and_manifest(handle)
        return locator, reference_id

    def resolve_checkpoint_manifest(
        self, handle: CheckpointHandle
    ) -> Mapping[str, Any]:
        _, _, manifest = self._resolve_handle_and_manifest(handle)
        if _strict_integer(manifest["format_version"], "format_version") != 3:
            raise RuntimeError("checkpoint manifest format version is unsupported")
        return copy.deepcopy(manifest)

    def acknowledge_store_mutations(
        self, store: Any, handle: CheckpointHandle, participant_index: int = 0
    ) -> None:
        self._require_mutation_allowed()
        self._ensure_tables()
        with self._commit_lock():
            manifest = self.resolve_checkpoint_manifest(handle)
            if (
                isinstance(participant_index, bool)
                or not isinstance(participant_index, int)
                or not 0
                <= participant_index
                < _strict_integer(
                    manifest["participant_count"], "participant_count", minimum=1
                )
            ):
                raise ValueError(
                    "checkpoint participant index is outside saved topology"
                )
            participants = self._validated_manifest_participants(manifest)
            rec = participants.record_for(participant_index)
            watermark = rec.watermark
            descriptor = participants.descriptor_for(participant_index)
            store.advance_checkpoint_baseline(
                watermark,
                self._checkpoint_baseline(manifest, participant_index),
                descriptor["capture_id"],
                descriptor["journal_identity"],
            )
            self._capture_cleanup.pop(
                (
                    str(manifest["checkpoint_name"]),
                    str(manifest["save_attempt"]),
                    participant_index,
                ),
                None,
            )

    def _checkpoint_baseline(
        self, manifest: Mapping[str, Any], participant_index: int
    ) -> str:
        rec = self._validated_manifest_participants(manifest).record_for(
            participant_index
        )
        return _canonical_digest(
            [
                {
                    "checkpoint_name": str(manifest["checkpoint_name"]),
                    "schema_digest": str(manifest["schema_digest"]),
                    "participant_index": participant_index,
                    "mutation_watermark": rec.watermark,
                }
            ]
        )

    def _validate_handle_storage_uri(self, storage_uri: str) -> None:
        expected_uri = self._workspace.uri("")
        if isinstance(self._workspace, LocalLanceWorkspace):
            if Path(storage_uri).expanduser().resolve() != Path(expected_uri).resolve():
                raise ValueError("checkpoint handle belongs to another storage root")
        elif storage_uri.rstrip("/") != expected_uri.rstrip("/"):
            raise ValueError("checkpoint handle belongs to another storage root")

    @staticmethod
    def _checkpoint_version_id(locator: SparseCheckpointLocator) -> bytes:
        return f"{locator.checkpoint_name}:{locator.checkpoint_version}".encode()

    @staticmethod
    def _owner_reference_id(identity: CheckpointSaveIdentity) -> str:
        return f"owner:{identity.checkpoint_name}:{identity.save_attempt}"

    def finalize_checkpoint(self, identity: CheckpointSaveIdentity) -> None:
        self._require_mutation_allowed()
        with self._commit_lock():
            self._require_authority(identity)

    def _commit_checkpoint_locked(
        self,
        identity: CheckpointSaveIdentity,
        fragments: Sequence[CheckpointFragment],
        lease_request: CheckpointLeaseRequest | None,
        reference_id: str | None,
    ) -> SparseCheckpointLocator:
        if lease_request is not None and lease_request.ttl_seconds <= 0:
            raise ValueError("lease ttl must be positive")
        self._ensure_tables()
        prepared_for_checkpoint = tuple(
            row
            for row in self._manifest_rows()
            if row["checkpoint_name"] == identity.checkpoint_name
            and row["commit_state"] == _PREPARED
        )
        if any(
            row["save_attempt"] != identity.save_attempt
            for row in prepared_for_checkpoint
        ):
            raise CheckpointConflictError(
                "checkpoint has multiple prepared commit intents"
            )
        recovered = self._recover_prepared(identity, lease_request, reference_id)
        if recovered is not None:
            return recovered
        outcome = self._pending_outcome(identity)
        if outcome is not None:
            if (reference_id or "") != outcome.reference_id:
                raise CheckpointConflictError(
                    "checkpoint commit reference does not match its durable outcome"
                )
            return outcome.locator
        intent_tags = self._intent_tags_for_attempt(identity)
        if intent_tags:
            raise CheckpointConflictError(
                "checkpoint commit intent has no matching manifest"
            )
        self._require_authority(identity)
        self._workspace.assert_commit_lock()
        self._reconcile_staging()
        validated, specification = self._validate_fragments(identity, fragments)
        manifest_rows = self._manifest_rows()
        participant_counts = {
            fragment.participant.participant_count for fragment in validated
        }
        if len(participant_counts) != 1:
            raise InvalidCheckpointFragment(
                "checkpoint fragments have inconsistent participant topology"
            )
        participant_count = participant_counts.pop()
        contexts = tuple(fragment.save_context for fragment in validated)
        context_contracts = {
            (
                item.mode,
                item.parent,
                item.parent_reference_id,
                item.include_page_images,
            )
            for item in contexts
        }
        if len(context_contracts) != 1:
            raise InvalidCheckpointFragment(
                "checkpoint fragments have inconsistent save contexts"
            )
        context = next(iter(contexts))
        parent = None
        if context.mode is CheckpointSaveMode.INCREMENTAL:
            if context.parent is None:
                raise RuntimeError("incremental checkpoint is missing its parent")
            parent_version = context.parent.checkpoint_version
            parent = self._manifest_for_version(
                context.parent.checkpoint_name, parent_version
            )
            if context.parent.checkpoint_name != identity.checkpoint_name:
                raise ValueError(
                    "incremental parent belongs to another checkpoint lineage"
                )
            self._assert_incremental_parent_head_locked(
                identity,
                SparseCheckpointLocator(
                    context.parent.checkpoint_name,
                    context.parent.checkpoint_version,
                ),
            )
            tags_set = set(
                self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list()
            )
            if (
                not self._is_acknowledged_committed_row(parent, tags_set)
                or parent["reference_id"] != context.parent_reference_id
            ):
                raise RuntimeError("incremental parent identity is not committed")
            if parent["schema_digest"] != _schema_digest(specification):
                raise RuntimeError("incremental parent schema mismatch")
            if _strict_integer(
                parent["participant_count"], "participant_count", minimum=1
            ) != len(validated):
                raise RuntimeError("incremental participant topology changed")
            for fragment in validated:
                parent_rec = self._validated_manifest_participants(parent).record_for(
                    fragment.participant.participant_index
                )
                if fragment.save_context.mutation_watermark < parent_rec.watermark:
                    raise RuntimeError(
                        "incremental participant watermark is retrograde"
                    )
                if context.include_page_images:
                    if fragment.page_dataset != parent_rec.page_dataset:
                        raise RuntimeError("incremental page dataset identity changed")
                    if (
                        fragment.page_layout_generation
                        != parent_rec.page_layout_generation
                    ):
                        raise RuntimeError("incremental page layout generation changed")
                    if fragment.page_version < parent_rec.page_version:
                        raise RuntimeError("incremental page dataset version regressed")
        else:
            parent_version = 0
        checkpoint_version = (
            max(
                (
                    _strict_integer(
                        row["checkpoint_version"], "checkpoint_version", minimum=1
                    )
                    for row in manifest_rows
                ),
                default=0,
            )
            + 1
        )
        fragment_ids = tuple(fragment.fragment_id for fragment in validated)
        self._check_storage_budget()
        self._workspace.assert_commit_lock()
        descriptors = [self._fragment_descriptor(fragment) for fragment in validated]
        fragment_digest = _canonical_digest(descriptors)
        manifest = {
            "checkpoint_name": identity.checkpoint_name,
            "checkpoint_version": checkpoint_version,
            "parent_checkpoint_version": parent_version,
            "participant_count": participant_count,
            "checkpoint_mode": context.mode.value,
            "schema_digest": _schema_digest(specification),
            "fragment_digest": fragment_digest,
            "save_attempt": identity.save_attempt,
            "authority_scope": identity.commit_authority.scope,
            "authority_generation": identity.commit_authority.generation,
            "commit_authority": identity.commit_authority.authority_id,
            "created_at": datetime.now(UTC),
            "dimensions": list(specification.dimensions),
            "embedding_dtype": specification.embedding_dtype,
            "optimizer_dtype": specification.optimizer_dtype or "",
            "score_policy": specification.score_policy,
            "key_dtype": specification.key_dtype,
            "reference_id": reference_id or "",
            "fragment_ids": list(fragment_ids),
            "compacted_version": 0,
            "commit_state": _PREPARED,
            "storage_mode": _FRAGMENT_STORAGE_MODE,
            "format_version": 3,
            "parent_checkpoint_name": parent["checkpoint_name"]
            if parent is not None
            else "",
            "parent_reference_id": parent["reference_id"] if parent is not None else "",
            "participant_watermarks": [
                fragment.save_context.mutation_watermark for fragment in validated
            ],
            "fragment_descriptors": [
                json.dumps(descriptor, sort_keys=True) for descriptor in descriptors
            ],
            "include_page_images": context.include_page_images,
            "participant_page_capacities": [
                fragment.save_context.entries_per_page for fragment in validated
            ],
            "participant_page_datasets": [
                fragment.page_dataset for fragment in validated
            ],
            "participant_page_versions": [
                fragment.page_version for fragment in validated
            ],
            "participant_page_layout_generations": [
                fragment.page_layout_generation for fragment in validated
            ],
        }
        manifest_before = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).version
        locator = SparseCheckpointLocator(identity.checkpoint_name, checkpoint_version)
        lease: CheckpointLease | None = None
        try:
            intent = self._intent_tag(identity, locator)
            tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
            if intent not in tags.list():
                tags.create(intent, manifest_before)
            elif tags.get_version(intent) > manifest_before:
                raise CheckpointConflictError(
                    "checkpoint commit intent points ahead of manifest"
                )
            self._workspace.assert_commit_lock()
            self._workspace.write_dataset(
                pa.Table.from_pylist([manifest], schema=MANIFEST_SCHEMA),
                SNAPSHOT_MANIFEST_DATASET,
                mode="append",
            )
            self._invalidate_manifest_cache()
            self._protect_manifest_pages(manifest)
            self._release_page_attempt_tags(
                identity, [fragment.page_dataset for fragment in validated]
            )
            self._release_page_write_intents(
                identity, [fragment.page_dataset for fragment in validated]
            )
            self._workspace.assert_commit_lock()
            if lease_request is not None:
                lease = self._acquire_checkpoint_lease_locked(locator, lease_request)
            if reference_id:
                self._create_reference_tags(locator, reference_id, manifest)
            if parent is not None:
                self._create_parent_chain_reference(manifest)
            self._workspace.assert_commit_lock()
            updated = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).update(
                {"commit_state": f"'{_COMMITTED}'"},
                where=(
                    f"checkpoint_name = '{self._sql_literal(identity.checkpoint_name)}' "
                    f"AND checkpoint_version = {checkpoint_version} "
                    f"AND commit_state = '{_PREPARED}'"
                ),
            )
            self._invalidate_manifest_cache()
            if updated["num_rows_updated"] != 1:
                raise CheckpointConflictError("checkpoint manifest update was lost")
            committed_tag = self._committed_tag(locator)
            if committed_tag not in tags.list():
                tags.create(
                    committed_tag,
                    self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).version,
                )
            self._record_pending_outcome(identity, locator, manifest_before)
        except BaseException as primary:
            actions: list[tuple[str, Any]] = []
            if lease is not None:
                actions.append(
                    (
                        "lease tag",
                        lambda: self._delete_lease_tag(lease, locator),
                    )
                )
            if reference_id:
                actions.append(
                    (
                        "reference tags",
                        lambda: self._delete_reference_tags(locator, reference_id),
                    )
                )
            if parent is not None:
                actions.append(
                    (
                        "parent chain reference",
                        lambda: self._delete_parent_chain_reference(manifest),
                    )
                )
            run_cleanup_preserving_primary(primary, actions, "checkpoint commit")
            raise
        self._workspace.assert_commit_lock()
        try:
            self._release_parent_head_reservations(identity)
            self._clear_authority(identity)
        except BaseException as primary:
            run_cleanup_preserving_primary(
                primary,
                [
                    ("clear authority", lambda: self._clear_authority(identity)),
                ],
                "post-commit resource release",
            )
            raise
        return locator

    def _delete_prepared_manifest_row(
        self, checkpoint_name: str, checkpoint_version: int
    ) -> None:
        self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).delete(
            f"checkpoint_name = '{self._sql_literal(checkpoint_name)}' "
            f"AND checkpoint_version = {checkpoint_version}"
        )
        self._invalidate_manifest_cache()

    def abort_checkpoint(self, identity: CheckpointSaveIdentity) -> None:
        self._require_mutation_allowed()
        with self._commit_lock():
            retained_committed = False
            reclaimed_committed = False
            tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
            attempt = self._attempt_prefix(identity)
            actions: list[tuple[str, Any]] = []
            global_actions = [
                (
                    "attempt prefix cleanup",
                    lambda: (
                        self._workspace.delete_prefix(attempt, missing_ok=True)
                        if (not retained_committed and self._workspace.exists(attempt))
                        else None
                    ),
                ),
                (
                    "authority cleanup",
                    lambda: (
                        self._clear_authority(identity)
                        if not retained_committed
                        else None
                    ),
                ),
                (
                    "parent head reservations cleanup",
                    lambda: self._release_parent_head_reservations(identity),
                ),
            ]
            capture_keys = tuple(
                key
                for key in self._capture_cleanup
                if key[0] == identity.checkpoint_name
                and key[1] == identity.save_attempt
            )
            for capture_key in capture_keys:
                capture = self._capture_cleanup[capture_key]
                actions.append(
                    (
                        "capture cancellation",
                        lambda capture=capture: capture[0].cancel_checkpoint_capture(
                            capture[1], capture[2]
                        ),
                    )
                )
                self._capture_cleanup.pop(capture_key, None)
            try:
                for row in self._manifest_rows():
                    if (
                        row["checkpoint_name"] == identity.checkpoint_name
                        and row["save_attempt"] == identity.save_attempt
                    ):
                        if row["commit_state"] == _COMMITTED:
                            locator = SparseCheckpointLocator(
                                identity.checkpoint_name,
                                _strict_integer(
                                    row["checkpoint_version"],
                                    "checkpoint_version",
                                    minimum=1,
                                ),
                            )
                            intent = self._intent_tag(identity, locator)
                            if intent not in tags.list():
                                retained_committed = True
                                continue
                            reclaimed_committed = True
                            actions.append(
                                (
                                    "reclaim orphan committed row",
                                    lambda r=row: self._reclaim_committed_row(r),
                                )
                            )
                            continue
                        if row["commit_state"] != _PREPARED:

                            def _corrupt() -> None:
                                raise RuntimeError(
                                    "checkpoint manifest commit state is corrupt"
                                )

                            actions.append(("validate commit state", _corrupt))
                            continue
                        version = _strict_integer(
                            row["checkpoint_version"], "checkpoint_version", minimum=1
                        )
                        locator = SparseCheckpointLocator(
                            identity.checkpoint_name, version
                        )
                        page_datasets = [
                            self._manifest_fragment_descriptor(row, participant_index)[
                                "page_dataset"
                            ]
                            for participant_index in range(
                                _strict_integer(
                                    row["participant_count"],
                                    "participant_count",
                                    minimum=1,
                                )
                            )
                        ]
                        actions.extend(
                            [
                                (
                                    "parent chain reference",
                                    lambda row=row: self._delete_parent_chain_reference(
                                        row
                                    ),
                                ),
                                (
                                    "reference tags",
                                    lambda locator=locator,
                                    ref=str(row["reference_id"]): (
                                        self._delete_reference_tags(locator, ref)
                                    ),
                                ),
                                (
                                    "manifest prepared row",
                                    lambda version=version: (
                                        self._delete_prepared_manifest_row(
                                            identity.checkpoint_name, version
                                        )
                                    ),
                                ),
                                (
                                    "intent tag",
                                    lambda version=version: self._delete_intent_tag(
                                        identity, version
                                    ),
                                ),
                                (
                                    "page write intents",
                                    lambda pds=page_datasets: (
                                        self._release_page_write_intents(identity, pds)
                                    ),
                                ),
                            ]
                        )
                if reclaimed_committed:
                    actions.append(
                        (
                            "reconcile page protection",
                            lambda: (
                                self._invalidate_manifest_cache(),
                                self._reconcile_page_protection(cleanup=True),
                            ),
                        )
                    )
            except BaseException as primary:
                run_cleanup_preserving_primary(
                    primary, [*actions, *global_actions], "checkpoint abort"
                )
                raise

            execute_cleanup_actions([*actions, *global_actions], "checkpoint abort")

    def acknowledge_checkpoint(self, identity: CheckpointSaveIdentity) -> None:
        self._require_mutation_allowed()
        with self._commit_lock():
            if self._pending_outcome(identity, require_identity=False) is not None:
                self._delete_pending_outcome(identity.checkpoint_name)
            for row in self._manifest_rows():
                if (
                    row["checkpoint_name"] == identity.checkpoint_name
                    and row["save_attempt"] == identity.save_attempt
                ):
                    self._delete_intent_tag(
                        identity,
                        _strict_integer(
                            row["checkpoint_version"], "checkpoint_version", minimum=1
                        ),
                    )

    def acquire_checkpoint_lease(
        self, handle: CheckpointHandle, request: CheckpointLeaseRequest
    ) -> CheckpointLease:
        self._require_mutation_allowed()
        with self._commit_lock():
            locator, _ = self._resolve_handle(handle)
            return self._acquire_checkpoint_lease_locked(locator, request)

    def _acquire_checkpoint_lease_locked(
        self, locator: SparseCheckpointLocator, request: CheckpointLeaseRequest
    ) -> CheckpointLease:
        manifest = self._resolve_manifest(locator)
        if self._locator_has_reservation(locator):
            raise CheckpointConflictError("checkpoint is reserved for reclaim")
        for record in self._active_checkpoint_lease_records_unlocked():
            existing = record.lease
            if existing.request_id != request.request_id:
                continue
            if (
                existing.lease_holder_id != request.lease_holder_id
                or record.locator != locator
            ):
                raise CheckpointConflictError(
                    "lease request ID belongs to another checkpoint or lease holder"
                )
            return existing
        lease = CheckpointLease(
            lease_id=uuid.uuid4().hex,
            lease_holder_id=request.lease_holder_id,
            expires_at=datetime.now(UTC) + timedelta(seconds=request.ttl_seconds),
            generation=0,
            request_id=request.request_id,
        )
        self._create_lease_tags(lease, locator, manifest)
        return lease

    def renew_checkpoint_lease(
        self, lease: CheckpointLease, ttl_seconds: float
    ) -> CheckpointLease:
        self._require_mutation_allowed()
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        with self._commit_lock():
            matches = tuple(
                record
                for record in self._active_checkpoint_lease_records_unlocked()
                if record.lease.lease_id == lease.lease_id
            )
            if len(matches) != 1:
                raise CheckpointConflictError("lease generation CAS conflict")
            (matched_record,) = matches
            if matched_record.lease != lease:
                raise CheckpointConflictError("lease generation CAS conflict")
            renewed = CheckpointLease(
                lease_id=lease.lease_id,
                lease_holder_id=lease.lease_holder_id,
                expires_at=datetime.now(UTC) + timedelta(seconds=ttl_seconds),
                generation=lease.generation + 1,
                request_id=lease.request_id,
            )
            self._create_lease_tags(
                renewed,
                matched_record.locator,
                self._resolve_manifest(matched_record.locator),
            )
            self._delete_lease_tag(lease, matched_record.locator)
            return renewed

    def release_checkpoint_lease(self, lease: CheckpointLease) -> None:
        self._require_mutation_allowed()
        with self._commit_lock():
            matches = tuple(
                record
                for record in self._active_checkpoint_lease_records_unlocked()
                if record.lease.lease_id == lease.lease_id
            )
            if not matches:
                return
            if len(matches) != 1:
                raise CheckpointConflictError("lease generation CAS conflict")
            (matched_record,) = matches
            if matched_record.lease != lease:
                raise CheckpointConflictError("lease generation CAS conflict")
            self._delete_lease_tag(lease, matched_record.locator)

    def _delete_lease_tag(
        self, lease: CheckpointLease, locator: SparseCheckpointLocator
    ) -> None:
        tag = self._lease_tag(lease, locator)
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        if tag in tags.list():
            tags.delete(tag)

    def active_checkpoint_leases(
        self, now: datetime | None = None
    ) -> tuple[CheckpointLease, ...]:
        self._ensure_tables()
        selected_now = now or datetime.now(UTC)
        if selected_now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self._commit_lock():
            return tuple(
                record.lease
                for record in self._active_checkpoint_lease_records_unlocked(
                    selected_now
                )
            )

    def _active_checkpoint_lease_records_unlocked(
        self, now: datetime | None = None
    ) -> tuple[_LeaseRecord, ...]:
        selected_now = now or datetime.now(UTC)
        dataset = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET)
        leases = []
        for tag in tuple(dataset.tags.list()):
            if not tag.startswith("selease."):
                continue
            record = self._decode_lease_tag(tag)
            if record.lease.expires_at > selected_now:
                leases.append(record)
        return tuple(sorted(leases, key=lambda item: item.lease.lease_id))

    def reclaim(self, *, older_than: timedelta = timedelta(0)) -> None:
        self._ensure_tables()
        self._require_exclusive_recovery()
        self._reconcile_reclaiming()
        self._reconcile_orphan_commits()
        self._reclaim_unlocked(older_than)
        self._reconcile_page_protection(cleanup=True)

    @contextmanager
    def exclusive_recovery(self):
        owner = threading.get_ident()
        if self._exclusive_recovery_owner is not None:
            raise CheckpointConflictError("exclusive recovery is already active")
        lock = self._workspace.commit_lock()
        lock.__enter__()
        self._exclusive_recovery_owner = owner
        primary: BaseException | None = None
        try:
            yield self
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                try:
                    lock.__exit__(None, None, None)
                except BaseException as release_error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "exclusive recovery lock release failed: "
                        f"{type(release_error).__name__}: {release_error}"
                    )
            finally:
                self._exclusive_recovery_owner = None

    def compact_checkpoints(
        self, handles: Sequence[CheckpointHandle], *, include_page_images: bool = False
    ) -> tuple[CheckpointHandle, ...]:
        self._require_mutation_allowed()
        return tuple(
            self.compact_checkpoint(handle, include_page_images=include_page_images)
            for handle in handles
        )

    def compact_fragments(
        self,
        source_path: str | Path,
        target_path: str | Path | None = None,
    ) -> str:
        self._require_mutation_allowed()
        with self._commit_lock():
            return self._compact_fragments_unlocked(source_path, target_path)

    def _compact_fragments_unlocked(
        self,
        source_path: str | Path,
        target_path: str | Path | None = None,
    ) -> str:
        source = self._workspace_path(source_path)
        roots = self._fragment_roots_in_workspace(source)
        target = self._workspace_path(
            target_path
            if target_path is not None
            else f"{source.rstrip('/')}.compacted"
        )
        if target == source or target.startswith(source.rstrip("/") + "/"):
            raise ValueError("compaction target must not overlap the source path")
        if self._workspace.exists(target):
            raise FileExistsError(f"compaction target already exists: {target}")
        parent, separator, name = target.rpartition("/")
        temporary = (
            f"{parent}{separator}.{name}.tmp-{uuid.uuid4().hex}"
            if separator
            else f".{name}.tmp-{uuid.uuid4().hex}"
        )
        if self._workspace.exists(temporary):
            raise FileExistsError(
                f"compaction temporary prefix already exists: {temporary}"
            )
        self._workspace.create_prefix(temporary)
        try:
            compacted = self._compact_fragment_roots(
                roots,
                temporary,
                expected_participant_count=None,
            )
            self._publish_local_validation_descriptor(temporary, compacted["metadata"])
            self._workspace.publish_prefix(temporary, target)
        except BaseException as primary:
            published_target = self._workspace.uri(target) == self._workspace.uri(
                temporary
            )
            if self._workspace.exists(temporary) and not published_target:
                run_cleanup_preserving_primary(
                    primary,
                    [
                        (
                            "compaction temporary prefix",
                            lambda: self._workspace.delete_prefix(
                                temporary, missing_ok=True
                            ),
                        )
                    ],
                    "fragment compaction",
                )
            raise
        return self._workspace.uri(target)

    async def compact_fragments_async(
        self,
        source_path: str | Path,
        target_path: str | Path | None = None,
    ) -> str:
        return await asyncio.to_thread(self.compact_fragments, source_path, target_path)

    def _workspace_path(self, path: str | Path) -> str:
        value = str(path)
        raw_root_uri = self._workspace.uri("")
        root_uri = raw_root_uri if raw_root_uri == "/" else raw_root_uri.rstrip("/")
        if "://" in value:
            if not value.startswith(root_uri + "/") and value != root_uri:
                raise ValueError("path is outside the configured Lance workspace")
            suffix = value[len(root_uri) + 1 :] if value != root_uri else ""
            return _validate_relative_path(suffix)
        if isinstance(self._workspace, LocalLanceWorkspace):
            root_path = Path(root_uri)
            raw_path = Path(value).expanduser()
            if raw_path.is_absolute():
                try:
                    relative = raw_path.relative_to(root_path)
                except ValueError as error:
                    raise ValueError(
                        "path is outside the configured Lance workspace"
                    ) from error
                return _validate_relative_path(relative.as_posix())
            return _validate_relative_path(value)
        return _validate_relative_path(value)

    def _fragment_roots_in_workspace(self, source: str) -> tuple[str, ...]:
        if self._workspace.exists(f"{source}/{ENTRIES_DATASET}"):
            nested = tuple(
                child
                for child in self._workspace.child_prefixes(source)
                if self._workspace.exists(f"{source}/{child}/{ENTRIES_DATASET}")
            )
            if nested:
                raise ValueError("fragment source path has an ambiguous nested layout")
            return (source,)
        roots = tuple(
            f"{source}/{child}"
            for child in sorted(self._workspace.child_prefixes(source))
            if self._workspace.exists(f"{source}/{child}/{ENTRIES_DATASET}")
        )
        if not roots:
            raise ValueError(
                "fragment source path must be a fragment root or contain participant fragments"
            )
        child_prefixes = self._workspace.child_prefixes(source)
        if any(
            child not in {posixpath.basename(root) for root in roots}
            for child in child_prefixes
        ):
            raise ValueError("fragment source path contains unrelated child prefixes")
        return roots

    def compact_checkpoint(
        self, handle: CheckpointHandle, *, include_page_images: bool = False
    ) -> CheckpointHandle:
        self._require_mutation_allowed()
        locator, _ = self._resolve_handle(handle)
        reference_id = f"compact-read:{uuid.uuid4().hex}"
        with self._commit_lock():
            source = self._resolve_manifest(locator)
            chain = self._manifest_chain(source)
            self._create_reference_tags(locator, reference_id, source)
        attempt = uuid.uuid4().hex
        identity = CheckpointSaveIdentity(
            str(source["checkpoint_name"]),
            attempt,
            CheckpointCommitAuthority(f"compact:{attempt}", 0, attempt),
        )
        try:
            for manifest in chain:
                self._validate_manifest_fragments(manifest)
            self._validate_chain_entries(chain)
            self.prepare_checkpoint(identity)
            specification = self._manifest_specification(source)
            participant_count = _strict_integer(
                source["participant_count"], "participant_count", minimum=1
            )
            fragments: list[CheckpointFragment] = []
            for participant_index in range(participant_count):
                participant = CheckpointParticipant(
                    participant_index, participant_count
                )
                fragment_id = self._fragment_id(identity, participant)
                root = f"{self._attempt_prefix(identity)}/{fragment_id}"
                self._workspace.create_prefix(root)
                sources = []
                for manifest in chain:
                    source_root = self._manifest_fragment_root_for_participant(
                        manifest, participant_index
                    )
                    source_entries = f"{source_root}/{ENTRIES_DATASET}"
                    if self._workspace.exists(source_entries):
                        sources.append(self._workspace.uri(source_entries))
                participant_rec = self._validated_manifest_participants(
                    source
                ).record_for(participant_index)
                page_dataset = ""
                page_version = 0
                page_layout_generation = 0
                native_include_page_images = include_page_images
                if include_page_images:
                    page_dataset = f"{root}/{PAGE_IMAGES_DATASET}"
                    if participant_rec.page_layout_generation == (1 << 64) - 1:
                        raise OverflowError("Page layout generation is exhausted")
                    page_layout_generation = participant_rec.page_layout_generation + 1
                artifact = _checkpoint_native.compact_native_checkpoint(
                    self._workspace.uri(root),
                    sources,
                    participant_index,
                    list(specification.dimensions),
                    specification.embedding_dtype,
                    specification.optimizer_dtype or "",
                    specification.score_policy,
                    specification.key_dtype,
                    _schema_digest(specification),
                    native_include_page_images,
                    participant_rec.page_capacity,
                    self._workspace.uri(page_dataset)
                    if native_include_page_images
                    else "",
                    dict(self._workspace.storage_options()),
                )
                compact_page_version = 0
                if (
                    include_page_images
                    and page_version == 0
                    and _strict_integer(artifact["page_count"], "page_count") > 0
                ):
                    compact_page_version = _strict_integer(
                        self._workspace.dataset(page_dataset).version,
                        "compact page dataset version",
                        minimum=1,
                    )
                    if compact_page_version <= 0:
                        raise InvalidCheckpointFragment(
                            "compact page dataset has no committed version"
                        )
                fragment = CheckpointFragment(
                    fragment_id,
                    identity.checkpoint_name,
                    attempt,
                    participant,
                    _strict_integer(artifact["entry_count"], "entry_count"),
                    _strict_integer(artifact["page_count"], "page_count"),
                    _schema_digest(specification),
                    str(artifact["content_digest"]),
                    specification,
                    CheckpointSaveContext(
                        CheckpointSaveMode.FULL,
                        None,
                        "",
                        participant_rec.watermark,
                        include_page_images,
                        participant_rec.page_capacity,
                        0,
                        "",
                    ),
                    page_dataset
                    if page_version > 0
                    or _strict_integer(artifact["page_count"], "page_count")
                    else "",
                    page_version if page_version > 0 else compact_page_version,
                    page_layout_generation,
                )
                self._publish_save_descriptor(root, fragment)
                fragments.append(fragment)
            committed = self._commit_checkpoint(identity, fragments)
        except BaseException as primary:
            run_cleanup_preserving_primary(
                primary,
                [
                    ("checkpoint abort", lambda: self.abort_checkpoint(identity)),
                    (
                        "compaction source reference",
                        lambda: self._delete_reference_tags_with_lock(
                            locator, reference_id
                        ),
                    ),
                ],
                "checkpoint compaction",
            )
            raise

        try:
            self.acknowledge_checkpoint(identity)
        except BaseException as primary:
            run_cleanup_preserving_primary(
                primary,
                [
                    (
                        "compaction source reference",
                        lambda: self._delete_reference_tags_with_lock(
                            locator, reference_id
                        ),
                    ),
                ],
                "checkpoint compaction acknowledgement",
            )
            raise

        with self._commit_lock():
            self._delete_reference_tags(locator, reference_id)
        return committed.handle

    async def compact_checkpoints_async(
        self, handles: Sequence[CheckpointHandle], *, include_page_images: bool = False
    ) -> tuple[CheckpointHandle, ...]:
        return await asyncio.to_thread(
            self.compact_checkpoints, handles, include_page_images=include_page_images
        )

    async def compact_checkpoint_async(
        self, handle: CheckpointHandle, *, include_page_images: bool = False
    ) -> CheckpointHandle:
        return await asyncio.to_thread(
            self.compact_checkpoint,
            handle,
            include_page_images=include_page_images,
        )

    def _reclaim_unlocked(self, older_than: timedelta) -> None:
        self._reclaim_unlocked_for(older_than, frozenset())

    def _reclaim_unlocked_for(
        self,
        older_than: timedelta,
        authorized_locators: frozenset[SparseCheckpointLocator],
    ) -> None:
        now = datetime.now(UTC)
        active_records = self._active_checkpoint_lease_records_unlocked(now)
        active_ids = {record.lease.lease_id for record in active_records}
        leased_locators = {record.locator for record in active_records}
        manifest_dataset = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET)
        rows = self._manifest_rows()
        tags = manifest_dataset.tags
        for tag in tuple(tags.list()):
            if tag.startswith("selease."):
                _, lease_id, *_ = tag.split(".", 2)
                if lease_id not in active_ids:
                    tags.delete(tag)
        protected = {
            locator
            for locator in (
                self._decode_reference_tag(tag)
                for tag in tuple(tags.list())
                if tag.startswith("sereference.")
            )
        }
        protected.update(
            SparseCheckpointLocator(
                str(row["parent_checkpoint_name"]),
                _strict_integer(
                    row["parent_checkpoint_version"], "parent_checkpoint_version"
                ),
            )
            for row in rows
            if row["commit_state"] == _PREPARED
            and row["checkpoint_mode"] == CheckpointSaveMode.INCREMENTAL.value
        )
        for row in rows:
            locator = SparseCheckpointLocator(
                row["checkpoint_name"],
                _strict_integer(
                    row["checkpoint_version"], "checkpoint_version", minimum=1
                ),
            )
            if row["commit_state"] != _COMMITTED:
                continue
            if (
                locator in leased_locators
                or locator in protected
                or (
                    locator not in authorized_locators
                    and self._locator_has_reservation(locator)
                )
                or self._locator_has_pending_intent(locator)
            ):
                continue
            tags_set = set(tags.list())
            latest_version = max(
                (
                    _strict_integer(
                        candidate["checkpoint_version"], "checkpoint_version", minimum=1
                    )
                    for candidate in rows
                    if candidate["checkpoint_name"] == locator.checkpoint_name
                    and self._is_acknowledged_committed_row(candidate, tags_set)
                ),
                default=0,
            )
            has_pending_reclaim = any(
                tag.startswith("sepending.")
                and tag.endswith(self._reservation_locator_suffix(locator))
                for tag in tags.list()
            )
            if (
                locator.checkpoint_version == latest_version
                and locator not in authorized_locators
                and not has_pending_reclaim
            ):
                continue
            created_at = row["created_at"]
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            if now - created_at < older_than:
                continue
            self._reclaim_committed_row(row)
        if self._exclusive_recovery_owner is not None:
            self._reconcile_staging(cleanup=True)

    def _reconcile_reclaiming(self) -> None:
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        for tag in tuple(tags.list()):
            if not tag.startswith("sereclaiming."):
                continue
            parts = tag.split(".")
            if len(parts) != 3:
                raise RuntimeError("checkpoint reclaim marker is corrupt")
            locator = self._reclaim_marker_locator(tag)
            try:
                manifest = self._manifest_for_version(
                    locator.checkpoint_name, locator.checkpoint_version
                )
            except LookupError:
                prior = self._workspace.dataset(
                    SNAPSHOT_MANIFEST_DATASET, version=tags.get_version(tag)
                )
                matches = [
                    row
                    for row in prior.to_table().to_pylist()
                    if row["checkpoint_name"] == locator.checkpoint_name
                    and _strict_integer(
                        row["checkpoint_version"], "checkpoint_version", minimum=1
                    )
                    == locator.checkpoint_version
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        "checkpoint reclaim marker has no durable manifest"
                    )
                (manifest,) = matches
            if manifest["commit_state"] != _COMMITTED:
                raise RuntimeError(
                    "checkpoint reclaim marker manifest is not committed"
                )
            self._reclaim_committed_row(manifest)

    def _reclaim_committed_row(self, row: Mapping[str, Any]) -> None:
        version = _strict_integer(
            row["checkpoint_version"], "checkpoint_version", minimum=1
        )
        locator = SparseCheckpointLocator(str(row["checkpoint_name"]), version)
        dataset = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET)
        tags = dataset.tags

        reclaim_tag = self._reclaim_pending_tag(locator)
        if reclaim_tag not in tags.list():
            tags.create(reclaim_tag, dataset.version)

        roots = self._manifest_fragment_roots(dict(row))
        for root in roots.values():
            if self._workspace.exists(root):
                self._workspace.delete_prefix(root)

        self._delete_parent_chain_reference(dict(row))
        if row["reference_id"]:
            self._delete_reference_tags(locator, str(row["reference_id"]))

        if bool(row["include_page_images"]):
            page_datasets = [
                self._manifest_fragment_descriptor(row, participant_index)[
                    "page_dataset"
                ]
                for participant_index in range(
                    _strict_integer(
                        row["participant_count"],
                        "participant_count",
                        minimum=1,
                    )
                )
            ]
            save_identity = CheckpointSaveIdentity(
                str(row["checkpoint_name"]),
                str(row["save_attempt"]),
                CheckpointCommitAuthority(
                    str(row["authority_scope"]),
                    _strict_integer(
                        row["authority_generation"], "authority_generation"
                    ),
                    str(row["commit_authority"]),
                ),
            )
            self._release_page_attempt_tags(save_identity, page_datasets)
            self._release_page_write_intents(save_identity, page_datasets)

        dataset.delete(
            f"checkpoint_name = '{self._sql_literal(locator.checkpoint_name)}' "
            f"AND checkpoint_version = {locator.checkpoint_version}"
        )
        self._invalidate_manifest_cache()

        attempt_token = self._token(str(row["save_attempt"]))
        intent_tag = f"seintent.{attempt_token}.{version}"
        if intent_tag in tags.list():
            tags.delete(intent_tag)

        self._delete_pending_outcome(locator.checkpoint_name)
        committed_tag = self._committed_tag(locator)
        if committed_tag in tags.list():
            tags.delete(committed_tag)

        if reclaim_tag in tags.list():
            tags.delete(reclaim_tag)

    def _reconcile_orphan_commits(self) -> None:
        dataset = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET)
        tags_set = set(dataset.tags.list())
        rows = self._manifest_rows()
        for row in rows:
            if row["commit_state"] != _COMMITTED:
                continue
            attempt_token = self._token(str(row["save_attempt"]))
            version = _strict_integer(
                row["checkpoint_version"], "checkpoint_version", minimum=1
            )
            intent_tag = f"seintent.{attempt_token}.{version}"
            if intent_tag not in tags_set:
                continue
            has_superseding_acknowledged = any(
                candidate["checkpoint_name"] == row["checkpoint_name"]
                and _strict_integer(
                    candidate["checkpoint_version"], "checkpoint_version", minimum=1
                )
                >= version
                and self._is_acknowledged_committed_row(candidate, tags_set)
                for candidate in rows
            )
            scope = str(row["authority_scope"])
            fence_prefix = f"sefence.{self._token(scope)}."
            fence_generations = [
                _serialized_integer(
                    tag.removeprefix(fence_prefix),
                    "authority generation",
                    maximum=(1 << 64) - 1,
                )
                for tag in tags_set
                if tag.startswith(fence_prefix)
            ]
            current_fence_gen = max(fence_generations, default=-1)
            row_gen = _strict_integer(
                row["authority_generation"], "authority_generation"
            )
            is_fenced = current_fence_gen > row_gen
            if has_superseding_acknowledged or is_fenced:
                self._reclaim_committed_row(row)

    @staticmethod
    def _reclaim_marker_locator(tag: str) -> SparseCheckpointLocator:
        marker = _ReclaimMarkerTag.parse(tag, _LanceCheckpointStorage._decode)
        return SparseCheckpointLocator(
            marker.checkpoint_name, marker.checkpoint_version
        )

    def reserve_cleanup(
        self,
        reservation_id: str,
        ordered_candidates: Sequence[CheckpointHandle],
        required_count: int,
    ) -> CleanupReservation:
        self._require_mutation_allowed()
        if required_count <= 0:
            raise ValueError("required_count must be positive")
        with self._commit_lock():
            leased = {
                record.locator
                for record in self._active_checkpoint_lease_records_unlocked()
            }
            tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
            names = tuple(tags.list())
            selected: list[SparseCheckpointLocator] = []
            selected_handles: list[CheckpointHandle] = []
            for handle in ordered_candidates:
                locator, _ = self._resolve_handle(handle)
                self._resolve_manifest(locator)
                if locator in leased:
                    continue
                expected = self._reservation_tag("reserved", reservation_id, locator)
                conflicts = tuple(
                    name
                    for name in names
                    if self._is_reservation_tag(name)
                    and name.endswith(self._reservation_locator_suffix(locator))
                    and name != expected
                )
                if conflicts:
                    continue
                selected.append(locator)
                selected_handles.append(handle)
                if len(selected) == required_count:
                    break
            if len(selected) != required_count:
                raise CheckpointConflictError(
                    "not enough unleased checkpoints are reclaimable"
                )
            owner_handle = self._reservation_owner_handle(
                reservation_id, tuple(selected_handles)
            )
            reservation = CleanupReservation(
                reservation_id, owner_handle, tuple(selected_handles)
            )
            manifest_version = self._workspace.dataset(
                SNAPSHOT_MANIFEST_DATASET
            ).version
            created = []
            try:
                for handle in reservation.handles:
                    locator, _ = self._resolve_handle(handle)
                    expected = self._reservation_tag(
                        "reserved", reservation.reservation_id, locator
                    )
                    if expected not in names:
                        tags.create(expected, manifest_version)
                        created.append(expected)
            except BaseException as primary:
                run_cleanup_preserving_primary(
                    primary,
                    [
                        (
                            f"cleanup reservation tag {tag}",
                            lambda tag=tag: (
                                tags.delete(tag) if tag in tags.list() else None
                            ),
                        )
                        for tag in reversed(created)
                    ],
                    "cleanup reservation",
                )
                raise
            return reservation

    def commit_cleanup(self, reservation: CleanupReservation) -> None:
        self._require_mutation_allowed()
        self._validate_cleanup_reservation(reservation)
        with self._commit_lock():
            tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
            for handle in reservation.handles:
                locator, _ = self._resolve_handle(handle)
                reserved = self._reservation_tag(
                    "reserved", reservation.reservation_id, locator
                )
                pending = self._reservation_tag(
                    "pending", reservation.reservation_id, locator
                )
                names = tuple(tags.list())
                if pending in names:
                    continue
                if reserved not in names:
                    raise CheckpointConflictError(
                        "checkpoint reclaim reservation is missing"
                    )
                tags.create(
                    pending,
                    self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).version,
                )
                tags.delete(reserved)

    def rollback_cleanup(self, reservation: CleanupReservation) -> None:
        self._require_mutation_allowed()
        self._validate_cleanup_reservation(reservation)
        with self._commit_lock():
            tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
            for handle in reservation.handles:
                locator, _ = self._resolve_handle(handle)
                for status in ("reserved", "pending"):
                    tag = self._reservation_tag(
                        status, reservation.reservation_id, locator
                    )
                    if tag in tags.list():
                        tags.delete(tag)

    def cleanup_checkpoint(
        self,
        reservation: CleanupReservation,
        handle: CheckpointHandle,
    ) -> None:
        self._require_mutation_allowed()
        if handle not in reservation.handles:
            raise ValueError("checkpoint handle is outside reclaim reservation")
        self._validate_cleanup_reservation(reservation)
        storage_uri, locator, reference_id = _decode_checkpoint_handle(handle)
        self._validate_handle_storage_uri(storage_uri)
        expected_owner = self._reservation_owner_handle(
            reservation.reservation_id, reservation.handles
        )
        if reservation.owner_handle != expected_owner:
            raise CheckpointConflictError("cleanup reservation owner handle is invalid")
        with self._commit_lock():
            tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
            pending = self._reservation_tag(
                "pending", reservation.reservation_id, locator
            )
            if pending not in tags.list():
                if self._reclaim_completed(reservation, locator):
                    return
                raise CheckpointConflictError(
                    "checkpoint reclaim reservation is not pending"
                )
            self._delete_reference_tags(locator, reference_id)
            self._reclaim_unlocked_for(timedelta(0), frozenset({locator}))
            self._record_reclaim_completion(reservation, locator)
            self._delete_pending_outcome(locator.checkpoint_name)
            if pending in tags.list():
                tags.delete(pending)

    def release_reference(self, handle: CheckpointHandle) -> None:
        self._require_mutation_allowed()
        with self._commit_lock():
            locator, reference_id = self._resolve_handle(handle)
            self._delete_reference_tags(locator, reference_id)
            self._reclaim_unlocked(timedelta(0))

    def _validate_cleanup_reservation(self, reservation: CleanupReservation) -> None:
        expected = self._reservation_owner_handle(
            reservation.reservation_id, reservation.handles
        )
        if reservation.owner_handle != expected:
            raise CheckpointConflictError("cleanup reservation owner handle is invalid")

    @staticmethod
    def _decode_restore_plan(restore: bytes) -> RestorePlan:
        if not isinstance(restore, bytes) or not restore:
            raise TypeError("restore options must be non-empty bytes")
        return RestorePlan.from_bytes(restore)

    @staticmethod
    def _execute_guarded_restore(store: Any, action: Callable[[], None]) -> None:
        try:
            action()
        except BaseException as primary:
            run_cleanup_preserving_primary(
                primary,
                [
                    ("store clear", store.clear),
                    ("store terminal failure", store.mark_terminal_failure),
                ],
                "checkpoint restore",
            )
            raise

    def load_component_checkpoint(
        self,
        store: Any,
        storage_locator: CheckpointHandle,
        restore: bytes,
        *,
        load_optimizer_state: bool = True,
    ) -> None:
        self._require_mutation_allowed()
        self._ensure_tables()
        with self._commit_lock():
            if not isinstance(load_optimizer_state, bool):
                raise TypeError("load_optimizer_state must be bool")
            plan = self._decode_restore_plan(restore)
            locator, _ = self._resolve_handle(storage_locator)
            manifest = self._resolve_manifest(locator)
            prepared = self._prepare_restore(
                store, manifest, plan, load_optimizer_state=load_optimizer_state
            )

            def _restore() -> None:
                self._restore_entries(store, prepared, plan, load_optimizer_state)
                if load_optimizer_state:
                    self._initialize_restore_origin(store, prepared, plan)

            self._execute_guarded_restore(store, _restore)

    def _initialize_restore_origin(
        self,
        store: Any,
        prepared: _PreparedRestore,
        plan: RestorePlan,
    ) -> None:
        chain = prepared.chain
        if not chain:
            raise RuntimeError("validated restore chain is empty")
        manifest = chain[-1]
        target = plan.target
        if prepared.preserves_source_ownership:
            participant_index = target.partition_id
            participant_rec = self._validated_manifest_participants(
                manifest
            ).record_for(participant_index)
            store.initialize_checkpoint_baseline(
                self._checkpoint_baseline(manifest, participant_index),
                participant_rec.watermark,
            )
            return
        baseline, watermark = self._target_restore_origin(chain, plan)
        store.initialize_checkpoint_baseline(baseline, watermark, resharded=True)

    @staticmethod
    def _target_restore_origin(
        chain: Sequence[Mapping[str, Any]],
        plan: RestorePlan,
    ) -> tuple[str, int]:
        """Start a target baseline when historical source ownership changes.

        Equal participant counts can still change ownership. Keep the complete
        validated chain and assignment in the origin and continue sequences above
        every saved participant cut; subsequent saves must start with a FULL baseline.
        """
        if not chain:
            raise RuntimeError("validated restore chain is empty")
        manifest = chain[-1]
        watermarks = tuple(
            _strict_integer(value, "participant watermark")
            for value in manifest["participant_watermarks"]
        )
        if not watermarks:
            raise RuntimeError("checkpoint manifest has no participant watermarks")
        baseline = _canonical_digest(
            [
                {
                    "manifest_chain": [
                        {
                            "checkpoint_name": str(item["checkpoint_name"]),
                            "checkpoint_version": _strict_integer(
                                item["checkpoint_version"],
                                "checkpoint_version",
                                minimum=1,
                            ),
                            "schema_digest": str(item["schema_digest"]),
                            "fragment_ids": tuple(
                                str(fragment_id) for fragment_id in item["fragment_ids"]
                            ),
                            "participant_watermarks": tuple(
                                _strict_integer(value, "participant watermark")
                                for value in item["participant_watermarks"]
                            ),
                            "checkpoint_mode": str(item["checkpoint_mode"]),
                        }
                        for item in chain
                    ],
                    "assignment_name": plan.target.spec.name,
                    "assignment_parameters": dict(plan.target.spec.parameters),
                    "partition_count": plan.target.partition_count,
                    "partition_id": plan.target.partition_id,
                    "key_dtype": plan.key_dtype,
                }
            ]
        )
        return baseline, max(watermarks)

    def load_inference_weights(
        self,
        store: Any,
        restore: bytes,
    ) -> None:
        self._require_mutation_allowed()
        if self._bound_handle is None:
            raise RuntimeError(
                "published weights require LocalLanceStorage.from_published_weight"
            )
        if not hasattr(store, "optimizer_enabled") or store.optimizer_enabled():
            raise ValueError(
                "published weights restore requires a target store with optimizer enabled=False"
            )
        self.load_component_checkpoint(
            store,
            self._bound_handle,
            restore,
            load_optimizer_state=False,
        )

    def publish_weight_handle(
        self,
        path: str | Path,
        handle: CheckpointHandle,
    ) -> None:
        self._require_mutation_allowed()
        with self._commit_lock():
            locator, _ = self._resolve_handle(handle)
            if not isinstance(self._workspace, LocalLanceWorkspace):
                raise NotImplementedError(
                    "published weight handle writing is only implemented for local Lance paths"
                )
            manifest = self._resolve_manifest(locator)
            if not str(manifest["reference_id"]):
                raise CheckpointConflictError(
                    "publishing weight handle requires a committed checkpoint reference"
                )
            PublishedWeightHandleAsset.publish(path, handle)

    def read_published_weight_handle(self, path: str | Path) -> CheckpointHandle:
        return PublishedWeightHandleAsset.read(path)

    def _restore_committed_record(
        self,
        store: Any,
        manifest: dict[str, Any],
        plan: RestorePlan,
        *,
        load_optimizer_state: bool,
    ) -> tuple[dict[str, Any], ...]:
        """Validate and restore a committed manifest chain into an empty target."""
        prepared = self._prepare_restore(
            store, manifest, plan, load_optimizer_state=load_optimizer_state
        )
        self._restore_entries(store, prepared, plan, load_optimizer_state)
        return prepared.chain

    def _prepare_restore(
        self,
        store: Any,
        manifest: dict[str, Any],
        plan: RestorePlan,
        *,
        load_optimizer_state: bool,
    ) -> _PreparedRestore:
        """Validate restore inputs before any target mutation is permitted."""
        specification = self._manifest_specification(manifest)
        if manifest["schema_digest"] != _schema_digest(specification):
            raise RuntimeError("checkpoint schema digest mismatch")
        if not hasattr(store, "optimizer_enabled") or not callable(
            store.optimizer_enabled
        ):
            raise TypeError("restore target must expose optimizer_enabled()")
        if bool(store.optimizer_enabled()) != load_optimizer_state:
            if load_optimizer_state:
                raise ValueError(
                    "load_optimizer_state requires a target store with optimizer enabled"
                )
            raise ValueError(
                "load_optimizer_state=False requires a target store with optimizer disabled"
            )
        if plan.key_dtype != specification.key_dtype:
            raise ValueError(
                "restore options sparse ID dtype does not match checkpoint"
            )
        self._validate_restore_target(
            store, specification, load_optimizer_state=load_optimizer_state
        )
        if manifest["commit_state"] != _COMMITTED:
            raise LookupError("checkpoint manifest is not committed")
        if manifest["storage_mode"] != _FRAGMENT_STORAGE_MODE:
            raise RuntimeError("checkpoint storage mode is unsupported")
        chain = self._manifest_chain(manifest)
        for chain_manifest in chain:
            self._validate_manifest_fragments(chain_manifest)
        preserves_source_ownership = self._validate_chain_entries(chain, plan.target)
        if store.total_size() != 0:
            raise RuntimeError("restore requires an empty target store")
        if not hasattr(store, "checkpoint_restore_is_pristine") or not callable(
            store.checkpoint_restore_is_pristine
        ):
            raise TypeError(
                "restore target must expose checkpoint_restore_is_pristine()"
            )
        if not store.checkpoint_restore_is_pristine():
            raise RuntimeError("restore requires a pristine checkpoint journal")
        if not hasattr(_checkpoint_native, "restore_native_partition_entries"):
            raise RuntimeError(
                "native checkpoint restore requires a Lance-enabled _core extension"
            )
        participant_rec = None
        if plan.engine is IngestionEngine.FAST_PAGE_CACHE:
            participant_rec = self._page_restore_participant(
                manifest, plan.target, preserves_source_ownership
            )
            if store.entries_per_page != participant_rec.page_capacity:
                raise ValueError("page-image geometry does not match store topology")
            if not bool(manifest["include_page_images"]):
                raise FileNotFoundError(
                    "checkpoint does not contain page_images cache for page-image materialization; "
                    "request IngestionEngine.RECONSTRUCT_ENTRIES to reconstruct from canonical entries"
                )
            if not hasattr(_checkpoint_native, "restore_native_page_images"):
                raise RuntimeError(
                    "native checkpoint restore requires a Lance-enabled _core extension"
                )
        return _PreparedRestore(chain, preserves_source_ownership, participant_rec)

    def _page_restore_participant(
        self,
        manifest: Mapping[str, Any],
        target: PartitionTarget,
        preserves_source_ownership: bool,
    ) -> ManifestParticipantRecord:
        saved_count = _strict_integer(
            manifest["participant_count"], "manifest participant_count", minimum=1
        )
        if target.partition_count != saved_count:
            raise ValueError(
                "page-image materialization requires the saved partition topology"
            )
        if not preserves_source_ownership:
            raise ValueError(
                "page-image materialization requires unchanged source partition ownership"
            )
        return self._validated_manifest_participants(manifest).record_for(
            target.partition_id
        )

    @classmethod
    def _validate_restore_target(
        cls,
        store: Any,
        specification: TableSpecification,
        *,
        load_optimizer_state: bool = True,
    ) -> None:
        target = cls._store_specification(store)
        if (
            target.dimensions != specification.dimensions
            or target.embedding_dtype != specification.embedding_dtype
            or target.score_policy != specification.score_policy
            or target.key_dtype != specification.key_dtype
            or (
                load_optimizer_state
                and target.optimizer_dtype != specification.optimizer_dtype
            )
        ):
            raise ValueError("restore target does not match checkpoint specification")

    def _ensure_tables(self) -> None:
        self._workspace.ensure_root()
        for name, schema in ((SNAPSHOT_MANIFEST_DATASET, MANIFEST_SCHEMA),):
            if not self._workspace.exists(name):
                self._workspace.write_dataset(_empty_table(schema), name, mode="create")
            elif self._workspace.dataset(name).schema != schema:
                raise RuntimeError(
                    f"checkpoint dataset {name!r} does not match protocol schema"
                )

    def _publish_authority(self, identity: CheckpointSaveIdentity) -> None:
        dataset = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET)
        tags = dataset.tags
        names = tuple(tags.list())
        current_generation = self._current_authority_generation(identity, names)
        incoming_generation = identity.commit_authority.generation
        if incoming_generation < current_generation:
            raise CheckpointConflictError("checkpoint commit authority is stale")
        fence_tag = self._fence_tag(identity)
        if incoming_generation > current_generation and fence_tag not in names:
            tags.create(fence_tag, dataset.version)
        for name in names:
            if name.startswith(self._fence_prefix(identity)) and name != fence_tag:
                tags.delete(name)
            if name.startswith(self._authority_scope_prefix(identity)):
                generation = self._authority_generation_from_tag(identity, name)
                if generation < incoming_generation:
                    tags.delete(name)
                elif generation > incoming_generation:
                    raise RuntimeError("checkpoint authority tag is ahead of its fence")
            if name.startswith(
                f"setransaction.{self._short_token(identity.checkpoint_name)}."
            ):
                parts = name.split(".")
                if len(parts) == 5:
                    gen = _serialized_integer(parts[2], "authority_generation")
                    if gen < incoming_generation:
                        tags.delete(name)
        names = tuple(tags.list())
        authority_tag = self._authority_tag(identity)
        checkpoint_prefix = self._authority_checkpoint_prefix(identity)
        conflicting = tuple(
            name
            for name in names
            if name.startswith(checkpoint_prefix) and name != authority_tag
        )
        if conflicting:
            raise CheckpointConflictError(
                "checkpoint name already has another commit authority"
            )
        if authority_tag not in names:
            tags.create(authority_tag, dataset.version)
        self._reconcile_staging()

    def _require_authority(self, identity: CheckpointSaveIdentity) -> None:
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list()
        if (
            self._current_authority_generation(identity, tuple(tags))
            != identity.commit_authority.generation
            or self._authority_tag(identity) not in tags
        ):
            raise CheckpointConflictError("checkpoint commit authority is stale")

    def _parent_head_tag(
        self, identity: CheckpointSaveIdentity, parent: SparseCheckpointLocator
    ) -> str:
        return ".".join(
            (
                "separenthead",
                self._token(parent.checkpoint_name),
                str(parent.checkpoint_version),
                self._token(self._attempt_prefix(identity)),
            )
        )

    def _parent_head_prefix(self, parent: SparseCheckpointLocator) -> str:
        return (
            f"separenthead.{self._token(parent.checkpoint_name)}."
            f"{parent.checkpoint_version}."
        )

    def _reserve_incremental_parent_head(
        self, identity: CheckpointSaveIdentity, parent: SparseCheckpointLocator
    ) -> None:
        if parent.checkpoint_name != identity.checkpoint_name:
            raise CheckpointConflictError(
                "incremental parent belongs to another checkpoint lineage"
            )
        with self._commit_lock():
            self._require_authority(identity)
            tags_list = tuple(
                self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list()
            )
            self._assert_incremental_parent_head_locked(
                identity,
                parent,
                require_reservation=False,
                existing_tags=tags_list,
            )
            own = self._parent_head_tag(identity, parent)
            if own not in tags_list:
                self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.create(
                    own, self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).version
                )

    def _assert_incremental_parent_head_locked(
        self,
        identity: CheckpointSaveIdentity,
        parent: SparseCheckpointLocator,
        *,
        require_reservation: bool = True,
        existing_tags: Sequence[str] | None = None,
    ) -> None:
        tags_list = (
            existing_tags
            if existing_tags is not None
            else tuple(self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list())
        )
        tags_set = set(tags_list)
        committed = [
            row
            for row in self._manifest_rows()
            if row["checkpoint_name"] == identity.checkpoint_name
            and self._is_acknowledged_committed_row(row, tags_set)
        ]
        if not committed:
            raise CheckpointConflictError(
                "incremental checkpoint has no committed lineage head"
            )
        head_version = max(
            _strict_integer(row["checkpoint_version"], "checkpoint_version", minimum=1)
            for row in committed
        )
        if head_version != parent.checkpoint_version:
            raise CheckpointConflictError(
                "incremental parent is not the committed lineage head"
            )
        prefix = self._parent_head_prefix(parent)
        own = self._parent_head_tag(identity, parent)
        reservations = tuple(tag for tag in tags_list if tag.startswith(prefix))
        if any(tag != own for tag in reservations):
            raise CheckpointConflictError(
                "incremental parent already has another child reservation"
            )
        if require_reservation and own not in tags_list:
            raise CheckpointConflictError("incremental parent reservation is missing")

    def _release_parent_head_reservations(
        self, identity: CheckpointSaveIdentity
    ) -> None:
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        suffix = f".{self._token(self._attempt_prefix(identity))}"
        for tag in tuple(tags.list()):
            if tag.startswith("separenthead.") and tag.endswith(suffix):
                tags.delete(tag)

    def _page_protection_tag(
        self, locator: SparseCheckpointLocator, participant_index: int
    ) -> str:
        if participant_index < 0:
            raise ValueError("page participant index must be non-negative")
        return ".".join(
            (
                _PAGE_PROTECTION_TAG_PREFIX,
                self._token(locator.checkpoint_name),
                str(locator.checkpoint_version),
                str(participant_index),
            )
        )

    def _page_attempt_tag(
        self,
        identity: CheckpointSaveIdentity,
        participant_index: int,
        page_version: int,
    ) -> str:
        if participant_index < 0 or page_version <= 0:
            raise ValueError("page attempt identity is invalid")
        return ".".join(
            (
                _PAGE_ATTEMPT_TAG_PREFIX,
                self._token(self._attempt_prefix(identity)),
                str(page_version),
                str(participant_index),
            )
        )

    def _page_write_intent_tag(
        self,
        identity: CheckpointSaveIdentity,
        participant_index: int,
        parent_version: int,
    ) -> str:
        if participant_index < 0 or parent_version <= 0:
            raise ValueError("page write intent identity is invalid")
        return ".".join(
            (
                _PAGE_WRITE_INTENT_TAG_PREFIX,
                self._token(self._attempt_prefix(identity)),
                str(participant_index),
                str(parent_version),
            )
        )

    def _create_page_write_intent(
        self,
        identity: CheckpointSaveIdentity,
        participant_index: int,
        page_dataset: str,
        parent_version: int,
    ) -> tuple[str, str]:
        tag = self._page_write_intent_tag(identity, participant_index, parent_version)
        with self._commit_lock():
            dataset = self._workspace.dataset(page_dataset, version=parent_version)
            tags = dataset.tags
            if tag not in tags.list():
                tags.create(tag, parent_version)
        return page_dataset, tag

    def _delete_page_write_intent(self, page_dataset: str, tag: str) -> None:
        with self._commit_lock():
            if not self._workspace.exists(page_dataset):
                return
            tags = self._workspace.dataset(page_dataset).tags
            if tag in tags.list():
                tags.delete(tag)

    def _release_page_write_intents(
        self, identity: CheckpointSaveIdentity, page_datasets: Sequence[str]
    ) -> None:
        attempt_token = self._token(self._attempt_prefix(identity))
        for page_dataset in set(page_datasets):
            if not page_dataset or not self._workspace.exists(page_dataset):
                continue
            tags = self._workspace.dataset(page_dataset).tags
            for tag in tuple(tags.list()):
                if not tag.startswith(f"{_PAGE_WRITE_INTENT_TAG_PREFIX}."):
                    continue
                if self._page_write_intent_attempt(tag) == attempt_token:
                    tags.delete(tag)

    @staticmethod
    def _page_write_intent_attempt(tag: str) -> str:
        return _PageWriteIntentTag.parse(tag).attempt_token

    def _protect_page_attempt(
        self,
        identity: CheckpointSaveIdentity,
        participant_index: int,
        page_dataset: str,
        page_version: int,
    ) -> tuple[str, str, int]:
        tag = self._page_attempt_tag(identity, participant_index, page_version)
        with self._commit_lock():
            dataset = self._workspace.dataset(page_dataset, version=page_version)
            if tag not in dataset.tags.list():
                dataset.tags.create(tag, page_version)
        return page_dataset, tag, page_version

    def _delete_page_attempt_tag(
        self, page_dataset: str, tag: str, page_version: int
    ) -> None:
        del page_version
        with self._commit_lock():
            dataset = self._workspace.dataset(page_dataset)
            if tag in dataset.tags.list():
                dataset.tags.delete(tag)

    def _protect_manifest_pages(self, manifest: Mapping[str, Any]) -> None:
        if not bool(manifest["include_page_images"]):
            return
        locator = SparseCheckpointLocator(
            str(manifest["checkpoint_name"]),
            _strict_integer(
                manifest["checkpoint_version"], "checkpoint_version", minimum=1
            ),
        )
        for participant_index in range(
            _strict_integer(
                manifest["participant_count"], "participant_count", minimum=1
            )
        ):
            descriptor = self._manifest_fragment_descriptor(manifest, participant_index)
            page_dataset = str(descriptor["page_dataset"])
            page_version = _strict_integer(descriptor["page_version"], "page_version")
            if not page_dataset.startswith(f"{_PAGE_DATASET_ROOT}/"):
                continue
            if page_version <= 0:
                continue
            dataset = self._workspace.dataset(page_dataset, version=page_version)
            tag = self._page_protection_tag(locator, participant_index)
            if tag not in dataset.tags.list():
                dataset.tags.create(tag, page_version)

    def _release_page_attempt_tags(
        self, identity: CheckpointSaveIdentity, page_datasets: Sequence[str]
    ) -> None:
        attempt_token = self._token(self._attempt_prefix(identity))
        for page_dataset in set(page_datasets):
            if not page_dataset:
                continue
            if not self._workspace.exists(page_dataset):
                continue
            dataset = self._workspace.dataset(page_dataset)
            for tag in tuple(dataset.tags.list()):
                if tag.startswith(f"{_PAGE_ATTEMPT_TAG_PREFIX}.{attempt_token}."):
                    dataset.tags.delete(tag)

    def _page_dataset_relatives(self) -> tuple[str, ...]:
        if not self._workspace.exists(_PAGE_DATASET_ROOT):
            return ()
        pending = [_PAGE_DATASET_ROOT]
        visited: set[str] = set()
        datasets: list[str] = []
        while pending:
            prefix = pending.pop()
            if prefix in visited:
                continue
            visited.add(prefix)
            candidate = f"{prefix}/{PAGE_IMAGES_DATASET}"
            if self._workspace.exists(candidate):
                datasets.append(candidate)
                continue
            pending.extend(
                f"{prefix}/{child}" for child in self._workspace.child_prefixes(prefix)
            )
        return tuple(sorted(datasets))

    def _reconcile_page_protection(self, *, cleanup: bool) -> None:
        rows = self._manifest_rows()
        required: dict[str, dict[str, int]] = {}
        for row in rows:
            if row["commit_state"] not in {_PREPARED, _COMMITTED}:
                continue
            if not bool(row["include_page_images"]):
                continue
            locator = SparseCheckpointLocator(
                str(row["checkpoint_name"]),
                _strict_integer(
                    row["checkpoint_version"], "checkpoint_version", minimum=1
                ),
            )
            for participant_index in range(
                _strict_integer(
                    row["participant_count"], "participant_count", minimum=1
                )
            ):
                descriptor = self._manifest_fragment_descriptor(row, participant_index)
                page_dataset = str(descriptor["page_dataset"])
                page_version = _strict_integer(
                    descriptor["page_version"], "page_version"
                )
                if (
                    not page_dataset.startswith(f"{_PAGE_DATASET_ROOT}/")
                    or page_version <= 0
                ):
                    continue
                required.setdefault(page_dataset, {})[
                    self._page_protection_tag(locator, participant_index)
                ] = page_version

        referenced_attempts = set()
        for row in rows:
            if row["commit_state"] in {_PREPARED, _COMMITTED}:
                for root in self._manifest_fragment_roots(row).values():
                    if root.startswith("_staging/"):
                        _, attempt_id, *_ = root.split("/", 2)
                        referenced_attempts.add(attempt_id)
        staging_attempts = (
            set(self._workspace.child_prefixes("_staging"))
            if self._workspace.exists("_staging")
            else set()
        )
        if cleanup and staging_attempts.difference(referenced_attempts):
            return

        for page_dataset in self._page_dataset_relatives():
            dataset = self._workspace.dataset(page_dataset)
            tags = dataset.tags
            current = tuple(tags.list())
            for tag, version in required.get(page_dataset, {}).items():
                if tag not in current:
                    tags.create(tag, version)
            active_attempts = {
                self._token(
                    self._attempt_prefix(
                        CheckpointSaveIdentity(
                            str(row["checkpoint_name"]),
                            str(row["save_attempt"]),
                            CheckpointCommitAuthority(
                                str(row["authority_scope"]),
                                _strict_integer(
                                    row["authority_generation"], "authority_generation"
                                ),
                                str(row["commit_authority"]),
                            ),
                        )
                    )
                )
                for row in rows
                if row["commit_state"] == _PREPARED
            }
            if self._workspace.exists("_staging"):
                active_attempts.update(
                    self._token(f"_staging/{attempt}")
                    for attempt in self._workspace.child_prefixes("_staging")
                )
            for tag in tuple(tags.list()):
                if tag.startswith(f"{_PAGE_PROTECTION_TAG_PREFIX}."):
                    if tag not in required.get(page_dataset, {}):
                        tags.delete(tag)
                    continue
                if tag.startswith(f"{_PAGE_WRITE_INTENT_TAG_PREFIX}."):
                    parsed_intent = _PageWriteIntentTag.parse(tag)
                    if parsed_intent.attempt_token not in active_attempts:
                        tags.delete(tag)
                    continue
                if not tag.startswith(f"{_PAGE_ATTEMPT_TAG_PREFIX}."):
                    continue
                parsed_attempt = _PageAttemptTag.parse(tag)
                if parsed_attempt.attempt_token not in active_attempts:
                    tags.delete(tag)
            if not cleanup:
                continue
            remaining = tuple(tags.list())
            if any(
                tag.startswith(
                    (
                        f"{_PAGE_WRITE_INTENT_TAG_PREFIX}.",
                        f"{_PAGE_ATTEMPT_TAG_PREFIX}.",
                    )
                )
                for tag in remaining
            ):
                continue
            has_protection = any(
                tag.startswith(
                    (
                        f"{_PAGE_PROTECTION_TAG_PREFIX}.",
                        f"{_PAGE_ATTEMPT_TAG_PREFIX}.",
                    )
                )
                for tag in remaining
            )
            if not has_protection and page_dataset not in required:
                self._workspace.delete_prefix(page_dataset, missing_ok=True)
                continue
            required_versions = tuple(required.get(page_dataset, {}).values())
            protected_versions = set(required_versions)
            for tag in dataset.tags.list():
                if tag.startswith(f"{_PAGE_ATTEMPT_TAG_PREFIX}."):
                    protected_versions.add(_PageAttemptTag.parse(tag).page_version)
            referenced_files: set[str] = set()
            for version in protected_versions:
                version_dataset = self._workspace.dataset(page_dataset, version=version)
                for fragment in version_dataset.get_fragments():
                    referenced_files.update(
                        Path(str(data_file.path)).name
                        for data_file in fragment.data_files()
                    )
            latest_files = {
                Path(str(data_file.path)).name
                for fragment in dataset.get_fragments()
                for data_file in fragment.data_files()
            }
            if (
                required_versions
                and dataset.version > max(required_versions)
                and latest_files.difference(referenced_files)
            ):
                dataset.checkout_version(max(required_versions)).restore()
                dataset = self._workspace.dataset(page_dataset)
            dataset.cleanup_old_versions(
                older_than=timedelta(0),
                delete_unverified=True,
                error_if_tagged_old_versions=False,
            )
            data_prefix = f"{page_dataset}/data"
            for file_name in self._workspace.child_files(data_prefix):
                if file_name not in referenced_files:
                    self._workspace.delete_prefix(
                        f"{data_prefix}/{file_name}", missing_ok=True
                    )

    def _clear_authority(self, identity: CheckpointSaveIdentity) -> None:
        if not self._workspace.exists(SNAPSHOT_MANIFEST_DATASET):
            return
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        name = self._authority_tag(identity)
        if name in tags.list():
            tags.delete(name)

    def _reconcile_staging(self, *, cleanup: bool = False) -> None:
        if not cleanup:
            return
        self._require_exclusive_recovery()
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list()
        active = {
            tag.removeprefix("seauthority.")
            for tag in tags
            if tag.startswith("seauthority.")
        }
        referenced = set()
        for row in self._manifest_rows():
            if row["commit_state"] in {_PREPARED, _COMMITTED}:
                for root in self._manifest_fragment_roots(row).values():
                    if root.startswith("_staging/"):
                        _, attempt_id, *_ = root.split("/", 2)
                        referenced.add(attempt_id)
        staging_attempts = (
            self._workspace.child_prefixes("_staging")
            if self._workspace.exists("_staging")
            else ()
        )
        if cleanup:
            referenced_orphans = set(referenced)
            for tag in tuple(tags):
                if not tag.startswith("seauthority."):
                    continue
                attempt = tag.removeprefix("seauthority.")
                if attempt not in referenced_orphans:
                    self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.delete(tag)
            for attempt in staging_attempts:
                if attempt not in active and attempt not in referenced:
                    self._workspace.delete_prefix(
                        f"_staging/{attempt}", missing_ok=True
                    )
        live_attempt_tokens = {
            self._token(f"_staging/{attempt}")
            for attempt in (
                self._workspace.child_prefixes("_staging")
                if self._workspace.exists("_staging")
                else ()
            )
        }
        manifest_dataset = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET)
        for tag in tuple(manifest_dataset.tags.list()):
            if not tag.startswith("separenthead."):
                continue
            parsed_parent = _ParentHeadReservationTag.parse(tag)
            if cleanup and parsed_parent.attempt_token not in live_attempt_tokens:
                manifest_dataset.tags.delete(tag)

    def _create_lease_tags(
        self,
        lease: CheckpointLease,
        locator: SparseCheckpointLocator,
        manifest: dict[str, Any],
    ) -> None:
        self._create_version_tags(self._lease_tag(lease, locator), manifest)

    def _create_reference_tags(
        self,
        locator: SparseCheckpointLocator,
        reference_id: str,
        manifest: dict[str, Any],
    ) -> None:
        if not reference_id:
            raise ValueError("reference_id must be non-empty")
        self._create_version_tags(self._reference_tag(locator, reference_id), manifest)

    def _create_version_tags(self, tag: str, manifest: dict[str, Any]) -> None:
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        if tag not in tags.list():
            tags.create(tag, self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).version)

    def _create_parent_chain_reference(self, manifest: Mapping[str, Any]) -> None:
        if manifest["checkpoint_mode"] == CheckpointSaveMode.FULL.value:
            return
        parent_name = str(manifest["parent_checkpoint_name"])
        parent_version = _strict_integer(
            manifest["parent_checkpoint_version"], "parent_checkpoint_version"
        )
        parent_reference = str(manifest["parent_reference_id"])
        if not parent_name or parent_version <= 0 or not parent_reference:
            raise RuntimeError("incremental parent identity is incomplete")
        parent = self._manifest_for_version(parent_name, parent_version)
        if str(parent["reference_id"]) != parent_reference:
            raise RuntimeError("checkpoint parent reference identity mismatch")
        child_locator = SparseCheckpointLocator(
            str(manifest["checkpoint_name"]),
            _strict_integer(
                manifest["checkpoint_version"], "checkpoint_version", minimum=1
            ),
        )
        self._create_reference_tags(
            SparseCheckpointLocator(parent_name, parent_version),
            f"chain:{child_locator.checkpoint_name}:{child_locator.checkpoint_version}",
            parent,
        )

    def _delete_parent_chain_reference(self, manifest: Mapping[str, Any]) -> None:
        if str(manifest["checkpoint_mode"]) != CheckpointSaveMode.INCREMENTAL.value:
            return
        parent_name = str(manifest["parent_checkpoint_name"])
        parent_version = _strict_integer(
            manifest["parent_checkpoint_version"], "parent_checkpoint_version"
        )
        if not parent_name or parent_version <= 0:
            raise RuntimeError("incremental parent identity is incomplete")
        self._delete_reference_tags(
            SparseCheckpointLocator(parent_name, parent_version),
            f"chain:{manifest['checkpoint_name']}:{manifest['checkpoint_version']}",
        )

    def _delete_reference_tags(
        self, locator: SparseCheckpointLocator, reference_id: str
    ) -> None:
        tag = self._reference_tag(locator, reference_id)
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        if tag in tags.list():
            tags.delete(tag)

    def _delete_reference_tags_with_lock(
        self, locator: SparseCheckpointLocator, reference_id: str
    ) -> None:
        with self._commit_lock():
            self._delete_reference_tags(locator, reference_id)

    @staticmethod
    def _token(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    def _authority_tag(self, identity: CheckpointSaveIdentity) -> str:
        authority = self._token(
            f"{identity.save_attempt}\0{identity.commit_authority.authority_id}"
        )
        return ".".join(
            (
                self._authority_checkpoint_prefix(identity).rstrip("."),
                str(identity.commit_authority.generation),
                authority,
            )
        )

    def _authority_scope_prefix(self, identity: CheckpointSaveIdentity) -> str:
        return f"seauthority.{self._token(identity.commit_authority.scope)}."

    def _authority_checkpoint_prefix(self, identity: CheckpointSaveIdentity) -> str:
        return (
            self._authority_scope_prefix(identity)
            + f"{self._token(identity.checkpoint_name)}."
        )

    def _fence_prefix(self, identity: CheckpointSaveIdentity) -> str:
        return f"sefence.{self._token(identity.commit_authority.scope)}."

    def _fence_tag(self, identity: CheckpointSaveIdentity) -> str:
        return self._fence_prefix(identity) + str(identity.commit_authority.generation)

    def _current_authority_generation(
        self, identity: CheckpointSaveIdentity, tags: tuple[str, ...]
    ) -> int:
        prefix = self._fence_prefix(identity)
        generations = []
        for tag in tags:
            if not tag.startswith(prefix):
                continue
            raw_generation = tag.removeprefix(prefix)
            try:
                generations.append(
                    _serialized_integer(
                        raw_generation,
                        "authority generation",
                        maximum=(1 << 64) - 1,
                    )
                )
            except ValueError as error:
                raise RuntimeError(
                    "checkpoint authority fence tag is corrupt"
                ) from error
        return max(generations, default=-1)

    def _authority_generation_from_tag(
        self, identity: CheckpointSaveIdentity, tag: str
    ) -> int:
        suffix = tag.removeprefix(self._authority_scope_prefix(identity))
        return _AuthorityTag.parse_suffix(suffix).generation

    @staticmethod
    def _encode(value: str) -> str:
        return urlsafe_b64encode(value.encode()).decode().rstrip("=")

    @staticmethod
    def _decode(value: str) -> str:
        return urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()

    def _lease_tag(
        self, lease: CheckpointLease, locator: SparseCheckpointLocator
    ) -> str:
        expires = int(lease.expires_at.timestamp() * 1_000_000)
        return ".".join(
            (
                "selease",
                lease.lease_id,
                str(lease.generation),
                str(expires),
                str(locator.checkpoint_version),
                self._encode(locator.checkpoint_name),
                self._encode(lease.lease_holder_id),
                self._encode(lease.request_id),
            )
        )

    def _decode_lease_tag(self, tag: str) -> _LeaseRecord:
        record = _LeaseTag.parse(tag, self._decode)
        lease = CheckpointLease(
            lease_id=record.lease_id,
            lease_holder_id=record.holder_id,
            expires_at=datetime.fromtimestamp(
                record.expires_at_micros / 1_000_000, UTC
            ),
            generation=record.generation,
            request_id=record.request_id,
        )
        return _LeaseRecord(
            lease,
            SparseCheckpointLocator(record.checkpoint_name, record.checkpoint_version),
        )

    def _reference_tag(
        self, locator: SparseCheckpointLocator, reference_id: str
    ) -> str:
        return ".".join(
            (
                "sereference",
                self._token(reference_id),
                str(locator.checkpoint_version),
                self._encode(locator.checkpoint_name),
            )
        )

    def _reservation_tag(
        self,
        status: str,
        reservation_id: str,
        locator: SparseCheckpointLocator,
    ) -> str:
        if status not in {"reserved", "pending"}:
            raise ValueError("unsupported reclaim reservation status")
        return ".".join(
            (
                f"se{status}",
                self._token(reservation_id),
                str(locator.checkpoint_version),
                self._encode(locator.checkpoint_name),
            )
        )

    @staticmethod
    def _reservation_owner_handle(
        reservation_id: str, handles: Sequence[CheckpointHandle]
    ) -> CheckpointHandle:
        if not reservation_id or any(
            not isinstance(handle, bytes) or not handle for handle in handles
        ):
            raise ValueError("cleanup reservation identity is invalid")
        digest = hashlib.sha256()
        digest.update(b"sparse_embedding.cleanup_owner\0")
        digest.update(reservation_id.encode("utf-8"))
        for handle in handles:
            digest.update(_UINT32.pack(len(handle)))
            digest.update(handle)
        return digest.digest()

    def _outcome_tag(
        self,
        checkpoint_name: str,
        checkpoint_generation: int,
        checkpoint_version: int,
        manifest_before: int,
    ) -> str:
        return ".".join(
            (
                "setransaction",
                self._short_token(checkpoint_name),
                str(checkpoint_generation),
                str(checkpoint_version),
                str(manifest_before),
            )
        )

    @staticmethod
    def _short_token(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()[:32]

    def _pending_outcome(
        self,
        identity: CheckpointSaveIdentity,
        *,
        require_identity: bool = True,
    ) -> _CommitOutcome | None:
        prefix = f"setransaction.{self._short_token(identity.checkpoint_name)}."
        tags = tuple(
            tag
            for tag in self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list()
            if tag.startswith(prefix)
        )
        if not tags:
            return None
        if len(tags) != 1:
            raise ValueError("durable checkpoint commit outcome is ambiguous")
        (tag,) = tags
        parts = tag.split(".")
        if len(parts) != 5:
            raise ValueError("durable checkpoint commit outcome is corrupt")
        (
            _,
            checkpoint_token,
            authority_generation,
            version,
            manifest_before,
        ) = parts
        if checkpoint_token != self._short_token(identity.checkpoint_name):
            raise CheckpointConflictError(
                "durable checkpoint outcome belongs to another checkpoint"
            )
        try:
            decoded = {
                "authority_generation": _serialized_integer(
                    authority_generation, "authority_generation"
                ),
                "checkpoint_version": _serialized_integer(
                    version, "checkpoint_version", minimum=1
                ),
                "manifest_before": _serialized_integer(
                    manifest_before, "manifest_before"
                ),
            }
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("durable checkpoint commit outcome is corrupt") from error
        if decoded["authority_generation"] != identity.commit_authority.generation:
            if not require_identity:
                return None
            if decoded["authority_generation"] < identity.commit_authority.generation:
                return None
            raise CheckpointConflictError(
                "durable checkpoint outcome belongs to another authority generation"
            )
        manifests = [
            row
            for row in self._manifest_rows()
            if row["checkpoint_name"] == identity.checkpoint_name
            and _strict_integer(
                row["checkpoint_version"], "checkpoint_version", minimum=1
            )
            == decoded["checkpoint_version"]
        ]
        if len(manifests) != 1:
            raise CheckpointConflictError(
                "durable checkpoint outcome has no unique manifest"
            )
        (manifest,) = manifests
        expected_identity = (
            manifest["save_attempt"] == identity.save_attempt
            and manifest["authority_scope"] == identity.commit_authority.scope
            and _strict_integer(
                manifest["authority_generation"], "authority_generation"
            )
            == identity.commit_authority.generation
            and manifest["commit_authority"] == identity.commit_authority.authority_id
        )
        if not expected_identity:
            if not require_identity:
                return None
            raise CheckpointConflictError(
                "checkpoint name belongs to another durable commit outcome"
            )
        if decoded["checkpoint_version"] <= 0 or decoded["manifest_before"] <= 0:
            raise ValueError("durable checkpoint commit outcome is corrupt")
        return _CommitOutcome(
            locator=SparseCheckpointLocator(
                identity.checkpoint_name, decoded["checkpoint_version"]
            ),
            reference_id=manifest["reference_id"],
            manifest_before=cast(int, decoded["manifest_before"]),
        )

    def _record_pending_outcome(
        self,
        identity: CheckpointSaveIdentity,
        locator: SparseCheckpointLocator,
        manifest_before: int,
    ) -> None:
        tag = self._outcome_tag(
            identity.checkpoint_name,
            identity.commit_authority.generation,
            locator.checkpoint_version,
            manifest_before,
        )
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        if tag not in tags.list():
            tags.create(tag, self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).version)

    def _delete_pending_outcome(self, checkpoint_name: str) -> None:
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        prefix = f"setransaction.{self._short_token(checkpoint_name)}."
        for tag in tuple(tags.list()):
            if tag.startswith(prefix):
                tags.delete(tag)

    def _reclaim_completion_tag(
        self,
        reservation: CleanupReservation,
        locator: SparseCheckpointLocator,
    ) -> str:
        return ".".join(
            (
                "sereclaim",
                self._token(reservation.reservation_id),
                str(locator.checkpoint_version),
                self._encode(locator.checkpoint_name),
            )
        )

    def _reclaim_completed(
        self,
        reservation: CleanupReservation,
        locator: SparseCheckpointLocator,
    ) -> bool:
        tag = self._reclaim_completion_tag(reservation, locator)
        return tag in self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list()

    def _record_reclaim_completion(
        self,
        reservation: CleanupReservation,
        locator: SparseCheckpointLocator,
    ) -> None:
        if self._reclaim_completed(reservation, locator):
            return
        self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.create(
            self._reclaim_completion_tag(reservation, locator),
            self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).version,
        )

    @staticmethod
    def _is_reservation_tag(tag: str) -> bool:
        return tag.startswith(("sereserved.", "sepending."))

    def _reservation_locator_suffix(self, locator: SparseCheckpointLocator) -> str:
        return f".{locator.checkpoint_version}.{self._encode(locator.checkpoint_name)}"

    def _locator_has_reservation(self, locator: SparseCheckpointLocator) -> bool:
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list()
        suffix = self._reservation_locator_suffix(locator)
        return any(
            self._is_reservation_tag(tag) and tag.endswith(suffix) for tag in tags
        )

    def _publish_save_descriptor(self, root: str, fragment: CheckpointFragment) -> None:
        descriptor = self._fragment_descriptor(fragment)
        metadata = {
            "sparse_embedding.save_descriptor": json.dumps(descriptor, sort_keys=True)
        }
        for name in (ENTRIES_DATASET, PAGE_IMAGES_DATASET):
            if self._workspace.exists(f"{root}/{name}"):
                self._workspace.dataset(f"{root}/{name}").update_schema_metadata(
                    metadata, replace=False
                )

    @staticmethod
    def _fragment_descriptor(fragment: CheckpointFragment) -> dict[str, Any]:
        specification = fragment.specification
        context = fragment.save_context
        return {
            "fragment_id": fragment.fragment_id,
            "participant_indices": [fragment.participant.participant_index],
            "participant_count": fragment.participant.participant_count,
            "entry_count": fragment.entry_count,
            "page_count": fragment.page_count,
            "schema_digest": fragment.schema_digest,
            "content_digest": fragment.content_digest,
            "dimensions": list(specification.dimensions),
            "embedding_dtype": specification.embedding_dtype,
            "optimizer_dtype": specification.optimizer_dtype or "",
            "score_policy": specification.score_policy,
            "key_dtype": specification.key_dtype,
            "mode": context.mode.value,
            "parent_checkpoint_name": context.parent.checkpoint_name
            if context.parent
            else "",
            "parent_checkpoint_version": context.parent.checkpoint_version
            if context.parent
            else 0,
            "parent_reference_id": context.parent_reference_id,
            "mutation_watermark": context.mutation_watermark,
            "include_page_images": context.include_page_images,
            "entries_per_page": context.entries_per_page,
            "capture_id": context.capture_id,
            "journal_identity": context.journal_identity,
            "page_dataset": fragment.page_dataset,
            "page_version": fragment.page_version,
            "page_layout_generation": fragment.page_layout_generation,
        }

    def _validate_fragment_payload(
        self, root: str, descriptor: Mapping[str, Any]
    ) -> None:
        specification = self._manifest_specification(dict(descriptor))
        entries_count = _strict_integer(descriptor["entry_count"], "entry_count")
        entries_exists = self._workspace.exists(f"{root}/{ENTRIES_DATASET}")
        if entries_exists != (entries_count > 0):
            raise InvalidCheckpointFragment(
                "fragment entry payload presence does not match its descriptor"
            )
        if entries_exists:
            entries = self._workspace.dataset(f"{root}/{ENTRIES_DATASET}")
            entry_schema_value = entry_schema(specification.key_dtype)
            if (
                entries.count_rows() != entries_count
                or entries.schema.remove_metadata()
                != entry_schema_value.remove_metadata()
            ):
                raise InvalidCheckpointFragment(
                    "fragment entry payload schema or row count mismatch"
                )
            if self._decode_specification(entries.schema.metadata) != specification:
                raise InvalidCheckpointFragment(
                    "fragment entry payload specification mismatch"
                )
            stored = entries.schema_metadata
            if "sparse_embedding.save_descriptor" not in stored:
                raise InvalidCheckpointFragment("fragment save context is missing")
            stored_descriptor = json.loads(stored["sparse_embedding.save_descriptor"])
            if _canonical_digest([stored_descriptor]) != _canonical_digest(
                [dict(descriptor)]
            ):
                raise InvalidCheckpointFragment(
                    "fragment save context does not match its descriptor"
                )

        page_count = _strict_integer(descriptor["page_count"], "page_count")
        page_dataset = descriptor["page_dataset"]
        if not isinstance(page_dataset, str):
            raise InvalidCheckpointFragment("page_dataset must be a string")
        page_version = _strict_integer(descriptor["page_version"], "page_version")
        local_page_dataset = f"{root}/{PAGE_IMAGES_DATASET}"
        has_local_page_dataset = self._workspace.exists(local_page_dataset)
        if has_local_page_dataset and page_dataset != local_page_dataset:
            raise InvalidCheckpointFragment(
                "staging page payload is not declared by the fragment"
            )
        if page_dataset == local_page_dataset and not has_local_page_dataset:
            raise InvalidCheckpointFragment("declared compact page payload is missing")
        if page_count > 0 and (not page_dataset or page_version <= 0):
            raise InvalidCheckpointFragment(
                "fragment page payload has no lineage dataset version"
            )
        if page_version <= 0:
            return
        if not page_dataset:
            raise InvalidCheckpointFragment("page dataset identity is missing")
        pages = self._workspace.dataset(page_dataset, version=page_version)
        if (
            pages.count_rows() == 0
            or pages.schema.remove_metadata() != PAGE_IMAGE_SCHEMA.remove_metadata()
        ):
            raise InvalidCheckpointFragment("page dataset schema or rows are invalid")
        if self._decode_specification(pages.schema.metadata) != specification:
            raise InvalidCheckpointFragment("page dataset specification mismatch")

    def _validate_fragments(
        self, identity: CheckpointSaveIdentity, fragments: Sequence[CheckpointFragment]
    ) -> tuple[list[CheckpointFragment], TableSpecification]:
        if not fragments:
            raise InvalidCheckpointFragment(
                "checkpoint requires participant descriptors"
            )
        fragment_ids = tuple(item.fragment_id for item in fragments)
        if len(set(fragment_ids)) != len(fragment_ids):
            raise InvalidCheckpointFragment("checkpoint fragment IDs are duplicated")
        fragments_by_participant = {
            item.participant.participant_index: item for item in fragments
        }
        if len(fragments_by_participant) != len(fragments):
            raise InvalidCheckpointFragment(
                "checkpoint participant coverage is invalid"
            )
        leader_fragment = fragments_by_participant.get(0)
        if leader_fragment is None:
            raise InvalidCheckpointFragment("checkpoint participant zero is missing")
        expected_count = leader_fragment.participant.participant_count
        if set(fragments_by_participant) != set(range(expected_count)):
            raise InvalidCheckpointFragment("checkpoint fragment coverage is invalid")
        result = tuple(
            fragments_by_participant[participant_index]
            for participant_index in range(expected_count)
        )
        specification = leader_fragment.specification
        context = leader_fragment.save_context
        for item in result:
            if (
                item.checkpoint_name != identity.checkpoint_name
                or item.save_attempt != identity.save_attempt
            ):
                raise InvalidCheckpointFragment("checkpoint fragment identity mismatch")
            if (
                item.participant.participant_count != expected_count
                or item.specification != specification
            ):
                raise InvalidCheckpointFragment(
                    "checkpoint participant topology or specification mismatch"
                )
            candidate = item.save_context
            if (
                candidate.mode,
                candidate.parent,
                candidate.parent_reference_id,
                candidate.include_page_images,
            ) != (
                context.mode,
                context.parent,
                context.parent_reference_id,
                context.include_page_images,
            ):
                raise InvalidCheckpointFragment(
                    "checkpoint participant save contexts differ"
                )
            if item.schema_digest != _schema_digest(
                specification
            ) or item.fragment_id != self._fragment_id(identity, item.participant):
                raise InvalidCheckpointFragment(
                    "checkpoint fragment schema or identifier mismatch"
                )
            self._validate_fragment_payload(
                f"{self._attempt_prefix(identity)}/{item.fragment_id}",
                self._fragment_descriptor(item),
            )
        return result, specification

    def _manifest_rows(self) -> list[dict[str, Any]]:
        dataset = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET)
        current_version = dataset.version
        if (
            self._manifest_cache_version == current_version
            and self._manifest_cache_rows is not None
        ):
            return self._manifest_cache_rows
        if self._manifest_cache_version != current_version:
            self._participant_view_cache.clear()
        rows = dataset.to_table().to_pylist()
        self._manifest_cache_version = current_version
        self._manifest_cache_rows = rows
        return rows

    def _manifest_for_version(
        self, checkpoint_name: str, version: int
    ) -> dict[str, Any]:
        matches = [
            row
            for row in self._manifest_rows()
            if row["checkpoint_name"] == checkpoint_name
            and _strict_integer(
                row["checkpoint_version"], "checkpoint_version", minimum=1
            )
            == version
        ]
        if len(matches) != 1:
            raise LookupError("checkpoint manifest is missing or ambiguous")
        (match,) = matches
        return match

    def _resolve_manifest(self, locator: SparseCheckpointLocator) -> dict[str, Any]:
        matches = [
            row
            for row in self._manifest_rows()
            if row["checkpoint_name"] == locator.checkpoint_name
            and _strict_integer(
                row["checkpoint_version"], "checkpoint_version", minimum=1
            )
            == locator.checkpoint_version
        ]
        if len(matches) != 1:
            raise LookupError("checkpoint locator is missing or ambiguous")
        (match,) = matches
        if match["commit_state"] != _COMMITTED:
            raise LookupError("checkpoint manifest is not committed")
        return match

    def resolve_lineage_head(self, checkpoint_name: str) -> Mapping[str, Any] | None:
        if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
            raise ValueError("checkpoint_name must be a non-empty string")
        self._ensure_tables()
        with self._commit_lock():
            tags_set = set(
                self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list()
            )
            committed = [
                row
                for row in self._manifest_rows()
                if row["checkpoint_name"] == checkpoint_name
                and self._is_acknowledged_committed_row(row, tags_set)
            ]
            if not committed:
                return None
            latest_version = max(
                _strict_integer(
                    row["checkpoint_version"], "checkpoint_version", minimum=1
                )
                for row in committed
            )
            latest = [
                row
                for row in committed
                if _strict_integer(
                    row["checkpoint_version"], "checkpoint_version", minimum=1
                )
                == latest_version
            ]
            if len(latest) != 1:
                raise RuntimeError("checkpoint lineage head is ambiguous")
            (manifest,) = latest
            locator = SparseCheckpointLocator(checkpoint_name, latest_version)
            if self._resolve_manifest(locator) != manifest:
                raise RuntimeError("checkpoint lineage head changed during resolution")
            self._manifest_chain(manifest)
            return dict(manifest)

    def resolve_lineage_head_handle(
        self, checkpoint_name: str
    ) -> CheckpointHandle | None:
        manifest = self.resolve_lineage_head(checkpoint_name)
        if manifest is None:
            return None
        return _encode_checkpoint_handle(
            self._workspace.uri(""),
            SparseCheckpointLocator(
                checkpoint_name,
                _strict_integer(
                    manifest["checkpoint_version"], "checkpoint_version", minimum=1
                ),
            ),
            str(manifest["reference_id"]),
        )

    def _manifest_chain(self, manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        chain: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        current = manifest
        while True:
            checkpoint_name = str(current["checkpoint_name"])
            checkpoint_version = _strict_integer(
                current["checkpoint_version"], "checkpoint_version", minimum=1
            )
            identity = (checkpoint_name, checkpoint_version)
            if identity in seen:
                raise RuntimeError("checkpoint parent chain contains a cycle")
            seen.add(identity)
            if _strict_integer(current["format_version"], "format_version") != 3:
                raise RuntimeError("checkpoint manifest format version is unsupported")
            if current["schema_digest"] != manifest["schema_digest"]:
                raise RuntimeError("checkpoint parent chain schema mismatch")
            chain.append(current)
            mode = str(current["checkpoint_mode"])
            if mode == CheckpointSaveMode.FULL.value:
                if (
                    _strict_integer(
                        current["parent_checkpoint_version"],
                        "parent_checkpoint_version",
                    )
                    != 0
                ):
                    raise RuntimeError("FULL checkpoint cannot have a parent")
                break
            if mode != CheckpointSaveMode.INCREMENTAL.value:
                raise RuntimeError("checkpoint mode is unsupported")
            parent_name = str(current["parent_checkpoint_name"])
            parent_version = _strict_integer(
                current["parent_checkpoint_version"], "parent_checkpoint_version"
            )
            parent_reference = str(current["parent_reference_id"])
            if not parent_name or parent_version <= 0 or not parent_reference:
                raise RuntimeError(
                    "incremental checkpoint has incomplete parent identity"
                )
            parent = self._manifest_for_version(parent_name, parent_version)
            if parent_name != str(manifest["checkpoint_name"]):
                raise RuntimeError("checkpoint parent belongs to another lineage")
            if str(parent["reference_id"]) != parent_reference:
                raise RuntimeError("checkpoint parent reference identity mismatch")
            if parent_version >= checkpoint_version:
                raise RuntimeError("checkpoint parent version does not precede child")
            current = parent
        chain.reverse()
        return tuple(chain)

    def _validate_chain_entries(
        self, chain: Sequence[dict[str, Any]], target: PartitionTarget | None = None
    ) -> bool:
        *_, target_manifest = chain
        target_options: dict[str, Any] = {}
        if target is not None:
            target_options = {
                "partition_count": target.partition_count,
                "assignment_name": target.spec.name,
                "assignment_parameters": dict(target.spec.parameters),
            }
        try:
            return _checkpoint_native.validate_native_checkpoint_chain(
                [
                    [
                        self._workspace.uri(f"{root}/{ENTRIES_DATASET}")
                        if self._workspace.exists(f"{root}/{ENTRIES_DATASET}")
                        else ""
                        for root in self._participant_roots_in_native_order(manifest)
                    ]
                    for manifest in chain
                ],
                [manifest["participant_watermarks"] for manifest in chain],
                list(target_manifest["dimensions"]),
                str(target_manifest["key_dtype"]),
                dict(self._workspace.storage_options()),
                **target_options,
            )
        except ValueError as error:
            raise InvalidCheckpointFragment(str(error)) from error

    def _manifest_fragment_roots(self, manifest: Mapping[str, Any]) -> dict[str, str]:
        raw_fragment_ids = manifest["fragment_ids"]
        if (
            not isinstance(raw_fragment_ids, Sequence)
            or isinstance(raw_fragment_ids, (str, bytes))
            or any(not isinstance(item, str) or not item for item in raw_fragment_ids)
        ):
            raise RuntimeError("checkpoint manifest fragment IDs are invalid")
        fragment_ids = tuple(raw_fragment_ids)
        if not fragment_ids:
            raise RuntimeError("checkpoint manifest has no fragments")
        if len(set(fragment_ids)) != len(fragment_ids):
            raise RuntimeError("checkpoint manifest has duplicate fragment IDs")
        if any(
            len(fragment_id) != 64
            or any(character not in "0123456789abcdef" for character in fragment_id)
            for fragment_id in fragment_ids
        ):
            raise RuntimeError("checkpoint manifest has a non-canonical fragment ID")
        identity = CheckpointSaveIdentity(
            manifest["checkpoint_name"],
            manifest["save_attempt"],
            CheckpointCommitAuthority(
                manifest["authority_scope"],
                _strict_integer(
                    manifest["authority_generation"], "authority_generation"
                ),
                manifest["commit_authority"],
            ),
        )
        return {
            fragment_id: f"{self._attempt_prefix(identity)}/{fragment_id}"
            for fragment_id in fragment_ids
        }

    def _participant_roots_in_native_order(
        self, manifest: Mapping[str, Any]
    ) -> tuple[str, ...]:
        participants = self._validated_manifest_participants(manifest)
        return tuple(
            participants.root_for(participant_index)
            for participant_index in range(
                _strict_integer(
                    manifest["participant_count"], "participant_count", minimum=1
                )
            )
        )

    def _manifest_fragment_descriptor(
        self, manifest: Mapping[str, Any], participant_index: int
    ) -> Mapping[str, Any]:
        return self._validated_manifest_participants(manifest).descriptor_for(
            participant_index
        )

    def _validated_manifest_participants(
        self, manifest: Mapping[str, Any]
    ) -> ValidatedManifestParticipants:
        cacheable = any(manifest is row for row in (self._manifest_cache_rows or ()))
        fingerprint = _canonical_digest(
            [
                {
                    field: manifest[field]
                    for field in (
                        "checkpoint_name",
                        "checkpoint_version",
                        "save_attempt",
                        "authority_scope",
                        "authority_generation",
                        "commit_authority",
                        "schema_digest",
                        "checkpoint_mode",
                        "participant_count",
                        "fragment_ids",
                        "fragment_descriptors",
                        "participant_watermarks",
                        "participant_page_capacities",
                        "participant_page_datasets",
                        "participant_page_versions",
                        "participant_page_layout_generations",
                    )
                }
            ]
        )
        identity = (
            _strict_string(manifest["checkpoint_name"], "checkpoint_name"),
            _strict_integer(manifest["checkpoint_version"], "checkpoint_version"),
            fingerprint,
        )
        if cacheable:
            cached = self._participant_view_cache.get(identity)
            if cached is not None:
                return cached
        roots_by_fragment_id = self._manifest_fragment_roots(manifest)
        view = ValidatedManifestParticipants.from_manifest(
            manifest, roots_by_fragment_id
        )
        if cacheable:
            self._participant_view_cache[identity] = view
        return view

    def _manifest_fragment_root_for_participant(
        self, manifest: Mapping[str, Any], participant_index: int
    ) -> str:
        return self._validated_manifest_participants(manifest).root_for(
            participant_index
        )

    def _intent_tags_for_attempt(
        self, identity: CheckpointSaveIdentity
    ) -> tuple[str, ...]:
        token = self._token(identity.save_attempt)
        return tuple(
            tag
            for tag in self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list()
            if tag.startswith(f"seintent.{token}.")
        )

    def _validate_manifest_fragments(self, manifest: dict[str, Any]) -> None:
        participants = self._validated_manifest_participants(manifest)
        descriptors = participants.descriptors()
        roots = tuple(participants.roots_by_fragment_id.values())
        participant_count = _strict_integer(
            manifest["participant_count"], "participant_count", minimum=1
        )
        if len(descriptors) != len(roots) or len(roots) != participant_count:
            raise RuntimeError("checkpoint manifest participant coverage is incomplete")
        if _canonical_digest(descriptors) != manifest["fragment_digest"]:
            raise RuntimeError("checkpoint manifest fragment digest mismatch")
        for field in (
            "participant_page_datasets",
            "participant_page_versions",
            "participant_page_layout_generations",
        ):
            if len(manifest[field]) != participant_count:
                raise RuntimeError(f"checkpoint {field} coverage is invalid")
        for descriptor in descriptors:
            participant_indices = descriptor["participant_indices"]
            if len(participant_indices) != 1:
                raise RuntimeError("checkpoint manifest participant routing is invalid")
            (participant_index,) = participant_indices
            if participant_index < 0 or participant_index >= participant_count:
                raise RuntimeError("checkpoint manifest participant routing is invalid")
            try:
                root = participants.roots_by_fragment_id[str(descriptor["fragment_id"])]
            except KeyError as error:
                raise RuntimeError(
                    "checkpoint manifest participant root is missing"
                ) from error
            if descriptor["participant_count"] != participant_count:
                raise RuntimeError("checkpoint participant topology mismatch")
            if descriptor["schema_digest"] != manifest[
                "schema_digest"
            ] or self._manifest_specification(
                descriptor
            ) != self._manifest_specification(manifest):
                raise RuntimeError("checkpoint fragment specification mismatch")
            participant_rec = participants.record_for(participant_index)
            if (
                descriptor["mode"],
                descriptor["parent_checkpoint_name"],
                descriptor["parent_checkpoint_version"],
                descriptor["parent_reference_id"],
                descriptor["mutation_watermark"],
                descriptor["include_page_images"],
            ) != (
                manifest["checkpoint_mode"],
                manifest["parent_checkpoint_name"],
                manifest["parent_checkpoint_version"],
                manifest["parent_reference_id"],
                participant_rec.watermark,
                manifest["include_page_images"],
            ):
                raise RuntimeError("checkpoint manifest save context mismatch")
            if descriptor["entries_per_page"] != participant_rec.page_capacity:
                raise RuntimeError("checkpoint participant Page geometry mismatch")
            if (
                descriptor["page_dataset"] != participant_rec.page_dataset
                or _strict_integer(descriptor["page_version"], "page_version")
                != participant_rec.page_version
                or _strict_integer(
                    descriptor["page_layout_generation"], "page_layout_generation"
                )
                != participant_rec.page_layout_generation
            ):
                raise RuntimeError("checkpoint Page snapshot metadata mismatch")
            self._validate_fragment_payload(root, descriptor)

    def _fragment_descriptor_for_root(self, root: str) -> dict[str, Any]:
        matches = []
        for manifest in self._manifest_rows():
            descriptor = self._validated_manifest_participants(
                manifest
            ).descriptors_by_root.get(root)
            if descriptor is not None:
                matches.append(dict(descriptor))
        if not matches:
            raise _ManifestFragmentDescriptorNotFound(
                "fragment descriptor is not referenced by a manifest"
            )
        if len(matches) > 1:
            raise InvalidCheckpointFragment(
                "fragment descriptor is referenced by multiple manifests"
            )
        (match,) = matches
        return match

    def _compact_fragment_native(
        self, compact_root: str, source_roots: Sequence[str]
    ) -> dict[str, Any]:
        if not hasattr(_checkpoint_native, "compact_native_lance_sources"):
            raise RuntimeError(
                "native checkpoint compaction requires a Lance-enabled _core extension"
            )
        source_uris = [
            self._workspace.uri(f"{root}/{ENTRIES_DATASET}") for root in source_roots
        ]
        entry_artifact = _checkpoint_native.compact_native_lance_sources(
            self._workspace.uri(f"{compact_root}/{ENTRIES_DATASET}"),
            source_uris,
            dict(self._workspace.storage_options()),
        )
        has_pages = all(
            self._workspace.exists(f"{root}/{PAGE_IMAGES_DATASET}")
            for root in source_roots
        )
        if has_pages:
            source_uris = [
                self._workspace.uri(f"{root}/{PAGE_IMAGES_DATASET}")
                for root in source_roots
            ]
            page_artifact = _checkpoint_native.compact_native_lance_sources(
                self._workspace.uri(f"{compact_root}/{PAGE_IMAGES_DATASET}"),
                source_uris,
                dict(self._workspace.storage_options()),
            )
            return {"entries": entry_artifact, "page_images": page_artifact}
        return {"entries": entry_artifact}

    def _compact_fragment_roots(
        self,
        source_roots: Sequence[str],
        compact_root: str,
        *,
        expected_participant_count: int | None,
    ) -> dict[str, Any]:
        artifact = self._compact_fragment_native(compact_root, source_roots)
        metadata = self._build_compacted_fragment_metadata(
            compact_root,
            source_roots,
            artifact,
            expected_participant_count,
        )
        return {"artifact": artifact, "metadata": metadata}

    def _build_compacted_fragment_metadata(
        self,
        compact_root: str,
        source_roots: Sequence[str],
        compaction_artifact: dict[str, Any],
        expected_participant_count: int | None,
    ) -> dict[str, Any]:
        if not source_roots:
            raise RuntimeError("fragment compaction requires source fragments")
        source_metadata = [
            self._compaction_descriptor_for_root(root) for root in source_roots
        ]
        source_metadata = [
            _strict_compaction_descriptor(metadata) for metadata in source_metadata
        ]
        participant_counts = {
            metadata["participant_count"] for metadata in source_metadata
        }
        if len(participant_counts) != 1:
            raise RuntimeError("fragment participant counts are inconsistent")
        participant_count = participant_counts.pop()
        if (
            expected_participant_count is not None
            and participant_count != expected_participant_count
        ):
            raise RuntimeError("compaction participant count mismatch")
        key_dtypes = {str(metadata["key_dtype"]) for metadata in source_metadata}
        if len(key_dtypes) != 1:
            raise RuntimeError("fragment key dtypes are inconsistent")
        key_dtype = key_dtypes.pop()
        schema_digests = {
            str(metadata["schema_digest"]) for metadata in source_metadata
        }
        if len(schema_digests) != 1:
            raise RuntimeError("fragment schema digests are inconsistent")
        schema_digest = schema_digests.pop()
        normalized_indices: list[tuple[int, ...]] = []
        for metadata in source_metadata:
            raw_indices = metadata["participant_indices"]
            if (
                not isinstance(raw_indices, Sequence)
                or isinstance(raw_indices, (str, bytes))
                or not raw_indices
            ):
                raise InvalidCheckpointFragment(
                    "compaction participant indices are invalid"
                )
            normalized_indices.append(
                tuple(
                    _strict_integer(index, "compaction participant index", minimum=0)
                    for index in raw_indices
                )
            )
        flattened_indices = tuple(
            index for indices in normalized_indices for index in indices
        )
        if len(set(flattened_indices)) != len(flattened_indices):
            raise InvalidCheckpointFragment(
                "compaction participant indices contain duplicates"
            )
        participant_indices = tuple(sorted(flattened_indices))
        if (expected_participant_count is not None or len(source_roots) > 1) and (
            participant_indices != tuple(range(participant_count))
        ):
            raise RuntimeError("compaction participant coverage is incomplete")
        entry_count = self._workspace.dataset(
            f"{compact_root}/{ENTRIES_DATASET}"
        ).count_rows()
        if (
            _strict_integer(
                compaction_artifact["entries"]["inserted_rows"], "inserted_rows"
            )
            != entry_count
        ):
            raise RuntimeError("compaction output row counts differ from sources")
        schema = self._workspace.dataset(f"{compact_root}/{ENTRIES_DATASET}").schema
        if schema.remove_metadata() != entry_schema(key_dtype).remove_metadata():
            raise RuntimeError("compaction output schema mismatch")
        specification = self._decode_specification(schema.metadata or {})
        if _schema_digest(specification) != schema_digest:
            raise RuntimeError("compaction output schema digest mismatch")

        if "page_images" in compaction_artifact:
            page_count = self._workspace.dataset(
                f"{compact_root}/{PAGE_IMAGES_DATASET}"
            ).count_rows()
            if (
                _strict_integer(
                    compaction_artifact["page_images"]["inserted_rows"], "inserted_rows"
                )
                != page_count
            ):
                raise RuntimeError("compaction output row counts differ from sources")
            page_schema = self._workspace.dataset(
                f"{compact_root}/{PAGE_IMAGES_DATASET}"
            ).schema
            if page_schema.remove_metadata() != PAGE_IMAGE_SCHEMA.remove_metadata():
                raise RuntimeError("compaction output schema mismatch")
            page_digest = str(compaction_artifact["page_images"]["content_digest"])
        else:
            page_count = sum(metadata["page_count"] for metadata in source_metadata)
            page_digest = ""

        content_digest = _canonical_digest(
            [
                {
                    "entries": str(compaction_artifact["entries"]["content_digest"]),
                    "page_images": page_digest,
                }
            ]
        )
        return {
            "participant_indices": participant_indices,
            "participant_count": participant_count,
            "entry_count": entry_count,
            "page_count": page_count,
            "schema_digest": schema_digest,
            "content_digest": content_digest,
            "dimensions": list(specification.dimensions),
            "embedding_dtype": specification.embedding_dtype,
            "optimizer_dtype": specification.optimizer_dtype or "",
            "score_policy": specification.score_policy,
            "key_dtype": specification.key_dtype,
        }

    def _compaction_descriptor_for_root(self, root: str) -> dict[str, Any]:
        manifest_descriptor: dict[str, Any] | None = None
        try:
            manifest_descriptor = self._fragment_descriptor_for_root(root)
        except _ManifestFragmentDescriptorNotFound:
            manifest_descriptor = None
        entries = self._workspace.dataset(f"{root}/{ENTRIES_DATASET}")
        metadata = entries.schema_metadata
        normalized_metadata = {
            key.decode("utf-8") if isinstance(key, bytes) else str(key): value
            for key, value in metadata.items()
        }
        raw = normalized_metadata.get(_LOCAL_VALIDATION_DESCRIPTOR)
        local_descriptor: dict[str, Any] | None = None
        if raw is not None:
            if not isinstance(raw, (str, bytes)):
                raise InvalidCheckpointFragment(
                    "fragment validation descriptor is not text"
                )
            try:
                decoded = json.loads(
                    raw.decode("utf-8") if isinstance(raw, bytes) else raw
                )
            except (TypeError, ValueError) as error:
                raise InvalidCheckpointFragment(
                    "fragment validation descriptor is invalid"
                ) from error
            if not isinstance(decoded, dict):
                raise InvalidCheckpointFragment(
                    "fragment validation descriptor is not an object"
                )
            local_fields = {
                "descriptor_version",
                "schema_digest",
                "dimensions",
                "embedding_dtype",
                "optimizer_dtype",
                "score_policy",
                "key_dtype",
                "entry_count",
                "page_count",
                "content_digest",
                "participant_count",
                "participant_indices",
            }
            if set(decoded) != local_fields:
                raise InvalidCheckpointFragment(
                    "fragment validation descriptor fields are invalid"
                )
            if (
                _strict_integer(
                    decoded["descriptor_version"],
                    "descriptor_version",
                    minimum=1,
                    maximum=1,
                )
                != 1
            ):
                raise InvalidCheckpointFragment(
                    "fragment validation descriptor version is unsupported"
                )
            local_descriptor = decoded
        if manifest_descriptor is None and local_descriptor is None:
            raise InvalidCheckpointFragment(
                "manifest-free fragment is missing local validation descriptor"
            )
        if manifest_descriptor is not None and local_descriptor is not None:
            manifest_descriptor = _strict_compaction_descriptor(manifest_descriptor)
            local_descriptor = _strict_compaction_descriptor(local_descriptor)
            if not _strict_equal_canonical_descriptors(
                manifest_descriptor, local_descriptor
            ):
                raise InvalidCheckpointFragment(
                    "manifest and local fragment descriptors disagree"
                )
        if manifest_descriptor is not None:
            return _strict_compaction_descriptor(manifest_descriptor)
        if local_descriptor is not None:
            return _strict_compaction_descriptor(local_descriptor)
        raise InvalidCheckpointFragment("fragment validation descriptor is unavailable")

    def _publish_local_validation_descriptor(
        self, root: str, metadata: Mapping[str, Any]
    ) -> None:
        specification = self._manifest_specification(dict(metadata))
        descriptor = {
            "descriptor_version": 1,
            "schema_digest": str(metadata["schema_digest"]),
            "dimensions": list(specification.dimensions),
            "embedding_dtype": specification.embedding_dtype,
            "optimizer_dtype": specification.optimizer_dtype or "",
            "score_policy": specification.score_policy,
            "key_dtype": specification.key_dtype,
            "entry_count": _strict_integer(metadata["entry_count"], "entry_count"),
            "page_count": _strict_integer(metadata["page_count"], "page_count"),
            "content_digest": str(metadata["content_digest"]),
            "participant_count": _strict_integer(
                metadata["participant_count"], "participant_count", minimum=1
            ),
            "participant_indices": [
                _strict_integer(index, "participant index", minimum=0)
                for index in metadata["participant_indices"]
            ],
        }
        encoded = json.dumps(descriptor, sort_keys=True)
        for name in (ENTRIES_DATASET, PAGE_IMAGES_DATASET):
            if self._workspace.exists(f"{root}/{name}"):
                self._workspace.dataset(f"{root}/{name}").update_schema_metadata(
                    {_LOCAL_VALIDATION_DESCRIPTOR: encoded}, replace=False
                )

    @staticmethod
    def _sql_literal(value: str) -> str:
        return value.replace("'", "''")

    def _is_acknowledged_committed_row(
        self,
        row: Mapping[str, Any],
        tags_set: Collection[str],
    ) -> bool:
        if row["commit_state"] != _COMMITTED:
            return False
        attempt_token = self._token(str(row["save_attempt"]))
        version = _strict_integer(
            row["checkpoint_version"], "checkpoint_version", minimum=1
        )
        intent_tag = f"seintent.{attempt_token}.{version}"
        if intent_tag in tags_set:
            return False
        locator = SparseCheckpointLocator(str(row["checkpoint_name"]), version)
        if self._reclaim_pending_tag(locator) in tags_set:
            return False
        return True

    @classmethod
    def _intent_tag(
        self, identity: CheckpointSaveIdentity, locator: SparseCheckpointLocator
    ) -> str:
        return f"seintent.{self._token(identity.save_attempt)}.{locator.checkpoint_version}"

    def _committed_tag(self, locator: SparseCheckpointLocator) -> str:
        return f"secommitted.{locator.checkpoint_version}.{self._encode(locator.checkpoint_name)}"

    def _reclaim_pending_tag(self, locator: SparseCheckpointLocator) -> str:
        return f"sereclaiming.{locator.checkpoint_version}.{self._encode(locator.checkpoint_name)}"

    def _locator_has_pending_intent(self, locator: SparseCheckpointLocator) -> bool:
        return any(
            row["checkpoint_name"] == locator.checkpoint_name
            and _strict_integer(
                row["checkpoint_version"], "checkpoint_version", minimum=1
            )
            == locator.checkpoint_version
            and row["commit_state"] == _PREPARED
            for row in self._manifest_rows()
        )

    def _delete_intent_tag(
        self, identity: CheckpointSaveIdentity, version: int
    ) -> None:
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        tag = self._intent_tag(
            identity, SparseCheckpointLocator(identity.checkpoint_name, version)
        )
        if tag in tags.list():
            tags.delete(tag)

    def _decode_reference_tag(self, tag: str) -> SparseCheckpointLocator:
        ref = _ReferenceTag.parse(tag, self._decode)
        return SparseCheckpointLocator(ref.checkpoint_name, ref.checkpoint_version)

    def _recover_prepared(
        self,
        identity: CheckpointSaveIdentity,
        lease_request: CheckpointLeaseRequest | None,
        reference_id: str | None,
    ) -> SparseCheckpointLocator | None:
        matches = [
            row
            for row in self._manifest_rows()
            if row["checkpoint_name"] == identity.checkpoint_name
            and row["save_attempt"] == identity.save_attempt
        ]
        if len(matches) > 1:
            raise CheckpointConflictError("checkpoint has multiple prepared manifests")
        if not matches:
            return None
        (row,) = matches
        if (
            row["authority_scope"],
            _strict_integer(row["authority_generation"], "authority_generation"),
            row["commit_authority"],
        ) != (
            identity.commit_authority.scope,
            identity.commit_authority.generation,
            identity.commit_authority.authority_id,
        ):
            raise CheckpointConflictError(
                "recovered checkpoint commit authority does not match"
            )
        locator = SparseCheckpointLocator(
            identity.checkpoint_name,
            _strict_integer(row["checkpoint_version"], "checkpoint_version", minimum=1),
        )
        if row["storage_mode"] != _FRAGMENT_STORAGE_MODE:
            raise RuntimeError("checkpoint manifest storage mode is corrupt")
        if row["commit_state"] == _COMMITTED:
            self._create_parent_chain_reference(row)
            self._validate_manifest_fragments(row)
            self._protect_manifest_pages(row)
            self._release_page_attempt_tags(
                identity,
                [
                    self._manifest_fragment_descriptor(row, participant_index)[
                        "page_dataset"
                    ]
                    for participant_index in range(
                        _strict_integer(
                            row["participant_count"], "participant_count", minimum=1
                        )
                    )
                ],
            )
            self._release_page_write_intents(
                identity,
                [
                    self._manifest_fragment_descriptor(row, participant_index)[
                        "page_dataset"
                    ]
                    for participant_index in range(
                        _strict_integer(
                            row["participant_count"], "participant_count", minimum=1
                        )
                    )
                ],
            )
            if (reference_id or "") != (row["reference_id"] or ""):
                raise CheckpointConflictError(
                    "recovered checkpoint reference does not match"
                )
            if lease_request is not None:
                self._acquire_checkpoint_lease_locked(locator, lease_request)
            if reference_id:
                self._create_reference_tags(locator, reference_id, row)
            tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
            intent = self._intent_tag(identity, locator)
            if intent in tags.list():
                intent_manifest_before = tags.get_version(intent)
            else:
                outcome = self._pending_outcome(identity)
                if outcome is None or outcome.locator != locator:
                    raise RuntimeError(
                        "committed checkpoint intent is missing during recovery"
                    )
                self._clear_authority(identity)
                self._release_parent_head_reservations(identity)
                return locator
            committed_tag = self._committed_tag(locator)
            tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
            if committed_tag not in tags.list():
                tags.create(
                    committed_tag,
                    self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).version,
                )
            self._record_pending_outcome(
                identity,
                locator,
                intent_manifest_before,
            )
            self._delete_intent_tag(identity, locator.checkpoint_version)
            self._release_parent_head_reservations(identity)
            self._clear_authority(identity)
            return locator
        if row["commit_state"] != _PREPARED:
            raise RuntimeError("checkpoint manifest commit state is corrupt")
        intent = self._intent_tag(identity, locator)
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        if intent not in tags.list():
            raise CheckpointConflictError(
                "prepared checkpoint intent is missing or ambiguous"
            )
        self._require_authority(identity)
        if row["checkpoint_mode"] == CheckpointSaveMode.INCREMENTAL.value:
            self._assert_incremental_parent_head_locked(
                identity,
                SparseCheckpointLocator(
                    str(row["parent_checkpoint_name"]),
                    _strict_integer(
                        row["parent_checkpoint_version"], "parent_checkpoint_version"
                    ),
                ),
            )
        self._validate_manifest_fragments(row)
        self._protect_manifest_pages(row)
        self._release_page_attempt_tags(
            identity,
            [
                self._manifest_fragment_descriptor(row, participant_index)[
                    "page_dataset"
                ]
                for participant_index in range(
                    _strict_integer(
                        row["participant_count"], "participant_count", minimum=1
                    )
                )
            ],
        )
        self._release_page_write_intents(
            identity,
            [
                self._manifest_fragment_descriptor(row, participant_index)[
                    "page_dataset"
                ]
                for participant_index in range(
                    _strict_integer(
                        row["participant_count"], "participant_count", minimum=1
                    )
                )
            ],
        )
        self._create_parent_chain_reference(row)
        if reference_id != (row["reference_id"] or ""):
            raise CheckpointConflictError(
                "prepared checkpoint reference does not match"
            )
        if lease_request is not None:
            self._acquire_checkpoint_lease_locked(locator, lease_request)
        if reference_id:
            self._create_reference_tags(locator, reference_id, row)
        updated = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).update(
            {"commit_state": f"'{_COMMITTED}'"},
            where=(
                f"checkpoint_name = '{self._sql_literal(identity.checkpoint_name)}' "
                f"AND checkpoint_version = {locator.checkpoint_version}"
            ),
        )
        if updated["num_rows_updated"] != 1:
            raise CheckpointConflictError(
                "prepared checkpoint manifest update was lost"
            )
        committed_tag = self._committed_tag(locator)
        if committed_tag not in tags.list():
            tags.create(
                committed_tag,
                self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).version,
            )
        intent = self._intent_tag(identity, locator)
        manifest_before = tags.get_version(intent)
        self._record_pending_outcome(
            identity,
            locator,
            manifest_before,
        )
        self._release_parent_head_reservations(identity)
        self._clear_authority(identity)
        return locator

    def _restore_entries(
        self,
        store: Any,
        prepared: _PreparedRestore,
        plan: RestorePlan,
        load_optimizer_state: bool,
    ) -> None:
        chain = prepared.chain
        target = plan.target
        if plan.engine is IngestionEngine.FAST_PAGE_CACHE:
            participant_rec = prepared.page_participant
            if participant_rec is None:
                raise RuntimeError(
                    "page-image restore requires a validated page descriptor"
                )
            page_dataset = participant_rec.page_dataset
            page_version = participant_rec.page_version
            page_sources = (
                [self._workspace.uri(page_dataset)] if page_version > 0 else []
            )
            _checkpoint_native.restore_native_page_images(
                store,
                page_sources,
                page_version,
                target.partition_id,
                plan.key_dtype,
                dict(self._workspace.storage_options()),
                load_optimizer_state,
                page_layout_generation=participant_rec.page_layout_generation,
            )
            return

        if plan.engine is not IngestionEngine.RECONSTRUCT_ENTRIES:
            raise ValueError(f"unsupported restore materialization: {plan.engine!r}")
        sources = [
            self._workspace.uri(f"{root}/{ENTRIES_DATASET}")
            for chain_manifest in chain
            for root in self._participant_roots_in_native_order(chain_manifest)
            if self._workspace.exists(f"{root}/{ENTRIES_DATASET}")
        ]
        assignment_name = target.spec.name
        assignment_parameters = dict(target.spec.parameters)
        _checkpoint_native.restore_native_partition_entries(
            store,
            sources,
            0,
            target.partition_count,
            target.partition_id,
            plan.key_dtype,
            None,
            dict(self._workspace.storage_options()),
            load_optimizer_state,
            assignment_name,
            assignment_parameters,
        )

    @staticmethod
    def _store_specification(store: Any) -> TableSpecification:
        """Extract and validate the canonical TableSpecification from an embedding store.

        The store must conform to the Store protocol, providing table_dims(),
        value_dtype(), optimizer_dtype(), score_policy(), and key_dtype().
        Dimensions must be a non-empty, strictly ascending sequence of unique positive
        integers to guarantee deterministic schema digests and external merge sort ordering.
        """
        required_methods = (
            "table_dims",
            "value_dtype",
            "optimizer_dtype",
            "score_policy",
            "key_dtype",
        )
        missing = [
            method
            for method in required_methods
            if not callable(getattr(store, method, None))
        ]
        if missing:
            raise TypeError(
                f"store object is missing required callable methods: {missing}. "
                "Ensure your store is an instance of se.EmbeddingStore or implements the Store protocol."
            )

        raw_dims = store.table_dims()
        if not raw_dims:
            raise ValueError(
                "store dimensions must not be empty. "
                "Please configure EmbeddingConfig with at least one dimension, e.g., EmbeddingConfig(dims=[128])."
            )

        dimensions = tuple(
            _strict_integer(dim, "store dimension", minimum=1) for dim in raw_dims
        )

        if len(set(dimensions)) != len(dimensions):
            raise ValueError(
                f"store dimensions must be unique, but duplicates were found: {dimensions}. "
                "Please deduplicate dimensions when configuring the store, e.g., EmbeddingConfig(dims=sorted(set(dims)))."
            )

        if tuple(sorted(dimensions)) != dimensions:
            raise ValueError(
                f"store dimensions must be in strictly ascending order, but received: {dimensions}. "
                f"Please initialize the store with sorted dimensions, e.g., EmbeddingConfig(dims={sorted(dimensions)})."
            )

        return TableSpecification(
            dimensions=dimensions,
            embedding_dtype=store.value_dtype(),
            optimizer_dtype=store.optimizer_dtype() or None,
            score_policy=store.score_policy(),
            key_dtype=store.key_dtype(),
        )

    @staticmethod
    def _decode_specification(
        metadata: dict[bytes, bytes] | None,
    ) -> TableSpecification:
        required = {
            b"sparse_embedding.dimensions",
            b"sparse_embedding.embedding_dtype",
            b"sparse_embedding.optimizer_dtype",
            b"sparse_embedding.score_policy",
            b"sparse_embedding.key_dtype",
        }
        if metadata is None or not required.issubset(metadata):
            raise InvalidCheckpointFragment(
                "checkpoint fragment specification is missing"
            )
        if any(not isinstance(metadata[field], bytes) for field in required):
            raise InvalidCheckpointFragment(
                "checkpoint fragment specification values must be bytes"
            )
        try:
            dimension_tokens = (
                metadata[b"sparse_embedding.dimensions"].decode().split(",")
            )
            if not dimension_tokens or any(not token for token in dimension_tokens):
                raise ValueError("dimensions contain an empty token")
            dimensions = tuple(
                _serialized_integer(token, "fragment dimension", minimum=1)
                for token in dimension_tokens
            )
            if tuple(sorted(dimensions)) != dimensions:
                raise ValueError("fragment dimensions must be sorted")
            return TableSpecification(
                dimensions=dimensions,
                embedding_dtype=metadata[b"sparse_embedding.embedding_dtype"].decode(),
                optimizer_dtype=metadata[b"sparse_embedding.optimizer_dtype"].decode()
                or None,
                score_policy=metadata[b"sparse_embedding.score_policy"].decode(),
                key_dtype=metadata[b"sparse_embedding.key_dtype"].decode(),
            )
        except (TypeError, ValueError) as error:
            raise InvalidCheckpointFragment(
                "checkpoint fragment specification is invalid"
            ) from error

    @staticmethod
    def _manifest_specification(manifest: dict[str, Any]) -> TableSpecification:
        try:
            dimensions = tuple(
                _strict_integer(dim, "manifest dimension", minimum=1)
                for dim in manifest["dimensions"]
            )
            if not dimensions or tuple(sorted(dimensions)) != dimensions:
                raise ValueError("manifest dimensions must be sorted")
            return TableSpecification(
                dimensions=dimensions,
                embedding_dtype=manifest["embedding_dtype"],
                optimizer_dtype=manifest["optimizer_dtype"] or None,
                score_policy=manifest["score_policy"],
                key_dtype=manifest["key_dtype"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "checkpoint manifest specification is invalid"
            ) from error

    def _attempt_prefix(self, identity: CheckpointSaveIdentity) -> str:
        authority = self._authority_tag(identity).removeprefix("seauthority.")
        return f"_staging/{authority}"

    @staticmethod
    def _fragment_id(
        identity: CheckpointSaveIdentity, participant: CheckpointParticipant
    ) -> str:
        return hashlib.sha256(
            f"{identity.checkpoint_name}\0{identity.save_attempt}\0{participant.participant_index}".encode()
        ).hexdigest()

    @staticmethod
    def _page_dataset_relative(lineage_identity: str, participant_index: int) -> str:
        if not lineage_identity:
            raise ValueError("page dataset lineage identity must be non-empty")
        if participant_index < 0:
            raise ValueError("page dataset participant index must be non-negative")
        lineage = hashlib.sha256(lineage_identity.encode()).hexdigest()
        return (
            f"{_PAGE_DATASET_ROOT}/{lineage}/{participant_index}/{PAGE_IMAGES_DATASET}"
        )

    def _check_storage_budget(self) -> None:
        if self._max_storage_bytes is None:
            return
        size = self._workspace.storage_bytes()
        if size > self._max_storage_bytes:
            raise StorageBudgetExceeded("checkpoint storage budget is exceeded")

    @contextmanager
    def _commit_lock(self):
        self._require_mutation_allowed()
        depth = getattr(self._commit_lock_state, "depth", 0)
        if depth:
            self._commit_lock_state.depth = depth + 1
            try:
                yield
            finally:
                self._commit_lock_state.depth = depth
            return
        self._commit_lock_state.depth = 1
        try:
            with self._workspace.commit_lock():
                yield
        finally:
            self._commit_lock_state.depth = 0

    def _require_mutation_allowed(self) -> None:
        if self._exclusive_recovery_owner is not None:
            raise CheckpointConflictError(
                "checkpoint operation is fenced by exclusive recovery"
            )

    def _require_exclusive_recovery(self) -> None:
        if self._exclusive_recovery_owner != threading.get_ident():
            raise CheckpointConflictError(
                "reclaim requires an active exclusive recovery context"
            )

    def close(self) -> None:
        self._workspace.close()


class LocalLanceStorage(_LanceCheckpointStorage):
    @classmethod
    def from_checkpoint_root(
        cls,
        root: str | Path,
        *,
        max_storage_bytes: int | None = None,
    ) -> LocalLanceStorage:
        return cls(root, max_storage_bytes=max_storage_bytes)

    @classmethod
    def from_published_weight(cls, path: str | Path) -> LocalLanceStorage:
        asset_root = Path(path).expanduser()
        handle = PublishedWeightHandleAsset.read(asset_root)
        storage_uri, _, _ = _decode_checkpoint_handle(handle)
        parsed = urlparse(storage_uri)
        if parsed.scheme not in ("", "file"):
            raise NotImplementedError(
                f"sparse published weight backend is unsupported: {parsed.scheme}"
            )
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            raise ValueError("sparse published weight file URI host is unsupported")
        backend_root = (
            Path(parsed.path if parsed.scheme == "file" else storage_uri)
            .expanduser()
            .resolve()
        )
        if backend_root == asset_root.resolve():
            raise ValueError("sparse published weight points to its own asset path")
        if not backend_root.is_dir():
            raise FileNotFoundError(
                f"sparse checkpoint root does not exist: {backend_root}"
            )
        storage = cls(backend_root)
        storage._bound_handle = handle
        return storage

    def __init__(
        self,
        root: str | Path,
        *,
        max_storage_bytes: int | None = None,
    ) -> None:
        super().__init__(
            LocalLanceWorkspace(root),
            max_storage_bytes=max_storage_bytes,
        )
