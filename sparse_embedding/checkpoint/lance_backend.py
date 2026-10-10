"""Local Lance implementation of the sparse checkpoint protocol."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import struct
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa

from sparse_embedding import _core as _native

from ._lance_workspace import LanceWorkspace, LocalLanceWorkspace, S3LanceWorkspace
from .base import (
    CheckpointCommitAuthority,
    CheckpointConflictError,
    CheckpointFragment,
    CheckpointHandle,
    CheckpointLease,
    CheckpointLeaseRequest,
    CheckpointParticipant,
    CheckpointSaveIdentity,
    CleanupReservation,
    CommittedCheckpoint,
    ElasticRestore,
    InvalidCheckpointFragment,
    PhysicalRestore,
    PhysicalRestoreStrategy,
    RestoreMode,
    SparseCheckpointLocator,
    StorageBudgetExceeded,
    TableSpecification,
)
from .restore_options import _decode_restore_options
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
_WEIGHT_REFERENCE_FILENAME = ".sparse_embedding_reference"
_CHECKPOINT_HANDLE_MAGIC = b"sparse_embedding.checkpoint_handle\0"
_UINT32 = struct.Struct("!I")
_UINT64 = struct.Struct("!Q")


class _RestoreScope(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"


@dataclass(frozen=True, slots=True)
class _CommitOutcome:
    locator: SparseCheckpointLocator
    reference_id: str
    manifest_before: int


@dataclass(frozen=True, slots=True)
class _LeaseRecord:
    lease: CheckpointLease
    locator: SparseCheckpointLocator


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
        if isinstance(value, dict):
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
        length = _UINT32.unpack_from(payload, offset)[0]
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
    checkpoint_version = _UINT64.unpack_from(payload, offset)[0]
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
    source = Path(source_path).expanduser().resolve()
    target = (
        Path(target_path).expanduser().resolve() if target_path is not None else None
    )
    workspace = LocalLanceWorkspace(Path("/"))
    engine = _LanceCheckpointStorage(workspace)
    published = engine.compact_fragments(source, target)
    return Path(workspace.uri(published)).resolve()


async def compact_fragments_async(
    source_path: str | Path,
    target_path: str | Path | None = None,
    *,
    storage_options: Mapping[str, str] | None = None,
) -> Path | str:
    return await asyncio.to_thread(
        compact_fragments, source_path, target_path, storage_options=storage_options
    )


def _native_page_checksum(row: dict[str, Any]) -> str:
    hash_value = 1469598103934665603
    for key in (
        "slot_ordinals",
        "keys",
        "sequences",
        "embedding_values",
        "scores",
        "optimizer_state_values",
        "steps",
    ):
        value = row[key]
        if not isinstance(value, (bytes, bytearray, memoryview)):
            value = np.ascontiguousarray(value).tobytes()
        for byte in bytes(value):
            hash_value ^= byte
            hash_value = (hash_value * 1099511628211) & ((1 << 64) - 1)
    return f"{hash_value:x}"


def _decode_value_bytes(payload: bytes, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return np.frombuffer(payload, dtype=np.float32)
    if dtype == "float16":
        return np.frombuffer(payload, dtype=np.float16).astype(np.float32)
    if dtype == "bfloat16":
        bits = np.frombuffer(payload, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32)
    raise ValueError(f"unsupported payload dtype: {dtype}")


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

    def prepare_checkpoint(self, identity: CheckpointSaveIdentity) -> None:
        self._ensure_tables()
        with self._commit_lock():
            self._reconcile_compaction_retirements()
            self._reconcile_staging()
            self._check_storage_budget()
            attempt = self._attempt_prefix(identity)
            if self._workspace.exists(attempt):
                raise CheckpointConflictError("checkpoint save attempt already exists")
            self._publish_authority(identity)
            self._workspace.create_prefix(attempt)

    def write_checkpoint_fragment(
        self,
        store: Any,
        identity: CheckpointSaveIdentity,
        participant: CheckpointParticipant,
        include_page_images: bool = False,
    ) -> CheckpointFragment:
        self._require_authority(identity)
        attempt = self._attempt_prefix(identity)
        if not self._workspace.exists(attempt):
            raise CheckpointConflictError("checkpoint must be prepared before writing")
        fragment_id = self._fragment_id(identity, participant)
        fragment_root = f"{attempt}/{fragment_id}"
        if self._workspace.exists(fragment_root):
            raise CheckpointConflictError("checkpoint fragment already exists")
        specification = self._store_specification(store)
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
        )
        if not isinstance(artifact, dict):
            raise TypeError("native fragment writer returned a non-mapping artifact")
        required = {
            "locator",
            "entries_version",
            "page_images_version",
            "entry_count",
            "page_count",
            "physical_bytes",
            "schema_digest",
            "content_digest",
        }
        if set(artifact) != required:
            raise ValueError("native fragment artifact fields are invalid")
        content_digest = str(artifact["content_digest"])
        self._publish_fragment_metadata(
            fragment_root,
            fragment_id,
            participant.participant_index,
            participant.participant_count,
            int(artifact["entry_count"]),
            int(artifact["page_count"]),
            str(artifact["schema_digest"]),
            content_digest,
        )
        fragment = CheckpointFragment(
            fragment_id=fragment_id,
            checkpoint_name=identity.checkpoint_name,
            save_attempt=identity.save_attempt,
            participant=participant,
            entry_count=int(artifact["entry_count"]),
            page_count=int(artifact["page_count"]),
            schema_digest=str(artifact["schema_digest"]),
            content_digest=content_digest,
        )
        self._require_authority(identity)
        return fragment

    def commit_checkpoint(
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

    def _committed_checkpoint(
        self, locator: SparseCheckpointLocator, reference_id: str
    ) -> CommittedCheckpoint:
        return CommittedCheckpoint(
            _encode_checkpoint_handle(self._workspace.uri(""), locator, reference_id),
            self._checkpoint_version_id(locator),
        )

    def _resolve_handle(
        self, handle: CheckpointHandle
    ) -> tuple[SparseCheckpointLocator, str]:
        storage_uri, locator, reference_id = _decode_checkpoint_handle(handle)
        self._validate_handle_storage_uri(storage_uri)
        manifest = self._resolve_manifest(locator)
        if str(manifest["reference_id"]) != reference_id:
            raise ValueError("checkpoint handle reference identity does not match")
        return locator, reference_id

    def _validate_handle_storage_uri(self, storage_uri: str) -> None:
        expected_uri = self._workspace.uri("")
        if isinstance(self._workspace, LocalLanceWorkspace):
            if Path(storage_uri).expanduser().resolve() != Path(expected_uri).resolve():
                raise ValueError("checkpoint handle belongs to another storage root")
        elif storage_uri.rstrip("/") != expected_uri.rstrip("/"):
            raise ValueError("checkpoint handle belongs to another storage root")

    @staticmethod
    def _checkpoint_version_id(locator: SparseCheckpointLocator) -> bytes:
        return f"{locator.checkpoint_name}:{locator.checkpoint_version}".encode("utf-8")

    @staticmethod
    def _owner_reference_id(identity: CheckpointSaveIdentity) -> str:
        return f"owner:{identity.checkpoint_name}:{identity.save_attempt}"

    def finalize_checkpoint(self, identity: CheckpointSaveIdentity) -> None:
        self._require_authority(identity)

    def _commit_checkpoint_locked(
        self,
        identity: CheckpointSaveIdentity,
        fragments: Sequence[CheckpointFragment],
        lease_request: CheckpointLeaseRequest | None,
        reference_id: str | None,
    ) -> SparseCheckpointLocator:
        if lease_request is not None:
            if lease_request.ttl_seconds <= 0:
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
        previous = max(
            (int(row["checkpoint_version"]) for row in manifest_rows), default=0
        )
        checkpoint_version = previous + 1
        fragment_ids = tuple(fragment.fragment_id for fragment in validated)
        fragment_roots = tuple(
            f"{self._attempt_prefix(identity)}/{fragment.fragment_id}"
            for fragment in validated
        )
        for root in fragment_roots:
            if not self._workspace.exists(f"{root}/{ENTRIES_DATASET}"):
                raise InvalidCheckpointFragment(
                    "checkpoint fragment payload is missing"
                )
        self._check_storage_budget()
        self._workspace.assert_commit_lock()
        fragment_digest = self._fragment_digest_for_metadata(
            fragment_ids, fragment_roots
        )
        manifest = {
            "checkpoint_name": identity.checkpoint_name,
            "checkpoint_version": checkpoint_version,
            "parent_checkpoint_version": previous,
            "participant_count": validated[0].participant.participant_count,
            "checkpoint_mode": "full",
            "schema_digest": _schema_digest(specification),
            "fragment_digest": fragment_digest,
            "save_attempt": identity.save_attempt,
            "authority_scope": identity.commit_authority.scope,
            "authority_generation": identity.commit_authority.generation,
            "commit_authority": identity.commit_authority.authority_id,
            "created_at": datetime.now(timezone.utc),
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
            self._workspace.assert_commit_lock()
            if lease_request is not None:
                lease = self._acquire_checkpoint_lease_locked(locator, lease_request)
            if reference_id:
                self._create_reference_tags(locator, reference_id, manifest)
            self._workspace.assert_commit_lock()
            updated = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).update(
                {"commit_state": f"'{_COMMITTED}'"},
                where=(
                    f"checkpoint_name = '{self._sql_literal(identity.checkpoint_name)}' "
                    f"AND checkpoint_version = {checkpoint_version} "
                    f"AND commit_state = '{_PREPARED}'"
                ),
            )
            if updated["num_rows_updated"] != 1:
                raise CheckpointConflictError("checkpoint manifest update was lost")
            committed_tag = self._committed_tag(locator)
            if committed_tag not in tags.list():
                tags.create(
                    committed_tag,
                    self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).version,
                )
            self._record_pending_outcome(identity, locator, manifest_before)
        except BaseException:
            if lease is not None:
                self._delete_lease_tag(lease, locator)
            if reference_id:
                self._delete_reference_tags(locator, reference_id)
            raise
        self._workspace.assert_commit_lock()
        self._clear_authority(identity)
        return locator

    def abort_checkpoint(self, identity: CheckpointSaveIdentity) -> None:
        with self._commit_lock():
            committed = False
            for row in self._manifest_rows():
                if (
                    row["checkpoint_name"] == identity.checkpoint_name
                    and row["save_attempt"] == identity.save_attempt
                ):
                    if row["commit_state"] == _COMMITTED:
                        committed = True
                        continue
                    if row["commit_state"] != _PREPARED:
                        raise RuntimeError(
                            "checkpoint manifest commit state is corrupt"
                        )
                    version = int(row["checkpoint_version"])
                    self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).delete(
                        f"checkpoint_name = '{self._sql_literal(identity.checkpoint_name)}' "
                        f"AND checkpoint_version = {version}"
                    )
                    self._delete_intent_tag(identity, version)
            attempt = self._attempt_prefix(identity)
            if not committed and self._workspace.exists(attempt):
                self._workspace.delete_prefix(attempt, missing_ok=True)
            if not committed:
                self._clear_authority(identity)

    def acknowledge_checkpoint(self, identity: CheckpointSaveIdentity) -> None:
        with self._commit_lock():
            if self._pending_outcome(identity, require_identity=False) is not None:
                self._delete_pending_outcome(identity.checkpoint_name)
            for row in self._manifest_rows():
                if (
                    row["checkpoint_name"] == identity.checkpoint_name
                    and row["save_attempt"] == identity.save_attempt
                ):
                    self._delete_intent_tag(identity, int(row["checkpoint_version"]))

    def acquire_checkpoint_lease(
        self, handle: CheckpointHandle, request: CheckpointLeaseRequest
    ) -> CheckpointLease:
        locator, _ = self._resolve_handle(handle)
        with self._commit_lock():
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
            expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=request.ttl_seconds),
            generation=0,
            request_id=request.request_id,
        )
        self._create_lease_tags(lease, locator, manifest)
        return lease

    def renew_checkpoint_lease(
        self, lease: CheckpointLease, ttl_seconds: float
    ) -> CheckpointLease:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        with self._commit_lock():
            matches = tuple(
                record
                for record in self._active_checkpoint_lease_records_unlocked()
                if record.lease.lease_id == lease.lease_id
            )
            if len(matches) != 1 or matches[0].lease != lease:
                raise CheckpointConflictError("lease generation CAS conflict")
            renewed = CheckpointLease(
                lease_id=lease.lease_id,
                lease_holder_id=lease.lease_holder_id,
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
                generation=lease.generation + 1,
                request_id=lease.request_id,
            )
            self._create_lease_tags(
                renewed, matches[0].locator, self._resolve_manifest(matches[0].locator)
            )
            self._delete_lease_tag(lease, matches[0].locator)
            return renewed

    def release_checkpoint_lease(self, lease: CheckpointLease) -> None:
        with self._commit_lock():
            matches = tuple(
                record
                for record in self._active_checkpoint_lease_records_unlocked()
                if record.lease.lease_id == lease.lease_id
            )
            if not matches:
                return
            if len(matches) != 1 or matches[0].lease != lease:
                raise CheckpointConflictError("lease generation CAS conflict")
            self._delete_lease_tag(lease, matches[0].locator)

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
        selected_now = now or datetime.now(timezone.utc)
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
        selected_now = now or datetime.now(timezone.utc)
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
        with self._commit_lock():
            self._reconcile_reclaiming()
            self._reconcile_compaction_retirements()
            self._reclaim_unlocked(older_than)

    def compact_checkpoints(self) -> None:
        self._ensure_tables()
        with self._commit_lock():
            rows = [
                row
                for row in self._manifest_rows()
                if row["commit_state"] == _COMMITTED
                and int(row["compacted_version"]) == 0
            ]
        self._compact_checkpoint_rows(rows)

    def compact_fragments(
        self,
        source_path: str | Path,
        target_path: str | Path | None = None,
    ) -> str:
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
            self._compact_fragment_roots(
                roots,
                temporary,
                fragment_id=target.rsplit("/", 1)[-1],
                expected_participant_count=None,
            )
            self._workspace.publish_prefix(temporary, target)
        except BaseException:
            published_target = self._workspace.uri(target) == self._workspace.uri(
                temporary
            )
            if self._workspace.exists(temporary) and not published_target:
                self._workspace.delete_prefix(temporary, missing_ok=True)
            raise
        return self._workspace.uri(target)

    async def compact_fragments_async(
        self,
        source_path: str | Path,
        target_path: str | Path | None = None,
    ) -> str:
        return await asyncio.to_thread(self.compact_fragments, source_path, target_path)

    def _workspace_path(self, path: str | Path) -> str:
        value = str(path).rstrip("/")
        raw_root_uri = self._workspace.uri("")
        root_uri = raw_root_uri if raw_root_uri == "/" else raw_root_uri.rstrip("/")
        if value.startswith(("s3://", "s3a://", "gs://", "http://", "https://")):
            if not value.startswith(root_uri + "/") and value != root_uri:
                raise ValueError("path is outside the configured Lance workspace")
            suffix = value[len(root_uri) :].lstrip("/")
            if any(part in {"", ".", ".."} for part in Path(suffix).parts):
                raise ValueError(
                    "path traversal is outside the configured Lance workspace"
                )
            return suffix
        if isinstance(self._workspace, LocalLanceWorkspace):
            root_path = Path(root_uri).resolve()
            candidate = (
                Path(value).expanduser().resolve()
                if Path(value).expanduser().is_absolute()
                else (root_path / value).resolve()
            )
            try:
                relative = candidate.relative_to(root_path)
            except ValueError as error:
                raise ValueError(
                    "path is outside the configured Lance workspace"
                ) from error
            return str(relative)
        if value.startswith("/"):
            raise ValueError(
                "absolute paths are unsupported for remote Lance workspaces"
            )
        if any(part == ".." for part in Path(value).parts):
            raise ValueError("path traversal is outside the configured Lance workspace")
        return value

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
            child not in {root.rsplit("/", 1)[-1] for root in roots}
            for child in child_prefixes
        ):
            raise ValueError("fragment source path contains unrelated child prefixes")
        return roots

    def compact_checkpoint(self, handle: CheckpointHandle) -> None:
        locator, _ = self._resolve_handle(handle)
        with self._commit_lock():
            row = self._resolve_manifest(locator)
            if int(row["compacted_version"]) != 0:
                return
        self._compact_checkpoint_rows((row,))

    def _compact_checkpoint_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        for row in rows:
            self._validate_manifest_fragments(row)
            version = int(row["checkpoint_version"])
            source_roots = self._manifest_fragment_roots(row)
            compact_root = (
                f"_compacted/{self._token(row['checkpoint_name'])}/v{version}/"
                f"{uuid.uuid4().hex}"
            )
            with self._commit_lock():
                current = self._manifest_for_version(row["checkpoint_name"], version)
                if current["compacted_version"] != 0:
                    continue
                self._validate_manifest_fragments(current)
                source_roots = self._manifest_fragment_roots(current)
                protections = self._compaction_tags(current, source_roots, compact_root)
                tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
                if any(tag in tags.list() for tag in protections):
                    raise CheckpointConflictError(
                        "checkpoint compaction intent already exists"
                    )
                for protection in protections:
                    tags.create(
                        protection,
                        self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).version,
                    )
                expected_digest = str(current["fragment_digest"])
            published = False
            try:
                self._workspace.create_prefix(compact_root)
                compaction_artifact = self._compact_fragment_native(
                    compact_root, source_roots
                )
                self._register_compacted_fragment_metadata(
                    compact_root,
                    current,
                    source_roots,
                    compaction_artifact,
                )
                compact_digest = self._fragment_digest_for_metadata(
                    (compact_root,), (compact_root,)
                )
                with self._commit_lock():
                    latest = self._manifest_for_version(row["checkpoint_name"], version)
                    if (
                        latest["commit_state"] != _COMMITTED
                        or int(latest["compacted_version"]) != 0
                        or str(latest["fragment_digest"]) != expected_digest
                    ):
                        raise CheckpointConflictError(
                            "checkpoint changed during compaction"
                        )
                    manifest_dataset = self._workspace.dataset(
                        SNAPSHOT_MANIFEST_DATASET
                    )
                    updated = manifest_dataset.update(
                        {
                            "fragment_ids": self._sql_list_literal([compact_root]),
                            "fragment_digest": f"'{compact_digest}'",
                            "compacted_version": str(version),
                        },
                        where=(
                            f"checkpoint_name = '{self._sql_literal(row['checkpoint_name'])}' "
                            f"AND checkpoint_version = {version} "
                            f"AND commit_state = '{_COMMITTED}' "
                            f"AND compacted_version = 0 "
                            f"AND fragment_digest = '{self._sql_literal(expected_digest)}'"
                        ),
                    )
                    if updated["num_rows_updated"] != 1:
                        raise CheckpointConflictError(
                            "checkpoint compaction publication was lost"
                        )
                    published = True
                try:
                    for root in source_roots:
                        self._workspace.delete_prefix(root)
                except BaseException as error:
                    raise RuntimeError(
                        "compaction retired fragment deletion was partial"
                    ) from error
                with self._commit_lock():
                    tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
                    for protection in protections:
                        if protection in tags.list():
                            tags.delete(protection)
            except BaseException:
                if not published:
                    cleanup_error: BaseException | None = None
                    try:
                        if self._workspace.exists(compact_root):
                            self._workspace.delete_prefix(compact_root)
                    except BaseException as error:
                        cleanup_error = error
                    if cleanup_error is None:
                        with self._commit_lock():
                            tags = self._workspace.dataset(
                                SNAPSHOT_MANIFEST_DATASET
                            ).tags
                            for protection in protections:
                                if protection in tags.list():
                                    tags.delete(protection)
                    else:
                        raise RuntimeError(
                            "unpublished compaction cleanup failed"
                        ) from cleanup_error
                raise

    async def compact_checkpoints_async(self) -> None:
        await asyncio.to_thread(self.compact_checkpoints)

    async def compact_checkpoint_async(self, handle: CheckpointHandle) -> None:
        await asyncio.to_thread(self.compact_checkpoint, handle)

    def _reclaim_unlocked(self, older_than: timedelta) -> None:
        self._reclaim_unlocked_for(older_than, frozenset())

    def _reclaim_unlocked_for(
        self,
        older_than: timedelta,
        authorized_locators: frozenset[SparseCheckpointLocator],
    ) -> None:
        now = datetime.now(timezone.utc)
        active_records = self._active_checkpoint_lease_records_unlocked(now)
        active_ids = {record.lease.lease_id for record in active_records}
        leased_locators = {record.locator for record in active_records}
        manifest_dataset = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET)
        rows = self._manifest_rows()
        tags = manifest_dataset.tags
        for tag in tuple(tags.list()):
            if tag.startswith("selease.") and tag.split(".", 2)[1] not in active_ids:
                tags.delete(tag)
        protected = {
            locator
            for locator in (
                self._decode_reference_tag(tag)
                for tag in tuple(tags.list())
                if tag.startswith("sereference.")
            )
        }
        for row in rows:
            locator = SparseCheckpointLocator(
                row["checkpoint_name"], int(row["checkpoint_version"])
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
                or self._locator_has_compaction(locator)
            ):
                continue
            latest_version = max(
                int(candidate["checkpoint_version"])
                for candidate in rows
                if candidate["checkpoint_name"] == locator.checkpoint_name
                and candidate["commit_state"] == _COMMITTED
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
                created_at = created_at.replace(tzinfo=timezone.utc)
            if now - created_at < older_than:
                continue
            roots = self._manifest_fragment_roots(row)
            reclaim_tag = self._reclaim_pending_tag(locator)
            if reclaim_tag not in tags.list():
                tags.create(reclaim_tag, manifest_dataset.version)
            try:
                for root in roots:
                    self._workspace.delete_prefix(root)
            except BaseException as error:
                raise RuntimeError(
                    "checkpoint fragment reclaim was partially completed"
                ) from error
            manifest_dataset.delete(
                f"checkpoint_name = '{self._sql_literal(locator.checkpoint_name)}' "
                f"AND checkpoint_version = {locator.checkpoint_version}"
            )
            if reclaim_tag in tags.list():
                tags.delete(reclaim_tag)
        self._reconcile_staging()

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
                raise RuntimeError("checkpoint reclaim marker has no manifest")
            if manifest["commit_state"] != _COMMITTED:
                raise RuntimeError(
                    "checkpoint reclaim marker manifest is not committed"
                )
            roots = self._manifest_fragment_roots(manifest)
            states = tuple(self._workspace.exists(root) for root in roots)
            if any(states) and not all(states):
                raise RuntimeError(
                    "checkpoint reclaim found partially deleted fragments"
                )
            try:
                for root in roots:
                    if self._workspace.exists(root):
                        self._workspace.delete_prefix(root)
            except BaseException as error:
                raise RuntimeError("checkpoint reclaim recovery was partial") from error
            self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).delete(
                f"checkpoint_name = '{self._sql_literal(locator.checkpoint_name)}' "
                f"AND checkpoint_version = {locator.checkpoint_version}"
            )
            if tag in tags.list():
                tags.delete(tag)

    @staticmethod
    def _reclaim_marker_locator(tag: str) -> SparseCheckpointLocator:
        parts = tag.split(".")
        if len(parts) != 3 or not parts[1].isdecimal():
            raise RuntimeError("checkpoint reclaim marker is corrupt")
        try:
            return SparseCheckpointLocator(
                urlsafe_b64decode(parts[2] + "=" * (-len(parts[2]) % 4)).decode(),
                int(parts[1]),
            )
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeError("checkpoint reclaim marker is corrupt") from error

    def _reconcile_compaction_retirements(self) -> None:
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        for tag in tuple(tags.list()):
            if not tag.startswith("secompact."):
                continue
            parts = tag.split(".")
            if len(parts) != 4:
                raise RuntimeError("checkpoint compaction marker is corrupt")
            locator = self._compaction_marker_locator(tag)
            try:
                target_root = self._decode(parts[3])
            except (TypeError, ValueError, UnicodeDecodeError) as error:
                raise RuntimeError("checkpoint compaction marker is corrupt") from error
            if not target_root.startswith("_compacted/"):
                raise RuntimeError("checkpoint compaction marker target is corrupt")
            if self._workspace_path(target_root) != target_root:
                raise RuntimeError("checkpoint compaction marker target is corrupt")
            try:
                manifest = self._manifest_for_version(
                    locator.checkpoint_name, locator.checkpoint_version
                )
            except LookupError:
                raise RuntimeError("checkpoint compaction marker has no manifest")
            if manifest["commit_state"] != _COMMITTED:
                raise RuntimeError(
                    "checkpoint compaction marker manifest is not committed"
                )
            current_roots = self._manifest_fragment_roots(manifest)
            if target_root not in current_roots:
                if self._workspace.exists(target_root):
                    self._workspace.delete_prefix(target_root)
                if tag in tags.list():
                    tags.delete(tag)
                continue
            if not self._workspace.exists(target_root):
                raise RuntimeError(
                    "checkpoint compaction marker target fragment is missing"
                )
            self._validate_manifest_fragments(manifest)
            source_roots = self._participant_fragment_roots(manifest)
            states = tuple(self._workspace.exists(root) for root in source_roots)
            if any(states) and not all(states):
                raise RuntimeError(
                    "checkpoint compaction found partially retired fragments"
                )
            try:
                for root in source_roots:
                    if self._workspace.exists(root):
                        self._workspace.delete_prefix(root)
            except BaseException as error:
                raise RuntimeError(
                    "checkpoint compaction recovery was partial"
                ) from error
            if tag in tags.list():
                tags.delete(tag)

    @staticmethod
    def _compaction_marker_locator(tag: str) -> SparseCheckpointLocator:
        parts = tag.split(".")
        if len(parts) != 4 or not parts[1].isdecimal():
            raise RuntimeError("checkpoint compaction marker is corrupt")
        try:
            return SparseCheckpointLocator(
                urlsafe_b64decode(parts[2] + "=" * (-len(parts[2]) % 4)).decode(),
                int(parts[1]),
            )
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeError("checkpoint compaction marker is corrupt") from error

    def reserve_cleanup(
        self,
        reservation_id: str,
        ordered_candidates: Sequence[CheckpointHandle],
        required_count: int,
    ) -> CleanupReservation:
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
            except BaseException:
                for tag in reversed(created):
                    if tag in tags.list():
                        tags.delete(tag)
                raise
            return reservation

    def commit_cleanup(self, reservation: CleanupReservation) -> None:
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
        locator, reference_id = self._resolve_handle(handle)
        with self._commit_lock():
            self._delete_reference_tags(locator, reference_id)
            self._reclaim_unlocked(timedelta(0))

    def _validate_cleanup_reservation(self, reservation: CleanupReservation) -> None:
        expected = self._reservation_owner_handle(
            reservation.reservation_id, reservation.handles
        )
        if reservation.owner_handle != expected:
            raise CheckpointConflictError("cleanup reservation owner handle is invalid")

    def load_component_checkpoint(
        self,
        store: Any,
        storage_locator: CheckpointHandle,
        owner_restore_options: bytes,
        *,
        load_optimizer_state: bool = True,
    ) -> None:
        if not isinstance(load_optimizer_state, bool):
            raise TypeError("load_optimizer_state must be bool")
        mode = _decode_restore_options(owner_restore_options)
        locator, _ = self._resolve_handle(storage_locator)
        self._load_component_checkpoint_by_locator(
            store, locator, mode, load_optimizer_state=load_optimizer_state
        )

    def _load_component_checkpoint_by_locator(
        self,
        store: Any,
        locator: SparseCheckpointLocator,
        mode: RestoreMode,
        *,
        load_optimizer_state: bool = True,
    ) -> None:
        manifest = self._resolve_manifest(locator)
        self._restore_committed_record(
            store, manifest, mode, _RestoreScope.TRAINING, load_optimizer_state
        )

    def load_inference_weights(
        self,
        store: Any,
        owner_restore_options: bytes,
    ) -> None:
        if self._bound_handle is None:
            raise RuntimeError(
                "inference weights require LocalLanceStorage.from_weight_path"
            )
        mode = _decode_restore_options(owner_restore_options)
        if not isinstance(mode, ElasticRestore):
            raise ValueError("inference weight restore requires an elastic mode")
        locator, _ = self._resolve_handle(self._bound_handle)
        manifest = self._resolve_manifest(locator)
        self._restore_committed_record(
            store, manifest, mode, _RestoreScope.INFERENCE, False
        )

    def write_weight_reference(
        self,
        path: str,
        handle: CheckpointHandle,
    ) -> None:
        locator, _ = self._resolve_handle(handle)
        if not isinstance(self._workspace, LocalLanceWorkspace):
            raise NotImplementedError(
                "weight reference writing is only implemented for local Lance paths"
            )
        manifest = self._resolve_manifest(locator)
        if not str(manifest["reference_id"]):
            raise CheckpointConflictError(
                "weight reference requires a committed checkpoint reference"
            )
        asset_root = Path(path).expanduser()
        if asset_root.is_symlink() or not asset_root.is_dir():
            raise FileNotFoundError(
                f"sparse weight reference directory does not exist: {asset_root}"
            )
        if any(
            (asset_root / name).exists()
            for name in (
                SNAPSHOT_MANIFEST_DATASET,
                ENTRIES_DATASET,
                PAGE_IMAGES_DATASET,
            )
        ):
            raise ValueError(
                "sparse weight path must contain a reference asset, not a checkpoint root"
            )
        reference_path = asset_root / _WEIGHT_REFERENCE_FILENAME
        if any(
            item.name != _WEIGHT_REFERENCE_FILENAME for item in asset_root.iterdir()
        ):
            raise ValueError("sparse weight reference directory has an unknown layout")
        if reference_path.is_symlink():
            raise ValueError("sparse weight reference must not be a symlink")
        payload = handle
        if reference_path.exists():
            if not reference_path.is_file() or reference_path.read_bytes() != payload:
                raise CheckpointConflictError(
                    "sparse weight reference already exists with another identity"
                )
            return
        temporary = asset_root / f".{_WEIGHT_REFERENCE_FILENAME}.tmp-{uuid.uuid4().hex}"
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            descriptor = os.open(temporary, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    descriptor = -1
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            os.replace(temporary, reference_path)
            directory_descriptor = os.open(asset_root, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            if reference_path.read_bytes() != payload:
                raise RuntimeError("sparse weight reference verification failed")
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise

    def _restore_committed_record(
        self,
        store: Any,
        manifest: dict[str, Any],
        mode: RestoreMode,
        scope: _RestoreScope,
        load_optimizer_state: bool,
    ) -> None:
        specification = self._manifest_specification(manifest)
        if manifest["schema_digest"] != _schema_digest(specification):
            raise RuntimeError("checkpoint schema digest mismatch")
        if load_optimizer_state and not bool(store.optimizer_enabled()):
            raise ValueError(
                "load_optimizer_state requires a target store with optimizer enabled"
            )
        mode_key_dtype = (
            mode.key_dtype
            if isinstance(mode, (PhysicalRestore, ElasticRestore))
            else None
        )
        if mode_key_dtype != specification.key_dtype:
            raise ValueError(
                "restore options sparse ID dtype does not match checkpoint"
            )
        self._validate_restore_target(store, specification, scope, load_optimizer_state)
        if manifest["commit_state"] != _COMMITTED:
            raise LookupError("checkpoint manifest is not committed")
        if manifest["storage_mode"] != _FRAGMENT_STORAGE_MODE:
            raise RuntimeError("checkpoint storage mode is unsupported")
        self._validate_manifest_fragments(manifest)
        if store.total_size() != 0:
            raise RuntimeError("restore requires an empty target store")
        if isinstance(mode, PhysicalRestore):
            if mode.participant.participant_index >= int(manifest["participant_count"]):
                raise ValueError("restore participant is outside saved topology")
            self._restore_physical(store, manifest, mode, load_optimizer_state)
            return
        if isinstance(mode, ElasticRestore):
            self._restore_elastic(store, manifest, mode, load_optimizer_state)
            return
        raise TypeError("unsupported restore mode")

    @classmethod
    def _validate_restore_target(
        cls,
        store: Any,
        specification: TableSpecification,
        scope: _RestoreScope,
        load_optimizer_state: bool = True,
    ) -> None:
        if scope is _RestoreScope.TRAINING:
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
                raise ValueError(
                    "restore target does not match checkpoint specification"
                )
            return
        if scope is not _RestoreScope.INFERENCE:
            raise ValueError(f"unsupported restore scope: {scope!r}")
        if not hasattr(store, "optimizer_enabled") or store.optimizer_enabled():
            raise ValueError("inference target store must have optimizer enabled=False")
        target = cls._store_specification(store)
        if (
            target.dimensions != specification.dimensions
            or target.embedding_dtype != specification.embedding_dtype
            or target.score_policy != specification.score_policy
            or target.key_dtype != specification.key_dtype
        ):
            raise ValueError("inference target does not match weight reference")

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
        if incoming_generation > current_generation:
            if fence_tag not in names:
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

    def _clear_authority(self, identity: CheckpointSaveIdentity) -> None:
        if not self._workspace.exists(SNAPSHOT_MANIFEST_DATASET):
            return
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        name = self._authority_tag(identity)
        if name in tags.list():
            tags.delete(name)

    def _reconcile_staging(self) -> None:
        if not self._workspace.exists("_staging"):
            return
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list()
        active = {
            ".".join(tag.split(".")[1:])
            for tag in tags
            if tag.startswith("seauthority.")
        }
        referenced = {
            root.split("/", 2)[1]
            for row in self._manifest_rows()
            if row["commit_state"] == _COMMITTED
            for root in self._manifest_fragment_roots(row)
            if root.startswith("_staging/") and len(root.split("/", 2)) >= 2
        }
        retirement_attempts: set[str] = set()
        for tag in tags:
            if not tag.startswith("secompact."):
                continue
            parts = tag.split(".")
            if len(parts) != 4:
                raise RuntimeError("checkpoint compaction marker is corrupt")
            try:
                locator = SparseCheckpointLocator(self._decode(parts[2]), int(parts[1]))
                manifest = self._manifest_for_version(
                    locator.checkpoint_name, locator.checkpoint_version
                )
            except (TypeError, ValueError, UnicodeDecodeError, LookupError) as error:
                raise RuntimeError("checkpoint compaction marker is corrupt") from error
            for root in self._participant_fragment_roots(manifest):
                if root.startswith("_staging/") and len(root.split("/", 2)) >= 2:
                    retirement_attempts.add(root.split("/", 2)[1])
        for attempt in self._workspace.child_prefixes("_staging"):
            if (
                attempt not in active
                and attempt not in referenced
                and attempt not in retirement_attempts
            ):
                self._workspace.delete_prefix(f"_staging/{attempt}", missing_ok=True)

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

    def _delete_reference_tags(
        self, locator: SparseCheckpointLocator, reference_id: str
    ) -> None:
        tag = self._reference_tag(locator, reference_id)
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags
        if tag in tags.list():
            tags.delete(tag)

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
            if not raw_generation.isdecimal():
                raise RuntimeError("checkpoint authority fence tag is corrupt")
            generations.append(int(raw_generation))
        return max(generations, default=-1)

    def _authority_generation_from_tag(
        self, identity: CheckpointSaveIdentity, tag: str
    ) -> int:
        suffix = tag.removeprefix(self._authority_scope_prefix(identity))
        parts = suffix.split(".")
        if len(parts) != 3 or not parts[1].isdecimal():
            raise RuntimeError("checkpoint authority tag is corrupt")
        return int(parts[1])

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
        parts = tag.split(".")
        if len(parts) != 8:
            raise RuntimeError("invalid sparse checkpoint lease tag")
        lease = CheckpointLease(
            lease_id=parts[1],
            lease_holder_id=self._decode(parts[6]),
            expires_at=datetime.fromtimestamp(int(parts[3]) / 1_000_000, timezone.utc),
            generation=int(parts[2]),
            request_id=self._decode(parts[7]),
        )
        return _LeaseRecord(
            lease,
            SparseCheckpointLocator(self._decode(parts[5]), int(parts[4])),
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
        parts = tags[0].split(".")
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
                "authority_generation": int(authority_generation),
                "checkpoint_version": int(version),
                "manifest_before": int(manifest_before),
            }
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("durable checkpoint commit outcome is corrupt") from error
        if decoded["authority_generation"] != identity.commit_authority.generation:
            if not require_identity:
                return None
            raise CheckpointConflictError(
                "durable checkpoint outcome belongs to another authority generation"
            )
        manifests = [
            row
            for row in self._manifest_rows()
            if row["checkpoint_name"] == identity.checkpoint_name
            and int(row["checkpoint_version"]) == decoded["checkpoint_version"]
        ]
        if len(manifests) != 1:
            raise CheckpointConflictError(
                "durable checkpoint outcome has no unique manifest"
            )
        manifest = manifests[0]
        expected_identity = (
            manifest["save_attempt"] == identity.save_attempt
            and manifest["authority_scope"] == identity.commit_authority.scope
            and int(manifest["authority_generation"])
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
        return tag.startswith("sereserved.") or tag.startswith("sepending.")

    def _reservation_locator_suffix(self, locator: SparseCheckpointLocator) -> str:
        return f".{locator.checkpoint_version}.{self._encode(locator.checkpoint_name)}"

    def _locator_has_reservation(self, locator: SparseCheckpointLocator) -> bool:
        tags = self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list()
        suffix = self._reservation_locator_suffix(locator)
        return any(
            self._is_reservation_tag(tag) and tag.endswith(suffix) for tag in tags
        )

    def _validate_fragments(
        self,
        identity: CheckpointSaveIdentity,
        fragments: Sequence[CheckpointFragment],
    ) -> tuple[list[CheckpointFragment], TableSpecification]:
        result = list(fragments)
        if not result:
            raise InvalidCheckpointFragment("checkpoint requires fragments")
        expected_count = result[0].participant.participant_count
        expected_indices = set(range(expected_count))
        actual_indices = {item.participant.participant_index for item in result}
        if len(actual_indices) != len(result) or actual_indices != expected_indices:
            raise InvalidCheckpointFragment("checkpoint fragment coverage is invalid")
        specification = self._fragment_specification(identity, result[0])
        for item in result:
            if (
                item.checkpoint_name != identity.checkpoint_name
                or item.save_attempt != identity.save_attempt
            ):
                raise InvalidCheckpointFragment("checkpoint fragment identity mismatch")
            if item.participant.participant_count != expected_count:
                raise InvalidCheckpointFragment("checkpoint participant count mismatch")
            if self._fragment_specification(identity, item) != specification:
                raise InvalidCheckpointFragment(
                    "checkpoint fragment specification mismatch"
                )
            if item.schema_digest != _schema_digest(specification):
                raise InvalidCheckpointFragment("checkpoint fragment schema mismatch")
            if item.fragment_id != self._fragment_id(identity, item.participant):
                raise InvalidCheckpointFragment(
                    "checkpoint fragment identifier mismatch"
                )
            root = f"{self._attempt_prefix(identity)}/{item.fragment_id}"
            metadata = self._metadata_for_root(root)
            if metadata["fragment_id"] != item.fragment_id:
                raise RuntimeError("checkpoint fragment metadata identifier mismatch")
            if metadata["schema_digest"] != item.schema_digest:
                raise RuntimeError("checkpoint fragment metadata schema mismatch")
            if metadata["key_dtype"] != specification.key_dtype:
                raise InvalidCheckpointFragment(
                    "checkpoint fragment key encoding mismatch"
                )
            if tuple(int(index) for index in metadata["participant_indices"]) != (
                item.participant.participant_index,
            ):
                raise RuntimeError("checkpoint fragment metadata participant mismatch")
            if (
                int(metadata["entry_count"])
                != self._workspace.dataset(f"{root}/{ENTRIES_DATASET}").count_rows()
            ):
                raise RuntimeError("checkpoint fragment entry count mismatch")
            if self._workspace.exists(f"{root}/{PAGE_IMAGES_DATASET}") and (
                int(metadata["page_count"])
                != self._workspace.dataset(f"{root}/{PAGE_IMAGES_DATASET}").count_rows()
            ):
                raise RuntimeError("checkpoint fragment page count mismatch")
            if (
                metadata["fragment_id"] != item.fragment_id
                or int(metadata["participant_count"]) != expected_count
                or int(metadata["entry_count"]) != item.entry_count
                or int(metadata["page_count"]) != item.page_count
                or metadata["schema_digest"] != item.schema_digest
                or metadata["content_digest"] != item.content_digest
                or metadata["key_dtype"] != specification.key_dtype
            ):
                raise InvalidCheckpointFragment("checkpoint fragment metadata mismatch")
            if not self._workspace.exists(f"{root}/{ENTRIES_DATASET}"):
                raise InvalidCheckpointFragment(
                    "checkpoint fragment payload is missing"
                )
        return (
            sorted(result, key=lambda item: item.participant.participant_index),
            specification,
        )

    def _manifest_rows(self) -> list[dict[str, Any]]:
        return self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).to_table().to_pylist()

    def _manifest_for_version(
        self, checkpoint_name: str, version: int
    ) -> dict[str, Any]:
        matches = [
            row
            for row in self._manifest_rows()
            if row["checkpoint_name"] == checkpoint_name
            and int(row["checkpoint_version"]) == version
        ]
        if len(matches) != 1:
            raise LookupError("checkpoint manifest is missing or ambiguous")
        return matches[0]

    def _resolve_manifest(self, locator: SparseCheckpointLocator) -> dict[str, Any]:
        matches = [
            row
            for row in self._manifest_rows()
            if row["checkpoint_name"] == locator.checkpoint_name
            and int(row["checkpoint_version"]) == locator.checkpoint_version
        ]
        if len(matches) != 1:
            raise LookupError("checkpoint locator is missing or ambiguous")
        if matches[0]["commit_state"] != _COMMITTED:
            raise LookupError("checkpoint manifest is not committed")
        return matches[0]

    def _manifest_fragment_roots(self, manifest: dict[str, Any]) -> tuple[str, ...]:
        fragment_ids = tuple(str(item) for item in manifest["fragment_ids"])
        if not fragment_ids:
            raise RuntimeError("checkpoint manifest has no fragments")
        absolute_roots = tuple(item.startswith("_") for item in fragment_ids)
        if any(absolute_roots) and not all(absolute_roots):
            raise RuntimeError("checkpoint manifest mixes fragment ID and root forms")
        if all(absolute_roots):
            if any("/" not in item for item in fragment_ids):
                raise RuntimeError("checkpoint manifest has an invalid fragment root")
            return fragment_ids
        identity = CheckpointSaveIdentity(
            manifest["checkpoint_name"],
            manifest["save_attempt"],
            CheckpointCommitAuthority(
                manifest["authority_scope"],
                int(manifest["authority_generation"]),
                manifest["commit_authority"],
            ),
        )
        return tuple(
            f"{self._attempt_prefix(identity)}/{fragment_id}"
            for fragment_id in fragment_ids
        )

    def _participant_fragment_roots(self, manifest: dict[str, Any]) -> tuple[str, ...]:
        identity = CheckpointSaveIdentity(
            manifest["checkpoint_name"],
            manifest["save_attempt"],
            CheckpointCommitAuthority(
                manifest["authority_scope"],
                int(manifest["authority_generation"]),
                manifest["commit_authority"],
            ),
        )
        return tuple(
            f"{self._attempt_prefix(identity)}/"
            f"{self._fragment_id(identity, CheckpointParticipant(index, int(manifest['participant_count'])))}"
            for index in range(int(manifest["participant_count"]))
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
        roots = self._manifest_fragment_roots(manifest)
        fragment_ids = tuple(str(item) for item in manifest["fragment_ids"])
        if len(fragment_ids) != len(set(fragment_ids)):
            raise RuntimeError("checkpoint manifest contains duplicate fragments")
        expected_digest = self._fragment_digest_for_metadata(fragment_ids, roots)
        if expected_digest != str(manifest["fragment_digest"]):
            raise RuntimeError("checkpoint manifest fragment digest mismatch")
        all_participants: set[int] = set()
        for fragment_id, root in zip(fragment_ids, roots, strict=True):
            entries = f"{root}/{ENTRIES_DATASET}"
            if not self._workspace.exists(entries):
                raise LookupError("checkpoint manifest references a missing fragment")
            dataset = self._workspace.dataset(entries)
            if (
                dataset.schema.remove_metadata()
                != entry_schema(manifest["key_dtype"]).remove_metadata()
            ):
                raise RuntimeError("checkpoint fragment schema mismatch")
            metadata = dataset.schema.metadata or {}
            specification = self._decode_specification(metadata)
            if specification.key_dtype != manifest["key_dtype"]:
                raise RuntimeError("checkpoint fragment key encoding mismatch")
            if _schema_digest(specification) != str(manifest["schema_digest"]):
                raise RuntimeError("checkpoint fragment schema digest mismatch")
            if self._workspace.exists(f"{root}/{PAGE_IMAGES_DATASET}"):
                pages_ds = self._workspace.dataset(f"{root}/{PAGE_IMAGES_DATASET}")
                if (
                    pages_ds.schema.remove_metadata()
                    != PAGE_IMAGE_SCHEMA.remove_metadata()
                ):
                    raise RuntimeError("checkpoint fragment schema mismatch")
            metadata = self._metadata_for_root(root)
            if metadata["fragment_id"] != fragment_id:
                raise RuntimeError("checkpoint fragment metadata identifier mismatch")
            if metadata["schema_digest"] != str(manifest["schema_digest"]):
                raise RuntimeError("checkpoint fragment metadata schema mismatch")
            if metadata["key_dtype"] != manifest["key_dtype"]:
                raise RuntimeError("checkpoint fragment key encoding mismatch")
            if (
                int(metadata["entry_count"])
                != self._workspace.dataset(f"{root}/{ENTRIES_DATASET}").count_rows()
            ):
                raise RuntimeError("checkpoint fragment entry count mismatch")
            if self._workspace.exists(f"{root}/{PAGE_IMAGES_DATASET}") and (
                int(metadata["page_count"])
                != self._workspace.dataset(f"{root}/{PAGE_IMAGES_DATASET}").count_rows()
            ):
                raise RuntimeError("checkpoint fragment page count mismatch")
            participants = {int(index) for index in metadata["participant_indices"]}
            expected_participants = set(range(int(manifest["participant_count"])))
            if not participants.issubset(expected_participants):
                raise RuntimeError(
                    "checkpoint fragment participant is outside topology"
                )
            all_participants.update(participants)
        expected_participants = set(range(int(manifest["participant_count"])))
        if len(fragment_ids) > 1 and len(fragment_ids) != int(
            manifest["participant_count"]
        ):
            raise RuntimeError("checkpoint manifest participant coverage is incomplete")
        if len(fragment_ids) > 1:
            participant_sets = [
                tuple(
                    sorted(
                        int(index)
                        for index in self._metadata_for_root(root)[
                            "participant_indices"
                        ]
                    )
                )
                for root in roots
            ]
            if any(len(indices) != 1 for indices in participant_sets) or len(
                {indices[0] for indices in participant_sets}
            ) != len(participant_sets):
                raise RuntimeError(
                    "checkpoint manifest fragment participant routing is ambiguous"
                )
        if all_participants != expected_participants:
            raise RuntimeError("checkpoint manifest participant coverage is incomplete")
        for root in roots:
            metadata = self._metadata_for_root(root)
            if int(metadata["participant_count"]) != int(manifest["participant_count"]):
                raise RuntimeError("checkpoint fragment participant count mismatch")

    def _fragment_digest_for_metadata(
        self, fragment_ids: Sequence[str], roots: Sequence[str]
    ) -> str:
        if not roots or len(fragment_ids) != len(roots):
            raise RuntimeError("fragment metadata and root counts differ")
        return _canonical_digest(
            [
                {
                    "fragment_id": fragment_id,
                    "content_digest": self._metadata_for_root(root)["content_digest"],
                }
                for fragment_id, root in zip(fragment_ids, roots, strict=True)
            ]
        )

    def _metadata_for_root(self, root: str) -> dict[str, Any]:
        entries = self._workspace.dataset(f"{root}/{ENTRIES_DATASET}")
        entry_metadata = entries.schema_metadata
        required = {
            "sparse_embedding.fragment_id",
            "sparse_embedding.participant_indices",
            "sparse_embedding.participant_count",
            "sparse_embedding.entry_count",
            "sparse_embedding.page_count",
            "sparse_embedding.fragment_schema_digest",
            "sparse_embedding.fragment_content_digest",
            "sparse_embedding.key_dtype",
        }
        if not required.issubset(entry_metadata):
            raise LookupError("fragment metadata is missing or inconsistent")
        if self._workspace.exists(f"{root}/{PAGE_IMAGES_DATASET}"):
            pages = self._workspace.dataset(f"{root}/{PAGE_IMAGES_DATASET}")
            page_metadata = pages.schema_metadata
            if not required.issubset(page_metadata) or entry_metadata != page_metadata:
                raise LookupError("fragment metadata is missing or inconsistent")

        def parse_indices(value: str) -> list[int]:
            values = [int(item) for item in value.split(",") if item]
            if not values or len(values) != len(set(values)):
                raise RuntimeError("fragment metadata participant coverage is invalid")
            return values

        return {
            "fragment_id": entry_metadata["sparse_embedding.fragment_id"],
            "participant_indices": parse_indices(
                entry_metadata["sparse_embedding.participant_indices"]
            ),
            "participant_count": int(
                entry_metadata["sparse_embedding.participant_count"]
            ),
            "entry_count": int(entry_metadata["sparse_embedding.entry_count"]),
            "page_count": int(entry_metadata["sparse_embedding.page_count"]),
            "schema_digest": entry_metadata["sparse_embedding.fragment_schema_digest"],
            "content_digest": entry_metadata[
                "sparse_embedding.fragment_content_digest"
            ],
            "key_dtype": entry_metadata["sparse_embedding.key_dtype"],
        }

    def _publish_fragment_metadata(
        self,
        root: str,
        fragment_id: str,
        participant_index: int,
        participant_count: int,
        entry_count: int,
        page_count: int,
        schema_digest: str,
        content_digest: str,
        participant_indices: Sequence[int] | None = None,
    ) -> None:
        indices = tuple(participant_indices or (participant_index,))
        if not indices or len(set(indices)) != len(indices):
            raise RuntimeError("fragment metadata participant coverage is invalid")
        metadata = {
            "sparse_embedding.fragment_id": fragment_id,
            "sparse_embedding.participant_indices": ",".join(
                str(index) for index in indices
            ),
            "sparse_embedding.participant_count": str(participant_count),
            "sparse_embedding.entry_count": str(entry_count),
            "sparse_embedding.page_count": str(page_count),
            "sparse_embedding.fragment_schema_digest": schema_digest,
            "sparse_embedding.fragment_content_digest": content_digest,
        }
        for name in (ENTRIES_DATASET, PAGE_IMAGES_DATASET):
            if not self._workspace.exists(f"{root}/{name}"):
                continue
            dataset = self._workspace.dataset(f"{root}/{name}")
            existing = dataset.schema_metadata
            base = dict(existing)
            base.update(metadata)
            dataset.update_schema_metadata(base, replace=True)

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
        fragment_id: str,
        expected_participant_count: int | None,
    ) -> dict[str, Any]:
        artifact = self._compact_fragment_native(compact_root, source_roots)
        metadata = self._publish_compacted_fragment_metadata(
            compact_root,
            source_roots,
            artifact,
            expected_participant_count=expected_participant_count,
            fragment_id=fragment_id,
        )
        return {"artifact": artifact, "metadata": metadata}

    def _register_compacted_fragment_metadata(
        self,
        compact_root: str,
        manifest: dict[str, Any],
        source_roots: Sequence[str],
        compaction_artifact: dict[str, Any],
    ) -> None:
        self._publish_compacted_fragment_metadata(
            compact_root,
            source_roots,
            compaction_artifact,
            expected_participant_count=int(manifest["participant_count"]),
            fragment_id=compact_root,
        )

    def _build_compacted_fragment_metadata(
        self,
        compact_root: str,
        source_roots: Sequence[str],
        compaction_artifact: dict[str, Any],
        expected_participant_count: int | None,
    ) -> dict[str, Any]:
        if not source_roots:
            raise RuntimeError("fragment compaction requires source fragments")
        source_metadata = [self._metadata_for_root(root) for root in source_roots]
        participant_counts = {
            int(metadata["participant_count"]) for metadata in source_metadata
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
        participant_indices = tuple(
            sorted(
                {
                    int(index)
                    for metadata in source_metadata
                    for index in metadata["participant_indices"]
                }
            )
        )
        if (expected_participant_count is not None or len(source_roots) > 1) and (
            participant_indices != tuple(range(participant_count))
        ):
            raise RuntimeError("compaction participant coverage is incomplete")
        entry_count = self._workspace.dataset(
            f"{compact_root}/{ENTRIES_DATASET}"
        ).count_rows()
        if int(compaction_artifact["entries"]["inserted_rows"]) != entry_count:
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
            if int(compaction_artifact["page_images"]["inserted_rows"]) != page_count:
                raise RuntimeError("compaction output row counts differ from sources")
            page_schema = self._workspace.dataset(
                f"{compact_root}/{PAGE_IMAGES_DATASET}"
            ).schema
            if page_schema.remove_metadata() != PAGE_IMAGE_SCHEMA.remove_metadata():
                raise RuntimeError("compaction output schema mismatch")
            page_digest = str(compaction_artifact["page_images"]["content_digest"])
        else:
            page_count = sum(
                int(self._metadata_for_root(root)["page_count"])
                for root in source_roots
            )
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
        }

    def _publish_compacted_fragment_metadata(
        self,
        compact_root: str,
        source_roots: Sequence[str],
        compaction_artifact: dict[str, Any],
        *,
        expected_participant_count: int | None,
        fragment_id: str,
    ) -> dict[str, Any]:
        metadata = self._build_compacted_fragment_metadata(
            compact_root,
            source_roots,
            compaction_artifact,
            expected_participant_count,
        )
        self._publish_fragment_metadata(
            compact_root,
            fragment_id,
            metadata["participant_indices"][0],
            metadata["participant_count"],
            metadata["entry_count"],
            metadata["page_count"],
            metadata["schema_digest"],
            metadata["content_digest"],
            metadata["participant_indices"],
        )
        return metadata

    @staticmethod
    def _sql_literal(value: str) -> str:
        return value.replace("'", "''")

    @classmethod
    def _sql_list_literal(cls, values: Sequence[str]) -> str:
        if not values:
            raise ValueError("SQL list literal requires at least one value")
        return (
            "["
            + ",".join("'" + cls._sql_literal(value) + "'" for value in values)
            + "]"
        )

    def _intent_tag(
        self, identity: CheckpointSaveIdentity, locator: SparseCheckpointLocator
    ) -> str:
        return f"seintent.{self._token(identity.save_attempt)}.{locator.checkpoint_version}"

    def _committed_tag(self, locator: SparseCheckpointLocator) -> str:
        return f"secommitted.{locator.checkpoint_version}.{self._encode(locator.checkpoint_name)}"

    def _compaction_tags(
        self,
        manifest: dict[str, Any],
        source_roots: Sequence[str],
        compact_root: str,
    ) -> tuple[str, ...]:
        return (
            "secompact."
            f"{manifest['checkpoint_version']}."
            f"{self._encode(str(manifest['checkpoint_name']))}."
            f"{self._encode(compact_root)}",
        )

    def _reclaim_pending_tag(self, locator: SparseCheckpointLocator) -> str:
        return f"sereclaiming.{locator.checkpoint_version}.{self._encode(locator.checkpoint_name)}"

    def _locator_has_pending_intent(self, locator: SparseCheckpointLocator) -> bool:
        return any(
            row["checkpoint_name"] == locator.checkpoint_name
            and int(row["checkpoint_version"]) == locator.checkpoint_version
            and row["commit_state"] == _PREPARED
            for row in self._manifest_rows()
        )

    def _locator_has_compaction(self, locator: SparseCheckpointLocator) -> bool:
        for tag in self._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list():
            if not tag.startswith("secompact."):
                continue
            parts = tag.split(".")
            if len(parts) != 4:
                raise RuntimeError("checkpoint compaction marker is corrupt")
            try:
                marker_locator = self._compaction_marker_locator(tag)
            except (TypeError, ValueError, UnicodeDecodeError, RuntimeError) as error:
                raise RuntimeError("checkpoint compaction marker is corrupt") from error
            if marker_locator == locator:
                return True
        return False

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
        parts = tag.split(".")
        if len(parts) != 4:
            raise RuntimeError("checkpoint reference tag is corrupt")
        return SparseCheckpointLocator(self._decode(parts[3]), int(parts[2]))

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
        row = matches[0]
        locator = SparseCheckpointLocator(
            identity.checkpoint_name, int(row["checkpoint_version"])
        )
        if row["storage_mode"] != _FRAGMENT_STORAGE_MODE:
            raise RuntimeError("checkpoint manifest storage mode is corrupt")
        if row["commit_state"] == _COMMITTED:
            self._validate_manifest_fragments(row)
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
        self._validate_manifest_fragments(row)
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
        self._clear_authority(identity)
        return locator

    def _restore_elastic(
        self,
        store: Any,
        manifest: dict[str, Any],
        mode: ElasticRestore,
        load_optimizer_state: bool,
    ) -> None:
        roots = self._manifest_fragment_roots(manifest)
        if not hasattr(_checkpoint_native, "restore_native_partition_entries"):
            raise RuntimeError(
                "native checkpoint restore requires a Lance-enabled _core extension"
            )
        if not hasattr(_checkpoint_native, "validate_active_entry_keys"):
            raise RuntimeError(
                "native checkpoint duplicate validation requires a Lance-enabled _core extension"
            )
        try:
            _checkpoint_native.validate_active_entry_keys(
                [self._workspace.uri(f"{root}/{ENTRIES_DATASET}") for root in roots],
                0,
                dict(self._workspace.storage_options()),
            )
        except ValueError as error:
            if "duplicate active key" not in str(error):
                raise
            raise InvalidCheckpointFragment(
                "duplicate active checkpoint key"
            ) from error
        assignment_name = mode.spec.name
        assignment_parameters = {
            str(k): str(v) for k, v in mode.spec.parameters.items()
        }
        for root in roots:
            _checkpoint_native.restore_native_partition_entries(
                store,
                self._workspace.uri(f"{root}/{ENTRIES_DATASET}"),
                0,
                mode.partition_count,
                mode.partition_id,
                mode.key_dtype,
                None,
                dict(self._workspace.storage_options()),
                load_optimizer_state,
                assignment_name,
                assignment_parameters,
            )

    def _restore_physical(
        self,
        store: Any,
        manifest: dict[str, Any],
        mode: PhysicalRestore,
        load_optimizer_state: bool,
    ) -> None:
        if mode.participant.participant_count != int(manifest["participant_count"]):
            raise ValueError("physical restore requires the saved participant topology")
        roots = self._manifest_fragment_roots(manifest)

        if mode.strategy == PhysicalRestoreStrategy.FAST_PAGE_CACHE:
            has_page_cache = all(
                self._workspace.exists(f"{root}/{PAGE_IMAGES_DATASET}")
                for root in roots
            )
            if not has_page_cache:
                raise FileNotFoundError(
                    "checkpoint does not contain page_images cache for physical restore; "
                    "request PhysicalRestoreStrategy.RECONSTRUCT_ENTRIES to reconstruct from canonical entries"
                )
            if not hasattr(_checkpoint_native, "restore_native_page_images"):
                raise RuntimeError(
                    "native checkpoint restore requires a Lance-enabled _core extension"
                )
            _checkpoint_native.restore_native_page_images(
                store,
                [
                    self._workspace.uri(f"{root}/{PAGE_IMAGES_DATASET}")
                    for root in roots
                ],
                0,
                mode.participant.participant_index,
                mode.key_dtype,
                dict(self._workspace.storage_options()),
                load_optimizer_state,
            )
            return

        if mode.strategy is not PhysicalRestoreStrategy.RECONSTRUCT_ENTRIES:
            raise ValueError(
                f"unsupported physical restore strategy: {mode.strategy!r}"
            )

        target_root = None
        for root in roots:
            meta = self._metadata_for_root(root)
            if mode.participant.participant_index in meta["participant_indices"]:
                target_root = root
                break
        if target_root is None:
            raise RuntimeError(
                f"no fragment root covers participant {mode.participant.participant_index}"
            )
        if not hasattr(_checkpoint_native, "restore_native_partition_entries"):
            raise RuntimeError(
                "native checkpoint restore requires a Lance-enabled _core extension"
            )
        _checkpoint_native.restore_native_partition_entries(
            store,
            self._workspace.uri(f"{target_root}/{ENTRIES_DATASET}"),
            0,
            1,
            0,
            mode.key_dtype,
            mode.participant.participant_index,
            dict(self._workspace.storage_options()),
            load_optimizer_state,
        )

    @staticmethod
    def _store_specification(store: Any) -> TableSpecification:
        dimensions = tuple(sorted(int(dim) for dim in store.table_dims()))
        if not hasattr(store, "key_dtype") or not callable(store.key_dtype):
            raise TypeError("store must expose key_dtype()")
        return TableSpecification(
            dimensions=dimensions,
            embedding_dtype=store.value_dtype(),
            optimizer_dtype=store.optimizer_dtype(),
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
        try:
            dimensions = tuple(
                int(item)
                for item in metadata[b"sparse_embedding.dimensions"].decode().split(",")
                if item
            )
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

    def _fragment_specification(
        self, identity: CheckpointSaveIdentity, fragment: CheckpointFragment
    ) -> TableSpecification:
        path = (
            f"{self._attempt_prefix(identity)}/{fragment.fragment_id}/{ENTRIES_DATASET}"
        )
        if not self._workspace.exists(path):
            raise InvalidCheckpointFragment("checkpoint fragment is missing")
        return self._decode_specification(self._workspace.dataset(path).schema.metadata)

    @staticmethod
    def _manifest_specification(manifest: dict[str, Any]) -> TableSpecification:
        try:
            return TableSpecification(
                dimensions=tuple(int(dim) for dim in manifest["dimensions"]),
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

    def _check_storage_budget(self) -> None:
        if self._max_storage_bytes is None:
            return
        size = self._workspace.storage_bytes()
        if size > self._max_storage_bytes:
            raise StorageBudgetExceeded("checkpoint storage budget is exceeded")

    def _commit_lock(self):
        return self._workspace.commit_lock()

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
    def from_weight_path(cls, path: str | Path) -> LocalLanceStorage:
        asset_root = Path(path).expanduser()
        if asset_root.is_symlink() or not asset_root.is_dir():
            raise FileNotFoundError(
                f"sparse weight reference directory does not exist: {asset_root}"
            )
        reference_path = asset_root / _WEIGHT_REFERENCE_FILENAME
        if not reference_path.is_file() or reference_path.is_symlink():
            raise FileNotFoundError(
                f"sparse weight reference is missing: {reference_path}"
            )
        if any(
            item.name.startswith(_WEIGHT_REFERENCE_FILENAME)
            and item.name != _WEIGHT_REFERENCE_FILENAME
            for item in asset_root.iterdir()
        ):
            raise LookupError("sparse weight reference directory is ambiguous")
        if any(
            item.name != _WEIGHT_REFERENCE_FILENAME for item in asset_root.iterdir()
        ):
            raise ValueError("sparse weight reference directory has an unknown layout")
        handle = reference_path.read_bytes()
        storage_uri, _, _ = _decode_checkpoint_handle(handle)
        parsed = urlparse(storage_uri)
        if parsed.scheme not in ("", "file"):
            raise NotImplementedError(
                f"sparse weight reference backend is unsupported: {parsed.scheme}"
            )
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            raise ValueError("sparse weight reference file URI host is unsupported")
        backend_root = (
            Path(parsed.path if parsed.scheme == "file" else storage_uri)
            .expanduser()
            .resolve()
        )
        if backend_root == asset_root.resolve():
            raise ValueError("sparse weight reference points to its own asset path")
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
