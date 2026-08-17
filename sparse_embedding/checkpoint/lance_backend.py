"""Local Lance implementation of the sparse checkpoint protocol."""

from __future__ import annotations

import hashlib
import json
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa

from sparse_embedding import _core as _native

from ._lance_workspace import LanceWorkspace, LocalLanceWorkspace
from .base import (
    CheckpointConflictError,
    CheckpointFragment,
    CheckpointLease,
    CheckpointLeaseRequest,
    CheckpointParticipant,
    CheckpointSaveIdentity,
    CleanupReservation,
    ElasticRestore,
    InvalidCheckpointFragment,
    PhysicalRestore,
    RestoreMode,
    SparseCheckpointLocator,
    StorageBudgetExceeded,
    TableSpecification,
)
from .schema import ENTRY_SCHEMA, MANIFEST_SCHEMA, PAGE_IMAGE_SCHEMA

_checkpoint_native = _native.checkpoint
_PROTOCOL_VERSION = 1
_ACTIVE = "active"
_DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class _CommitOutcome:
    locator: SparseCheckpointLocator
    reference_id: str
    entries_before: int
    page_images_before: int
    manifest_before: int


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.Table.from_arrays(
        [pa.array([], type=field.type) for field in schema], schema=schema
    )


def _schema_digest(specification: TableSpecification | None = None) -> str:
    payload = b"".join(
        schema.serialize().to_pybytes()
        for schema in (ENTRY_SCHEMA, PAGE_IMAGE_SCHEMA, MANIFEST_SCHEMA)
    )
    if specification is not None:
        payload += _canonical_digest(
            [
                {
                    "dimensions": specification.dimensions,
                    "embedding_dtype": specification.embedding_dtype,
                    "optimizer_dtype": specification.optimizer_dtype,
                    "score_policy": specification.score_policy,
                    "protocol_version": specification.protocol_version,
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

    def prepare_checkpoint(self, identity: CheckpointSaveIdentity) -> None:
        self._ensure_tables()
        with self._commit_lock():
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
            specification.protocol_version,
            _schema_digest(specification),
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
        fragment = CheckpointFragment(
            fragment_id=fragment_id,
            checkpoint_name=identity.checkpoint_name,
            save_attempt=identity.save_attempt,
            participant=participant,
            entry_count=int(artifact["entry_count"]),
            page_count=int(artifact["page_count"]),
            schema_digest=str(artifact["schema_digest"]),
            content_digest=str(artifact["content_digest"]),
        )
        self._require_authority(identity)
        return fragment

    def commit_checkpoint(
        self,
        identity: CheckpointSaveIdentity,
        fragments: Sequence[CheckpointFragment],
        lease_request: CheckpointLeaseRequest | None = None,
        reference_id: str | None = None,
    ) -> SparseCheckpointLocator:
        with self._commit_lock():
            return self._commit_checkpoint_locked(
                identity, fragments, lease_request, reference_id
            )

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
        outcome = self._pending_outcome(identity)
        if outcome is not None:
            if (reference_id or "") != outcome.reference_id:
                raise CheckpointConflictError(
                    "checkpoint commit reference does not match its durable outcome"
                )
            return outcome.locator
        self._require_authority(identity)
        self._workspace.assert_commit_lock()
        self._reconcile_staging()
        validated, specification = self._validate_fragments(identity, fragments)
        self._ensure_payload_tables(validated, identity)
        manifest_rows = self._manifest_rows()
        previous = max(
            (int(row["checkpoint_version"]) for row in manifest_rows), default=0
        )
        checkpoint_version = previous + 1
        fragment_roots = [
            f"{self._attempt_prefix(identity)}/{fragment.fragment_id}"
            for fragment in validated
        ]
        self._workspace.assert_commit_lock()
        entries_before = self._workspace.dataset("entries").version
        pages_before = self._workspace.dataset("page_images").version
        try:
            entry_artifact = _checkpoint_native.merge_native_lance_sources(
                self._workspace.uri("entries"),
                [self._workspace.uri(f"{root}/entries") for root in fragment_roots],
                ["dim", "key"],
                checkpoint_version,
            )
            page_artifact = _checkpoint_native.merge_native_lance_sources(
                self._workspace.uri("page_images"),
                [self._workspace.uri(f"{root}/page_images") for root in fragment_roots],
                [
                    "source_participant_index",
                    "embedding_dim",
                    "physical_page_id",
                ],
                checkpoint_version,
            )
            entries_version = int(entry_artifact["target_version"])
            pages_version = int(page_artifact["target_version"])
            self._check_storage_budget()
            self._workspace.assert_commit_lock()
        except Exception as error:
            try:
                for name, version in (
                    ("entries", entries_before),
                    ("page_images", pages_before),
                ):
                    self._restore_dataset_version(name, version)
            except Exception as rollback_error:
                raise RuntimeError(
                    "checkpoint merge rollback failed after commit preparation error"
                ) from rollback_error
            if "duplicate checkpoint key" in str(error):
                raise InvalidCheckpointFragment(
                    "checkpoint fragments contain a duplicate key"
                ) from error
            raise
        fragment_digest = _canonical_digest(
            [
                {
                    "fragment_id": fragment.fragment_id,
                    "content_digest": fragment.content_digest,
                }
                for fragment in validated
            ]
        )
        manifest = {
            "protocol_version": _PROTOCOL_VERSION,
            "checkpoint_name": identity.checkpoint_name,
            "checkpoint_version": checkpoint_version,
            "parent_checkpoint_version": previous,
            "entries_lance_version": entries_version,
            "page_images_lance_version": pages_version,
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
            "reference_id": reference_id or "",
        }
        manifest_before = self._workspace.dataset("snapshot_manifest").version
        locator = SparseCheckpointLocator(identity.checkpoint_name, checkpoint_version)
        lease: CheckpointLease | None = None
        try:
            self._workspace.assert_commit_lock()
            self._workspace.write_dataset(
                pa.Table.from_pylist([manifest], schema=MANIFEST_SCHEMA),
                "snapshot_manifest",
                mode="append",
            )
            self._workspace.assert_commit_lock()
            if lease_request is not None:
                lease = self._acquire_checkpoint_lease_locked(locator, lease_request)
            if reference_id:
                self._create_reference_tags(locator, reference_id, manifest)
            self._workspace.assert_commit_lock()
            self._record_pending_outcome(
                identity,
                locator,
                entries_before,
                pages_before,
                manifest_before,
            )
        except BaseException:
            if lease is not None:
                self._delete_lease_tag(lease)
            if reference_id:
                self._delete_reference_tags(locator, reference_id)
            try:
                for name, version in (
                    ("entries", entries_before),
                    ("page_images", pages_before),
                    ("snapshot_manifest", manifest_before),
                ):
                    self._restore_dataset_version(name, version)
            except BaseException as rollback_error:
                raise RuntimeError(
                    "checkpoint rollback failed after commit finalization error"
                ) from rollback_error
            raise
        self._workspace.assert_commit_lock()
        self._workspace.delete_prefix(self._attempt_prefix(identity))
        self._workspace.assert_commit_lock()
        self._clear_authority(identity)
        return locator

    def abort_checkpoint(self, identity: CheckpointSaveIdentity) -> None:
        with self._commit_lock():
            outcome = self._pending_outcome(identity, require_identity=False)
            if outcome is not None:
                if outcome.reference_id:
                    self._delete_reference_tags(outcome.locator, outcome.reference_id)
                for name, version in (
                    ("entries", outcome.entries_before),
                    ("page_images", outcome.page_images_before),
                    ("snapshot_manifest", outcome.manifest_before),
                ):
                    self._restore_dataset_version(name, version)
                self._reclaim_unlocked(timedelta(0))
                self._delete_pending_outcome(identity.checkpoint_name)
            attempt = self._attempt_prefix(identity)
            if self._workspace.exists(attempt):
                self._workspace.delete_prefix(attempt)
            self._clear_authority(identity)

    def acknowledge_checkpoint(self, identity: CheckpointSaveIdentity) -> None:
        with self._commit_lock():
            if self._pending_outcome(identity, require_identity=False) is not None:
                self._delete_pending_outcome(identity.checkpoint_name)

    def acquire_checkpoint_lease(
        self, locator: SparseCheckpointLocator, request: CheckpointLeaseRequest
    ) -> CheckpointLease:
        with self._commit_lock():
            return self._acquire_checkpoint_lease_locked(locator, request)

    def _acquire_checkpoint_lease_locked(
        self, locator: SparseCheckpointLocator, request: CheckpointLeaseRequest
    ) -> CheckpointLease:
        manifest = self._resolve_manifest(locator)
        if self._locator_has_reservation(locator):
            raise CheckpointConflictError("checkpoint is reserved for reclaim")
        for existing in self._active_checkpoint_leases_unlocked():
            if existing.request_id != request.request_id:
                continue
            if (
                existing.lease_holder_id != request.lease_holder_id
                or existing.locator != locator
            ):
                raise CheckpointConflictError(
                    "lease request ID belongs to another checkpoint or lease holder"
                )
            return existing
        lease = CheckpointLease(
            lease_id=uuid.uuid4().hex,
            locator=locator,
            lease_holder_id=request.lease_holder_id,
            expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=request.ttl_seconds),
            generation=0,
            request_id=request.request_id,
        )
        self._create_lease_tags(lease, manifest)
        return lease

    def renew_checkpoint_lease(
        self, lease: CheckpointLease, ttl_seconds: float
    ) -> CheckpointLease:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        with self._commit_lock():
            matches = tuple(
                active
                for active in self._active_checkpoint_leases_unlocked()
                if active.lease_id == lease.lease_id
            )
            if len(matches) != 1 or matches[0] != lease:
                raise CheckpointConflictError("lease generation CAS conflict")
            renewed = CheckpointLease(
                lease_id=lease.lease_id,
                locator=lease.locator,
                lease_holder_id=lease.lease_holder_id,
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
                generation=lease.generation + 1,
                request_id=lease.request_id,
            )
            self._create_lease_tags(renewed, self._resolve_manifest(lease.locator))
            self._delete_lease_tag(lease)
            return renewed

    def release_checkpoint_lease(self, lease: CheckpointLease) -> None:
        with self._commit_lock():
            matches = tuple(
                active
                for active in self._active_checkpoint_leases_unlocked()
                if active.lease_id == lease.lease_id
            )
            if not matches:
                return
            if len(matches) != 1 or matches[0] != lease:
                raise CheckpointConflictError("lease generation CAS conflict")
            self._delete_lease_tag(lease)

    def _delete_lease_tag(self, lease: CheckpointLease) -> None:
        tag = self._lease_tag(lease)
        for name in ("entries", "page_images", "snapshot_manifest"):
            tags = self._workspace.dataset(name).tags
            if tag in tags.list():
                tags.delete(tag)

    def active_checkpoint_leases(
        self, now: datetime | None = None
    ) -> tuple[CheckpointLease, ...]:
        selected_now = now or datetime.now(timezone.utc)
        if selected_now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self._commit_lock():
            return self._active_checkpoint_leases_unlocked(selected_now)

    def _active_checkpoint_leases_unlocked(
        self, now: datetime | None = None
    ) -> tuple[CheckpointLease, ...]:
        selected_now = now or datetime.now(timezone.utc)
        dataset = self._workspace.dataset("snapshot_manifest")
        leases = []
        for tag in tuple(dataset.tags.list()):
            if not tag.startswith("selease."):
                continue
            lease = self._decode_lease_tag(tag)
            if lease.expires_at > selected_now:
                leases.append(lease)
        return tuple(sorted(leases, key=lambda item: item.lease_id))

    def reclaim(self, *, older_than: timedelta = timedelta(0)) -> None:
        with self._commit_lock():
            self._reclaim_unlocked(older_than)

    def _reclaim_unlocked(self, older_than: timedelta) -> None:
        now = datetime.now(timezone.utc)
        active_ids = {
            lease.lease_id for lease in self._active_checkpoint_leases_unlocked(now)
        }
        for name in ("entries", "page_images", "snapshot_manifest"):
            dataset = self._workspace.dataset(name)
            for tag in tuple(dataset.tags.list()):
                if (
                    tag.startswith("selease.")
                    and tag.split(".", 2)[1] not in active_ids
                ):
                    dataset.tags.delete(tag)
            if name != "snapshot_manifest":
                dataset.optimize.compact_files(materialize_deletions=True)
            dataset.cleanup_old_versions(
                older_than=older_than,
                error_if_tagged_old_versions=False,
            )
        self._reconcile_staging()

    def reserve_cleanup(
        self,
        reservation_id: str,
        ordered_candidates: Sequence[SparseCheckpointLocator],
        required_count: int,
    ) -> CleanupReservation:
        if required_count <= 0:
            raise ValueError("required_count must be positive")
        with self._commit_lock():
            leased = {
                lease.locator for lease in self._active_checkpoint_leases_unlocked()
            }
            tags = self._workspace.dataset("snapshot_manifest").tags
            names = tuple(tags.list())
            selected = []
            for locator in ordered_candidates:
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
                if len(selected) == required_count:
                    break
            if len(selected) != required_count:
                raise CheckpointConflictError(
                    "not enough unleased checkpoints are reclaimable"
                )
            reservation = CleanupReservation(reservation_id, tuple(selected))
            manifest_version = self._workspace.dataset("snapshot_manifest").version
            created = []
            try:
                for locator in reservation.locators:
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
        with self._commit_lock():
            tags = self._workspace.dataset("snapshot_manifest").tags
            for locator in reservation.locators:
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
                    self._workspace.dataset("snapshot_manifest").version,
                )
                tags.delete(reserved)

    def rollback_cleanup(self, reservation: CleanupReservation) -> None:
        with self._commit_lock():
            tags = self._workspace.dataset("snapshot_manifest").tags
            for locator in reservation.locators:
                for status in ("reserved", "pending"):
                    tag = self._reservation_tag(
                        status, reservation.reservation_id, locator
                    )
                    if tag in tags.list():
                        tags.delete(tag)

    def cleanup_checkpoint(
        self,
        reservation: CleanupReservation,
        locator: SparseCheckpointLocator,
        reference_id: str,
    ) -> None:
        if locator not in reservation.locators:
            raise ValueError("checkpoint locator is outside reclaim reservation")
        if not reference_id:
            raise ValueError("reference_id must be non-empty")
        with self._commit_lock():
            tags = self._workspace.dataset("snapshot_manifest").tags
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
            self._reclaim_unlocked(timedelta(0))
            self._record_reclaim_completion(reservation, locator)
            self._delete_pending_outcome(locator.checkpoint_name)
            if pending in tags.list():
                tags.delete(pending)

    def release_reference(
        self, locator: SparseCheckpointLocator, reference_id: str
    ) -> None:
        if not reference_id:
            raise ValueError("reference_id must be non-empty")
        with self._commit_lock():
            self._delete_reference_tags(locator, reference_id)
            self._reclaim_unlocked(timedelta(0))

    def restore_checkpoint(
        self,
        store: Any,
        locator: SparseCheckpointLocator,
        mode: RestoreMode,
    ) -> None:
        manifest = self._resolve_manifest(locator)
        specification = self._manifest_specification(manifest)
        if manifest["schema_digest"] != _schema_digest(specification):
            raise RuntimeError("checkpoint schema digest mismatch")
        if self._store_specification(store) != specification:
            raise ValueError("restore target does not match checkpoint specification")
        if isinstance(mode, PhysicalRestore):
            self._restore_physical(store, manifest, mode)
            return
        if isinstance(mode, ElasticRestore):
            self._restore_elastic(store, manifest, mode)
            return
        raise TypeError("unsupported restore mode")

    def _ensure_tables(self) -> None:
        self._workspace.ensure_root()
        for name, schema in (("snapshot_manifest", MANIFEST_SCHEMA),):
            if not self._workspace.exists(name):
                self._workspace.write_dataset(_empty_table(schema), name, mode="create")
            elif self._workspace.dataset(name).schema != schema:
                raise RuntimeError(
                    f"checkpoint dataset {name!r} does not match protocol schema"
                )
        for name, schema in (
            ("entries", ENTRY_SCHEMA),
            ("page_images", PAGE_IMAGE_SCHEMA),
        ):
            if self._workspace.exists(name):
                actual = self._workspace.dataset(name).schema
                if actual.remove_metadata() != schema.remove_metadata():
                    raise RuntimeError(
                        f"checkpoint dataset {name!r} does not match protocol schema"
                    )

    def _ensure_payload_tables(
        self,
        fragments: Sequence[CheckpointFragment],
        identity: CheckpointSaveIdentity,
    ) -> None:
        first_root = f"{self._attempt_prefix(identity)}/{fragments[0].fragment_id}"
        for name, canonical in (
            ("entries", ENTRY_SCHEMA),
            ("page_images", PAGE_IMAGE_SCHEMA),
        ):
            source_schema = self._workspace.dataset(f"{first_root}/{name}").schema
            if source_schema.remove_metadata() != canonical.remove_metadata():
                raise InvalidCheckpointFragment(
                    f"checkpoint fragment {name!r} schema fields are invalid"
                )
            if self._workspace.exists(name):
                target_schema = self._workspace.dataset(name).schema
                if target_schema != source_schema:
                    raise InvalidCheckpointFragment(
                        f"checkpoint target {name!r} schema metadata is inconsistent"
                    )
                continue
            self._workspace.write_dataset(
                _empty_table(source_schema), name, mode="create"
            )

    def _publish_authority(self, identity: CheckpointSaveIdentity) -> None:
        dataset = self._workspace.dataset("snapshot_manifest")
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
        tags = self._workspace.dataset("snapshot_manifest").tags.list()
        if (
            self._current_authority_generation(identity, tuple(tags))
            != identity.commit_authority.generation
            or self._authority_tag(identity) not in tags
        ):
            raise CheckpointConflictError("checkpoint commit authority is stale")

    def _clear_authority(self, identity: CheckpointSaveIdentity) -> None:
        if not self._workspace.exists("snapshot_manifest"):
            return
        tags = self._workspace.dataset("snapshot_manifest").tags
        name = self._authority_tag(identity)
        if name in tags.list():
            tags.delete(name)

    def _reconcile_staging(self) -> None:
        if not self._workspace.exists("_staging"):
            return
        tags = self._workspace.dataset("snapshot_manifest").tags.list()
        active = {
            ".".join(tag.split(".")[1:])
            for tag in tags
            if tag.startswith("seauthority.")
        }
        for attempt in self._workspace.child_prefixes("_staging"):
            if attempt not in active:
                self._workspace.delete_prefix(f"_staging/{attempt}")

    def _create_lease_tags(
        self, lease: CheckpointLease, manifest: dict[str, Any]
    ) -> None:
        self._create_version_tags(self._lease_tag(lease), manifest)

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
        versions = {
            "entries": int(manifest["entries_lance_version"]),
            "page_images": int(manifest["page_images_lance_version"]),
            "snapshot_manifest": self._workspace.dataset("snapshot_manifest").version,
        }
        created = []
        try:
            for name, version in versions.items():
                self._workspace.dataset(name).tags.create(tag, version)
                created.append(name)
        except Exception:
            for name in reversed(created):
                self._workspace.dataset(name).tags.delete(tag)
            raise

    def _delete_reference_tags(
        self, locator: SparseCheckpointLocator, reference_id: str
    ) -> None:
        tag = self._reference_tag(locator, reference_id)
        for name in ("entries", "page_images", "snapshot_manifest"):
            tags = self._workspace.dataset(name).tags
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

    def _lease_tag(self, lease: CheckpointLease) -> str:
        expires = int(lease.expires_at.timestamp() * 1_000_000)
        return ".".join(
            (
                "selease",
                lease.lease_id,
                str(lease.generation),
                str(expires),
                str(lease.locator.checkpoint_version),
                self._encode(lease.locator.checkpoint_name),
                self._encode(lease.lease_holder_id),
                self._encode(lease.request_id),
            )
        )

    def _decode_lease_tag(self, tag: str) -> CheckpointLease:
        parts = tag.split(".")
        if len(parts) != 8:
            raise RuntimeError("invalid sparse checkpoint lease tag")
        return CheckpointLease(
            lease_id=parts[1],
            locator=SparseCheckpointLocator(self._decode(parts[5]), int(parts[4])),
            lease_holder_id=self._decode(parts[6]),
            expires_at=datetime.fromtimestamp(int(parts[3]) / 1_000_000, timezone.utc),
            generation=int(parts[2]),
            request_id=self._decode(parts[7]),
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

    def _outcome_tag(
        self,
        checkpoint_name: str,
        checkpoint_generation: int,
        checkpoint_version: int,
        entries_before: int,
        page_images_before: int,
        manifest_before: int,
    ) -> str:
        return ".".join(
            (
                "setransaction",
                self._short_token(checkpoint_name),
                str(checkpoint_generation),
                str(checkpoint_version),
                str(entries_before),
                str(page_images_before),
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
            for tag in self._workspace.dataset("snapshot_manifest").tags.list()
            if tag.startswith(prefix)
        )
        if not tags:
            return None
        if len(tags) != 1:
            raise ValueError("durable checkpoint commit outcome is ambiguous")
        parts = tags[0].split(".")
        if len(parts) != 7:
            raise ValueError("durable checkpoint commit outcome is corrupt")
        (
            _,
            checkpoint_token,
            authority_generation,
            version,
            entries_before,
            page_images_before,
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
                "entries_before": int(entries_before),
                "page_images_before": int(page_images_before),
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
        version_fields = {
            name: decoded[name]
            for name in ("entries_before", "page_images_before", "manifest_before")
        }
        if decoded["checkpoint_version"] <= 0 or any(
            value <= 0 for value in version_fields.values()
        ):
            raise ValueError("durable checkpoint commit outcome is corrupt")
        return _CommitOutcome(
            locator=SparseCheckpointLocator(
                identity.checkpoint_name, decoded["checkpoint_version"]
            ),
            reference_id=manifest["reference_id"],
            entries_before=cast(int, version_fields["entries_before"]),
            page_images_before=cast(int, version_fields["page_images_before"]),
            manifest_before=cast(int, version_fields["manifest_before"]),
        )

    def _record_pending_outcome(
        self,
        identity: CheckpointSaveIdentity,
        locator: SparseCheckpointLocator,
        entries_before: int,
        page_images_before: int,
        manifest_before: int,
    ) -> None:
        tag = self._outcome_tag(
            identity.checkpoint_name,
            identity.commit_authority.generation,
            locator.checkpoint_version,
            entries_before,
            page_images_before,
            manifest_before,
        )
        tags = self._workspace.dataset("snapshot_manifest").tags
        if tag not in tags.list():
            tags.create(tag, self._workspace.dataset("snapshot_manifest").version)

    def _delete_pending_outcome(self, checkpoint_name: str) -> None:
        tags = self._workspace.dataset("snapshot_manifest").tags
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
        return tag in self._workspace.dataset("snapshot_manifest").tags.list()

    def _record_reclaim_completion(
        self,
        reservation: CleanupReservation,
        locator: SparseCheckpointLocator,
    ) -> None:
        if self._reclaim_completed(reservation, locator):
            return
        self._workspace.dataset("snapshot_manifest").tags.create(
            self._reclaim_completion_tag(reservation, locator),
            self._workspace.dataset("snapshot_manifest").version,
        )

    @staticmethod
    def _is_reservation_tag(tag: str) -> bool:
        return tag.startswith("sereserved.") or tag.startswith("sepending.")

    def _reservation_locator_suffix(self, locator: SparseCheckpointLocator) -> str:
        return f".{locator.checkpoint_version}.{self._encode(locator.checkpoint_name)}"

    def _locator_has_reservation(self, locator: SparseCheckpointLocator) -> bool:
        tags = self._workspace.dataset("snapshot_manifest").tags.list()
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
        return (
            sorted(result, key=lambda item: item.participant.participant_index),
            specification,
        )

    def _restore_dataset_version(self, name: str, version: int) -> None:
        self._workspace.dataset(name).checkout_version(version).restore()

    def _manifest_rows(self) -> list[dict[str, Any]]:
        return self._workspace.dataset("snapshot_manifest").to_table().to_pylist()

    def _resolve_manifest(self, locator: SparseCheckpointLocator) -> dict[str, Any]:
        matches = [
            row
            for row in self._manifest_rows()
            if row["checkpoint_name"] == locator.checkpoint_name
            and int(row["checkpoint_version"]) == locator.checkpoint_version
        ]
        if len(matches) != 1:
            raise LookupError("checkpoint locator is missing or ambiguous")
        return matches[0]

    def _restore_elastic(
        self, store: Any, manifest: dict[str, Any], mode: ElasticRestore
    ) -> None:
        if not hasattr(_checkpoint_native, "active_entry_keys"):
            raise RuntimeError(
                "native checkpoint restore requires a Lance-enabled _core extension"
            )
        keys_capsule = _checkpoint_native.active_entry_keys(
            self._workspace.uri("entries"),
            int(manifest["entries_lance_version"]),
        )
        keys = np.from_dlpack(keys_capsule).astype(np.uint64, copy=False)
        first = np.asarray(mode.partition_fn(keys, mode.partition_count))
        second = np.asarray(mode.partition_fn(keys, mode.partition_count))
        if first.shape != keys.shape:
            raise ValueError("partition_fn output length does not match keys")
        if first.dtype.kind not in "iu":
            raise TypeError("partition_fn must return integer partition IDs")
        if not np.array_equal(first, second):
            raise ValueError("partition_fn is not stable")
        if np.any(first < 0) or np.any(first >= mode.partition_count):
            raise ValueError("partition_fn returned an out-of-range partition ID")
        selected_keys = np.ascontiguousarray(
            keys[first == mode.target_partition_id], dtype=np.uint64
        )
        if selected_keys.size == 0:
            return
        _checkpoint_native.restore_native_entries(
            store,
            self._workspace.uri("entries"),
            int(manifest["entries_lance_version"]),
            selected_keys,
        )

    def _restore_physical(
        self, store: Any, manifest: dict[str, Any], mode: PhysicalRestore
    ) -> None:
        if mode.participant.participant_count != int(manifest["participant_count"]):
            raise ValueError("physical restore requires the saved participant topology")
        if store.total_size() != 0:
            raise RuntimeError("physical restore requires an empty store")
        if not hasattr(_checkpoint_native, "restore_native_page_images"):
            raise RuntimeError(
                "native checkpoint restore requires a Lance-enabled _core extension"
            )
        _checkpoint_native.restore_native_page_images(
            store,
            self._workspace.uri("page_images"),
            int(manifest["page_images_lance_version"]),
            mode.participant.participant_index,
        )

    @staticmethod
    def _store_specification(store: Any) -> TableSpecification:
        dimensions = tuple(sorted(int(dim) for dim in store.table_dims()))
        return TableSpecification(
            dimensions=dimensions,
            embedding_dtype=store.value_dtype(),
            optimizer_dtype=store.optimizer_dtype(),
            score_policy=store.score_policy(),
            protocol_version=_PROTOCOL_VERSION,
        )

    @staticmethod
    def _decode_specification(
        metadata: dict[bytes, bytes] | None,
    ) -> TableSpecification:
        required = {
            b"sparse_embedding.protocol_version",
            b"sparse_embedding.dimensions",
            b"sparse_embedding.embedding_dtype",
            b"sparse_embedding.optimizer_dtype",
            b"sparse_embedding.score_policy",
        }
        if metadata is None or not required.issubset(metadata):
            raise InvalidCheckpointFragment(
                "checkpoint fragment specification is missing"
            )
        try:
            parsed_protocol_version = int(
                metadata[b"sparse_embedding.protocol_version"]
            )
            if parsed_protocol_version != _PROTOCOL_VERSION:
                raise InvalidCheckpointFragment(
                    "unsupported checkpoint protocol version"
                )
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
                protocol_version=parsed_protocol_version,
            )
        except (TypeError, ValueError) as error:
            raise InvalidCheckpointFragment(
                "checkpoint fragment specification is invalid"
            ) from error

    def _fragment_specification(
        self, identity: CheckpointSaveIdentity, fragment: CheckpointFragment
    ) -> TableSpecification:
        path = f"{self._attempt_prefix(identity)}/{fragment.fragment_id}/entries"
        if not self._workspace.exists(path):
            raise InvalidCheckpointFragment("checkpoint fragment is missing")
        return self._decode_specification(self._workspace.dataset(path).schema.metadata)

    @staticmethod
    def _manifest_specification(manifest: dict[str, Any]) -> TableSpecification:
        try:
            if int(manifest["protocol_version"]) != _PROTOCOL_VERSION:
                raise RuntimeError("unsupported checkpoint protocol version")
            return TableSpecification(
                dimensions=tuple(int(dim) for dim in manifest["dimensions"]),
                embedding_dtype=manifest["embedding_dtype"],
                optimizer_dtype=manifest["optimizer_dtype"] or None,
                score_policy=manifest["score_policy"],
                protocol_version=int(manifest["protocol_version"]),
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
