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

_FORMAT_VERSION = 9
_ACTIVE = "active"
_DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class _CommitOutcome:
    locator: SparseCheckpointLocator
    reference_id: str
    entries_before: int
    page_images_before: int
    manifest_before: int


ENTRY_SCHEMA = pa.schema(
    [
        pa.field("dim", pa.uint32()),
        pa.field("key", pa.uint64()),
        pa.field("embedding_values", pa.list_(pa.float32())),
        pa.field("embedding_dtype", pa.string()),
        pa.field("optimizer_state_values", pa.list_(pa.float32())),
        pa.field("optimizer_state_dtype", pa.string()),
        pa.field("optimizer_state_stride", pa.uint32()),
        pa.field("step", pa.uint32()),
        pa.field("score", pa.uint64()),
        pa.field("sequence", pa.uint64()),
        pa.field("physical_page_id", pa.uint64()),
        pa.field("page_slot", pa.uint32()),
        pa.field("row_state", pa.string()),
        pa.field("last_updated_checkpoint_version", pa.uint64()),
        pa.field("source_participant_index", pa.uint32()),
    ]
)

PAGE_IMAGE_SCHEMA = pa.schema(
    [
        pa.field("source_participant_index", pa.uint32()),
        pa.field("dim", pa.uint32()),
        pa.field("physical_page_id", pa.uint64()),
        pa.field("embedding_stride", pa.uint32()),
        pa.field("embedding_dtype", pa.string()),
        pa.field("slot_ordinals", pa.list_(pa.uint32())),
        pa.field("keys", pa.list_(pa.uint64())),
        pa.field("sequences", pa.list_(pa.uint64())),
        pa.field("embedding_values", pa.list_(pa.float32())),
        pa.field("scores", pa.list_(pa.uint64())),
        pa.field("optimizer_state_values", pa.list_(pa.float32())),
        pa.field("optimizer_state_dtype", pa.string()),
        pa.field("optimizer_state_stride", pa.uint32()),
        pa.field("steps", pa.list_(pa.uint32())),
        pa.field("checksum", pa.string()),
        pa.field("row_state", pa.string()),
        pa.field("last_updated_checkpoint_version", pa.uint64()),
    ]
)

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("protocol_version", pa.uint32()),
        pa.field("checkpoint_name", pa.string()),
        pa.field("checkpoint_version", pa.uint64()),
        pa.field("parent_checkpoint_version", pa.uint64()),
        pa.field("entries_lance_version", pa.uint64()),
        pa.field("page_images_lance_version", pa.uint64()),
        pa.field("participant_count", pa.uint32()),
        pa.field("checkpoint_mode", pa.string()),
        pa.field("schema_digest", pa.string()),
        pa.field("fragment_digest", pa.string()),
        pa.field("save_attempt", pa.string()),
        pa.field("authority_scope", pa.string()),
        pa.field("authority_generation", pa.uint64()),
        pa.field("commit_authority", pa.string()),
        pa.field("created_at", pa.timestamp("us", tz="UTC")),
        pa.field("dimensions", pa.list_(pa.uint32())),
        pa.field("embedding_dtype", pa.string()),
        pa.field("optimizer_dtype", pa.string()),
    ]
)


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.Table.from_arrays(
        [pa.array([], type=field.type) for field in schema], schema=schema
    )


def _schema_digest(specification: TableSpecification | None = None) -> str:
    payload = (
        ENTRY_SCHEMA.serialize().to_pybytes()
        + PAGE_IMAGE_SCHEMA.serialize().to_pybytes()
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
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _page_checksum(row: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in row.items()
        if key not in {"checksum", "row_state", "last_updated_checkpoint_version"}
    }
    return _canonical_digest([payload])


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
        entries, pages = self._encode_store(store, participant, specification)
        metadata = self._specification_metadata(specification)
        self._workspace.create_prefix(fragment_root)
        self._workspace.write_dataset(
            entries.replace_schema_metadata(metadata),
            f"{fragment_root}/entries",
            mode="create",
        )
        self._workspace.write_dataset(
            pages.replace_schema_metadata(metadata),
            f"{fragment_root}/page_images",
            mode="create",
        )
        stored_entries = self._workspace.dataset(f"{fragment_root}/entries").to_table()
        stored_pages = self._workspace.dataset(
            f"{fragment_root}/page_images"
        ).to_table()
        rows = stored_entries.to_pylist() + stored_pages.to_pylist()
        fragment = CheckpointFragment(
            fragment_id=fragment_id,
            checkpoint_name=identity.checkpoint_name,
            save_attempt=identity.save_attempt,
            participant=participant,
            entry_count=stored_entries.num_rows,
            page_count=stored_pages.num_rows,
            schema_digest=_schema_digest(specification),
            content_digest=_canonical_digest(rows),
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
        manifest_rows = self._manifest_rows()
        previous = max(
            (int(row["checkpoint_version"]) for row in manifest_rows), default=0
        )
        checkpoint_version = previous + 1
        entry_rows: list[dict[str, Any]] = []
        page_rows: list[dict[str, Any]] = []
        for fragment in validated:
            root = f"{self._attempt_prefix(identity)}/{fragment.fragment_id}"
            entries = self._workspace.dataset(f"{root}/entries").to_table().to_pylist()
            pages = (
                self._workspace.dataset(f"{root}/page_images").to_table().to_pylist()
            )
            digest = _canonical_digest(entries + pages)
            if digest != fragment.content_digest:
                raise InvalidCheckpointFragment("checkpoint fragment digest mismatch")
            if (
                len(entries) != fragment.entry_count
                or len(pages) != fragment.page_count
            ):
                raise InvalidCheckpointFragment(
                    "checkpoint fragment row count mismatch"
                )
            entry_rows.extend(entries)
            page_rows.extend(pages)
        entry_keys = [(row["dim"], row["key"]) for row in entry_rows]
        if len(entry_keys) != len(set(entry_keys)):
            raise InvalidCheckpointFragment("duplicate (dim, key) across fragments")
        self._apply_checkpoint_version(entry_rows, checkpoint_version)
        self._apply_checkpoint_version(page_rows, checkpoint_version)
        entry_source = self._with_entry_tombstones(entry_rows, checkpoint_version)
        page_source = self._with_page_tombstones(page_rows, checkpoint_version)
        self._workspace.assert_commit_lock()
        entries_before = self._workspace.dataset("entries").version
        pages_before = self._workspace.dataset("page_images").version
        try:
            entries_version = self._merge(
                "entries", ENTRY_SCHEMA, ("dim", "key"), entry_source
            )
            pages_version = self._merge(
                "page_images",
                PAGE_IMAGE_SCHEMA,
                ("source_participant_index", "dim", "physical_page_id"),
                page_source,
            )
            self._check_storage_budget()
            self._workspace.assert_commit_lock()
        except Exception:
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
            "protocol_version": _FORMAT_VERSION,
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
                reference_id or "",
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
                self._workspace.delete_record(
                    self._outcome_key(identity.checkpoint_name)
                )
            attempt = self._attempt_prefix(identity)
            if self._workspace.exists(attempt):
                self._workspace.delete_prefix(attempt)
            self._clear_authority(identity)

    def acknowledge_checkpoint(self, identity: CheckpointSaveIdentity) -> None:
        with self._commit_lock():
            if self._pending_outcome(identity, require_identity=False) is not None:
                self._workspace.delete_record(
                    self._outcome_key(identity.checkpoint_name)
                )

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
            self._workspace.delete_record(self._outcome_key(locator.checkpoint_name))
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
        for name, schema in (
            ("entries", ENTRY_SCHEMA),
            ("page_images", PAGE_IMAGE_SCHEMA),
            ("snapshot_manifest", MANIFEST_SCHEMA),
        ):
            if not self._workspace.exists(name):
                self._workspace.write_dataset(_empty_table(schema), name, mode="create")

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

    @staticmethod
    def _outcome_key(checkpoint_name: str) -> str:
        digest = hashlib.sha256(checkpoint_name.encode()).hexdigest()
        return f".transactions/{digest}.json"

    def _pending_outcome(
        self,
        identity: CheckpointSaveIdentity,
        *,
        require_identity: bool = True,
    ) -> _CommitOutcome | None:
        stored = self._workspace.read_record(
            self._outcome_key(identity.checkpoint_name)
        )
        if stored is None:
            return None
        payload = json.loads(stored)
        expected_identity = {
            "checkpoint_name": identity.checkpoint_name,
            "save_attempt": identity.save_attempt,
            "authority_scope": identity.commit_authority.scope,
            "authority_generation": identity.commit_authority.generation,
            "commit_authority": identity.commit_authority.authority_id,
        }
        if not isinstance(payload, dict):
            raise ValueError("durable checkpoint commit outcome is corrupt")
        if any(payload.get(name) != value for name, value in expected_identity.items()):
            if not require_identity:
                return None
            raise CheckpointConflictError(
                "checkpoint name belongs to another durable commit outcome"
            )
        reference_id = payload.get("reference_id")
        version = payload.get("checkpoint_version")
        version_fields = {
            name: payload.get(name)
            for name in (
                "entries_before",
                "page_images_before",
                "manifest_before",
            )
        }
        if (
            not isinstance(reference_id, str)
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version <= 0
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in version_fields.values()
            )
        ):
            raise ValueError("durable checkpoint commit outcome is corrupt")
        return _CommitOutcome(
            locator=SparseCheckpointLocator(identity.checkpoint_name, version),
            reference_id=reference_id,
            entries_before=cast(int, version_fields["entries_before"]),
            page_images_before=cast(int, version_fields["page_images_before"]),
            manifest_before=cast(int, version_fields["manifest_before"]),
        )

    def _record_pending_outcome(
        self,
        identity: CheckpointSaveIdentity,
        locator: SparseCheckpointLocator,
        reference_id: str,
        entries_before: int,
        page_images_before: int,
        manifest_before: int,
    ) -> None:
        payload = {
            "checkpoint_name": identity.checkpoint_name,
            "checkpoint_version": locator.checkpoint_version,
            "save_attempt": identity.save_attempt,
            "authority_scope": identity.commit_authority.scope,
            "authority_generation": identity.commit_authority.generation,
            "commit_authority": identity.commit_authority.authority_id,
            "reference_id": reference_id,
            "entries_before": entries_before,
            "page_images_before": page_images_before,
            "manifest_before": manifest_before,
        }
        self._workspace.write_record_once(
            self._outcome_key(identity.checkpoint_name),
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        )

    def _reclaim_completion_key(
        self,
        reservation: CleanupReservation,
        locator: SparseCheckpointLocator,
    ) -> str:
        identity = "\x00".join(
            (
                reservation.reservation_id,
                locator.checkpoint_name,
                str(locator.checkpoint_version),
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f".retention/completed/{digest}.json"

    def _reclaim_completed(
        self,
        reservation: CleanupReservation,
        locator: SparseCheckpointLocator,
    ) -> bool:
        key = self._reclaim_completion_key(reservation, locator)
        stored = self._workspace.read_record(key)
        if stored is None:
            return False
        payload = json.loads(stored)
        expected = {
            "reservation_id": reservation.reservation_id,
            "checkpoint_name": locator.checkpoint_name,
            "checkpoint_version": locator.checkpoint_version,
        }
        if payload != expected:
            raise ValueError(f"reclaim completion record is corrupt: {key}")
        return True

    def _record_reclaim_completion(
        self,
        reservation: CleanupReservation,
        locator: SparseCheckpointLocator,
    ) -> None:
        if self._reclaim_completed(reservation, locator):
            return
        payload = {
            "reservation_id": reservation.reservation_id,
            "checkpoint_name": locator.checkpoint_name,
            "checkpoint_version": locator.checkpoint_version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        self._workspace.write_record_once(
            self._reclaim_completion_key(reservation, locator), encoded
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

    def _encode_store(
        self,
        store: Any,
        participant: CheckpointParticipant,
        specification: TableSpecification,
    ) -> tuple[pa.Table, pa.Table]:
        entry_rows: list[dict[str, Any]] = []
        page_rows: list[dict[str, Any]] = []
        images = store.export_page_images()
        if not isinstance(images, list):
            raise TypeError("export_page_images must return a list")
        for image in images:
            dim = int(image["dim"])
            page_id = int(image["physical_page_id"])
            slots = np.asarray(image["slot_ordinals"], dtype=np.uint32)
            keys = np.asarray(image["keys"], dtype=np.uint64)
            sequences = np.asarray(image["seq_counts"], dtype=np.uint64)
            values = np.asarray(image["values"], dtype=np.float32).reshape(
                len(keys), dim
            )
            scores = np.asarray(image.get("scores", []), dtype=np.uint64)
            optimizer = np.asarray(image.get("optimizer_states", []), dtype=np.float32)
            stride = optimizer.size // len(keys) if len(keys) else 0
            optimizer = optimizer.reshape(len(keys), stride) if stride else None
            steps = np.asarray(image.get("steps", []), dtype=np.uint32)
            for index, key in enumerate(keys):
                entry_rows.append(
                    {
                        "dim": dim,
                        "key": int(key),
                        "embedding_values": values[index].tolist(),
                        "embedding_dtype": specification.embedding_dtype,
                        "optimizer_state_values": optimizer[index].tolist()
                        if optimizer is not None
                        else [],
                        "optimizer_state_dtype": specification.optimizer_dtype or "",
                        "optimizer_state_stride": stride,
                        "step": int(steps[index]) if steps.size else 0,
                        "score": int(scores[index]) if scores.size else 0,
                        "sequence": int(sequences[index]),
                        "physical_page_id": page_id,
                        "page_slot": int(slots[index]),
                        "row_state": _ACTIVE,
                        "last_updated_checkpoint_version": 0,
                        "source_participant_index": participant.participant_index,
                    }
                )
            page_row = {
                "source_participant_index": participant.participant_index,
                "dim": dim,
                "physical_page_id": page_id,
                "embedding_stride": int(image["vector_stride"]),
                "embedding_dtype": specification.embedding_dtype,
                "slot_ordinals": slots.tolist(),
                "keys": keys.tolist(),
                "sequences": sequences.tolist(),
                "embedding_values": np.asarray(
                    image["values"], dtype=np.float32
                ).tolist(),
                "scores": scores.tolist(),
                "optimizer_state_values": np.asarray(
                    image.get("optimizer_states", []), dtype=np.float32
                ).tolist(),
                "optimizer_state_dtype": specification.optimizer_dtype or "",
                "optimizer_state_stride": stride,
                "steps": steps.tolist(),
                "checksum": "",
                "row_state": _ACTIVE,
                "last_updated_checkpoint_version": 0,
            }
            page_row["checksum"] = _page_checksum(page_row)
            page_rows.append(page_row)
        return (
            pa.Table.from_pylist(entry_rows, schema=ENTRY_SCHEMA)
            if entry_rows
            else _empty_table(ENTRY_SCHEMA),
            pa.Table.from_pylist(page_rows, schema=PAGE_IMAGE_SCHEMA)
            if page_rows
            else _empty_table(PAGE_IMAGE_SCHEMA),
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

    @staticmethod
    def _apply_checkpoint_version(rows: list[dict[str, Any]], version: int) -> None:
        for row in rows:
            row["last_updated_checkpoint_version"] = version

    def _with_entry_tombstones(
        self, rows: list[dict[str, Any]], version: int
    ) -> list[dict[str, Any]]:
        current = self._workspace.dataset("entries").to_table().to_pylist()
        incoming = {(row["dim"], row["key"]) for row in rows}
        tombstones = []
        for row in current:
            if row["row_state"] == _ACTIVE and (row["dim"], row["key"]) not in incoming:
                deleted = dict(row)
                deleted["row_state"] = _DELETED
                deleted["last_updated_checkpoint_version"] = version
                tombstones.append(deleted)
        return rows + tombstones

    def _with_page_tombstones(
        self, rows: list[dict[str, Any]], version: int
    ) -> list[dict[str, Any]]:
        current = self._workspace.dataset("page_images").to_table().to_pylist()
        incoming = {
            (row["source_participant_index"], row["dim"], row["physical_page_id"])
            for row in rows
        }
        tombstones = []
        for row in current:
            key = (row["source_participant_index"], row["dim"], row["physical_page_id"])
            if row["row_state"] == _ACTIVE and key not in incoming:
                deleted = dict(row)
                deleted["row_state"] = _DELETED
                deleted["last_updated_checkpoint_version"] = version
                tombstones.append(deleted)
        return rows + tombstones

    def _merge(
        self,
        name: str,
        schema: pa.Schema,
        keys: tuple[str, ...],
        rows: list[dict[str, Any]],
    ) -> int:
        dataset = self._workspace.dataset(name)
        if rows:
            source = pa.Table.from_pylist(rows, schema=schema)
            dataset.merge_insert(
                list(keys)
            ).when_matched_update_all().when_not_matched_insert_all().execute(source)
        return self._workspace.dataset(name).version

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
        dataset = self._workspace.dataset(
            "entries",
            version=int(manifest["entries_lance_version"]),
        )
        routing = dataset.to_table(columns=["key", "row_state"])
        active_positions = np.flatnonzero(routing["row_state"].to_numpy() == _ACTIVE)
        keys = (
            routing["key"]
            .take(pa.array(active_positions))
            .to_numpy()
            .astype(np.uint64, copy=False)
        )
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
        selected_positions = active_positions[first == mode.target_partition_id]
        if selected_positions.size == 0:
            return
        selected = dataset.take(
            selected_positions,
            columns=[
                "dim",
                "key",
                "embedding_values",
                "optimizer_state_values",
                "optimizer_state_stride",
                "step",
            ],
        ).to_pylist()
        for dim in sorted({int(row["dim"]) for row in selected}):
            dim_rows = [row for row in selected if int(row["dim"]) == dim]
            dim_keys = np.ascontiguousarray(
                [row["key"] for row in dim_rows], dtype=np.uint64
            )
            values = np.ascontiguousarray(
                [row["embedding_values"] for row in dim_rows], dtype=np.float32
            )
            store.assign(dim_keys, values)
            stride = int(dim_rows[0]["optimizer_state_stride"])
            if stride:
                optimizer = np.ascontiguousarray(
                    [row["optimizer_state_values"] for row in dim_rows],
                    dtype=np.float32,
                ).reshape(-1)
                steps = np.ascontiguousarray(
                    [row["step"] for row in dim_rows], dtype=np.uint32
                )
                store.assign_opt_state(dim_keys, optimizer, dim, steps)

    def _restore_physical(
        self, store: Any, manifest: dict[str, Any], mode: PhysicalRestore
    ) -> None:
        if mode.participant.participant_count != int(manifest["participant_count"]):
            raise ValueError("physical restore requires the saved participant topology")
        if store.total_size() != 0:
            raise RuntimeError("physical restore requires an empty store")
        rows = (
            self._workspace.dataset(
                "page_images",
                version=int(manifest["page_images_lance_version"]),
            )
            .to_table(
                filter=f"row_state = 'active' AND source_participant_index = {mode.participant.participant_index}"
            )
            .to_pylist()
        )
        pages = []
        for row in rows:
            if _page_checksum({**row, "checksum": ""}) != row["checksum"]:
                raise RuntimeError("page image checksum mismatch")
            pages.append(
                {
                    "dim": row["dim"],
                    "physical_page_id": row["physical_page_id"],
                    "vector_dim": row["dim"],
                    "vector_stride": row["embedding_stride"],
                    "slot_ordinals": row["slot_ordinals"],
                    "keys": row["keys"],
                    "seq_counts": row["sequences"],
                    "values": row["embedding_values"],
                    "scores": row["scores"],
                    "optimizer_states": row["optimizer_state_values"],
                    "steps": row["steps"],
                }
            )
        store.import_page_images(pages)

    @staticmethod
    def _store_specification(store: Any) -> TableSpecification:
        dimensions = tuple(sorted(int(dim) for dim in store.table_dims()))
        return TableSpecification(
            dimensions=dimensions,
            embedding_dtype=store.value_dtype(),
            optimizer_dtype=store.optimizer_dtype(),
            score_policy="uint64",
            protocol_version=_FORMAT_VERSION,
        )

    @staticmethod
    def _specification_metadata(
        specification: TableSpecification,
    ) -> dict[bytes, bytes]:
        return {
            b"sparse_embedding.specification": "\0".join(
                (
                    ",".join(str(dim) for dim in specification.dimensions),
                    specification.embedding_dtype,
                    specification.optimizer_dtype or "",
                    specification.score_policy,
                    str(specification.protocol_version),
                )
            ).encode()
        }

    @staticmethod
    def _decode_specification(
        metadata: dict[bytes, bytes] | None,
    ) -> TableSpecification:
        if metadata is None or b"sparse_embedding.specification" not in metadata:
            raise InvalidCheckpointFragment(
                "checkpoint fragment specification is missing"
            )
        payload = metadata[b"sparse_embedding.specification"].decode().split("\0")
        if len(payload) != 5:
            raise InvalidCheckpointFragment(
                "checkpoint fragment specification is invalid"
            )
        dimensions, embedding_dtype, optimizer_dtype, score_policy, protocol_version = (
            payload
        )
        try:
            return TableSpecification(
                dimensions=tuple(int(item) for item in dimensions.split(",") if item),
                embedding_dtype=embedding_dtype,
                optimizer_dtype=optimizer_dtype or None,
                score_policy=score_policy,
                protocol_version=int(protocol_version),
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
            return TableSpecification(
                dimensions=tuple(int(dim) for dim in manifest["dimensions"]),
                embedding_dtype=manifest["embedding_dtype"],
                optimizer_dtype=manifest["optimizer_dtype"] or None,
                score_policy="uint64",
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
