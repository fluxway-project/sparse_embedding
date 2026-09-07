"""Framework-neutral sparse checkpoint domain types."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, TypeAlias


class SparseCheckpointError(RuntimeError):
    pass


class CheckpointConflictError(SparseCheckpointError):
    pass


class InvalidCheckpointFragment(SparseCheckpointError):
    pass


class UnsupportedCheckpointMode(SparseCheckpointError):
    pass


class StorageBudgetExceeded(SparseCheckpointError):
    pass


class UnsupportedValueType(SparseCheckpointError):
    pass


SUPPORTED_VALUE_DTYPES = frozenset({"float16", "float32", "bfloat16"})


@dataclass(frozen=True, slots=True)
class TableSpecification:
    dimensions: tuple[int, ...]
    embedding_dtype: str
    optimizer_dtype: str | None
    score_policy: str
    protocol_version: int

    def __post_init__(self) -> None:
        if not self.dimensions or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
            for dim in self.dimensions
        ):
            raise ValueError("dimensions must contain positive integers")
        if tuple(sorted(set(self.dimensions))) != self.dimensions:
            raise ValueError("dimensions must be sorted and unique")
        if self.embedding_dtype not in SUPPORTED_VALUE_DTYPES:
            raise UnsupportedValueType(
                f"unsupported embedding dtype: {self.embedding_dtype}"
            )
        if (
            self.optimizer_dtype is not None
            and self.optimizer_dtype not in SUPPORTED_VALUE_DTYPES
        ):
            raise UnsupportedValueType(
                f"unsupported optimizer dtype: {self.optimizer_dtype}"
            )
        if not self.score_policy.strip():
            raise ValueError("score_policy must be non-empty")
        if self.protocol_version != 1:
            raise ValueError("unsupported checkpoint protocol version")


@dataclass(frozen=True, slots=True)
class CheckpointCommitAuthority:
    scope: str
    generation: int
    authority_id: str

    def __post_init__(self) -> None:
        for name in ("scope", "authority_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
            or self.generation >= 1 << 64
        ):
            raise ValueError("generation must be an unsigned 64-bit integer")


@dataclass(frozen=True, slots=True)
class CheckpointSaveIdentity:
    checkpoint_name: str
    save_attempt: str
    commit_authority: CheckpointCommitAuthority

    def __post_init__(self) -> None:
        for name in ("checkpoint_name", "save_attempt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.commit_authority, CheckpointCommitAuthority):
            raise TypeError("commit_authority must be CheckpointCommitAuthority")


@dataclass(frozen=True, slots=True)
class CheckpointParticipant:
    participant_index: int
    participant_count: int

    def __post_init__(self) -> None:
        if isinstance(self.participant_index, bool) or not isinstance(
            self.participant_index, int
        ):
            raise TypeError("participant_index must be an integer")
        if isinstance(self.participant_count, bool) or not isinstance(
            self.participant_count, int
        ):
            raise TypeError("participant_count must be an integer")
        if self.participant_count <= 0:
            raise ValueError("participant_count must be positive")
        if not 0 <= self.participant_index < self.participant_count:
            raise ValueError("participant_index is outside participant_count")


@dataclass(frozen=True, slots=True)
class CheckpointFragment:
    fragment_id: str
    checkpoint_name: str
    save_attempt: str
    participant: CheckpointParticipant
    entry_count: int
    page_count: int
    schema_digest: str
    content_digest: str

    def __post_init__(self) -> None:
        for name in (
            "fragment_id",
            "checkpoint_name",
            "save_attempt",
            "schema_digest",
            "content_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("entry_count", "page_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SparseCheckpointLocator:
    checkpoint_name: str
    checkpoint_version: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.checkpoint_name, str)
            or not self.checkpoint_name.strip()
        ):
            raise ValueError("checkpoint_name must be a non-empty string")
        if (
            isinstance(self.checkpoint_version, bool)
            or not isinstance(self.checkpoint_version, int)
            or self.checkpoint_version <= 0
        ):
            raise ValueError("checkpoint_version must be a positive integer")


@dataclass(frozen=True, slots=True)
class SparseWeightSnapshot:
    """Immutable content identity used by inference weight sources."""

    snapshot_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.snapshot_digest, str)
            or not self.snapshot_digest.strip()
        ):
            raise ValueError("snapshot_digest must be a non-empty string")


PartitionFn: TypeAlias = Callable[[Any, int], Any]


class PhysicalRestoreStrategy(str, Enum):
    RECONSTRUCT_ENTRIES = "reconstruct_entries"
    FAST_PAGE_CACHE = "fast_page_cache"


@dataclass(frozen=True, slots=True)
class PhysicalRestore:
    participant: CheckpointParticipant
    strategy: PhysicalRestoreStrategy = PhysicalRestoreStrategy.RECONSTRUCT_ENTRIES

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, PhysicalRestoreStrategy):
            raise TypeError("strategy must be a PhysicalRestoreStrategy")


@dataclass(frozen=True, slots=True)
class ElasticRestore:
    partition_fn: PartitionFn
    partition_count: int
    target_partition_id: int

    def __post_init__(self) -> None:
        if not callable(self.partition_fn):
            raise TypeError("partition_fn must be callable")
        if isinstance(self.partition_count, bool) or not isinstance(
            self.partition_count, int
        ):
            raise TypeError("partition_count must be an integer")
        if self.partition_count <= 0:
            raise ValueError("partition_count must be positive")
        if isinstance(self.target_partition_id, bool) or not isinstance(
            self.target_partition_id, int
        ):
            raise TypeError("target_partition_id must be an integer")
        if not 0 <= self.target_partition_id < self.partition_count:
            raise ValueError("target_partition_id is outside partition_count")


RestoreMode: TypeAlias = PhysicalRestore | ElasticRestore


@dataclass(frozen=True, slots=True)
class CheckpointLeaseRequest:
    lease_holder_id: str
    ttl_seconds: float
    request_id: str

    def __post_init__(self) -> None:
        for name in ("lease_holder_id", "request_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")


@dataclass(frozen=True, slots=True)
class CheckpointLease:
    lease_id: str
    locator: SparseCheckpointLocator
    lease_holder_id: str
    expires_at: datetime
    generation: int
    request_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.lease_id, str) or not self.lease_id.strip():
            raise ValueError("lease_id must be a non-empty string")
        if (
            not isinstance(self.lease_holder_id, str)
            or not self.lease_holder_id.strip()
        ):
            raise ValueError("lease_holder_id must be a non-empty string")
        if self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.generation < 0:
            raise ValueError("lease generation must be non-negative")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CleanupReservation:
    reservation_id: str
    locators: tuple[SparseCheckpointLocator, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reservation_id, str) or not self.reservation_id.strip():
            raise ValueError("reservation_id must be a non-empty string")
        if not self.locators:
            raise ValueError("reclaim reservation requires locators")
        if len(set(self.locators)) != len(self.locators):
            raise ValueError("reclaim reservation locators must be unique")


class CheckpointLeaseStore(Protocol):
    def acquire_checkpoint_lease(
        self,
        locator: SparseCheckpointLocator,
        request: CheckpointLeaseRequest,
    ) -> CheckpointLease: ...

    def renew_checkpoint_lease(
        self, lease: CheckpointLease, ttl_seconds: float
    ) -> CheckpointLease: ...

    def release_checkpoint_lease(self, lease: CheckpointLease) -> None: ...

    def active_checkpoint_leases(
        self, now: datetime | None = None
    ) -> Sequence[CheckpointLease]: ...


class CheckpointCleanup(Protocol):
    def reserve_cleanup(
        self,
        reservation_id: str,
        ordered_candidates: Sequence[SparseCheckpointLocator],
        required_count: int,
    ) -> CleanupReservation: ...

    def commit_cleanup(self, reservation: CleanupReservation) -> None: ...

    def rollback_cleanup(self, reservation: CleanupReservation) -> None: ...

    def cleanup_checkpoint(
        self,
        reservation: CleanupReservation,
        locator: SparseCheckpointLocator,
        reference_id: str,
    ) -> None: ...

    def release_reference(
        self, locator: SparseCheckpointLocator, reference_id: str
    ) -> None: ...


class SparseCheckpointProtocol(Protocol):
    def prepare_checkpoint(self, identity: CheckpointSaveIdentity) -> None: ...

    def write_checkpoint_fragment(
        self,
        store: Any,
        identity: CheckpointSaveIdentity,
        participant: CheckpointParticipant,
        include_page_images: bool = False,
    ) -> CheckpointFragment: ...

    def finalize_checkpoint(self, identity: CheckpointSaveIdentity) -> None: ...

    def commit_checkpoint(
        self,
        identity: CheckpointSaveIdentity,
        fragments: Sequence[CheckpointFragment],
        lease_request: CheckpointLeaseRequest | None = None,
        reference_id: str | None = None,
    ) -> SparseCheckpointLocator: ...

    def abort_checkpoint(self, identity: CheckpointSaveIdentity) -> None: ...

    def acknowledge_checkpoint(self, identity: CheckpointSaveIdentity) -> None: ...

    def restore_checkpoint(
        self,
        store: Any,
        locator: SparseCheckpointLocator,
        mode: RestoreMode,
        load_optimizer_state: bool = True,
    ) -> None: ...

    def restore_weight_snapshot(
        self,
        store: Any,
        snapshot: SparseWeightSnapshot,
        mode: RestoreMode,
    ) -> None: ...


class SparseCheckpointStorage(
    SparseCheckpointProtocol,
    CheckpointLeaseStore,
    CheckpointCleanup,
    Protocol,
):
    def close(self) -> None: ...

    def compact_checkpoints(self) -> None: ...

    def compact_checkpoint(self, locator: SparseCheckpointLocator) -> None: ...

    def compact_fragments(
        self,
        source_path: str,
        target_path: str | None = None,
    ) -> str: ...

    async def compact_checkpoints_async(self) -> None: ...

    async def compact_checkpoint_async(
        self, locator: SparseCheckpointLocator
    ) -> None: ...

    async def compact_fragments_async(
        self,
        source_path: str,
        target_path: str | None = None,
    ) -> str: ...


class SparseWeightSnapshotReader(Protocol):
    def restore_weight_snapshot(
        self,
        store: Any,
        snapshot: SparseWeightSnapshot,
        mode: RestoreMode,
    ) -> None: ...
