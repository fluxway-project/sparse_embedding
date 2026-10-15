"""Framework-neutral sparse checkpoint domain types."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol


def attach_cleanup_errors(
    primary: BaseException,
    cleanup_errors: Sequence[BaseException],
    operation: str,
) -> None:
    """Keep the operation failure primary while reporting failed cleanup."""
    for error in cleanup_errors:
        primary.add_note(f"{operation} cleanup failed: {type(error).__name__}: {error}")


def run_cleanup_preserving_primary(
    primary: BaseException,
    actions: Sequence[tuple[str, Any]],
    operation: str,
) -> None:
    """Run every cleanup action and retain all secondary failures."""
    failures: list[BaseException] = []
    for name, action in actions:
        try:
            action()
        except BaseException as error:
            failures.append(RuntimeError(f"{name}: {error}"))
    if failures:
        attach_cleanup_errors(primary, failures, operation)


def execute_cleanup_actions(
    actions: Sequence[tuple[str, Any]],
    operation: str,
) -> None:
    """Execute cleanup actions in sequence, preserving the first failure as primary."""
    for index, (_, action) in enumerate(actions):
        try:
            action()
        except BaseException as primary:
            run_cleanup_preserving_primary(primary, actions[index + 1 :], operation)
            raise


SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME = "sparse_embedding.modulo"
SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME = "sparse_embedding.hash"
SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME = "sparse_embedding.fast_range"
PUBLISHED_WEIGHT_DESCRIPTOR_FILENAME = ".sparse_embedding_reference"
_MAX_ASSIGNMENT_NAME_BYTES = 128
_MAX_ASSIGNMENT_MAP_ITEMS = 8
_MAX_ASSIGNMENT_ARRAY_ITEMS = 8
_MAX_ASSIGNMENT_STRING_BYTES = 128
_MAX_ASSIGNMENT_DEPTH = 8


def _freeze_assignment_value(value: object, path: str, depth: int = 0) -> object:
    if depth > _MAX_ASSIGNMENT_DEPTH:
        raise ValueError(f"{path} exceeds assignment nesting depth limit")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value < -(1 << 63) or value >= 1 << 64:
            raise ValueError(f"{path} integer is out of range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, str):
        if not value or len(value.encode("utf-8")) > _MAX_ASSIGNMENT_STRING_BYTES:
            raise ValueError(f"{path} has an invalid string")
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_ASSIGNMENT_MAP_ITEMS:
            raise ValueError(f"{path} exceeds map size limit")
        keys = tuple(value)
        if any(not isinstance(key, str) or not key for key in keys):
            raise TypeError(f"{path} keys must be non-empty strings")
        if keys != tuple(sorted(keys)):
            raise ValueError(f"{path} keys must be sorted")
        return MappingProxyType(
            {
                key: _freeze_assignment_value(value[key], f"{path}.{key}", depth + 1)
                for key in keys
            }
        )
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_ASSIGNMENT_ARRAY_ITEMS:
            raise ValueError(f"{path} exceeds array size limit")
        return tuple(
            _freeze_assignment_value(item, f"{path}[{index}]", depth + 1)
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} contains an unsupported value")


class SparseCheckpointError(RuntimeError):
    pass


class CheckpointConflictError(SparseCheckpointError):
    pass


class InvalidCheckpointFragment(SparseCheckpointError):
    pass


class UnsupportedCheckpointMode(SparseCheckpointError):
    pass


class CheckpointSaveMode(str, Enum):
    FULL = "full"
    INCREMENTAL = "incremental"


class StorageBudgetExceeded(SparseCheckpointError):
    pass


class UnsupportedValueType(SparseCheckpointError):
    pass


SUPPORTED_VALUE_DTYPES = frozenset({"float16", "float32", "bfloat16"})
SUPPORTED_KEY_DTYPES = frozenset({"int32", "uint32", "int64", "uint64"})


@dataclass(frozen=True, slots=True)
class TableSpecification:
    dimensions: tuple[int, ...]
    embedding_dtype: str
    optimizer_dtype: str | None
    score_policy: str
    key_dtype: str

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
        if self.key_dtype not in SUPPORTED_KEY_DTYPES:
            raise ValueError(f"unsupported key dtype: {self.key_dtype}")


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
class CheckpointSaveContext:
    mode: CheckpointSaveMode
    parent: SparseCheckpointLocator | None
    parent_reference_id: str
    mutation_watermark: int
    include_page_images: bool
    entries_per_page: int
    capture_id: int
    journal_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CheckpointSaveMode):
            raise TypeError("mode must be CheckpointSaveMode")
        if self.mode is CheckpointSaveMode.INCREMENTAL:
            if self.parent is None or not self.parent_reference_id:
                raise ValueError(
                    "incremental context requires immutable parent identity"
                )
        elif self.parent is not None or self.parent_reference_id:
            raise ValueError("FULL context cannot have a parent")
        if (
            isinstance(self.mutation_watermark, bool)
            or not isinstance(self.mutation_watermark, int)
            or not 0 <= self.mutation_watermark < 1 << 64
        ):
            raise ValueError("mutation watermark must be an unsigned 64-bit integer")
        if (
            not isinstance(self.entries_per_page, int)
            or isinstance(self.entries_per_page, bool)
            or self.entries_per_page <= 0
            or self.entries_per_page & (self.entries_per_page - 1)
        ):
            raise ValueError("entries_per_page must be a positive power of two")
        if (
            isinstance(self.capture_id, bool)
            or not isinstance(self.capture_id, int)
            or not 0 <= self.capture_id < 1 << 64
        ):
            raise ValueError("capture_id must be an unsigned 64-bit integer")
        if not isinstance(self.journal_identity, str) or bool(self.capture_id) != bool(
            self.journal_identity
        ):
            raise ValueError(
                "capture_id and journal_identity must describe the same live cut"
            )
        if self.mode is CheckpointSaveMode.INCREMENTAL and not self.capture_id:
            raise ValueError("incremental context requires a live capture")
        if not isinstance(self.include_page_images, bool):
            raise TypeError("include_page_images must be boolean")


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
    specification: TableSpecification
    save_context: CheckpointSaveContext
    page_dataset: str = ""
    page_version: int = 0
    page_layout_generation: int = 0

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
        if not isinstance(self.page_dataset, str):
            raise TypeError("page_dataset must be a string")
        for name in ("page_version", "page_layout_generation"):
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


type CheckpointHandle = bytes


@dataclass(frozen=True, slots=True)
class CommittedCheckpoint:
    """Owner-issued opaque handle and its logical version identity."""

    handle: CheckpointHandle
    version_id: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.handle, bytes) or not self.handle:
            raise ValueError("checkpoint handle must be non-empty bytes")
        if not isinstance(self.version_id, bytes) or not self.version_id:
            raise ValueError("checkpoint version_id must be non-empty bytes")


class IngestionEngine(str, Enum):
    RECONSTRUCT_ENTRIES = "reconstruct_entries"
    FAST_PAGE_CACHE = "fast_page_cache"


class PartitionSpec:
    """Declarative partition specification serving as Single Source of Truth."""

    __slots__ = ("_name", "_parameters")

    def __init__(
        self,
        name: str,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("partition spec name must be a string")
        if not name.strip():
            raise ValueError("partition spec name must be non-empty")
        if len(name.encode("utf-8")) > _MAX_ASSIGNMENT_NAME_BYTES:
            raise ValueError("partition spec name is too long")
        params = parameters if parameters is not None else {}
        if not isinstance(params, Mapping):
            raise TypeError("partition spec parameters must be a mapping")

        frozen_parameters = _freeze_assignment_value(params, "parameters")
        if not isinstance(frozen_parameters, Mapping):
            raise TypeError("partition spec parameters must be a mapping")

        _APPROVED_ASSIGNMENTS = {
            SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
            SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME,
            SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME,
        }
        if name not in _APPROVED_ASSIGNMENTS:
            try:
                import sparse_embedding as _se

                if not _se.is_partitioner_registered(name):
                    raise ValueError(
                        "sparse assignment is not an owner-approved executable contract"
                    )
            except (ImportError, AttributeError):
                raise ValueError(
                    "sparse assignment is not an owner-approved executable contract"
                )

        self._name = name
        self._parameters = frozen_parameters

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> Mapping[str, object]:
        return self._parameters

    @property
    def sparse_assignment_name(self) -> str:
        return self._name

    @property
    def sparse_assignment_parameters(self) -> Mapping[str, object]:
        return self._parameters

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PartitionSpec):
            return self._name == other._name and self._parameters == other._parameters
        return False

    def __hash__(self) -> int:
        return hash((self._name, tuple(sorted(self._parameters.items()))))

    def __repr__(self) -> str:
        return (
            f"PartitionSpec(name={self._name!r}, parameters={dict(self._parameters)!r})"
        )

    @classmethod
    def modulo(cls) -> PartitionSpec:
        return cls(SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME, {})

    @classmethod
    def hash(cls, *, seed: int | None = None) -> PartitionSpec:
        params: dict[str, object] = {}
        if seed is not None:
            params["seed"] = str(seed)
        return cls(SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME, params)

    @classmethod
    def fast_range(cls) -> PartitionSpec:
        return cls(SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME, {})

    @classmethod
    def from_strategy(cls, strategy: Any) -> PartitionSpec:
        if hasattr(strategy, "spec") and isinstance(strategy.spec, PartitionSpec):
            return strategy.spec
        raise TypeError(
            f"Expected PartitionStrategy with .spec, got {type(strategy).__name__}"
        )


class RestorePlan(ABC):
    """Abstract base class for typed checkpoint restore plans."""

    __slots__ = ()

    @property
    @abstractmethod
    def restore_kind(self) -> str:
        """Identifier for the canonical topology routing mode."""
        raise NotImplementedError

    @abstractmethod
    def encode(self) -> bytes:
        """Encode this restore plan into canonical opaque bytes."""
        raise NotImplementedError

    @classmethod
    def from_bytes(cls, payload: bytes) -> RestorePlan:
        """Decode canonical opaque bytes into a concrete RestorePlan."""
        from .restore_options import _decode_restore_options

        return _decode_restore_options(payload)

    @classmethod
    def from_participant(
        cls,
        participant: CheckpointParticipant,
        *,
        key_dtype: str = "uint64",
        engine: IngestionEngine = IngestionEngine.RECONSTRUCT_ENTRIES,
    ) -> RankRestore:
        return RankRestore(participant, engine=engine, key_dtype=key_dtype)

    @classmethod
    def from_partition(
        cls,
        spec: PartitionSpec,
        partition_count: int,
        partition_id: int,
        *,
        key_dtype: str = "uint64",
        engine: IngestionEngine = IngestionEngine.RECONSTRUCT_ENTRIES,
    ) -> ReshardRestore:
        return ReshardRestore(
            spec,
            partition_count,
            partition_id,
            key_dtype,
            engine=engine,
        )

    @classmethod
    def from_topology(
        cls,
        *,
        saved_participant_count: int | None,
        participant: CheckpointParticipant,
        spec: PartitionSpec,
        key_dtype: str = "uint64",
        engine: IngestionEngine = IngestionEngine.RECONSTRUCT_ENTRIES,
    ) -> RestorePlan:
        """Select rank restore only when the saved and current topologies match."""
        if not isinstance(spec, PartitionSpec):
            raise TypeError("spec must be a PartitionSpec")
        if not isinstance(participant, CheckpointParticipant):
            raise TypeError("participant must be a CheckpointParticipant")
        if saved_participant_count is not None:
            if (
                isinstance(saved_participant_count, bool)
                or not isinstance(saved_participant_count, int)
                or saved_participant_count <= 0
            ):
                raise ValueError("saved_participant_count must be positive")
        if saved_participant_count == participant.participant_count:
            return cls.from_participant(
                participant,
                key_dtype=key_dtype,
                engine=engine,
            )
        return cls.from_partition(
            spec,
            participant.participant_count,
            participant.participant_index,
            key_dtype=key_dtype,
            engine=engine,
        )

    decode = from_bytes
    rank = from_participant
    reshard = from_partition


class RankRestore(RestorePlan):
    """Rank-preserving topology restore with a selectable ingestion engine."""

    __slots__ = ("_key_dtype", "_participant", "_engine")

    def __init__(
        self,
        participant: CheckpointParticipant,
        engine: IngestionEngine = IngestionEngine.RECONSTRUCT_ENTRIES,
        key_dtype: str = "uint64",
    ) -> None:
        if not isinstance(participant, CheckpointParticipant):
            raise TypeError(
                f"participant must be a CheckpointParticipant; got {type(participant).__name__}"
            )
        if not isinstance(engine, IngestionEngine):
            raise TypeError("engine must be an IngestionEngine")
        if key_dtype not in SUPPORTED_KEY_DTYPES:
            raise ValueError(f"unsupported key dtype: {key_dtype}")

        self._participant = participant
        self._engine = engine
        self._key_dtype = key_dtype

    @property
    def restore_kind(self) -> str:
        return "rank"

    @property
    def participant(self) -> CheckpointParticipant:
        return self._participant

    @property
    def engine(self) -> IngestionEngine:
        return self._engine

    @property
    def key_dtype(self) -> str:
        return self._key_dtype

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RankRestore):
            return (
                self._participant == other._participant
                and self._engine == other._engine
                and self._key_dtype == other._key_dtype
            )
        return False

    def __hash__(self) -> int:
        return hash((self._participant, self._engine, self._key_dtype))

    def __repr__(self) -> str:
        return (
            f"RankRestore(participant={self._participant!r}, "
            f"engine={self._engine!r}, key_dtype={self._key_dtype!r})"
        )

    def encode(self) -> bytes:
        from .restore_options import encode_restore_options

        return encode_restore_options(self)


class ReshardRestore(RestorePlan):
    """M:N partition-reshard topology restore using canonical entry reconstruction."""

    __slots__ = (
        "_engine",
        "_key_dtype",
        "_partition_count",
        "_partition_id",
        "_spec",
    )

    def __init__(
        self,
        spec: PartitionSpec,
        partition_count: int,
        partition_id: int,
        key_dtype: str = "uint64",
        *,
        engine: IngestionEngine = IngestionEngine.RECONSTRUCT_ENTRIES,
    ) -> None:
        if not isinstance(spec, PartitionSpec):
            raise TypeError(f"spec must be a PartitionSpec; got {type(spec).__name__}")
        if key_dtype not in SUPPORTED_KEY_DTYPES:
            raise ValueError(f"unsupported key dtype: {key_dtype}")
        if not isinstance(engine, IngestionEngine):
            raise TypeError("engine must be an IngestionEngine")
        if engine is IngestionEngine.FAST_PAGE_CACHE:
            raise ValueError("FAST_PAGE_CACHE is only valid for rank restore")
        if isinstance(partition_count, bool) or not isinstance(partition_count, int):
            raise TypeError("partition_count must be an integer")
        if partition_count <= 0:
            raise ValueError("partition_count must be positive")
        if isinstance(partition_id, bool) or not isinstance(partition_id, int):
            raise TypeError("partition_id must be an integer")
        if not 0 <= partition_id < partition_count:
            raise ValueError("partition_id is outside partition_count")

        self._spec = spec
        self._engine = engine
        self._partition_count = partition_count
        self._partition_id = partition_id
        self._key_dtype = key_dtype

    @property
    def restore_kind(self) -> str:
        return "reshard"

    @property
    def spec(self) -> PartitionSpec:
        return self._spec

    @property
    def engine(self) -> IngestionEngine:
        return self._engine

    @property
    def key_dtype(self) -> str:
        return self._key_dtype

    @property
    def partition_count(self) -> int:
        return self._partition_count

    @property
    def partition_id(self) -> int:
        return self._partition_id

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ReshardRestore):
            return (
                self._spec == other._spec
                and self._engine == other._engine
                and self._key_dtype == other._key_dtype
                and self._partition_count == other._partition_count
                and self._partition_id == other._partition_id
            )
        return False

    def __hash__(self) -> int:
        return hash(
            (
                self._spec,
                self._engine,
                self._key_dtype,
                self._partition_count,
                self._partition_id,
            )
        )

    def __repr__(self) -> str:
        return (
            f"ReshardRestore(spec={self._spec!r}, "
            f"engine={self._engine!r}, "
            f"key_dtype={self._key_dtype!r}, "
            f"partition_count={self._partition_count}, "
            f"partition_id={self._partition_id})"
        )

    def encode(self) -> bytes:
        from .restore_options import encode_restore_options

        return encode_restore_options(self)


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
    owner_handle: CheckpointHandle
    handles: tuple[CheckpointHandle, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reservation_id, str) or not self.reservation_id.strip():
            raise ValueError("reservation_id must be a non-empty string")
        if not isinstance(self.owner_handle, bytes) or not self.owner_handle:
            raise ValueError("cleanup owner handle must be non-empty bytes")
        if not self.handles:
            raise ValueError("reclaim reservation requires locators")
        if any(not isinstance(handle, bytes) or not handle for handle in self.handles):
            raise ValueError("cleanup handles must be non-empty bytes")
        if len(set(self.handles)) != len(self.handles):
            raise ValueError("reclaim reservation handles must be unique")


class CheckpointLeaseStore(Protocol):
    def acquire_checkpoint_lease(
        self,
        handle: CheckpointHandle,
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
        ordered_candidates: Sequence[CheckpointHandle],
        required_count: int,
    ) -> CleanupReservation: ...

    def commit_cleanup(self, reservation: CleanupReservation) -> None: ...

    def rollback_cleanup(self, reservation: CleanupReservation) -> None: ...

    def cleanup_checkpoint(
        self,
        reservation: CleanupReservation,
        handle: CheckpointHandle,
    ) -> None: ...

    def release_reference(self, handle: CheckpointHandle) -> None: ...


class SparseCheckpointProtocol(Protocol):
    def resolve_lineage_head_handle(
        self, checkpoint_name: str
    ) -> CheckpointHandle | None: ...

    def acknowledge_store_mutations(
        self, store: Any, handle: CheckpointHandle, participant_index: int = 0
    ) -> None: ...

    def prepare_checkpoint(self, identity: CheckpointSaveIdentity) -> None: ...

    def write_checkpoint_fragment_bytes(
        self,
        store: Any,
        identity: CheckpointSaveIdentity,
        participant: CheckpointParticipant,
        include_page_images: bool = False,
        mode: CheckpointSaveMode = CheckpointSaveMode.FULL,
        parent_handle: CheckpointHandle | None = None,
    ) -> bytes: ...

    def finalize_checkpoint(self, identity: CheckpointSaveIdentity) -> None: ...

    def commit_checkpoint_bytes(
        self,
        identity: CheckpointSaveIdentity,
        fragments: Sequence[bytes],
        lease_request: CheckpointLeaseRequest | None = None,
    ) -> CommittedCheckpoint: ...

    def abort_checkpoint(self, identity: CheckpointSaveIdentity) -> None: ...

    def acknowledge_checkpoint(self, identity: CheckpointSaveIdentity) -> None: ...

    def load_component_checkpoint(
        self,
        store: Any,
        storage_locator: CheckpointHandle,
        restore: bytes,
        *,
        load_optimizer_state: bool = True,
    ) -> None: ...

    def publish_weight_handle(
        self,
        path: str | Path,
        handle: CheckpointHandle,
    ) -> None: ...

    def read_published_weight_handle(
        self,
        path: str | Path,
    ) -> CheckpointHandle: ...

    def load_inference_weights(
        self,
        store: Any,
        restore: bytes,
    ) -> None: ...


class SparseCheckpointStorage(
    SparseCheckpointProtocol,
    CheckpointLeaseStore,
    CheckpointCleanup,
    Protocol,
):
    def close(self) -> None: ...

    def compact_checkpoints(
        self, handles: Sequence[CheckpointHandle], *, include_page_images: bool = False
    ) -> tuple[CheckpointHandle, ...]: ...

    def compact_checkpoint(
        self, handle: CheckpointHandle, *, include_page_images: bool = False
    ) -> CheckpointHandle: ...

    def compact_fragments(
        self,
        source_path: str,
        target_path: str | None = None,
    ) -> str: ...

    async def compact_checkpoints_async(
        self, handles: Sequence[CheckpointHandle], *, include_page_images: bool = False
    ) -> tuple[CheckpointHandle, ...]: ...

    async def compact_checkpoint_async(
        self, handle: CheckpointHandle, *, include_page_images: bool = False
    ) -> CheckpointHandle: ...

    async def compact_fragments_async(
        self,
        source_path: str,
        target_path: str | None = None,
    ) -> str: ...
