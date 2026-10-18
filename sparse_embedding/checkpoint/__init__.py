"""Algorithm-facing sparse checkpoint API."""

from .base import (
    PUBLISHED_WEIGHT_DESCRIPTOR_FILENAME,
    SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME,
    SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME,
    SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
    SUPPORTED_KEY_DTYPES,
    CheckpointableStore,
    CheckpointHandle,
    CheckpointParticipant,
    CheckpointSaveMode,
    CommittedCheckpoint,
    IngestionEngine,
    PartitionSpec,
    PartitionTarget,
    RestorePlan,
    resolve_effective_save_mode,
)
from .facade import SparseCheckpoint
from .gravitino_lance.storage import GravitinoLanceStorage
from .lance_backend import LocalLanceStorage, compact_fragments
from .reference_vectors import generate_reference_vectors
from .restore_options import encode_restore_options

__all__ = [
    "PUBLISHED_WEIGHT_DESCRIPTOR_FILENAME",
    "SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME",
    "SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME",
    "SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME",
    "SUPPORTED_KEY_DTYPES",
    "CheckpointHandle",
    "CheckpointParticipant",
    "CheckpointSaveMode",
    "CheckpointableStore",
    "CommittedCheckpoint",
    "GravitinoLanceStorage",
    "IngestionEngine",
    "LocalLanceStorage",
    "PartitionSpec",
    "PartitionTarget",
    "RestorePlan",
    "SparseCheckpoint",
    "compact_fragments",
    "encode_restore_options",
    "generate_reference_vectors",
    "resolve_effective_save_mode",
]
