"""Algorithm-facing sparse checkpoint API."""

from .base import (
    SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME,
    SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME,
    SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
    SUPPORTED_KEY_DTYPES,
    CheckpointHandle,
    CheckpointParticipant,
    CheckpointSaveMode,
    CommittedCheckpoint,
    ElasticRestore,
    PartitionSpec,
    PhysicalRestore,
    PhysicalRestoreStrategy,
    RestoreMode,
    RestorePlan,
    SparsePartitionAssignmentSpec,
)
from .facade import SparseCheckpoint
from .gravitino_lance.storage import GravitinoLanceStorage
from .lance_backend import LocalLanceStorage
from .reference_vectors import generate_reference_vectors
from .restore_options import encode_restore_options

__all__ = [
    "CheckpointHandle",
    "CheckpointSaveMode",
    "CheckpointParticipant",
    "CommittedCheckpoint",
    "ElasticRestore",
    "PartitionSpec",
    "PhysicalRestore",
    "PhysicalRestoreStrategy",
    "RestoreMode",
    "RestorePlan",
    "SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME",
    "SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME",
    "SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME",
    "SUPPORTED_KEY_DTYPES",
    "SparseCheckpoint",
    "LocalLanceStorage",
    "GravitinoLanceStorage",
    "SparsePartitionAssignmentSpec",
    "encode_restore_options",
    "generate_reference_vectors",
]
