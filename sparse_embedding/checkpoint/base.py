import abc
from typing import List, Union


class CheckpointBackend(abc.ABC):
    """
    Base checkpoint backend abstraction.
    Supports different storage strategies: per-slot files, unified dataset, etc.
    Concrete implementations decide the physical layout.
    """

    @abc.abstractmethod
    def save(self, store, uri: str, **kwargs):
        """Single-node save. Backend decides per-slot vs. unified layout."""

    @abc.abstractmethod
    def load(self, store, uri: str, **kwargs):
        """Single-node load."""


class DistributedCheckpointBackend(CheckpointBackend):
    """
    Extension for distributed two-phase commit.
    Typically coordinated by an SPMD Driver across multiple workers.
    """

    @abc.abstractmethod
    def init_dataset(self, store, uri: str, **kwargs):
        """Phase 0: Initialize persistent storage skeleton. Called by Driver."""

    @abc.abstractmethod
    def export_fragments(self, store, uri: str, **kwargs) -> List[bytes]:
        """Phase 1: Export local data and return opaque serialized metadata. Called by each worker."""

    @abc.abstractmethod
    def commit_fragments(
        self,
        uri: str,
        all_fragment_metas: Union[List[bytes], List[List[bytes]]],
        **kwargs,
    ):
        """Phase 2: Gather all metadata and perform atomic commit. Called by Driver."""
