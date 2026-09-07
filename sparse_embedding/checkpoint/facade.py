"""Algorithm-facing sparse checkpoint facade."""

from __future__ import annotations

import uuid
from typing import Any

from .base import (
    CheckpointCommitAuthority,
    CheckpointParticipant,
    CheckpointSaveIdentity,
    PhysicalRestore,
    RestoreMode,
    SparseCheckpointLocator,
    SparseCheckpointProtocol,
    SparseCheckpointStorage,
    SparseWeightSnapshot,
)


class SparseCheckpoint:
    def __init__(self, storage: SparseCheckpointStorage) -> None:
        self._protocol = storage

    @property
    def integration_protocol(self) -> SparseCheckpointProtocol:
        return self._protocol

    def save(
        self,
        store: Any,
        name: str,
        include_page_images: bool = False,
    ) -> SparseCheckpointLocator:
        attempt = uuid.uuid4().hex
        identity = CheckpointSaveIdentity(
            name,
            attempt,
            CheckpointCommitAuthority(
                scope=f"standalone:{attempt}",
                generation=0,
                authority_id=attempt,
            ),
        )
        participant = CheckpointParticipant(0, 1)
        self._protocol.prepare_checkpoint(identity)
        try:
            fragment = self._protocol.write_checkpoint_fragment(
                store, identity, participant, include_page_images=include_page_images
            )
            locator = self._protocol.commit_checkpoint(identity, (fragment,))
            self._protocol.acknowledge_checkpoint(identity)
            return locator
        except Exception:
            self._protocol.abort_checkpoint(identity)
            raise

    def restore(
        self,
        store: Any,
        locator: SparseCheckpointLocator,
        mode: RestoreMode | None = None,
        *,
        load_optimizer_state: bool = True,
    ) -> None:
        selected_mode = mode or PhysicalRestore(CheckpointParticipant(0, 1))
        self._protocol.restore_checkpoint(
            store, locator, selected_mode, load_optimizer_state
        )

    def restore_weight_snapshot(
        self,
        store: Any,
        snapshot: SparseWeightSnapshot,
        mode: RestoreMode | None = None,
    ) -> None:
        selected_mode = mode or PhysicalRestore(CheckpointParticipant(0, 1))
        self._protocol.restore_weight_snapshot(store, snapshot, selected_mode)
