"""Algorithm-facing sparse checkpoint facade."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .base import (
    CheckpointCommitAuthority,
    CheckpointHandle,
    CheckpointParticipant,
    CheckpointSaveIdentity,
    CommittedCheckpoint,
    RestorePlan,
    SparseCheckpointProtocol,
    SparseCheckpointStorage,
)


class SparseCheckpoint:
    def __init__(self, storage: SparseCheckpointStorage) -> None:
        self._protocol = storage

    @property
    def integration_protocol(self) -> SparseCheckpointProtocol:
        return self._protocol

    def save_training_checkpoint(
        self,
        store: Any,
        name: str,
        *,
        weight_path: str | Path | None = None,
        include_page_images: bool = False,
    ) -> CommittedCheckpoint:
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
        committed: CommittedCheckpoint | None = None
        try:
            fragment = self._protocol.write_checkpoint_fragment(
                store, identity, participant, include_page_images=include_page_images
            )
            self._protocol.finalize_checkpoint(identity)
            committed = self._protocol.commit_checkpoint(
                identity,
                (fragment,),
            )
            if weight_path is not None:
                self._protocol.write_weight_reference(
                    str(weight_path), committed.handle
                )
            self._protocol.acknowledge_checkpoint(identity)
            if weight_path is None:
                self._protocol.release_reference(committed.handle)
            return committed
        except Exception:
            self._protocol.abort_checkpoint(identity)
            raise

    def load_training_checkpoint(
        self,
        store: Any,
        storage_locator: CheckpointHandle,
        restore: RestorePlan | bytes,
        *,
        load_optimizer_state: bool = True,
    ) -> None:
        if isinstance(restore, RestorePlan):
            options = restore.encode()
        elif isinstance(restore, bytes):
            options = restore
        else:
            raise TypeError(
                f"restore must be a RestorePlan instance or bytes; got {type(restore).__name__}"
            )
        self._protocol.load_component_checkpoint(
            store,
            storage_locator,
            options,
            load_optimizer_state=load_optimizer_state,
        )
