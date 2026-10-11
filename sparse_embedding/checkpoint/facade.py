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
    CheckpointSaveMode,
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
        mode: CheckpointSaveMode = CheckpointSaveMode.FULL,
        parent_handle: CheckpointHandle | None = None,
    ) -> CommittedCheckpoint:
        if not isinstance(mode, CheckpointSaveMode):
            raise TypeError("mode must be a CheckpointSaveMode")
        if mode is CheckpointSaveMode.FULL and parent_handle is not None:
            raise ValueError("full checkpoint must not specify parent_handle")
        if mode is CheckpointSaveMode.INCREMENTAL and parent_handle is None:
            raise ValueError("incremental checkpoint requires parent_handle")
        parent_manifest = (
            self._protocol.resolve_checkpoint_manifest(parent_handle)
            if parent_handle is not None
            else None
        )
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
        fragment = None
        try:
            fragment = self._protocol.write_checkpoint_fragment(
                store,
                identity,
                participant,
                include_page_images=include_page_images,
                mode=mode,
                parent_manifest=parent_manifest,
            )
            self._protocol.finalize_checkpoint(identity)
            committed = self._protocol.commit_checkpoint(
                identity,
                (fragment,),
            )
            self._protocol.acknowledge_store_mutations(store, committed.handle)
            if weight_path is not None:
                self._protocol.write_weight_reference(
                    str(weight_path), committed.handle
                )
            self._protocol.acknowledge_checkpoint(identity)
            if weight_path is None:
                self._protocol.release_reference(committed.handle)
            return committed
        except Exception:
            if fragment is not None:
                store.cancel_checkpoint_capture(
                    fragment.save_context.capture_id,
                    fragment.save_context.journal_identity,
                )
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
