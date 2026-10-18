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
    resolve_effective_save_mode,
    run_cleanup_preserving_primary,
)


class SparseCheckpoint:
    def __init__(self, storage: SparseCheckpointStorage) -> None:
        self._protocol = storage

    @property
    def integration_protocol(self) -> SparseCheckpointProtocol:
        return self._protocol

    def requires_full_checkpoint(self, store: Any) -> bool:
        return self._protocol.requires_full_checkpoint(store)

    def save_training_checkpoint(
        self,
        store: Any,
        name: str,
        *,
        publish_weight_path: str | Path | None = None,
        include_page_images: bool = False,
        mode: CheckpointSaveMode | None = None,
        parent_handle: CheckpointHandle | None = None,
    ) -> CommittedCheckpoint:
        if mode is None:
            effective_mode, effective_parent = resolve_effective_save_mode(
                CheckpointSaveMode.INCREMENTAL,
                self.requires_full_checkpoint(store),
                parent_handle,
            )
        else:
            if not isinstance(mode, CheckpointSaveMode):
                raise TypeError("mode must be a CheckpointSaveMode")
            if mode is CheckpointSaveMode.FULL:
                if parent_handle is not None:
                    raise ValueError("full checkpoint must not specify parent_handle")
                effective_mode, effective_parent = resolve_effective_save_mode(
                    mode,
                    False,
                    None,
                )
            elif mode is CheckpointSaveMode.INCREMENTAL:
                if parent_handle is None:
                    raise ValueError("incremental checkpoint requires parent_handle")
                effective_mode, effective_parent = resolve_effective_save_mode(
                    mode,
                    self.requires_full_checkpoint(store),
                    parent_handle,
                )
                if effective_mode is CheckpointSaveMode.FULL:
                    raise ValueError(
                        "reshard restore requires the next training checkpoint to be FULL"
                    )
            else:
                raise TypeError("mode must be a CheckpointSaveMode")
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
        fragment: bytes
        try:
            fragment = self._protocol.write_checkpoint_fragment_bytes(
                store,
                identity,
                participant,
                include_page_images=include_page_images,
                mode=effective_mode,
                parent_handle=effective_parent,
            )
            self._protocol.finalize_checkpoint(identity)
            committed = self._protocol.commit_checkpoint_bytes(
                identity,
                (fragment,),
            )
            self._protocol.acknowledge_store_mutations(store, committed.handle)
            if publish_weight_path is not None:
                self._protocol.publish_weight_handle(
                    str(publish_weight_path), committed.handle
                )
            self._protocol.acknowledge_checkpoint(identity)
            if publish_weight_path is None:
                self._protocol.release_reference(committed.handle)
            return committed
        except BaseException as primary:
            actions = []
            if committed is None:
                actions.append(
                    (
                        "checkpoint abort",
                        lambda: self._protocol.abort_checkpoint(identity),
                    )
                )
            run_cleanup_preserving_primary(primary, actions, "training checkpoint save")
            raise

    def load_training_checkpoint(
        self,
        store: Any,
        storage_locator: CheckpointHandle,
        restore: RestorePlan,
        *,
        load_optimizer_state: bool = True,
    ) -> None:
        self._load_checkpoint(
            store,
            storage_locator,
            restore,
            load_optimizer_state=load_optimizer_state,
        )

    def load_inference_checkpoint(
        self,
        store: Any,
        storage_locator: CheckpointHandle,
        restore: RestorePlan,
    ) -> None:
        self._load_checkpoint(
            store,
            storage_locator,
            restore,
            load_optimizer_state=False,
        )

    def _load_checkpoint(
        self,
        store: Any,
        storage_locator: CheckpointHandle,
        restore: RestorePlan,
        *,
        load_optimizer_state: bool,
    ) -> None:
        if not isinstance(restore, RestorePlan):
            raise TypeError(
                f"restore must be a RestorePlan instance; got {type(restore).__name__}"
            )
        self._protocol.load_component_checkpoint(
            store,
            storage_locator,
            restore.encode(),
            load_optimizer_state=load_optimizer_state,
        )
