import numpy as np
import pytest

import sparse_embedding as se
from sparse_embedding.checkpoint import (
    CheckpointParticipant,
    LocalLanceStorage,
    PartitionSpec,
    RankRestore,
    ReshardRestore,
    SparseCheckpoint,
    encode_restore_options,
    lance_backend,
)


def _config(*, optimizer: bool = True) -> se.EmbeddingConfig:
    return se.EmbeddingConfig(
        dims=[4],
        thread_budget=1,
        value_dtype="float32",
        key_dtype="uint64",
        enable_optimizer=optimizer,
    )


def _store(*, optimizer: bool, key: int, value: float) -> se.EmbeddingStore:
    store = se.EmbeddingStore(_config(optimizer=optimizer))
    store.assign(
        np.asarray([key], dtype=np.uint64),
        np.full((1, 4), value, dtype=np.float32),
    )
    return store


def _elastic_options() -> bytes:
    return encode_restore_options(
        ReshardRestore(
            PartitionSpec.modulo(),
            1,
            0,
            "uint64",
        )
    )


@pytest.mark.parametrize(
    "options",
    [
        encode_restore_options(RankRestore(CheckpointParticipant(0, 1))),
        _elastic_options(),
    ],
)
def test_training_restore_rejects_nonempty_target_before_native_scan(
    tmp_path, monkeypatch, options
):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    committed = SparseCheckpoint(storage).save_training_checkpoint(
        _store(optimizer=True, key=1, value=2), "source"
    )
    target = _store(optimizer=True, key=9, value=3)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("native restore must not run for a non-empty target")

    monkeypatch.setattr(
        lance_backend._checkpoint_native,
        "restore_native_partition_entries",
        fail_if_called,
    )
    with pytest.raises(RuntimeError, match="restore requires an empty target store"):
        storage.load_component_checkpoint(target, committed.handle, options)
    assert target.is_operational() is True
    assert target.total_size() == 1


def test_inference_restore_rejects_nonempty_target_before_native_scan(
    tmp_path, monkeypatch
):
    root = tmp_path / "checkpoint"
    asset = tmp_path / "asset"
    asset.mkdir()
    storage = LocalLanceStorage.from_checkpoint_root(root)
    SparseCheckpoint(storage).save_training_checkpoint(
        _store(optimizer=True, key=1, value=2), "source", publish_weight_path=asset
    )
    target = _store(optimizer=False, key=9, value=3)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("native restore must not run for a non-empty target")

    monkeypatch.setattr(
        lance_backend._checkpoint_native,
        "restore_native_partition_entries",
        fail_if_called,
    )
    with pytest.raises(RuntimeError, match="restore requires an empty target store"):
        LocalLanceStorage.from_published_weight(asset).load_inference_weights(
            target, _elastic_options()
        )
    assert target.is_operational() is True
    assert target.total_size() == 1
