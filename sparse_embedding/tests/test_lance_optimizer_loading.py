import asyncio

import numpy as np
import pytest

import sparse_embedding as se
from sparse_embedding.checkpoint import (
    CheckpointParticipant,
    ElasticRestore,
    PhysicalRestore,
    PhysicalRestoreStrategy,
    SparseCheckpoint,
    SparsePartitionAssignmentSpec,
    encode_restore_options,
)
from sparse_embedding.checkpoint.lance_backend import LocalLanceStorage
from sparse_embedding.checkpoint.schema import ENTRIES_DATASET


def _modulo_restore(partition_count: int, partition_id: int) -> ElasticRestore:
    return ElasticRestore(
        SparsePartitionAssignmentSpec.modulo(),
        partition_count,
        partition_id,
        "uint64",
    )


def _config(*, optimizer_dtype="float32", enable_optimizer=True):
    return se.EmbeddingConfig(
        dims=[4],
        thread_budget=1,
        value_dtype="float32",
        optimizer_dtype=optimizer_dtype,
        key_dtype="uint64",
        enable_optimizer=enable_optimizer,
    )


def _store_with_optimizer_state():
    store = se.EmbeddingStore(
        _config(optimizer_dtype="float16"), optimizer_config=se.Adam(lr=0.01)
    )
    keys = np.asarray([7], dtype=np.uint64)
    store.lookup_or_insert(keys, dim=4)
    gradients = np.ones((1, 4), dtype=np.float32)
    store.accumulate_gradients(keys, gradients)
    store.optimizer_step()
    return store, keys, gradients


def _lookup(store, keys):
    values, _ = store.lookup(np.asarray(keys, dtype=np.uint64), dim=4)
    return np.from_dlpack(values).reshape(-1, 4)


def test_default_and_disabled_optimizer_metadata_and_operations():
    enabled = se.EmbeddingStore(_config())
    disabled = se.EmbeddingStore(_config(enable_optimizer=False))
    assert enabled.optimizer_enabled() is True
    assert disabled.optimizer_enabled() is False

    keys = np.asarray([1], dtype=np.uint64)
    gradients = np.ones((1, 4), dtype=np.float32)
    disabled.lookup_or_insert(keys, dim=4)
    assert disabled.export_page_images()[0]["optimizer_states"] == []
    with pytest.raises(RuntimeError, match="optimizer enabled"):
        disabled.accumulate_gradients(keys, gradients)
    with pytest.raises(RuntimeError, match="optimizer enabled"):
        disabled.optimizer_step()
    with pytest.raises(RuntimeError, match="optimizer enabled"):
        asyncio.run(disabled.async_accumulate_gradients(keys, gradients))
    with pytest.raises(RuntimeError, match="optimizer enabled"):
        asyncio.run(disabled.async_optimizer_step())
    with pytest.raises(ValueError, match="optimizer_config cannot be provided"):
        se.EmbeddingStore(_config(enable_optimizer=False), optimizer_config=se.Adam())


def test_entries_snapshot_skips_optimizer_state_for_disabled_target(tmp_path):
    source, keys, _ = _store_with_optimizer_state()
    storage = LocalLanceStorage(tmp_path / ENTRIES_DATASET)
    locator = SparseCheckpoint(storage).save_training_checkpoint(
        source, "optimizer-loading"
    )
    target = se.EmbeddingStore(
        _config(optimizer_dtype="float32", enable_optimizer=False)
    )

    storage.load_component_checkpoint(
        target,
        locator.handle,
        encode_restore_options(_modulo_restore(1, 0)),
        load_optimizer_state=False,
    )

    np.testing.assert_allclose(_lookup(target, keys), _lookup(source, keys))
    with pytest.raises(RuntimeError, match="optimizer enabled"):
        target.optimizer_step()


def test_page_image_snapshot_skips_optimizer_state_for_disabled_target(tmp_path):
    source, keys, _ = _store_with_optimizer_state()
    storage = LocalLanceStorage(tmp_path / "page-images")
    locator = SparseCheckpoint(storage).save_training_checkpoint(
        source, "optimizer-loading", include_page_images=True
    )
    target = se.EmbeddingStore(
        _config(optimizer_dtype="float32", enable_optimizer=False)
    )

    storage.load_component_checkpoint(
        target,
        locator.handle,
        encode_restore_options(
            PhysicalRestore(
                CheckpointParticipant(0, 1),
                strategy=PhysicalRestoreStrategy.FAST_PAGE_CACHE,
            )
        ),
        load_optimizer_state=False,
    )

    np.testing.assert_allclose(_lookup(target, keys), _lookup(source, keys))


def test_training_restore_default_loads_optimizer_state(tmp_path):
    source, keys, _ = _store_with_optimizer_state()
    gradients = np.asarray([[2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    storage = LocalLanceStorage(tmp_path / "training")
    locator = SparseCheckpoint(storage).save_training_checkpoint(source, "training")
    target = se.EmbeddingStore(
        _config(optimizer_dtype="float16"), optimizer_config=se.Adam(lr=0.01)
    )

    SparseCheckpoint(storage).load_training_checkpoint(
        target, locator.handle, encode_restore_options(_modulo_restore(1, 0))
    )
    source.accumulate_gradients(keys, gradients)
    source.optimizer_step()
    target.accumulate_gradients(keys, gradients)
    target.optimizer_step()

    np.testing.assert_allclose(_lookup(target, keys), _lookup(source, keys), atol=1e-3)


def test_training_restore_explicit_weight_only_does_not_load_optimizer_state(tmp_path):
    source, keys, _ = _store_with_optimizer_state()
    gradients = np.asarray([[2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    storage = LocalLanceStorage(tmp_path / "weight-only")
    locator = SparseCheckpoint(storage).save_training_checkpoint(source, "weight-only")
    target = se.EmbeddingStore(
        _config(optimizer_dtype="float16"), optimizer_config=se.Adam(lr=0.01)
    )

    SparseCheckpoint(storage).load_training_checkpoint(
        target,
        locator.handle,
        encode_restore_options(_modulo_restore(1, 0)),
        load_optimizer_state=False,
    )
    source.accumulate_gradients(keys, gradients)
    source.optimizer_step()
    target.accumulate_gradients(keys, gradients)
    target.optimizer_step()

    assert not np.allclose(_lookup(target, keys), _lookup(source, keys), atol=1e-3)


def test_training_weight_only_restore_accepts_disabled_or_different_optimizer(tmp_path):
    source, keys, _ = _store_with_optimizer_state()
    storage = LocalLanceStorage(tmp_path / "weight-only-targets")
    locator = SparseCheckpoint(storage).save_training_checkpoint(
        source, "weight-only-targets"
    )

    for target in (
        se.EmbeddingStore(_config(enable_optimizer=False)),
        se.EmbeddingStore(
            _config(optimizer_dtype="float32"), optimizer_config=se.Adam(lr=0.01)
        ),
    ):
        SparseCheckpoint(storage).load_training_checkpoint(
            target,
            locator.handle,
            encode_restore_options(_modulo_restore(1, 0)),
            load_optimizer_state=False,
        )
        np.testing.assert_allclose(_lookup(target, keys), _lookup(source, keys))


def test_training_restore_rejects_optimizer_state_for_disabled_target(tmp_path):
    source, _, _ = _store_with_optimizer_state()
    storage = LocalLanceStorage(tmp_path / "strict")
    locator = SparseCheckpoint(storage).save_training_checkpoint(source, "strict")
    target = se.EmbeddingStore(_config(enable_optimizer=False))

    with pytest.raises(ValueError, match="load_optimizer_state requires"):
        SparseCheckpoint(storage).load_training_checkpoint(
            target, locator.handle, encode_restore_options(_modulo_restore(1, 0))
        )
