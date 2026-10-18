import copy
import inspect
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import MappingProxyType

import lance
import msgpack
import numpy as np
import pyarrow as pa
import pytest

import sparse_embedding as se
from sparse_embedding.checkpoint import (
    IngestionEngine,
    PartitionSpec,
    PartitionTarget,
    RestorePlan,
    SparseCheckpoint,
    encode_restore_options,
    lance_backend,
)
from sparse_embedding.checkpoint.base import (
    CheckpointCommitAuthority,
    CheckpointConflictError,
    CheckpointFragment,
    CheckpointLeaseRequest,
    CheckpointParticipant,
    CheckpointSaveIdentity,
    CheckpointSaveMode,
    InvalidCheckpointFragment,
    StorageBudgetExceeded,
)
from sparse_embedding.checkpoint.lance_backend import (
    LocalLanceStorage,
    _decode_checkpoint_handle,
)
from sparse_embedding.checkpoint.restore_options import _decode_restore_options
from sparse_embedding.checkpoint.schema import (
    ENTRIES_DATASET,
    PAGE_IMAGES_DATASET,
    SNAPSHOT_MANIFEST_DATASET,
)


def config(dims=(4,)):
    return se.EmbeddingConfig(
        dims=list(dims), thread_budget=1, value_dtype="float32", key_dtype="uint64"
    )


def inference_config(dims=(4,)):
    return se.EmbeddingConfig(
        dims=list(dims),
        thread_budget=1,
        value_dtype="float32",
        key_dtype="uint64",
        enable_optimizer=False,
    )


def authority(generation=0, authority_id="authority"):
    return CheckpointCommitAuthority("test", generation, authority_id)


def populated_store(keys, value, *, dims=(4,)):
    store = se.EmbeddingStore(config(dims))
    keys = np.asarray(tuple(keys), dtype=np.uint64)
    for dim in dims:
        if keys.size:
            values = np.full((keys.size, dim), value, dtype=np.float32)
            store.assign(keys, values)
    return store


def modulo_restore(partition_count: int, partition_id: int) -> RestorePlan:
    return RestorePlan.from_partition(
        PartitionSpec.modulo(),
        partition_count,
        partition_id,
        key_dtype="uint64",
    )


def target_restore(
    participant: CheckpointParticipant,
    *,
    key_dtype: str = "uint64",
    engine: IngestionEngine = IngestionEngine.RECONSTRUCT_ENTRIES,
) -> RestorePlan:
    return RestorePlan.from_partition(
        PartitionSpec.modulo(),
        participant.participant_count,
        participant.participant_index,
        key_dtype=key_dtype,
        engine=engine,
    )


def lookup(store, keys, dim=4):
    values, _ = store.lookup(np.asarray(keys, dtype=np.uint64), dim=dim)
    return np.from_dlpack(values).reshape(-1, dim)


def handle(committed):
    return committed.handle


def test_user_facade_roundtrip(tmp_path):
    checkpoint = SparseCheckpoint(LocalLanceStorage(tmp_path / "checkpoint"))
    source = populated_store((1, 2), 3)
    locator = checkpoint.save_training_checkpoint(source, "step-1")
    restored = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        restored,
        handle(locator),
        target_restore(CheckpointParticipant(0, 1)),
    )
    np.testing.assert_allclose(lookup(restored, (1, 2)), 3)


def test_exclusive_recovery_fences_direct_mutation_entries(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    source = populated_store((1,), 1)
    identity = CheckpointSaveIdentity("fenced", "attempt", authority())
    participant = CheckpointParticipant(0, 1)
    storage._ensure_tables()
    with storage.exclusive_recovery():
        with pytest.raises(CheckpointConflictError, match="fenced"):
            storage.write_checkpoint_fragment_bytes(source, identity, participant)
        with pytest.raises(CheckpointConflictError, match="fenced"):
            storage.finalize_checkpoint(identity)
        with pytest.raises(CheckpointConflictError, match="fenced"):
            storage.acknowledge_store_mutations(source, b"invalid")
        with pytest.raises(CheckpointConflictError, match="fenced"):
            storage.load_component_checkpoint(source, b"invalid", b"invalid")

    with pytest.raises(CheckpointConflictError, match="authority"):
        storage.finalize_checkpoint(identity)


def test_incremental_checkpoint_restores_parent_and_delta(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    source = populated_store((1,), 1)
    parent = checkpoint.save_training_checkpoint(source, "incremental")
    source.assign(
        np.asarray([2], dtype=np.uint64), np.full((1, 4), 2, dtype=np.float32)
    )
    child = checkpoint.save_training_checkpoint(
        source,
        "incremental",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
    )
    restored_parent = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        restored_parent,
        parent.handle,
        modulo_restore(1, 0),
    )
    restored_child = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        restored_child,
        child.handle,
        modulo_restore(1, 0),
    )
    assert restored_parent.total_size() == 1
    assert restored_child.total_size() == 2
    np.testing.assert_allclose(lookup(restored_child, (2,)), 2)


def test_incremental_tombstone_is_dimension_scoped(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    source = populated_store((7,), 1, dims=(2, 4))
    parent = checkpoint.save_training_checkpoint(source, "tombstone")
    source.erase_dimension(np.asarray([7], dtype=np.uint64), dimension=2)
    child = checkpoint.save_training_checkpoint(
        source,
        "tombstone",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
    )
    restored = se.EmbeddingStore(config((2, 4)))
    checkpoint.load_training_checkpoint(
        restored,
        child.handle,
        modulo_restore(1, 0),
    )
    assert restored.total_size() == 1
    np.testing.assert_allclose(lookup(restored, (7,), dim=4), 1)


def test_incremental_requires_store_parent_baseline(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    source = populated_store((1,), 1)
    parent = checkpoint.save_training_checkpoint(source, "baseline")
    restarted = se.EmbeddingStore(config())
    with pytest.raises(RuntimeError, match="store baseline"):
        checkpoint.save_training_checkpoint(
            restarted,
            "baseline",
            mode=CheckpointSaveMode.INCREMENTAL,
            parent_handle=parent.handle,
        )


def test_empty_incremental_checkpoint_has_no_entry_payload(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    source = populated_store((1,), 1)
    parent = checkpoint.save_training_checkpoint(source, "empty-delta")
    child = checkpoint.save_training_checkpoint(
        source,
        "empty-delta",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
    )
    manifest = storage.resolve_checkpoint_manifest(child.handle)
    root = storage._manifest_fragment_root_for_participant(manifest, 0)
    descriptor = json.loads(next(iter(manifest["fragment_descriptors"])))
    assert descriptor["entry_count"] == 0
    assert not (tmp_path / "checkpoint" / root / ENTRIES_DATASET).exists()
    assert not (tmp_path / "checkpoint" / root / PAGE_IMAGES_DATASET).exists()
    restored = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        restored, child.handle, target_restore(CheckpointParticipant(0, 1))
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 1)


def test_checkpoint_publication_uses_opaque_handle_and_owner_version(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    committed = SparseCheckpoint(storage).save_training_checkpoint(
        populated_store((1,), 3), "opaque"
    )
    assert isinstance(committed.handle, bytes)
    assert isinstance(committed.version_id, bytes)
    assert committed.version_id == b"opaque:1"
    for name in (
        "encode_checkpoint_locator",
        "decode_checkpoint_locator",
        "checkpoint_locator_version_id",
        "checkpoint_locator_reference_id",
    ):
        assert not hasattr(storage, name)


def test_partition_assignment_spec_is_immutable_and_owner_approved():
    spec = PartitionSpec.modulo()
    assert spec.name == "sparse_embedding.modulo"
    assert dict(spec.parameters) == {}
    with pytest.raises(ValueError, match="unregistered partition algorithm"):
        PartitionSpec("unknown", {})
    with pytest.raises(TypeError, match="unsupported value"):
        PartitionSpec("custom", {"callback": object()})


def test_owner_restore_options_are_canonical_and_opaque():
    encoded = encode_restore_options(modulo_restore(3, 1))
    assert isinstance(encoded, bytes)
    noncanonical = msgpack.packb(
        {
            "restore_kind": "partition",
            "engine": "reconstruct_entries",
            "key_dtype": "uint64",
            "partition_count": 3,
            "partition_id": 1,
            "sparse_assignment_name": "sparse_embedding.modulo",
            "sparse_assignment_parameters": {},
        },
        use_bin_type=True,
    )
    assert noncanonical != encoded


@pytest.mark.parametrize("key_dtype", ["int32", "uint32", "int64", "uint64"])
def test_modulo_restore_options_preserve_key_dtype(key_dtype):
    encoded = encode_restore_options(
        RestorePlan.from_partition(
            PartitionSpec.modulo(),
            3,
            1,
            key_dtype=key_dtype,
        )
    )
    decoded = msgpack.unpackb(encoded, raw=False)
    assert decoded["key_dtype"] == key_dtype
    assert decoded["restore_kind"] == "partition"


def test_restore_options_reject_typed_mode_and_unknown_fields(tmp_path):
    with pytest.raises(TypeError, match="bytes"):
        _decode_restore_options(modulo_restore(1, 0))
    unknown = msgpack.packb(
        {
            "restore_kind": "partition",
            "engine": "reconstruct_entries",
            "key_dtype": "uint64",
            "partition_count": 1,
            "partition_id": 0,
            "sparse_assignment_name": "sparse_embedding.modulo",
            "sparse_assignment_parameters": {},
            "unexpected": True,
        },
        use_bin_type=True,
    )
    with pytest.raises(ValueError, match="canonical"):
        _decode_restore_options(unknown)


def test_opaque_handle_rejects_another_storage_root(tmp_path):
    first = LocalLanceStorage(tmp_path / "first")
    committed = SparseCheckpoint(first).save_training_checkpoint(
        populated_store((1,), 3), "opaque"
    )
    second = LocalLanceStorage(tmp_path / "second")
    with pytest.raises(ValueError, match="another storage root"):
        second.load_component_checkpoint(
            se.EmbeddingStore(config()),
            committed.handle,
            encode_restore_options(target_restore(CheckpointParticipant(0, 1))),
        )


def test_fragment_manifests_restore_exact_checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    checkpoint = SparseCheckpoint(storage)
    first = checkpoint.save_training_checkpoint(populated_store((1, 2), 1), "first")
    second = checkpoint.save_training_checkpoint(populated_store((2, 3), 2), "second")
    manifests = (
        lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET)).to_table().to_pylist()
    )
    assert len(manifests[0]["fragment_ids"]) == 1
    assert len(manifests[1]["fragment_ids"]) == 1
    assert manifests[0]["fragment_ids"] != manifests[1]["fragment_ids"]
    assert all(
        (root / r).exists()
        for r in storage._manifest_fragment_roots(manifests[0]).values()
    )
    assert all(
        (root / r).exists()
        for r in storage._manifest_fragment_roots(manifests[1]).values()
    )
    assert not (root / ENTRIES_DATASET).exists()
    assert not (root / PAGE_IMAGES_DATASET).exists()
    old_store = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        old_store,
        handle(first),
        target_restore(CheckpointParticipant(0, 1)),
    )
    np.testing.assert_allclose(lookup(old_store, (1, 2)), 1)
    new_store = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        new_store,
        handle(second),
        target_restore(CheckpointParticipant(0, 1)),
    )
    np.testing.assert_allclose(lookup(new_store, (2, 3)), 2)
    elastic_store = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        elastic_store,
        handle(second),
        modulo_restore(1, 0),
    )
    np.testing.assert_allclose(lookup(elastic_store, (2, 3)), 2)


def test_restore_filters_target_partition(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    source = populated_store(range(96), 5)
    committed = SparseCheckpoint(storage).save_training_checkpoint(source, "bounded")

    restored = se.EmbeddingStore(config())
    SparseCheckpoint(storage).load_training_checkpoint(
        restored,
        handle(committed),
        modulo_restore(3, 1),
    )
    expected = np.asarray([key for key in range(96) if key % 3 == 1], dtype=np.uint64)
    assert restored.total_size() == expected.size
    np.testing.assert_allclose(lookup(restored, expected), 5)


def test_inference_weights_restore_from_physical_reference(tmp_path):
    root = tmp_path / "checkpoint"
    asset = tmp_path / "asset"
    asset.mkdir()
    storage = LocalLanceStorage.from_checkpoint_root(root)
    checkpoint = SparseCheckpoint(storage)
    checkpoint.save_training_checkpoint(
        populated_store((1, 2), 7), "training-name", publish_weight_path=asset
    )
    restored = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_published_weight(asset).load_inference_weights(
        restored,
        encode_restore_options(modulo_restore(1, 0)),
    )
    np.testing.assert_allclose(lookup(restored, (1, 2)), 7)
    physical_restored = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_published_weight(asset).load_inference_weights(
        physical_restored,
        encode_restore_options(target_restore(CheckpointParticipant(0, 1))),
    )
    np.testing.assert_allclose(lookup(physical_restored, (1, 2)), 7)
    with pytest.raises(ValueError, match="optimizer enabled"):
        LocalLanceStorage.from_published_weight(asset).load_inference_weights(
            se.EmbeddingStore(config()),
            encode_restore_options(modulo_restore(1, 0)),
        )


def test_inference_weights_do_not_require_training_optimizer_spec(tmp_path):
    root = tmp_path / "checkpoint"
    asset = tmp_path / "asset"
    asset.mkdir()
    storage = LocalLanceStorage.from_checkpoint_root(root)
    source_config = se.EmbeddingConfig(
        dims=[4],
        thread_budget=1,
        value_dtype="float32",
        optimizer_dtype="float16",
        key_dtype="uint64",
    )
    source = se.EmbeddingStore(source_config)
    keys = np.asarray([3], dtype=np.uint64)
    source.assign(keys, np.ones((1, 4), dtype=np.float32))
    locator = SparseCheckpoint(storage).save_training_checkpoint(
        source, "training-config", publish_weight_path=asset
    )
    inference_target = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_published_weight(asset).load_inference_weights(
        inference_target,
        encode_restore_options(modulo_restore(1, 0)),
    )
    np.testing.assert_allclose(lookup(inference_target, (3,)), 1)
    with pytest.raises(ValueError, match="restore target does not match"):
        storage.load_component_checkpoint(
            se.EmbeddingStore(config()),
            handle(locator),
            encode_restore_options(modulo_restore(1, 0)),
        )


def test_inference_weights_restore_into_optimizer_disabled_store(tmp_path):
    root = tmp_path / "checkpoint"
    source = se.EmbeddingStore(
        se.EmbeddingConfig(
            dims=[4],
            thread_budget=1,
            value_dtype="float32",
            optimizer_dtype="float16",
            key_dtype="uint64",
        )
    )
    keys = np.asarray([3], dtype=np.uint64)
    source.assign(keys, np.ones((1, 4), dtype=np.float32))
    storage = LocalLanceStorage.from_checkpoint_root(root)
    asset = tmp_path / "asset"
    asset.mkdir()
    checkpoint = SparseCheckpoint(storage)
    locator = checkpoint.save_training_checkpoint(
        source, "inference-only", publish_weight_path=asset
    )

    target = se.EmbeddingStore(
        se.EmbeddingConfig(
            dims=[4], thread_budget=1, key_dtype="uint64", enable_optimizer=False
        )
    )
    assert target.optimizer_enabled() is False
    LocalLanceStorage.from_published_weight(asset).load_inference_weights(
        target,
        encode_restore_options(modulo_restore(1, 0)),
    )
    np.testing.assert_allclose(lookup(target, (3,)), 1)
    with pytest.raises(ValueError, match="optimizer enabled"):
        storage.load_component_checkpoint(
            target,
            handle(locator),
            encode_restore_options(modulo_restore(1, 0)),
        )


def test_training_restore_can_skip_optimizer_state(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    source = se.EmbeddingStore(config(), optimizer_config=se.Adam(lr=0.01))
    keys = np.asarray([5], dtype=np.uint64)
    source.assign(keys, np.ones((1, 4), dtype=np.float32))
    locator = SparseCheckpoint(storage).save_training_checkpoint(source, "weights-only")
    target = se.EmbeddingStore(inference_config())
    SparseCheckpoint(storage).load_training_checkpoint(
        target,
        handle(locator),
        modulo_restore(1, 0),
        load_optimizer_state=False,
    )
    np.testing.assert_allclose(lookup(target, (5,)), 1)


def test_page_image_restore_can_skip_optimizer_state(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    source = se.EmbeddingStore(config(), optimizer_config=se.Adam(lr=0.01))
    keys = np.asarray([6], dtype=np.uint64)
    source.assign(keys, np.ones((1, 4), dtype=np.float32))
    locator = SparseCheckpoint(storage).save_training_checkpoint(
        source, "page-weights-only", include_page_images=True
    )
    target = se.EmbeddingStore(inference_config())
    SparseCheckpoint(storage).load_training_checkpoint(
        target,
        handle(locator),
        target_restore(
            CheckpointParticipant(0, 1),
            engine=IngestionEngine.FAST_PAGE_CACHE,
        ),
        load_optimizer_state=False,
    )
    np.testing.assert_allclose(lookup(target, (6,)), 1)


def test_four_participants_restore_to_seven_partitions(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("elastic", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragments = []
    for participant_index in range(4):
        keys = np.arange(participant_index, 100, 4, dtype=np.uint64)
        store = se.EmbeddingStore(config())
        store.assign(keys, np.repeat(keys[:, None].astype(np.float32), 4, axis=1))
        fragments.append(
            storage._write_checkpoint_fragment(
                store,
                identity,
                CheckpointParticipant(participant_index, 4),
            )
        )
    locator = storage._commit_checkpoint(identity, fragments)

    seen = set()
    all_keys = np.arange(100, dtype=np.uint64)
    for partition_id in range(7):
        restored = se.EmbeddingStore(config())
        storage.load_component_checkpoint(
            restored,
            handle(locator),
            encode_restore_options(modulo_restore(7, partition_id)),
        )
        expected = all_keys[all_keys % 7 == partition_id]
        assert restored.total_size() == expected.size
        assert seen.isdisjoint(expected.tolist())
        seen.update(expected.tolist())
        if expected.size:
            np.testing.assert_allclose(
                lookup(restored, expected),
                np.repeat(expected[:, None].astype(np.float32), 4, axis=1),
            )
    assert seen == set(range(100))


def test_physical_restore_preserves_source_participant_after_compaction(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("physical-compaction", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragments = []
    for participant_index in range(2):
        keys = np.asarray([participant_index, participant_index + 2], dtype=np.uint64)
        source = se.EmbeddingStore(config())
        source.assign(keys, np.repeat(keys[:, None].astype(np.float32), 4, axis=1))
        fragments.append(
            storage._write_checkpoint_fragment(
                source,
                identity,
                CheckpointParticipant(participant_index, 2),
            )
        )
    locator = storage._commit_checkpoint(identity, fragments)
    storage.acknowledge_checkpoint(identity)
    compacted = storage.compact_checkpoint(handle(locator))

    for participant_index in range(2):
        restored = se.EmbeddingStore(config())
        storage.load_component_checkpoint(
            restored,
            compacted,
            encode_restore_options(
                target_restore(CheckpointParticipant(participant_index, 2))
            ),
        )
        expected = np.asarray(
            [participant_index, participant_index + 2], dtype=np.uint64
        )
        assert restored.total_size() == expected.size
        np.testing.assert_allclose(
            lookup(restored, expected),
            np.repeat(expected[:, None].astype(np.float32), 4, axis=1),
        )


def test_restore_merges_multiple_participants(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("duplicate", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragments = [
        storage._write_checkpoint_fragment(
            populated_store((index,), value),
            identity,
            CheckpointParticipant(index, 2),
        )
        for index, value in enumerate((1, 2))
    ]
    locator = storage._commit_checkpoint(identity, fragments)

    restored = se.EmbeddingStore(config())
    storage.load_component_checkpoint(
        restored,
        handle(locator),
        encode_restore_options(modulo_restore(1, 0)),
    )
    assert restored.total_size() == 2
    np.testing.assert_allclose(lookup(restored, (0,)), 1)
    np.testing.assert_allclose(lookup(restored, (1,)), 2)


def test_elastic_restore_rejects_duplicate_active_keys(tmp_path, monkeypatch):
    temporary_root = tmp_path / "native-scratch"
    temporary_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary_root))
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    temporary_runs_before = set(temporary_root.glob("sparse_embedding_checkpoint-*"))
    identity = CheckpointSaveIdentity("duplicate", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragments = [
        storage._write_checkpoint_fragment(
            populated_store((7,), value),
            identity,
            CheckpointParticipant(index, 2),
        )
        for index, value in enumerate((1, 2))
    ]
    locator = storage._commit_checkpoint(identity, fragments)
    restored = se.EmbeddingStore(config())
    with pytest.raises(InvalidCheckpointFragment, match="duplicate active"):
        storage.load_component_checkpoint(
            restored,
            handle(locator),
            encode_restore_options(modulo_restore(1, 0)),
        )
    assert restored.is_operational() is True
    assert set(temporary_root.glob("sparse_embedding_checkpoint-*")) == (
        temporary_runs_before
    )


def test_elastic_restore_rejects_duplicate_keys_across_external_runs(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("duplicate-spill", "attempt", authority())
    storage.prepare_checkpoint(identity)
    first = np.arange(65536, dtype=np.uint64)
    second = np.arange(65535, 131072, dtype=np.uint64)
    fragments = [
        storage._write_checkpoint_fragment(
            populated_store(keys, index + 1),
            identity,
            CheckpointParticipant(index, 2),
        )
        for index, keys in enumerate((first, second))
    ]
    locator = storage._commit_checkpoint(identity, fragments)
    restored = se.EmbeddingStore(config())
    with pytest.raises(InvalidCheckpointFragment, match="duplicate active"):
        storage.load_component_checkpoint(
            restored,
            handle(locator),
            encode_restore_options(modulo_restore(1, 0)),
        )
    assert restored.is_operational() is True


def test_legacy_restore_fields_reject_before_store_mutation(tmp_path):
    checkpoint = SparseCheckpoint(LocalLanceStorage(tmp_path / "checkpoint"))
    locator = checkpoint.save_training_checkpoint(
        populated_store((1, 2), 1), "partition"
    )
    restored = se.EmbeddingStore(config())

    unknown_router = msgpack.packb(
        dict(
            sorted(
                {
                    "restore_kind": "partition",
                    "engine": "reconstruct_entries",
                    "key_dtype": "uint64",
                    "partition_count": 2,
                    "partition_id": 0,
                    "sparse_assignment_name": "sparse_embedding.modulo",
                    "sparse_assignment_parameters": {},
                    "router_id": "unknown",
                }.items()
            )
        ),
        use_bin_type=True,
    )
    with pytest.raises(ValueError, match="unknown fields"):
        checkpoint.integration_protocol.load_component_checkpoint(
            restored, handle(locator), unknown_router
        )
    assert restored.total_size() == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dimension_scope", 0, "unknown fields"),
        ("static_config", {"salt": 1}, "unknown fields"),
    ),
)
def test_modulo_restore_options_validate_descriptor_fields(
    tmp_path, field, value, message
):
    checkpoint = SparseCheckpoint(LocalLanceStorage(tmp_path / "checkpoint"))
    locator = checkpoint.save_training_checkpoint(
        populated_store((1, 2), 1), "descriptor-validation"
    )
    payload = {
        "restore_kind": "partition",
        "engine": "reconstruct_entries",
        "key_dtype": "uint64",
        "partition_count": 1,
        "partition_id": 0,
        "sparse_assignment_name": "sparse_embedding.modulo",
        "sparse_assignment_parameters": {},
    }
    payload[field] = value
    encoded = msgpack.packb(dict(sorted(payload.items())), use_bin_type=True)
    with pytest.raises(ValueError, match=message):
        checkpoint.integration_protocol.load_component_checkpoint(
            se.EmbeddingStore(config()), handle(locator), encoded
        )


def test_checkpoint_fragment_contains_no_physical_location(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("opaque", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragment = storage._write_checkpoint_fragment(
        populated_store((1,), 1), identity, CheckpointParticipant(0, 1)
    )
    assert isinstance(fragment, CheckpointFragment)
    assert str(tmp_path) not in repr(fragment)
    assert not hasattr(fragment, "uri")


def test_repeated_checkpoints_keep_manifest_fragments_independent(tmp_path):
    root = tmp_path / "checkpoint"
    checkpoint = SparseCheckpoint(LocalLanceStorage(root))
    for version in range(30):
        checkpoint.save_training_checkpoint(
            populated_store(range(64), version), f"v{version}"
        )
    manifests = (
        lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET)).to_table().to_pylist()
    )
    assert len(manifests) == 30
    assert all(len(row["fragment_ids"]) == 1 for row in manifests)
    assert not (root / ENTRIES_DATASET).exists()
    assert not (root / PAGE_IMAGES_DATASET).exists()


def test_reclaim_removes_released_fragment_manifests(tmp_path):
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    checkpoint = SparseCheckpoint(storage)
    for version in range(20):
        checkpoint.save_training_checkpoint(
            populated_store(range(32), version), "series"
        )
    with storage.exclusive_recovery():
        storage.reclaim(older_than=timedelta(0))
    manifests = (
        lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET)).to_table().to_pylist()
    )
    assert len(manifests) == 1
    assert manifests[0]["checkpoint_name"] == "series"


def test_stale_commit_authority_is_rejected(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    old = CheckpointSaveIdentity("name", "old", authority(1, "authority-old"))
    storage.prepare_checkpoint(old)
    old_fragment = storage._write_checkpoint_fragment(
        populated_store((1,), 1), old, CheckpointParticipant(0, 1)
    )
    current = CheckpointSaveIdentity(
        "name", "current", authority(2, "authority-current")
    )
    storage.prepare_checkpoint(current)
    current_fragment = storage._write_checkpoint_fragment(
        populated_store((1,), 2), current, CheckpointParticipant(0, 1)
    )
    delayed = CheckpointSaveIdentity(
        "delayed", "delayed", authority(1, "authority-delayed")
    )
    with pytest.raises(CheckpointConflictError, match="stale"):
        storage.prepare_checkpoint(delayed)
    with pytest.raises(CheckpointConflictError, match="stale"):
        storage._commit_checkpoint(old, (old_fragment,))
    storage._commit_checkpoint(current, (current_fragment,))
    storage.abort_checkpoint(old)


def test_commit_authority_reconciles_partial_generation_takeover(tmp_path):
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    old = CheckpointSaveIdentity("name", "old", authority(1, "old"))
    current = CheckpointSaveIdentity("name", "current", authority(2, "current"))
    storage.prepare_checkpoint(old)
    manifest = lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET))
    manifest.tags.create(storage._fence_tag(current), manifest.version)

    storage.prepare_checkpoint(current)

    tags = manifest.tags.list()
    assert storage._authority_tag(current) in tags
    assert storage._authority_tag(old) not in tags


def test_retention_lease_pins_exact_version(tmp_path):
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    checkpoint = SparseCheckpoint(storage)
    old = checkpoint.save_training_checkpoint(populated_store((1,), 1), "series")
    lease = storage.acquire_checkpoint_lease(
        handle(old), CheckpointLeaseRequest("trainer", 60, "request-1")
    )
    checkpoint.save_training_checkpoint(populated_store((1,), 2), "series")
    manifest_rows = (
        lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET)).to_table().to_pylist()
    )
    old_fragment_id = next(
        row["fragment_ids"][0]
        for row in manifest_rows
        if old.version_id
        == f"{row['checkpoint_name']}:{row['checkpoint_version']}".encode()
    )
    with storage.exclusive_recovery():
        storage.reclaim(older_than=timedelta(0))
    assert old_fragment_id in {
        item
        for row in lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET))
        .to_table()
        .to_pylist()
        for item in row["fragment_ids"]
    }
    storage.release_checkpoint_lease(lease)
    with storage.exclusive_recovery():
        storage.reclaim(older_than=timedelta(0))
    assert old_fragment_id not in {
        item
        for row in lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET))
        .to_table()
        .to_pylist()
        for item in row["fragment_ids"]
    }


def test_lease_restart_uses_only_owner_lease_capability(tmp_path):
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    committed = SparseCheckpoint(storage).save_training_checkpoint(
        populated_store((1,), 1), "leased"
    )
    lease = storage.acquire_checkpoint_lease(
        committed.handle, CheckpointLeaseRequest("reader", 60, "restart")
    )
    restarted = LocalLanceStorage(root)
    renewed = restarted.renew_checkpoint_lease(lease, 60)
    assert renewed.generation == 1
    restarted.release_checkpoint_lease(renewed)
    assert restarted.active_checkpoint_leases() == ()


def test_reclaim_reservation_removes_only_released_reference(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")

    def save(name, value):
        identity = CheckpointSaveIdentity(name, name, authority())
        storage.prepare_checkpoint(identity)
        fragment = storage._write_checkpoint_fragment(
            populated_store((1,), value),
            identity,
            CheckpointParticipant(0, 1),
        )
        return storage._commit_checkpoint(identity, (fragment,))

    old = save("old", 1)
    current = save("current", 2)
    reservation = storage.reserve_cleanup("reservation", (handle(old),), 1)
    storage.commit_cleanup(reservation)
    storage.cleanup_checkpoint(reservation, handle(old))
    storage.cleanup_checkpoint(reservation, handle(old))

    restored = se.EmbeddingStore(config())
    storage.load_component_checkpoint(
        restored,
        handle(current),
        encode_restore_options(target_restore(CheckpointParticipant(0, 1))),
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 2)
    with pytest.raises(LookupError):
        storage.load_component_checkpoint(
            se.EmbeddingStore(config()),
            handle(old),
            encode_restore_options(target_restore(CheckpointParticipant(0, 1))),
        )


def test_reclaim_requires_published_reservation(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("checkpoint", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragment = storage._write_checkpoint_fragment(
        populated_store((1,), 1),
        identity,
        CheckpointParticipant(0, 1),
    )
    locator = storage._commit_checkpoint(identity, (fragment,))
    reservation = storage.reserve_cleanup("reservation", (locator.handle,), 1)

    with pytest.raises(CheckpointConflictError, match="not pending"):
        storage.cleanup_checkpoint(reservation, locator.handle)


def test_cleanup_reservation_owner_capability_is_validated(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("checkpoint", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragment = storage._write_checkpoint_fragment(
        populated_store((1,), 1), identity, CheckpointParticipant(0, 1)
    )
    committed = storage._commit_checkpoint(identity, (fragment,))
    reservation = storage.reserve_cleanup("reservation", (committed.handle,), 1)
    invalid = type(reservation)(
        reservation.reservation_id, b"invalid", reservation.handles
    )
    with pytest.raises(CheckpointConflictError, match="owner handle"):
        storage.commit_cleanup(invalid)


def test_active_lease_blocks_reclaim_reservation(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    locator = checkpoint.save_training_checkpoint(populated_store((1,), 1), "leased")
    lease = storage.acquire_checkpoint_lease(
        locator.handle, CheckpointLeaseRequest("reader", 60, "request")
    )
    assert (
        storage.acquire_checkpoint_lease(
            locator.handle, CheckpointLeaseRequest("reader", 60, "request")
        )
        == lease
    )
    with pytest.raises(CheckpointConflictError, match="unleased"):
        storage.reserve_cleanup("reservation", (locator.handle,), 1)
    renewed = storage.renew_checkpoint_lease(lease, 60)
    assert renewed.generation == lease.generation + 1
    with pytest.raises(CheckpointConflictError, match="generation CAS"):
        storage.renew_checkpoint_lease(lease, 60)
    storage.release_checkpoint_lease(renewed)


def test_committed_attempt_is_idempotent_and_not_abortable_after_storage_restart(
    tmp_path,
):
    root = tmp_path / "checkpoint"
    identity = CheckpointSaveIdentity("snapshot", "attempt", authority())
    first_storage = LocalLanceStorage(root)
    first_storage.prepare_checkpoint(identity)
    fragment = first_storage._write_checkpoint_fragment(
        populated_store((1, 2), 4),
        identity,
        CheckpointParticipant(0, 1),
    )
    first_storage.finalize_checkpoint(identity)
    locator = first_storage._commit_checkpoint(
        identity,
        (fragment,),
    )

    restarted = LocalLanceStorage(root)
    assert (
        restarted._commit_checkpoint(
            identity,
            (),
        )
        == locator
    )
    restarted.abort_checkpoint(identity)
    restarted.load_component_checkpoint(
        se.EmbeddingStore(config()),
        locator.handle,
        encode_restore_options(target_restore(CheckpointParticipant(0, 1))),
    )

    checkpoint = SparseCheckpoint(LocalLanceStorage(root))
    replacement = checkpoint.save_training_checkpoint(
        populated_store((3, 4), 8), "replacement"
    )
    assert replacement.version_id == b"replacement:2"


def test_concurrent_commits_receive_unique_publication_versions(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    prepared = []
    for index in range(2):
        identity = CheckpointSaveIdentity(
            f"checkpoint-{index}",
            f"attempt-{index}",
            authority(0, f"authority-{index}"),
        )
        storage.prepare_checkpoint(identity)
        fragment = storage._write_checkpoint_fragment(
            populated_store((index + 1,), index + 1),
            identity,
            CheckpointParticipant(0, 1),
        )
        prepared.append((identity, fragment))
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(storage._commit_checkpoint, identity, (fragment,))
            for identity, fragment in prepared
        ]
        locators = [future.result() for future in futures]
    assert {item.version_id.rsplit(b":", 1)[1] for item in locators} == {b"1", b"2"}


def test_manifest_records_table_specification_and_creation_time(tmp_path):
    root = tmp_path / "checkpoint"
    checkpoint = SparseCheckpoint(LocalLanceStorage(root))
    checkpoint.save_training_checkpoint(
        populated_store((1,), 1, dims=(4, 8)), "specification"
    )
    manifest = (
        lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET)).to_table().to_pylist()
    )
    assert len(manifest) == 1
    assert manifest[0]["dimensions"] == [4, 8]
    assert manifest[0]["embedding_dtype"] == "float32"
    assert manifest[0]["optimizer_dtype"] == "float32"
    assert manifest[0]["created_at"] is not None


def test_restore_rejects_incompatible_target_specification_before_mutation(tmp_path):
    checkpoint = SparseCheckpoint(LocalLanceStorage(tmp_path / "checkpoint"))
    locator = checkpoint.save_training_checkpoint(
        populated_store((1,), 1, dims=(4,)), "source"
    )
    target = se.EmbeddingStore(config((8,)))
    with pytest.raises(ValueError, match="target does not match"):
        checkpoint.load_training_checkpoint(
            target,
            handle(locator),
            target_restore(CheckpointParticipant(0, 1)),
        )
    assert target.is_operational() is True


def test_commit_rejects_mixed_participant_specifications(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("schema", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragments = (
        storage._write_checkpoint_fragment(
            populated_store((1,), 1, dims=(4,)),
            identity,
            CheckpointParticipant(0, 2),
        ),
        storage._write_checkpoint_fragment(
            populated_store((2,), 2, dims=(8,)),
            identity,
            CheckpointParticipant(1, 2),
        ),
    )
    with pytest.raises(InvalidCheckpointFragment, match="specification"):
        storage._commit_checkpoint(identity, fragments)


def test_commit_rejects_post_merge_storage_budget(tmp_path):
    root = tmp_path / "checkpoint"
    source = populated_store((1,), 1)
    SparseCheckpoint(LocalLanceStorage(root)).save_training_checkpoint(
        source, "baseline"
    )
    current_size = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    storage = LocalLanceStorage(root, max_storage_bytes=current_size + 1)
    checkpoint = SparseCheckpoint(storage)
    with pytest.raises(StorageBudgetExceeded):
        checkpoint.save_training_checkpoint(source, "over-budget")
    manifests = (
        lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET)).to_table().to_pylist()
    )
    assert [row["checkpoint_name"] for row in manifests] == ["baseline"]
    assert not (root / ENTRIES_DATASET).exists()
    assert not (root / PAGE_IMAGES_DATASET).exists()


def test_compact_checkpoint_publishes_new_full_version(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    source = checkpoint.save_training_checkpoint(populated_store((1,), 1), "compact-me")
    compacted_handle = storage.compact_checkpoint(handle(source))
    source_manifest = storage._resolve_manifest(
        _decode_checkpoint_handle(handle(source))[1]
    )
    compacted_manifest = storage._resolve_manifest(
        _decode_checkpoint_handle(compacted_handle)[1]
    )
    assert source_manifest["compacted_version"] == 0
    assert compacted_manifest["checkpoint_version"] == 2
    assert compacted_manifest["checkpoint_mode"] == "full"


def test_compact_checkpoint_can_materialize_page_images(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    source = checkpoint.save_training_checkpoint(
        populated_store((1, 2), 3), "compact-pages"
    )
    compacted = storage.compact_checkpoint(handle(source), include_page_images=True)
    manifest = storage._resolve_manifest(_decode_checkpoint_handle(compacted)[1])
    roots = storage._manifest_fragment_roots(manifest).values()
    assert manifest["checkpoint_mode"] == "full"
    assert manifest["compacted_version"] == 0
    assert all(
        storage._workspace.exists(f"{root}/{PAGE_IMAGES_DATASET}") for root in roots
    )
    restored = se.EmbeddingStore(config())
    storage.load_component_checkpoint(
        restored,
        compacted,
        encode_restore_options(
            target_restore(
                CheckpointParticipant(0, 1),
                engine=IngestionEngine.FAST_PAGE_CACHE,
            )
        ),
    )
    np.testing.assert_allclose(lookup(restored, (1, 2)), 3)


def test_inference_storage_requires_a_published_weight_path(tmp_path):
    assert (
        "bound_locator" not in inspect.signature(LocalLanceStorage.__init__).parameters
    )
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    store = se.EmbeddingStore(config())
    options = encode_restore_options(
        RestorePlan.from_partition(PartitionSpec.modulo(), 1, 0, key_dtype="uint64")
    )
    with pytest.raises(RuntimeError, match="from_published_weight"):
        storage.load_inference_weights(store, options)


def test_weight_reference_selects_exact_logical_version_and_survives_compaction(
    tmp_path,
):
    root = tmp_path / "checkpoint"
    first_asset = tmp_path / "asset-first"
    second_asset = tmp_path / "asset-second"
    first_asset.mkdir()
    second_asset.mkdir()
    storage = LocalLanceStorage.from_checkpoint_root(root)
    checkpoint = SparseCheckpoint(storage)
    first = checkpoint.save_training_checkpoint(
        populated_store((1,), 1), "series", publish_weight_path=first_asset
    )
    checkpoint.save_training_checkpoint(
        populated_store((1,), 2), "series", publish_weight_path=second_asset
    )

    for asset, expected in ((first_asset, 1), (second_asset, 2)):
        restored = se.EmbeddingStore(inference_config())
        LocalLanceStorage.from_published_weight(asset).load_inference_weights(
            restored,
            encode_restore_options(modulo_restore(1, 0)),
        )
        np.testing.assert_allclose(lookup(restored, (1,)), expected)

    storage.compact_checkpoint(handle(first))
    restored_after_compaction = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_published_weight(first_asset).load_inference_weights(
        restored_after_compaction,
        encode_restore_options(modulo_restore(1, 0)),
    )
    np.testing.assert_allclose(lookup(restored_after_compaction, (1,)), 1)


def test_weight_reference_is_immutable_and_pins_version_until_release(tmp_path):
    root = tmp_path / "checkpoint"
    first_asset = tmp_path / "asset-first"
    second_asset = tmp_path / "asset-second"
    first_asset.mkdir()
    second_asset.mkdir()
    storage = LocalLanceStorage.from_checkpoint_root(root)

    def save(value, asset):
        identity = CheckpointSaveIdentity("series", f"attempt-{value}", authority())
        storage.prepare_checkpoint(identity)
        fragment = storage._write_checkpoint_fragment(
            populated_store((1,), value),
            identity,
            CheckpointParticipant(0, 1),
        )
        locator = storage._commit_checkpoint(identity, (fragment,))
        storage.publish_weight_handle(asset, locator.handle)
        storage.acknowledge_checkpoint(identity)
        return locator

    first = save(1, first_asset)
    second = save(2, second_asset)
    with pytest.raises(CheckpointConflictError, match="another identity"):
        storage.publish_weight_handle(first_asset, second.handle)

    with storage.exclusive_recovery():
        storage.reclaim(older_than=timedelta(0))
    restored = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_published_weight(first_asset).load_inference_weights(
        restored,
        encode_restore_options(modulo_restore(1, 0)),
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 1)

    storage.release_reference(first.handle)
    with pytest.raises(LookupError, match="missing or ambiguous"):
        LocalLanceStorage.from_published_weight(first_asset).load_inference_weights(
            se.EmbeddingStore(inference_config()),
            encode_restore_options(modulo_restore(1, 0)),
        )


def test_weight_reference_restores_across_world_size(tmp_path):
    root = tmp_path / "checkpoint"
    asset = tmp_path / "asset"
    asset.mkdir()
    storage = LocalLanceStorage.from_checkpoint_root(root)
    identity = CheckpointSaveIdentity("elastic-asset", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragments = []
    for participant_index in range(4):
        keys = np.arange(participant_index, 24, 4, dtype=np.uint64)
        source = se.EmbeddingStore(config())
        source.assign(keys, np.repeat(keys[:, None].astype(np.float32), 4, axis=1))
        fragments.append(
            storage._write_checkpoint_fragment(
                source,
                identity,
                CheckpointParticipant(participant_index, 4),
            )
        )
    locator = storage._commit_checkpoint(identity, fragments)
    storage.publish_weight_handle(asset, locator.handle)
    storage.acknowledge_checkpoint(identity)

    for partition_id in range(7):
        restored = se.EmbeddingStore(inference_config())
        LocalLanceStorage.from_published_weight(asset).load_inference_weights(
            restored,
            encode_restore_options(modulo_restore(7, partition_id)),
        )
        expected = np.arange(24, dtype=np.uint64)
        expected = expected[expected % 7 == partition_id]
        assert restored.total_size() == expected.size
        if expected.size:
            np.testing.assert_allclose(
                lookup(restored, expected),
                np.repeat(expected[:, None].astype(np.float32), 4, axis=1),
            )


def test_weight_reference_rejects_missing_malformed_and_ambiguous_assets(tmp_path):
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(FileNotFoundError, match="reference"):
        LocalLanceStorage.from_published_weight(missing)

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / ".sparse_embedding_reference").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="unsupported format"):
        LocalLanceStorage.from_published_weight(malformed)

    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    (ambiguous / ".sparse_embedding_reference").write_bytes(b"invalid")
    (ambiguous / ".sparse_embedding_reference.tmp").write_bytes(b"invalid")
    with pytest.raises(LookupError, match="ambiguous"):
        LocalLanceStorage.from_published_weight(ambiguous)


def test_inference_weight_load_is_read_only_and_does_not_create_leases(tmp_path):
    root = tmp_path / "checkpoint"
    asset = tmp_path / "asset"
    asset.mkdir()
    storage = LocalLanceStorage.from_checkpoint_root(root)
    checkpoint = SparseCheckpoint(storage)
    checkpoint.save_training_checkpoint(
        populated_store((1,), 3), "readonly", publish_weight_path=asset
    )
    before = lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET)).version
    restored = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_published_weight(asset).load_inference_weights(
        restored,
        encode_restore_options(modulo_restore(1, 0)),
    )
    after = lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET)).version
    assert before == after
    assert storage.active_checkpoint_leases() == ()


def test_weight_reference_failure_keeps_committed_record_for_reconcile(tmp_path):
    root = tmp_path / "checkpoint"
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "unexpected").write_text("invalid", encoding="utf-8")
    storage = LocalLanceStorage.from_checkpoint_root(root)
    checkpoint = SparseCheckpoint(storage)
    with pytest.raises(ValueError, match="unknown layout"):
        checkpoint.save_training_checkpoint(
            populated_store((1,), 5), "reconcile", publish_weight_path=asset
        )
    rows = lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET)).to_table().to_pylist()
    assert len(rows) == 1
    assert rows[0]["commit_state"] == "COMMITTED"
    assert rows[0]["reference_id"]


def test_acknowledge_failure_preserves_reference_for_restart_reconcile(
    tmp_path, monkeypatch
):
    root = tmp_path / "checkpoint"
    asset = tmp_path / "asset"
    asset.mkdir()
    storage = LocalLanceStorage.from_checkpoint_root(root)
    identity = CheckpointSaveIdentity("acknowledge", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragment = storage._write_checkpoint_fragment(
        populated_store((1,), 7), identity, CheckpointParticipant(0, 1)
    )
    locator = storage._commit_checkpoint(identity, (fragment,))
    storage.publish_weight_handle(asset, locator.handle)

    def fail_acknowledge(_identity):
        raise OSError("injected acknowledge failure")

    monkeypatch.setattr(storage, "acknowledge_checkpoint", fail_acknowledge)
    with pytest.raises(OSError, match="injected acknowledge failure"):
        storage.acknowledge_checkpoint(identity)

    restarted = LocalLanceStorage.from_checkpoint_root(root)
    restarted.acknowledge_checkpoint(identity)
    restored = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_published_weight(asset).load_inference_weights(
        restored,
        encode_restore_options(modulo_restore(1, 0)),
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 7)


def test_partition_spec_declarative_contracts():
    modulo_spec = PartitionSpec.modulo()
    assert modulo_spec.name == "sparse_embedding.modulo"
    assert modulo_spec.parameters == {}

    hash_spec = PartitionSpec.hash(seed=1024)
    assert hash_spec.name == "sparse_embedding.hash"
    assert hash_spec.parameters == {"seed": "1024"}

    range_spec = PartitionSpec.fast_range()
    assert range_spec.name == "sparse_embedding.fast_range"

    strategy = se.ModuloPartition()
    assert PartitionSpec.from_strategy(strategy) == modulo_spec

    with pytest.raises(TypeError):
        PartitionSpec.from_strategy("not_a_strategy")

    with pytest.raises(ValueError):
        PartitionSpec("unregistered.partitioner")

    with pytest.raises(TypeError):
        PartitionSpec(name=123)  # type: ignore

    with pytest.raises(
        TypeError, match="type 'PartitionSpec' is not an acceptable base type"
    ):

        class _IllegalSubclass(PartitionSpec):
            pass


def test_partition_spec_copy_and_serialization_contracts():
    specs = [
        PartitionSpec.modulo(),
        PartitionSpec.hash(seed=1024),
        PartitionSpec.fast_range(),
        PartitionSpec(
            "sparse_embedding.modulo",
            {"level1": {"level2": [1, 2, 3], "tag": "test"}},
        ),
    ]

    for spec in specs:
        assert copy.copy(spec) is spec
        assert copy.deepcopy(spec) is spec
        assert copy.deepcopy(spec, {}) is spec

        pickled = pickle.dumps(spec)
        restored = pickle.loads(pickled)
        assert restored == spec
        assert hash(restored) == hash(spec)
        assert isinstance(restored.parameters, MappingProxyType)

        with pytest.raises(TypeError):
            restored.parameters["mutation"] = "illegal"  # type: ignore

        target = PartitionTarget(spec, 4, 1)
        assert copy.copy(target).spec is spec
        assert copy.deepcopy(target).spec is spec
        target_restored = pickle.loads(pickle.dumps(target))
        assert target_restored == target
        assert target_restored.spec == spec

        plan = RestorePlan.from_partition(spec, 4, 1)
        assert copy.copy(plan).target.spec is spec
        assert copy.deepcopy(plan).target.spec is spec
        plan_restored = pickle.loads(pickle.dumps(plan))
        assert plan_restored == plan
        assert plan_restored.target.spec == spec


def test_restore_plan_polymorphism_and_validation():
    strategy = se.ModuloPartition()
    plan = RestorePlan.from_partition(
        spec=strategy.spec,
        partition_count=4,
        partition_id=1,
        key_dtype="uint64",
    )
    assert plan.target.spec == PartitionSpec.modulo()
    assert plan.target.partition_count == 4
    assert plan.target.partition_id == 1

    encoded = plan.encode()
    assert isinstance(encoded, bytes)
    assert encoded == encode_restore_options(plan)

    decoded = RestorePlan.from_bytes(encoded)
    assert isinstance(decoded, RestorePlan)
    assert decoded == plan

    factory_plan = RestorePlan.from_partition(
        strategy.spec,
        4,
        1,
        key_dtype="uint64",
    )
    assert factory_plan == plan

    # Strict bounds and dtype assertions without fallback
    with pytest.raises(ValueError):
        RestorePlan.from_partition(strategy.spec, 4, 4)

    with pytest.raises(ValueError):
        RestorePlan.from_partition(strategy.spec, 0, 0)

    with pytest.raises(TypeError):
        RestorePlan.from_partition(strategy.spec, 4.0, 0)  # type: ignore

    with pytest.raises(ValueError):
        RestorePlan.from_partition(strategy.spec, 4, 0, key_dtype="invalid_dtype")


def test_restore_plan_uses_one_target_contract():
    plan = RestorePlan(
        PartitionTarget(PartitionSpec.modulo(), 2, 1),
        key_dtype="uint64",
    )
    assert plan.target.partition_count == 2
    assert plan.target.partition_id == 1
    assert RestorePlan.from_bytes(plan.encode()) == plan
    with pytest.raises(TypeError):
        RestorePlan("invalid")  # type: ignore


@pytest.mark.parametrize("key_dtype", ["int32", "uint32", "int64", "uint64"])
@pytest.mark.parametrize("engine", list(IngestionEngine))
def test_restore_plan_constructors_agree(key_dtype, engine):
    spec = PartitionSpec.hash(seed=42)
    direct = RestorePlan(
        PartitionTarget(spec, 3, 1), key_dtype=key_dtype, engine=engine
    )
    assert direct == RestorePlan.from_partition(
        spec, 3, 1, key_dtype=key_dtype, engine=engine
    )
    assert RestorePlan.from_bytes(direct.encode()) == direct


def test_restore_plan_rejects_positional_policy_arguments():
    spec = PartitionSpec.modulo()
    target = PartitionTarget(spec, 3, 1)
    with pytest.raises(TypeError, match="positional"):
        RestorePlan(target, "uint64")  # type: ignore
    with pytest.raises(TypeError, match="positional"):
        RestorePlan.from_partition(spec, 3, 1, "uint64")  # type: ignore


def test_sparse_checkpoint_load_with_typed_restore_plan(tmp_path):
    root = tmp_path / "typed_checkpoint"
    storage = LocalLanceStorage.from_checkpoint_root(root)
    checkpoint = SparseCheckpoint(storage)

    store = populated_store((1, 2, 3), 9.0)
    committed = checkpoint.save_training_checkpoint(store, "typed_ckpt")

    restore_store = se.EmbeddingStore(config())
    typed_plan = RestorePlan.from_partition(
        PartitionSpec.modulo(),
        1,
        0,
        key_dtype="uint64",
    )
    checkpoint.load_training_checkpoint(restore_store, committed.handle, typed_plan)
    np.testing.assert_allclose(lookup(restore_store, (1, 2, 3)), 9.0)

    with pytest.raises(TypeError):
        checkpoint.load_training_checkpoint(
            restore_store,
            committed.handle,
            "unsupported_restore_type",  # type: ignore
        )


def test_same_and_changed_target_topologies_share_native_entry_path(
    tmp_path, monkeypatch
):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    committed = checkpoint.save_training_checkpoint(
        populated_store(range(8), 3.0), "unified-targets"
    )
    native_restore = lance_backend._checkpoint_native.restore_native_partition_entries
    calls = []

    def record_call(*args, **kwargs):
        calls.append(
            {
                "partition_count": args[3],
                "partition_id": args[4],
                "source_participant_index": args[6],
                "assignment_name": args[9],
            }
        )
        return native_restore(*args, **kwargs)

    monkeypatch.setattr(
        lance_backend._checkpoint_native,
        "restore_native_partition_entries",
        record_call,
    )

    same_target = RestorePlan.from_partition(PartitionSpec.modulo(), 1, 0)
    changed_target = RestorePlan.from_partition(PartitionSpec.modulo(), 2, 0)
    assert isinstance(same_target, RestorePlan)
    assert isinstance(changed_target, RestorePlan)

    same_store = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(same_store, committed.handle, same_target)
    changed_store = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(changed_store, committed.handle, changed_target)

    assert calls == [
        {
            "partition_count": 1,
            "partition_id": 0,
            "source_participant_index": None,
            "assignment_name": "sparse_embedding.modulo",
        },
        {
            "partition_count": 2,
            "partition_id": 0,
            "source_participant_index": None,
            "assignment_name": "sparse_embedding.modulo",
        },
    ]
    assert same_store.total_size() == 8
    assert changed_store.total_size() == 4
    np.testing.assert_allclose(lookup(changed_store, (0, 2, 4, 6)), 3.0)


@pytest.mark.parametrize("include_page_images", [False, True])
def test_incremental_commit_context_survives_coordinator_restart(
    tmp_path, include_page_images
):
    root = tmp_path / "checkpoint"
    writer = LocalLanceStorage(root)
    checkpoint = SparseCheckpoint(writer)
    source = populated_store((1, 2), 1)
    parent = checkpoint.save_training_checkpoint(
        source, "restart-delta", include_page_images=include_page_images
    )
    source.assign(
        np.asarray([2], dtype=np.uint64), np.full((1, 4), 9, dtype=np.float32)
    )
    identity = CheckpointSaveIdentity("restart-delta", "delta", authority(1))
    writer.prepare_checkpoint(identity)
    fragment = writer._write_checkpoint_fragment(
        source,
        identity,
        CheckpointParticipant(0, 1),
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_manifest=writer.resolve_checkpoint_manifest(parent.handle),
        include_page_images=include_page_images,
    )
    committer = LocalLanceStorage(root)
    child = committer._commit_checkpoint(identity, [fragment])
    manifest = committer.resolve_checkpoint_manifest(child.handle)
    assert manifest["checkpoint_mode"] == "incremental"
    assert manifest["participant_watermarks"] == [
        fragment.save_context.mutation_watermark
    ]
    assert fragment.entry_count == 1
    restored = se.EmbeddingStore(config())
    strategy = (
        IngestionEngine.FAST_PAGE_CACHE
        if include_page_images
        else IngestionEngine.RECONSTRUCT_ENTRIES
    )
    SparseCheckpoint(committer).load_training_checkpoint(
        restored,
        child.handle,
        target_restore(CheckpointParticipant(0, 1), engine=strategy),
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 1)
    np.testing.assert_allclose(lookup(restored, (2,)), 9)


@pytest.mark.parametrize("equal_watermarks", [False, True])
def test_participant_local_sequences_and_unequal_watermarks(tmp_path, equal_watermarks):
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    stores = [
        populated_store((1,), 1),
        populated_store((2,) if equal_watermarks else (2, 3), 2),
    ]
    first_identity = CheckpointSaveIdentity("participants", "full", authority())
    storage.prepare_checkpoint(first_identity)
    fragments = [
        storage._write_checkpoint_fragment(
            store, first_identity, CheckpointParticipant(index, 2)
        )
        for index, store in enumerate(stores)
    ]
    parent = storage._commit_checkpoint(first_identity, fragments)
    storage.acknowledge_checkpoint(first_identity)
    for index, store in enumerate(stores):
        storage.acknowledge_store_mutations(
            store, parent.handle, participant_index=index
        )
    stores[0].assign(
        np.asarray([1], dtype=np.uint64), np.full((1, 4), 7, dtype=np.float32)
    )
    stores[1].assign(
        np.asarray([2], dtype=np.uint64), np.full((1, 4), 8, dtype=np.float32)
    )
    second_identity = CheckpointSaveIdentity("participants", "delta", authority(1))
    storage.prepare_checkpoint(second_identity)
    fragments = [
        storage._write_checkpoint_fragment(
            store,
            second_identity,
            CheckpointParticipant(index, 2),
            mode=CheckpointSaveMode.INCREMENTAL,
            parent_manifest=storage.resolve_checkpoint_manifest(parent.handle),
        )
        for index, store in enumerate(stores)
    ]
    child = LocalLanceStorage(root)._commit_checkpoint(second_identity, fragments)
    assert storage.resolve_checkpoint_manifest(child.handle)[
        "participant_watermarks"
    ] == [fragment.save_context.mutation_watermark for fragment in fragments]
    restored = se.EmbeddingStore(config())
    SparseCheckpoint(storage).load_training_checkpoint(
        restored, child.handle, modulo_restore(1, 0)
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 7)
    np.testing.assert_allclose(lookup(restored, (2,)), 8)
    if equal_watermarks:
        assert (
            fragments[0].save_context.mutation_watermark
            == fragments[1].save_context.mutation_watermark
        )
    else:
        np.testing.assert_allclose(lookup(restored, (3,)), 2)


def test_parent_protection_survives_prepared_commit_retry(tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    source = populated_store((1,), 1)
    parent = SparseCheckpoint(storage).save_training_checkpoint(source, "retry-parent")
    source.assign(
        np.asarray([1], dtype=np.uint64), np.full((1, 4), 2, dtype=np.float32)
    )
    identity = CheckpointSaveIdentity("retry-parent", "delta", authority(1))
    storage.prepare_checkpoint(identity)
    fragment = storage._write_checkpoint_fragment(
        source,
        identity,
        CheckpointParticipant(0, 1),
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_manifest=storage.resolve_checkpoint_manifest(parent.handle),
    )
    original = storage._create_parent_chain_reference

    def fault(manifest):
        original(manifest)
        raise RuntimeError("injected parent protection fault")

    monkeypatch.setattr(storage, "_create_parent_chain_reference", fault)
    with pytest.raises(RuntimeError, match="injected parent"):
        storage._commit_checkpoint(identity, [fragment])
    restarted = LocalLanceStorage(root)
    child = restarted._commit_checkpoint(identity, ())
    restarted.acknowledge_checkpoint(identity)
    restarted.release_reference(child.handle)
    with restarted.exclusive_recovery():
        restarted.reclaim()
    restored = se.EmbeddingStore(config())
    SparseCheckpoint(restarted).load_training_checkpoint(
        restored, child.handle, modulo_restore(1, 0)
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 2)
    assert (
        restarted.resolve_checkpoint_manifest(parent.handle)["checkpoint_version"] == 1
    )


def test_optimizer_tracks_only_updated_keys(tmp_path):
    source = se.EmbeddingStore(config(), optimizer_config=se.Adam(lr=0.01))
    source.assign(
        np.asarray([1, 2], dtype=np.uint64), np.ones((2, 4), dtype=np.float32)
    )
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    parent = checkpoint.save_training_checkpoint(source, "optimizer-dirty")
    source.optimizer_step(1)
    assert source.checkpoint_mutations() == []
    source.accumulate_gradients(
        np.asarray([2], dtype=np.uint64), np.ones((1, 4), dtype=np.float32)
    )
    source.optimizer_step(1)
    assert [row["key"] for row in source.checkpoint_mutations()] == [2]
    child = checkpoint.save_training_checkpoint(
        source,
        "optimizer-dirty",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
    )
    descriptor = json.loads(
        next(
            iter(
                storage.resolve_checkpoint_manifest(child.handle)[
                    "fragment_descriptors"
                ]
            )
        )
    )
    assert descriptor["entry_count"] == 1


def test_lookup_score_survives_incremental_restore(tmp_path):
    source = se.EmbeddingStore(config(), score_policy="lfu")
    source.assign(
        np.asarray([1, 2], dtype=np.uint64), np.ones((2, 4), dtype=np.float32)
    )
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    parent = checkpoint.save_training_checkpoint(source, "score-dirty")
    lookup(source, (1,))
    assert [row["key"] for row in source.checkpoint_mutations()] == [1]
    child = checkpoint.save_training_checkpoint(
        source,
        "score-dirty",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
    )
    restored = se.EmbeddingStore(config(), score_policy="lfu")
    checkpoint.load_training_checkpoint(
        restored, child.handle, target_restore(CheckpointParticipant(0, 1))
    )
    assert restored.total_size() == 2
    image = restored.export_page_images()[0]
    assert dict(zip(image["keys"], image["scores"], strict=True)) == {1: 1, 2: 0}


def test_mutation_after_capture_survives_acknowledgement(tmp_path, monkeypatch):
    source = populated_store((1,), 1)
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    parent = checkpoint.save_training_checkpoint(source, "later-mutation")
    source.assign(
        np.asarray([1], dtype=np.uint64), np.full((1, 4), 2, dtype=np.float32)
    )
    original = storage._commit_checkpoint

    def publish(identity, fragments, lease_request=None):
        source.assign(
            np.asarray([1], dtype=np.uint64), np.full((1, 4), 3, dtype=np.float32)
        )
        return original(identity, fragments, lease_request)

    monkeypatch.setattr(storage, "_commit_checkpoint", publish)
    child = checkpoint.save_training_checkpoint(
        source,
        "later-mutation",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
    )
    assert len(source.checkpoint_mutations()) == 1
    restored = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        restored, child.handle, target_restore(CheckpointParticipant(0, 1))
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 2)
    np.testing.assert_allclose(lookup(source, (1,)), 3)


def test_stale_acknowledgement_cannot_roll_back_baseline(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    store = populated_store((1,), 1)
    first_identity = CheckpointSaveIdentity("first-cut", "first", authority())
    storage.prepare_checkpoint(first_identity)
    first_fragment = storage._write_checkpoint_fragment(
        store, first_identity, CheckpointParticipant(0, 1)
    )
    first = storage._commit_checkpoint(first_identity, [first_fragment])
    storage.acknowledge_checkpoint(first_identity)
    store.assign(np.asarray([1], dtype=np.uint64), np.full((1, 4), 2, dtype=np.float32))
    second_identity = CheckpointSaveIdentity("second-cut", "second", authority(1))
    storage.prepare_checkpoint(second_identity)
    second_fragment = storage._write_checkpoint_fragment(
        store, second_identity, CheckpointParticipant(0, 1)
    )
    second = storage._commit_checkpoint(second_identity, [second_fragment])
    storage.acknowledge_store_mutations(store, second.handle)
    baseline = store.checkpoint_baseline()
    with pytest.raises(ValueError, match="stale"):
        storage.acknowledge_store_mutations(store, first.handle)
    assert store.checkpoint_baseline() == baseline


@pytest.mark.parametrize("engine", tuple(IngestionEngine))
@pytest.mark.parametrize("load_optimizer_state", [True, False])
def test_restore_requires_target_optimizer_state(
    tmp_path, engine, load_optimizer_state
):
    source = se.EmbeddingStore(config(), optimizer_config=se.Sgd(lr=0.01))
    source.assign(np.asarray([7], dtype=np.uint64), np.ones((1, 4), dtype=np.float32))
    checkpoint = SparseCheckpoint(LocalLanceStorage(tmp_path / "checkpoint"))
    saved = checkpoint.save_training_checkpoint(
        source, "optimizer-layout", include_page_images=True
    )
    target = (
        se.EmbeddingStore(config(), optimizer_config=se.Adam(lr=0.01))
        if load_optimizer_state
        else se.EmbeddingStore(inference_config())
    )
    plan = target_restore(CheckpointParticipant(0, 1), engine=engine)
    if load_optimizer_state:
        with pytest.raises(ValueError, match="optimizer state"):
            checkpoint.load_training_checkpoint(
                target, saved.handle, plan, load_optimizer_state=True
            )
    else:
        checkpoint.load_training_checkpoint(
            target, saved.handle, plan, load_optimizer_state=False
        )
        np.testing.assert_array_equal(lookup(target, [7]), np.ones((1, 4), np.float32))


def test_compact_pages_respect_explicit_geometry(tmp_path):
    tuning = se.StorageTuningOptions(
        entries_per_page=4096, max_pages=64, calibrate_page_geometry=False
    )
    cfg = se.EmbeddingConfig(
        dims=[4], thread_budget=1, key_dtype="uint64", storage_tuning=tuning
    )
    source = se.EmbeddingStore(cfg, optimizer_config=se.Adam(lr=0.01))
    count = 17 * 4096 + 3
    keys = np.arange(count, dtype=np.uint64)
    source.assign(keys, np.ones((count, 4), dtype=np.float32))
    gradients = np.arange(count * 4, dtype=np.float32).reshape(count, 4) / (count * 4)
    source.accumulate_gradients(keys, gradients)
    source.optimizer_step(1)
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    parent = checkpoint.save_training_checkpoint(source, "geometry")
    compacted = storage.compact_checkpoint(parent.handle, include_page_images=True)
    restored = se.EmbeddingStore(cfg, optimizer_config=se.Adam(lr=0.01))
    checkpoint.load_training_checkpoint(
        restored,
        compacted,
        target_restore(
            CheckpointParticipant(0, 1),
            engine=IngestionEngine.FAST_PAGE_CACHE,
        ),
        load_optimizer_state=True,
    )
    assert restored.total_size() == count
    np.testing.assert_allclose(lookup(restored, keys), lookup(source, keys))
    assert sorted(len(image["keys"]) for image in restored.export_page_images()) == [
        3,
        *([4096] * 17),
    ]
    for store in (source, restored):
        store.accumulate_gradients(keys, gradients)
        store.optimizer_step(1)
    np.testing.assert_allclose(lookup(restored, keys), lookup(source, keys))
    incompatible = se.EmbeddingStore(
        se.EmbeddingConfig(
            dims=[4],
            thread_budget=1,
            key_dtype="uint64",
            storage_tuning=se.StorageTuningOptions(
                entries_per_page=8192, max_pages=64, calibrate_page_geometry=False
            ),
        )
    )
    with pytest.raises(ValueError, match="geometry"):
        checkpoint.load_training_checkpoint(
            incompatible,
            compacted,
            target_restore(
                CheckpointParticipant(0, 1),
                engine=IngestionEngine.FAST_PAGE_CACHE,
            ),
        )
    assert incompatible.is_operational() is True


def test_page_image_materialization_rejects_changed_target_topology(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    committed = checkpoint.save_training_checkpoint(
        populated_store((1, 2), 4.0), "page-topology", include_page_images=True
    )
    target = se.EmbeddingStore(config())
    plan = RestorePlan.from_partition(
        PartitionSpec.modulo(),
        partition_count=2,
        partition_id=0,
        engine=IngestionEngine.FAST_PAGE_CACHE,
    )
    with pytest.raises(
        ValueError,
        match="page-image materialization requires the saved partition topology",
    ):
        checkpoint.load_training_checkpoint(target, committed.handle, plan)
    assert target.is_operational() is True
    assert target.total_size() == 0


@pytest.mark.parametrize("include_page_images", [False, True])
def test_native_compact_rejects_optimizer_stride_change_after_page_flush(
    tmp_path, include_page_images
):
    source = se.EmbeddingStore(config(), optimizer_config=se.Adam(lr=0.01))
    keys = np.arange(4097, dtype=np.uint64)
    source.assign(keys, np.ones((len(keys), 4), dtype=np.float32))
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    saved = SparseCheckpoint(storage).save_training_checkpoint(source, "stride-check")
    manifest = storage.resolve_checkpoint_manifest(saved.handle)
    root = storage._manifest_fragment_root_for_participant(manifest, 0)
    table = lance.dataset(
        storage._workspace.uri(f"{root}/{ENTRIES_DATASET}")
    ).to_table()
    first_uri = str(tmp_path / "first" / ENTRIES_DATASET)
    second_uri = str(tmp_path / "second" / ENTRIES_DATASET)
    lance.write_dataset(table.slice(0, 4096), first_uri)
    last = table.slice(4096, 1).to_pylist()[0]
    last["optimizer_state_stride"] //= 2
    last["optimizer_state_values"] = last["optimizer_state_values"][
        : len(last["optimizer_state_values"]) // 2
    ]
    lance.write_dataset(pa.Table.from_pylist([last], schema=table.schema), second_uri)
    specification = storage._manifest_specification(manifest)
    with pytest.raises(RuntimeError, match="optimizer strides differ"):
        lance_backend._checkpoint_native.compact_native_checkpoint(
            str(tmp_path / "target"),
            [first_uri, second_uri],
            0,
            list(specification.dimensions),
            specification.embedding_dtype,
            specification.optimizer_dtype,
            specification.score_policy,
            specification.key_dtype,
            lance_backend._schema_digest(specification),
            include_page_images,
            4096,
            str(tmp_path / "target-pages") if include_page_images else "",
            {},
        )


@pytest.mark.parametrize("failure_boundary", ["payload", "manifest", "post_manifest"])
def test_reclaim_recovery_releases_parent_chain_reference(
    tmp_path, monkeypatch, failure_boundary
):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    source = populated_store((1,), 1)
    checkpoint = SparseCheckpoint(storage)
    parent = checkpoint.save_training_checkpoint(source, "reclaim-chain")
    source.assign(
        np.asarray([1], dtype=np.uint64), np.full((1, 4), 2, dtype=np.float32)
    )
    child = checkpoint.save_training_checkpoint(
        source,
        "reclaim-chain",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
    )
    lease = storage.acquire_checkpoint_lease(
        child.handle, CheckpointLeaseRequest("fault-test", 60, "retain-child")
    )
    checkpoint.save_training_checkpoint(source, "reclaim-chain")
    child_manifest = storage.resolve_checkpoint_manifest(child.handle)
    child_root = storage._manifest_fragment_root_for_participant(child_manifest, 0)
    if failure_boundary == "payload":
        original = storage._workspace.delete_prefix

        def fault(root, **kwargs):
            original(root, **kwargs)
            if root == child_root:
                raise RuntimeError("injected reclaim payload fault")

        monkeypatch.setattr(storage._workspace, "delete_prefix", fault)
    elif failure_boundary == "manifest":
        original = storage._delete_parent_chain_reference

        def fault(manifest):
            original(manifest)
            if manifest["checkpoint_version"] == child_manifest["checkpoint_version"]:
                raise RuntimeError("injected reclaim manifest fault")

        monkeypatch.setattr(storage, "_delete_parent_chain_reference", fault)
    else:
        dataset_type = type(
            lance.dataset(str(tmp_path / "checkpoint" / SNAPSHOT_MANIFEST_DATASET))
        )
        original = dataset_type.delete

        def fault(dataset, predicate, *args, **kwargs):
            result = original(dataset, predicate, *args, **kwargs)
            if (
                f"checkpoint_version = {child_manifest['checkpoint_version']}"
                in predicate
            ):
                raise RuntimeError("injected post-manifest reclaim fault")
            return result

        monkeypatch.setattr(dataset_type, "delete", fault)
    storage.release_checkpoint_lease(lease)
    with pytest.raises(RuntimeError):
        with storage.exclusive_recovery():
            storage.reclaim()
    monkeypatch.undo()
    restarted = LocalLanceStorage(tmp_path / "checkpoint")
    with restarted.exclusive_recovery():
        restarted.reclaim()
        restarted.reclaim()
    with pytest.raises(LookupError):
        restarted.resolve_checkpoint_manifest(parent.handle)
    with pytest.raises(LookupError):
        restarted.resolve_checkpoint_manifest(child.handle)


def test_rejected_native_ack_preserves_resharded_full_requirement():
    store = se.EmbeddingStore(config())
    store.initialize_checkpoint_baseline("resharded-base", 0, resharded=True)
    baseline = store.checkpoint_baseline()
    assert store.checkpoint_requires_full()

    with pytest.raises(ValueError, match="journal incarnation"):
        store.advance_checkpoint_baseline(0, "next-base", 999, "other-journal")

    assert store.checkpoint_baseline() == baseline
    assert store.checkpoint_requires_full()


@pytest.mark.parametrize("include_page_images", [False, True])
@pytest.mark.parametrize("failure_boundary", ["marker", "payload", "post_manifest"])
def test_abort_reclaim_failure_preserves_lineage_and_recovers(
    tmp_path, monkeypatch, include_page_images, failure_boundary
):
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    source = populated_store((1,), 1)
    parent = SparseCheckpoint(storage).save_training_checkpoint(
        source, "abort-chain", include_page_images=include_page_images
    )
    source.assign(
        np.asarray([1], dtype=np.uint64), np.full((1, 4), 2, dtype=np.float32)
    )
    identity = CheckpointSaveIdentity("abort-chain", "orphan", authority(2))
    storage.prepare_checkpoint(identity)
    fragment = storage.write_checkpoint_fragment_bytes(
        source,
        identity,
        CheckpointParticipant(0, 1),
        include_page_images=include_page_images,
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
    )
    storage.finalize_checkpoint(identity)
    committed = storage.commit_checkpoint_bytes(identity, [fragment])
    locator, _ = storage._resolve_handle(committed.handle)
    manifest = storage.resolve_checkpoint_manifest(committed.handle)
    marker = storage._reclaim_pending_tag(locator)
    intent = storage._intent_tag(identity, locator)
    dataset = storage._workspace.dataset(SNAPSHOT_MANIFEST_DATASET)

    if failure_boundary == "marker":
        tag_type = type(dataset.tags)
        original = tag_type.create

        def fault(tags, tag, version):
            if tag == marker:
                raise OSError("injected abort reclaim fault")
            return original(tags, tag, version)

        monkeypatch.setattr(tag_type, "create", fault)
    elif failure_boundary == "payload":
        roots = set(storage._manifest_fragment_roots(manifest).values())
        original = storage._workspace.delete_prefix

        def fault(path, **kwargs):
            result = original(path, **kwargs)
            if path in roots:
                raise OSError("injected abort reclaim fault")
            return result

        monkeypatch.setattr(storage._workspace, "delete_prefix", fault)
    else:
        dataset_type = type(dataset)
        original = dataset_type.delete

        def fault(dataset, predicate, *args, **kwargs):
            result = original(dataset, predicate, *args, **kwargs)
            if f"AND checkpoint_version = {locator.checkpoint_version}" in predicate:
                raise OSError("injected abort reclaim fault")
            return result

        monkeypatch.setattr(dataset_type, "delete", fault)

    with pytest.raises(OSError, match="injected abort reclaim fault"):
        storage.abort_checkpoint(identity)
    tags = dataset.tags.list()
    assert intent in tags
    assert (marker in tags) is (failure_boundary != "marker")
    assert storage.resolve_lineage_head("abort-chain")["checkpoint_version"] == 1

    monkeypatch.undo()
    storage.close()
    restarted = LocalLanceStorage(root)
    try:
        restarted.abort_checkpoint(identity)
        with restarted.exclusive_recovery():
            restarted.reclaim()
            restarted.reclaim()
        tags = restarted._workspace.dataset(SNAPSHOT_MANIFEST_DATASET).tags.list()
        assert intent not in tags
        assert marker not in tags
        assert restarted.resolve_lineage_head("abort-chain")["checkpoint_version"] == 1
        with pytest.raises(LookupError):
            restarted.resolve_checkpoint_manifest(committed.handle)
        target = se.EmbeddingStore(config())
        SparseCheckpoint(restarted).load_training_checkpoint(
            target, parent.handle, modulo_restore(1, 0)
        )
        np.testing.assert_allclose(lookup(target, (1,)), 1)
    finally:
        restarted.close()


@pytest.mark.parametrize("capture_target", [False, True])
def test_acknowledgement_rejects_other_journal_incarnation(tmp_path, capture_target):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    source = populated_store((1,), 1)
    parent = SparseCheckpoint(storage).save_training_checkpoint(source, "incarnation")
    target = populated_store((1,), 1)
    if capture_target:
        identity = CheckpointSaveIdentity("other", "other", authority(1))
        storage.prepare_checkpoint(identity)
        storage._write_checkpoint_fragment(
            target, identity, CheckpointParticipant(0, 1)
        )
        storage.abort_checkpoint(identity)
    with pytest.raises(ValueError, match="incarnation"):
        storage.acknowledge_store_mutations(target, parent.handle)
    assert target.checkpoint_baseline() == ""
    assert target.checkpoint_mutations()


def test_page_restore_requires_new_full_baseline(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    store = populated_store((1,), 1)
    parent = checkpoint.save_training_checkpoint(
        store, "physical-lineage", include_page_images=True
    )
    store = se.EmbeddingStore(inference_config())
    checkpoint.load_training_checkpoint(
        store,
        parent.handle,
        target_restore(
            CheckpointParticipant(0, 1),
            engine=IngestionEngine.FAST_PAGE_CACHE,
        ),
        load_optimizer_state=False,
    )
    assert store.checkpoint_baseline() == ""
    np.testing.assert_allclose(lookup(store, (1,)), 1)
    with pytest.raises(RuntimeError, match="store baseline"):
        checkpoint.save_training_checkpoint(
            store,
            "physical-lineage",
            mode=CheckpointSaveMode.INCREMENTAL,
            parent_handle=parent.handle,
        )
    fresh = checkpoint.save_training_checkpoint(store, "physical-lineage")
    child = checkpoint.save_training_checkpoint(
        store,
        "physical-lineage",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=fresh.handle,
    )
    restored = se.EmbeddingStore(inference_config())
    checkpoint.load_training_checkpoint(
        restored, child.handle, modulo_restore(1, 0), load_optimizer_state=False
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 1)


def test_physical_weight_only_restore_does_not_install_baseline(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    committed = checkpoint.save_training_checkpoint(
        populated_store((1,), 1), "physical-inference"
    )
    target = se.EmbeddingStore(inference_config())
    checkpoint.load_training_checkpoint(
        target,
        committed.handle,
        target_restore(CheckpointParticipant(0, 1)),
        load_optimizer_state=False,
    )
    assert target.checkpoint_baseline() == ""
    np.testing.assert_allclose(lookup(target, (1,)), 1)


def test_elastic_training_restore_establishes_origin_and_requires_full(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    source = populated_store((1, 2), 1)
    committed = checkpoint.save_training_checkpoint(source, "elastic-training")
    target = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        target,
        committed.handle,
        modulo_restore(2, 0),
    )
    assert target.checkpoint_baseline()
    assert checkpoint.requires_full_checkpoint(target)
    assert storage.requires_full_checkpoint(target)
    target.assign(
        np.asarray([3], dtype=np.uint64), np.full((1, 4), 2, dtype=np.float32)
    )
    with pytest.raises(
        ValueError, match="requires the next training checkpoint to be FULL"
    ):
        checkpoint.save_training_checkpoint(
            target,
            "elastic-training",
            mode=CheckpointSaveMode.INCREMENTAL,
            parent_handle=committed.handle,
        )
    auto_full = checkpoint.save_training_checkpoint(
        target, "elastic-training", parent_handle=committed.handle
    )
    assert auto_full.handle
    assert not checkpoint.requires_full_checkpoint(target)
    assert not storage.requires_full_checkpoint(target)

    target.assign(
        np.asarray([4], dtype=np.uint64), np.full((1, 4), 3, dtype=np.float32)
    )
    incremental = checkpoint.save_training_checkpoint(
        target, "elastic-training", parent_handle=auto_full.handle
    )
    assert incremental.handle


def test_elastic_origin_uses_validated_chain_and_maximum_watermark():
    chain = (
        {
            "checkpoint_name": "chain",
            "checkpoint_version": 1,
            "schema_digest": "schema",
            "fragment_ids": ["p0", "p1"],
            "participant_watermarks": [3, 8],
            "checkpoint_mode": "full",
        },
        {
            "checkpoint_name": "chain",
            "checkpoint_version": 2,
            "schema_digest": "schema",
            "fragment_ids": ["p0", "p1"],
            "participant_watermarks": [5, 13],
            "checkpoint_mode": "incremental",
        },
    )
    plan = modulo_restore(2, 1)
    baseline, watermark = LocalLanceStorage._target_restore_origin(chain, plan)
    assert baseline
    assert watermark == 13


def test_elastic_origin_uses_final_manifest_watermark_only():
    chain = (
        {
            "checkpoint_name": "chain",
            "checkpoint_version": 1,
            "schema_digest": "schema",
            "fragment_ids": ["p0"],
            "participant_watermarks": [99],
            "checkpoint_mode": "full",
        },
        {
            "checkpoint_name": "chain",
            "checkpoint_version": 2,
            "schema_digest": "schema",
            "fragment_ids": ["p0"],
            "participant_watermarks": [13],
            "checkpoint_mode": "incremental",
        },
    )
    _, watermark = LocalLanceStorage._target_restore_origin(chain, modulo_restore(1, 0))
    assert watermark == 13


def test_elastic_weight_only_restore_does_not_install_baseline(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    committed = checkpoint.save_training_checkpoint(
        populated_store((1,), 1), "elastic-inference"
    )
    target = se.EmbeddingStore(inference_config())
    checkpoint.load_training_checkpoint(
        target,
        committed.handle,
        modulo_restore(1, 0),
        load_optimizer_state=False,
    )
    assert target.checkpoint_baseline() == ""
    np.testing.assert_allclose(lookup(target, (1,)), 1)


@pytest.mark.parametrize("include_page_images", [False, True])
def test_empty_full_delta_and_compact_without_payload(tmp_path, include_page_images):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    store = se.EmbeddingStore(config())
    parent = checkpoint.save_training_checkpoint(
        store, "empty-chain", include_page_images=include_page_images
    )
    child = checkpoint.save_training_checkpoint(
        store,
        "empty-chain",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
        include_page_images=include_page_images,
    )
    compacted = storage.compact_checkpoint(
        child.handle, include_page_images=include_page_images
    )
    for saved in (parent.handle, child.handle, compacted):
        manifest = storage.resolve_checkpoint_manifest(saved)
        root = storage._manifest_fragment_root_for_participant(manifest, 0)
        assert not storage._workspace.exists(f"{root}/{ENTRIES_DATASET}")
        assert not storage._workspace.exists(f"{root}/{PAGE_IMAGES_DATASET}")
        target = se.EmbeddingStore(config())
        checkpoint.load_training_checkpoint(
            target,
            saved,
            target_restore(
                CheckpointParticipant(0, 1),
                engine=IngestionEngine.FAST_PAGE_CACHE
                if include_page_images
                else IngestionEngine.RECONSTRUCT_ENTRIES,
            ),
        )
        assert target.total_size() == 0
    if include_page_images:
        resumed = checkpoint.save_training_checkpoint(
            target,
            "restored-empty",
            include_page_images=True,
        )
        assert (
            storage.resolve_checkpoint_manifest(resumed.handle)[
                "participant_page_layout_generations"
            ]
            == storage.resolve_checkpoint_manifest(compacted)[
                "participant_page_layout_generations"
            ]
        )


@pytest.mark.parametrize("include_page_images", [False, True])
def test_compact_preserves_delta_tombstones_optimizer_and_scores(
    tmp_path, include_page_images
):
    cfg = config()
    source = se.EmbeddingStore(
        cfg, optimizer_config=se.Adam(lr=0.01), score_policy="lfu"
    )
    source.assign(
        np.asarray([1, 2], dtype=np.uint64), np.ones((2, 4), dtype=np.float32)
    )
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    parent = checkpoint.save_training_checkpoint(source, "merge-state")
    source.erase_dimension(np.asarray([2], dtype=np.uint64), dimension=4)
    source.assign(
        np.asarray([3], dtype=np.uint64), np.full((1, 4), 3, dtype=np.float32)
    )
    source.accumulate_gradients(
        np.asarray([1], dtype=np.uint64), np.ones((1, 4), dtype=np.float32)
    )
    source.optimizer_step(1)
    lookup(source, (1,))
    child = checkpoint.save_training_checkpoint(
        source,
        "merge-state",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
    )
    compacted = storage.compact_checkpoint(
        child.handle, include_page_images=include_page_images
    )
    original = se.EmbeddingStore(
        cfg, optimizer_config=se.Adam(lr=0.01), score_policy="lfu"
    )
    merged = se.EmbeddingStore(
        cfg, optimizer_config=se.Adam(lr=0.01), score_policy="lfu"
    )
    checkpoint.load_training_checkpoint(original, child.handle, modulo_restore(1, 0))
    checkpoint.load_training_checkpoint(
        merged,
        compacted,
        target_restore(
            CheckpointParticipant(0, 1),
            engine=IngestionEngine.FAST_PAGE_CACHE
            if include_page_images
            else IngestionEngine.RECONSTRUCT_ENTRIES,
        ),
    )
    assert original.total_size() == merged.total_size() == 2

    def scores(store):
        return {
            int(key): int(score)
            for image in store.export_page_images()
            for key, score in zip(image["keys"], image["scores"], strict=True)
        }

    assert scores(original) == scores(merged)
    np.testing.assert_allclose(lookup(original, (1, 3)), lookup(merged, (1, 3)))
    for target in (original, merged):
        target.accumulate_gradients(
            np.asarray([1], dtype=np.uint64), np.ones((1, 4), dtype=np.float32)
        )
        target.optimizer_step(1)
    np.testing.assert_allclose(lookup(original, (1, 3)), lookup(merged, (1, 3)))


@pytest.mark.parametrize("failure_boundary", ["write", "commit"])
def test_compact_failure_preserves_source_chain(
    tmp_path, monkeypatch, failure_boundary
):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    source = populated_store((1,), 1)
    parent = checkpoint.save_training_checkpoint(source, "compact-fault")
    source.assign(
        np.asarray([1], dtype=np.uint64), np.full((1, 4), 2, dtype=np.float32)
    )
    child = checkpoint.save_training_checkpoint(
        source,
        "compact-fault",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
    )

    def fault(*args, **kwargs):
        raise RuntimeError("injected compact fault")

    if failure_boundary == "write":
        monkeypatch.setattr(
            lance_backend._checkpoint_native, "compact_native_checkpoint", fault
        )
    else:
        monkeypatch.setattr(storage, "_commit_checkpoint", fault)
    with pytest.raises(RuntimeError, match="injected compact"):
        storage.compact_checkpoint(child.handle, include_page_images=True)
    for saved, value in ((parent.handle, 1), (child.handle, 2)):
        target = se.EmbeddingStore(config())
        checkpoint.load_training_checkpoint(target, saved, modulo_restore(1, 0))
        np.testing.assert_allclose(lookup(target, (1,)), value)


@pytest.mark.parametrize("operation", ["clear", "reserve"])
def test_structural_publication_preserves_incremental_baseline(tmp_path, operation):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    store = populated_store((1,), 1)
    parent = checkpoint.save_training_checkpoint(store, "structural")
    if operation == "clear":
        store.clear()
    else:
        store.reserve(128, dim=4)
        store.assign(
            np.asarray([1], dtype=np.uint64), np.full((1, 4), 2, dtype=np.float32)
        )
    child = checkpoint.save_training_checkpoint(
        store,
        "structural",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
    )
    restored = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(restored, child.handle, modulo_restore(1, 0))
    assert restored.total_size() == (0 if operation == "clear" else 1)
    if operation == "reserve":
        np.testing.assert_allclose(lookup(restored, (1,)), 2)


def test_acknowledgement_idempotence_validates_watermark(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    source = populated_store((1,), 1)
    saved = SparseCheckpoint(storage).save_training_checkpoint(source, "strict-cut")
    manifest = storage.resolve_checkpoint_manifest(saved.handle)
    import json

    descriptor = json.loads(manifest["fragment_descriptors"][0])
    storage.acknowledge_store_mutations(source, saved.handle)
    with pytest.raises(ValueError, match="stale"):
        source.advance_checkpoint_baseline(
            descriptor["mutation_watermark"] + 1,
            source.checkpoint_baseline(),
            descriptor["capture_id"],
            descriptor["journal_identity"],
        )


def test_cancelled_capture_keeps_dirty_generation(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    source = populated_store((1,), 1)
    identity = CheckpointSaveIdentity("cancelled", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragment = storage._write_checkpoint_fragment(
        source, identity, CheckpointParticipant(0, 1)
    )
    source.cancel_checkpoint_capture(
        fragment.save_context.capture_id, fragment.save_context.journal_identity
    )
    source.cancel_checkpoint_capture(
        fragment.save_context.capture_id, fragment.save_context.journal_identity
    )
    assert source.checkpoint_mutations()
    committed = storage._commit_checkpoint(identity, [fragment])
    with pytest.raises(ValueError, match="stale"):
        storage.acknowledge_store_mutations(source, committed.handle)
    assert source.checkpoint_baseline() == ""


def test_restore_allocates_only_surviving_keys(tmp_path):
    source = populated_store(range(20000), 1)
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    parent = checkpoint.save_training_checkpoint(source, "capacity")
    source.erase_dimension(np.arange(19999, dtype=np.uint64), dimension=4)
    child = checkpoint.save_training_checkpoint(
        source,
        "capacity",
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=parent.handle,
    )
    tuning = se.StorageTuningOptions(
        entries_per_page=4096, max_pages=2, calibrate_page_geometry=False
    )
    target = se.EmbeddingStore(
        se.EmbeddingConfig(
            dims=[4], thread_budget=1, key_dtype="uint64", storage_tuning=tuning
        )
    )
    checkpoint.load_training_checkpoint(target, child.handle, modulo_restore(1, 0))
    assert target.total_size() == 1
    np.testing.assert_allclose(lookup(target, (19999,)), 1)


def test_native_restore_uses_one_explicit_lance_version(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    saved = checkpoint.save_training_checkpoint(
        populated_store((1,), 1), "scan-version"
    )
    root = storage._manifest_fragment_root_for_participant(
        storage.resolve_checkpoint_manifest(saved.handle), 0
    )
    uri = storage._workspace.uri(f"{root}/{ENTRIES_DATASET}")
    dataset = lance.dataset(uri)
    version = dataset.version
    dataset.delete("key = 1")
    assert dataset.count_rows() == 0
    target = se.EmbeddingStore(config())
    lance_backend._checkpoint_native.restore_native_partition_entries(
        target, [uri], version, 1, 0, "uint64"
    )
    assert target.total_size() == 1
    np.testing.assert_allclose(lookup(target, (1,)), 1)


@pytest.mark.parametrize("key_dtype", ["int64", "uint64"])
def test_native_entry_restore_preserves_multi_batch_dimension_state(
    tmp_path, key_dtype
):
    cfg = se.EmbeddingConfig(
        dims=[3, 5],
        thread_budget=1,
        key_dtype=key_dtype,
        optimizer_dtype="float16",
    )
    source = se.EmbeddingStore(
        cfg, optimizer_config=se.Adam(lr=0.01), score_policy="lfu"
    )
    keys = np.arange(8200, dtype=key_dtype)
    keys += -4100 if key_dtype == "int64" else 1 << 63
    for dim in (3, 5):
        values = np.arange(len(keys) * dim, dtype=np.float32).reshape(-1, dim) / 256
        source.assign(keys, values)
        source.accumulate_gradients(keys, np.full_like(values, dim / 10))
    source.optimizer_step(1)
    for dim in (3, 3, 5):
        source.lookup(keys, dim=dim)

    def scores(store):
        return {
            (image["embedding_dim"], int(key)): int(score)
            for image in store.export_page_images()
            for key, score in zip(image["keys"], image["scores"], strict=True)
        }

    expected_scores = scores(source)
    first_key = int(keys[0]) % (1 << 64)
    assert expected_scores[(3, first_key)] != expected_scores[(5, first_key)]
    checkpoint = SparseCheckpoint(LocalLanceStorage(tmp_path / "checkpoint"))
    saved = checkpoint.save_training_checkpoint(source, "multi-batch-state")
    target = se.EmbeddingStore(
        cfg, optimizer_config=se.Adam(lr=0.01), score_policy="lfu"
    )
    checkpoint.load_training_checkpoint(
        target,
        saved.handle,
        RestorePlan.from_partition(PartitionSpec.modulo(), 1, 0, key_dtype=key_dtype),
    )
    assert target.total_size() == len(keys) * 2
    assert scores(target) == expected_scores
    for store in (source, target):
        for dim in (3, 5):
            store.accumulate_gradients(
                keys, np.full((len(keys), dim), dim / 7, dtype=np.float32)
            )
        store.optimizer_step(1)
    for dim in (3, 5):
        np.testing.assert_array_equal(
            np.from_dlpack(target.lookup(keys, dim=dim)[0]),
            np.from_dlpack(source.lookup(keys, dim=dim)[0]),
        )


def _commit_partitioned_restore_fixture(
    storage, stores, name, attempt, generation, *, parent=None, pages=False
):
    identity = CheckpointSaveIdentity(name, attempt, authority(generation))
    storage.prepare_checkpoint(identity)
    fragments = [
        storage._write_checkpoint_fragment(
            store,
            identity,
            CheckpointParticipant(index, len(stores)),
            include_page_images=pages,
            mode=CheckpointSaveMode.INCREMENTAL if parent else CheckpointSaveMode.FULL,
            parent_manifest=storage.resolve_checkpoint_manifest(parent.handle)
            if parent
            else None,
        )
        for index, store in enumerate(stores)
    ]
    committed = storage._commit_checkpoint(identity, fragments)
    storage.acknowledge_checkpoint(identity)
    for index, store in enumerate(stores):
        storage.acknowledge_store_mutations(
            store, committed.handle, participant_index=index
        )
    return committed


def test_equal_count_reassignment_rejects_pages_and_requires_full(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    sources = [populated_store((0, 2), 3), populated_store((1, 3), 3)]
    parent = _commit_partitioned_restore_fixture(
        storage, sources, "layout", "full", 0, pages=True
    )
    checkpoint = SparseCheckpoint(storage)
    target = se.EmbeddingStore(config())
    plan = RestorePlan.from_partition(
        PartitionSpec.fast_range(), 2, 0, engine=IngestionEngine.FAST_PAGE_CACHE
    )
    with pytest.raises(ValueError, match="source partition ownership"):
        checkpoint.load_training_checkpoint(target, parent.handle, plan)
    assert target.is_operational() is True
    assert target.total_size() == 0
    restored = []
    for index in range(2):
        target = se.EmbeddingStore(config())
        checkpoint.load_training_checkpoint(
            target,
            parent.handle,
            RestorePlan.from_partition(PartitionSpec.fast_range(), 2, index),
        )
        restored.append(target)
    assert [store.total_size() for store in restored] == [4, 0]
    np.testing.assert_allclose(lookup(restored[0], (0, 1, 2, 3)), 3)
    with pytest.raises(
        ValueError, match="requires the next training checkpoint to be FULL"
    ):
        checkpoint.save_training_checkpoint(
            restored[0],
            "layout",
            mode=CheckpointSaveMode.INCREMENTAL,
            parent_handle=parent.handle,
        )
    full = _commit_partitioned_restore_fixture(
        storage, restored, "new-layout", "full", 0
    )
    restored[0].assign(np.asarray([1], dtype=np.uint64), np.full((1, 4), 9, np.float32))
    child = _commit_partitioned_restore_fixture(
        storage, restored, "new-layout", "delta", 1, parent=full
    )
    result = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        result,
        child.handle,
        RestorePlan.from_partition(PartitionSpec.fast_range(), 2, 0),
    )
    np.testing.assert_allclose(lookup(result, (1,)), 9)
    np.testing.assert_allclose(lookup(result, (0, 2, 3)), 3)


@pytest.mark.parametrize("key_dtype", ["int32", "uint32", "int64", "uint64"])
def test_native_layout_proof_preserves_typed_fast_range_pages(tmp_path, key_dtype):
    dtype = np.dtype(key_dtype)
    high = np.iinfo(dtype).min if dtype.kind == "i" else 1 << (dtype.itemsize * 8 - 1)
    groups = [
        np.asarray([0, 1], dtype=dtype),
        np.asarray([high, high + 1], dtype=dtype),
    ]
    cfg = se.EmbeddingConfig(dims=[4], thread_budget=1, key_dtype=key_dtype)
    sources = [se.EmbeddingStore(cfg) for _ in groups]
    for source, keys in zip(sources, groups, strict=True):
        source.assign(keys, np.full((len(keys), 4), 3, np.float32))
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    parent = _commit_partitioned_restore_fixture(
        storage, sources, "typed-layout", "full", 0, pages=True
    )
    restored = []
    for index, keys in enumerate(groups):
        target = se.EmbeddingStore(cfg)
        checkpoint.load_training_checkpoint(
            target,
            parent.handle,
            RestorePlan.from_partition(
                PartitionSpec.fast_range(),
                2,
                index,
                key_dtype=key_dtype,
                engine=IngestionEngine.FAST_PAGE_CACHE,
            ),
        )
        assert target.total_size() == len(keys)
        np.testing.assert_allclose(np.from_dlpack(target.lookup(keys, dim=4)[0]), 3)
        target.assign(keys, np.full((len(keys), 4), 9, np.float32))
        restored.append(target)
    child = _commit_partitioned_restore_fixture(
        storage, restored, "typed-layout", "delta", 1, parent=parent
    )
    for index, keys in enumerate(groups):
        target = se.EmbeddingStore(cfg)
        checkpoint.load_training_checkpoint(
            target,
            child.handle,
            RestorePlan.from_partition(
                PartitionSpec.fast_range(), 2, index, key_dtype=key_dtype
            ),
        )
        assert target.total_size() == len(keys)
        np.testing.assert_allclose(np.from_dlpack(target.lookup(keys, dim=4)[0]), 9)


def test_layout_proof_includes_deleted_identities(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    high = 1 << 63
    sources = [populated_store((0, high), 3), se.EmbeddingStore(config())]
    parent = _commit_partitioned_restore_fixture(
        storage, sources, "deleted-layout", "full", 0, pages=True
    )
    sources[0].erase_dimension(np.asarray([high], dtype=np.uint64), dimension=4)
    child = _commit_partitioned_restore_fixture(
        storage, sources, "deleted-layout", "delta", 1, parent=parent, pages=True
    )
    checkpoint = SparseCheckpoint(storage)
    target = se.EmbeddingStore(config())
    with pytest.raises(ValueError, match="source partition ownership"):
        checkpoint.load_training_checkpoint(
            target,
            child.handle,
            RestorePlan.from_partition(
                PartitionSpec.fast_range(), 2, 0, engine=IngestionEngine.FAST_PAGE_CACHE
            ),
        )
    assert target.is_operational() is True
    assert target.total_size() == 0
    checkpoint.load_training_checkpoint(
        target,
        child.handle,
        RestorePlan.from_partition(PartitionSpec.fast_range(), 2, 0),
    )
    assert target.total_size() == 1
    with pytest.raises(
        ValueError, match="requires the next training checkpoint to be FULL"
    ):
        checkpoint.save_training_checkpoint(
            target,
            "deleted-layout",
            mode=CheckpointSaveMode.INCREMENTAL,
            parent_handle=child.handle,
        )


@pytest.mark.parametrize(
    ("partition_count", "partition_id"),
    [(1 << 32, 0), ((1 << 32) + 1, 0), ((1 << 32) + 1, 1 << 32)],
)
def test_partition_count_rejects_native_narrowing(partition_count, partition_id):
    with pytest.raises(ValueError, match="UINT32_MAX"):
        RestorePlan.from_partition(
            PartitionSpec.modulo(), partition_count, partition_id
        )
    wire = msgpack.unpackb(encode_restore_options(modulo_restore(1, 0)), raw=False)
    wire["partition_count"] = partition_count
    wire["partition_id"] = partition_id
    with pytest.raises(ValueError, match="UINT32_MAX"):
        RestorePlan.from_bytes(msgpack.packb(wire, use_bin_type=True))
    target = se.EmbeddingStore(config())
    with pytest.raises(ValueError, match="UINT32_MAX"):
        lance_backend._checkpoint_native.restore_native_partition_entries(
            target, [], 0, partition_count, partition_id, "uint64"
        )
    assert target.is_operational() is True
    assert target.total_size() == 0


def test_maximum_native_partition_count_round_trips_and_routes(tmp_path):
    plan = RestorePlan.from_partition(PartitionSpec.modulo(), (1 << 32) - 1, 0)
    assert RestorePlan.from_bytes(encode_restore_options(plan)) == plan
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    saved = checkpoint.save_training_checkpoint(
        populated_store((0, 1, 2), 3), "max-count"
    )
    target = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(target, saved.handle, plan)
    assert target.total_size() == 1
    np.testing.assert_allclose(lookup(target, (0,)), 3)


def test_orphan_head_lineage_resolution_and_reclaim(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    store = populated_store((0, 1), 1)
    lineage = "test_lineage"

    # Save 1: acknowledged FULL commit
    auth1 = CheckpointCommitAuthority("scope", 1, "auth1")
    id1 = CheckpointSaveIdentity(lineage, "attempt1", auth1)
    storage.prepare_checkpoint(id1)
    frag1 = storage.write_checkpoint_fragment_bytes(
        store, id1, CheckpointParticipant(0, 1), mode=CheckpointSaveMode.FULL
    )
    storage.finalize_checkpoint(id1)
    committed1 = storage.commit_checkpoint_bytes(id1, [frag1])
    storage.acknowledge_store_mutations(store, committed1.handle, 0)
    storage.acknowledge_checkpoint(id1)

    # Save 2: committed but NOT acknowledged (simulates crash before publish/ack)
    auth2 = CheckpointCommitAuthority("scope", 2, "auth2")
    id2 = CheckpointSaveIdentity(lineage, "attempt2", auth2)
    storage.prepare_checkpoint(id2)
    frag2 = storage.write_checkpoint_fragment_bytes(
        store,
        id2,
        CheckpointParticipant(0, 1),
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=committed1.handle,
    )
    storage.finalize_checkpoint(id2)
    _ = storage.commit_checkpoint_bytes(id2, [frag2])

    # Unacknowledged Save 2 must not be resolved as lineage head
    head = storage.resolve_lineage_head(lineage)
    assert head is not None
    assert head["checkpoint_version"] == 1

    # Save 3: recovery attempt saves on top of acknowledged Save 1
    auth3 = CheckpointCommitAuthority("scope", 3, "auth3")
    id3 = CheckpointSaveIdentity(lineage, "attempt3", auth3)
    storage.prepare_checkpoint(id3)
    frag3 = storage.write_checkpoint_fragment_bytes(
        store,
        id3,
        CheckpointParticipant(0, 1),
        mode=CheckpointSaveMode.INCREMENTAL,
        parent_handle=committed1.handle,
    )
    storage.finalize_checkpoint(id3)
    committed3 = storage.commit_checkpoint_bytes(id3, [frag3])
    storage.acknowledge_store_mutations(store, committed3.handle, 0)
    storage.acknowledge_checkpoint(id3)

    head3 = storage.resolve_lineage_head(lineage)
    assert head3 is not None
    assert head3["checkpoint_version"] == 3
    assert head3["parent_checkpoint_version"] == 1

    # Reclaim purges orphaned Save 2
    with storage.exclusive_recovery():
        storage.reclaim()

    rows = storage._manifest_rows()
    versions = [
        r["checkpoint_version"] for r in rows if r["checkpoint_name"] == lineage
    ]
    assert sorted(versions) == [1, 3]
    storage.close()
