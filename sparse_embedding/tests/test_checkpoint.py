import inspect
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import lance
import msgpack
import numpy as np
import pytest

import sparse_embedding as se
from sparse_embedding.checkpoint import (
    ElasticRestore,
    PartitionSpec,
    PhysicalRestore,
    PhysicalRestoreStrategy,
    RestorePlan,
    SparseCheckpoint,
    SparsePartitionAssignmentSpec,
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


def modulo_restore(partition_count: int, partition_id: int) -> ElasticRestore:
    return ElasticRestore(
        PartitionSpec.modulo(),
        partition_count,
        partition_id,
        "uint64",
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
        encode_restore_options(PhysicalRestore(CheckpointParticipant(0, 1))),
    )
    np.testing.assert_allclose(lookup(restored, (1, 2)), 3)


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
    spec = SparsePartitionAssignmentSpec.modulo()
    assert spec.sparse_assignment_name == "sparse_embedding.modulo"
    assert dict(spec.sparse_assignment_parameters) == {}
    with pytest.raises(ValueError, match="owner-approved"):
        SparsePartitionAssignmentSpec("unknown", {})
    with pytest.raises(TypeError, match="unsupported value"):
        SparsePartitionAssignmentSpec("custom", {"callback": object()})


def test_owner_restore_options_are_canonical_and_opaque():
    encoded = encode_restore_options(modulo_restore(3, 1))
    assert isinstance(encoded, bytes)
    noncanonical = msgpack.packb(
        {
            "restore_kind": "elastic",
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
        ElasticRestore(
            PartitionSpec.modulo(),
            3,
            1,
            key_dtype=key_dtype,
        )
    )
    decoded = msgpack.unpackb(encoded, raw=False)
    assert decoded["key_dtype"] == key_dtype
    assert decoded["restore_kind"] == "elastic"


def test_restore_options_reject_typed_mode_and_unknown_fields(tmp_path):
    with pytest.raises(TypeError, match="bytes"):
        _decode_restore_options(modulo_restore(1, 0))
    unknown = msgpack.packb(
        {
            "restore_kind": "elastic",
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
            encode_restore_options(PhysicalRestore(CheckpointParticipant(0, 1))),
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
        (root / r).exists() for r in storage._manifest_fragment_roots(manifests[0])
    )
    assert all(
        (root / r).exists() for r in storage._manifest_fragment_roots(manifests[1])
    )
    assert not (root / ENTRIES_DATASET).exists()
    assert not (root / PAGE_IMAGES_DATASET).exists()
    old_store = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        old_store,
        handle(first),
        encode_restore_options(PhysicalRestore(CheckpointParticipant(0, 1))),
    )
    np.testing.assert_allclose(lookup(old_store, (1, 2)), 1)
    new_store = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        new_store,
        handle(second),
        encode_restore_options(PhysicalRestore(CheckpointParticipant(0, 1))),
    )
    np.testing.assert_allclose(lookup(new_store, (2, 3)), 2)
    elastic_store = se.EmbeddingStore(config())
    checkpoint.load_training_checkpoint(
        elastic_store,
        handle(second),
        encode_restore_options(modulo_restore(1, 0)),
    )
    np.testing.assert_allclose(lookup(elastic_store, (2, 3)), 2)


def test_elastic_restore_uses_bounded_native_partition_scan(tmp_path, monkeypatch):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    source = populated_store(range(96), 5)
    committed = SparseCheckpoint(storage).save_training_checkpoint(source, "bounded")

    def reject_materialized_keys(*args, **kwargs):
        raise AssertionError("elastic restore must not materialize active keys")

    monkeypatch.setattr(
        lance_backend._checkpoint_native,
        "active_entry_keys",
        reject_materialized_keys,
    )
    restored = se.EmbeddingStore(config())
    SparseCheckpoint(storage).load_training_checkpoint(
        restored,
        handle(committed),
        encode_restore_options(modulo_restore(3, 1)),
    )
    expected = np.asarray([key for key in range(96) if key % 3 == 1], dtype=np.uint64)
    np.testing.assert_allclose(lookup(restored, expected), 5)


def test_inference_weights_restore_from_physical_reference(tmp_path):
    root = tmp_path / "checkpoint"
    asset = tmp_path / "asset"
    asset.mkdir()
    storage = LocalLanceStorage.from_checkpoint_root(root)
    checkpoint = SparseCheckpoint(storage)
    checkpoint.save_training_checkpoint(
        populated_store((1, 2), 7), "training-name", weight_path=asset
    )
    restored = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_weight_path(asset).load_inference_weights(
        restored,
        encode_restore_options(modulo_restore(1, 0)),
    )
    np.testing.assert_allclose(lookup(restored, (1, 2)), 7)
    with pytest.raises(ValueError, match="requires an elastic mode"):
        LocalLanceStorage.from_weight_path(asset).load_inference_weights(
            se.EmbeddingStore(inference_config()),
            encode_restore_options(PhysicalRestore(CheckpointParticipant(0, 1))),
        )
    with pytest.raises(ValueError, match="optimizer enabled"):
        LocalLanceStorage.from_weight_path(asset).load_inference_weights(
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
        source, "training-config", weight_path=asset
    )
    inference_target = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_weight_path(asset).load_inference_weights(
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
        source, "inference-only", weight_path=asset
    )

    target = se.EmbeddingStore(
        se.EmbeddingConfig(
            dims=[4], thread_budget=1, key_dtype="uint64", enable_optimizer=False
        )
    )
    assert target.optimizer_enabled() is False
    LocalLanceStorage.from_weight_path(asset).load_inference_weights(
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
    target = se.EmbeddingStore(config(), optimizer_config=se.Adam(lr=0.01))
    SparseCheckpoint(storage).load_training_checkpoint(
        target,
        handle(locator),
        encode_restore_options(modulo_restore(1, 0)),
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
    target = se.EmbeddingStore(config(), optimizer_config=se.Adam(lr=0.01))
    SparseCheckpoint(storage).load_training_checkpoint(
        target,
        handle(locator),
        encode_restore_options(
            PhysicalRestore(
                CheckpointParticipant(0, 1),
                strategy=PhysicalRestoreStrategy.FAST_PAGE_CACHE,
            )
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
            storage.write_checkpoint_fragment(
                store,
                identity,
                CheckpointParticipant(participant_index, 4),
            )
        )
    locator = storage.commit_checkpoint(identity, fragments)

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
            storage.write_checkpoint_fragment(
                source,
                identity,
                CheckpointParticipant(participant_index, 2),
            )
        )
    locator = storage.commit_checkpoint(identity, fragments)
    storage.compact_checkpoint(handle(locator))

    for participant_index in range(2):
        restored = se.EmbeddingStore(config())
        storage.load_component_checkpoint(
            restored,
            handle(locator),
            encode_restore_options(
                PhysicalRestore(CheckpointParticipant(participant_index, 2))
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


def test_elastic_restore_does_not_materialize_global_active_keys(tmp_path, monkeypatch):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("duplicate", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragments = [
        storage.write_checkpoint_fragment(
            populated_store((index,), value),
            identity,
            CheckpointParticipant(index, 2),
        )
        for index, value in enumerate((1, 2))
    ]
    locator = storage.commit_checkpoint(identity, fragments)

    def reject_materialized_keys(*args, **kwargs):
        raise AssertionError("elastic restore must not materialize active keys")

    monkeypatch.setattr(
        lance_backend._checkpoint_native,
        "active_entry_keys",
        reject_materialized_keys,
    )
    restored = se.EmbeddingStore(config())
    storage.load_component_checkpoint(
        restored,
        handle(locator),
        encode_restore_options(modulo_restore(1, 0)),
    )
    assert restored.total_size() == 2


def test_elastic_restore_rejects_duplicate_active_keys(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    temporary_root = Path(tempfile.gettempdir())
    temporary_runs_before = set(
        temporary_root.glob("sparse_embedding_duplicate_validation-*")
    )
    identity = CheckpointSaveIdentity("duplicate", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragments = [
        storage.write_checkpoint_fragment(
            populated_store((7,), value),
            identity,
            CheckpointParticipant(index, 2),
        )
        for index, value in enumerate((1, 2))
    ]
    locator = storage.commit_checkpoint(identity, fragments)
    restored = se.EmbeddingStore(config())
    with pytest.raises(InvalidCheckpointFragment, match="duplicate active"):
        storage.load_component_checkpoint(
            restored,
            handle(locator),
            encode_restore_options(modulo_restore(1, 0)),
        )
    assert restored.total_size() == 0
    assert set(temporary_root.glob("sparse_embedding_duplicate_validation-*")) == (
        temporary_runs_before
    )


def test_elastic_restore_rejects_duplicate_keys_across_external_runs(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("duplicate-spill", "attempt", authority())
    storage.prepare_checkpoint(identity)
    first = np.arange(65536, dtype=np.uint64)
    second = np.arange(65535, 131072, dtype=np.uint64)
    fragments = [
        storage.write_checkpoint_fragment(
            populated_store(keys, index + 1),
            identity,
            CheckpointParticipant(index, 2),
        )
        for index, keys in enumerate((first, second))
    ]
    locator = storage.commit_checkpoint(identity, fragments)
    restored = se.EmbeddingStore(config())
    with pytest.raises(InvalidCheckpointFragment, match="duplicate active"):
        storage.load_component_checkpoint(
            restored,
            handle(locator),
            encode_restore_options(modulo_restore(1, 0)),
        )
    assert restored.total_size() == 0


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
                    "restore_kind": "elastic",
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
        checkpoint.load_training_checkpoint(restored, handle(locator), unknown_router)
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
        "restore_kind": "elastic",
        "key_dtype": "uint64",
        "partition_count": 1,
        "partition_id": 0,
        "sparse_assignment_name": "sparse_embedding.modulo",
        "sparse_assignment_parameters": {},
    }
    payload[field] = value
    encoded = msgpack.packb(dict(sorted(payload.items())), use_bin_type=True)
    with pytest.raises(ValueError, match=message):
        checkpoint.load_training_checkpoint(
            se.EmbeddingStore(config()), handle(locator), encoded
        )


def test_checkpoint_fragment_contains_no_physical_location(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("opaque", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragment = storage.write_checkpoint_fragment(
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
    old_fragment = storage.write_checkpoint_fragment(
        populated_store((1,), 1), old, CheckpointParticipant(0, 1)
    )
    current = CheckpointSaveIdentity(
        "name", "current", authority(2, "authority-current")
    )
    storage.prepare_checkpoint(current)
    current_fragment = storage.write_checkpoint_fragment(
        populated_store((1,), 2), current, CheckpointParticipant(0, 1)
    )
    delayed = CheckpointSaveIdentity(
        "delayed", "delayed", authority(1, "authority-delayed")
    )
    with pytest.raises(CheckpointConflictError, match="stale"):
        storage.prepare_checkpoint(delayed)
    with pytest.raises(CheckpointConflictError, match="stale"):
        storage.commit_checkpoint(old, (old_fragment,))
    storage.commit_checkpoint(current, (current_fragment,))
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
    storage.reclaim(older_than=timedelta(0))
    assert old_fragment_id in {
        item
        for row in lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET))
        .to_table()
        .to_pylist()
        for item in row["fragment_ids"]
    }
    storage.release_checkpoint_lease(lease)
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
        fragment = storage.write_checkpoint_fragment(
            populated_store((1,), value),
            identity,
            CheckpointParticipant(0, 1),
        )
        return storage.commit_checkpoint(identity, (fragment,))

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
        encode_restore_options(PhysicalRestore(CheckpointParticipant(0, 1))),
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 2)
    with pytest.raises(LookupError):
        storage.load_component_checkpoint(
            se.EmbeddingStore(config()),
            handle(old),
            encode_restore_options(PhysicalRestore(CheckpointParticipant(0, 1))),
        )


def test_reclaim_requires_published_reservation(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("checkpoint", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragment = storage.write_checkpoint_fragment(
        populated_store((1,), 1),
        identity,
        CheckpointParticipant(0, 1),
    )
    locator = storage.commit_checkpoint(identity, (fragment,))
    reservation = storage.reserve_cleanup("reservation", (locator.handle,), 1)

    with pytest.raises(CheckpointConflictError, match="not pending"):
        storage.cleanup_checkpoint(reservation, locator.handle)


def test_cleanup_reservation_owner_capability_is_validated(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("checkpoint", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragment = storage.write_checkpoint_fragment(
        populated_store((1,), 1), identity, CheckpointParticipant(0, 1)
    )
    committed = storage.commit_checkpoint(identity, (fragment,))
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
    fragment = first_storage.write_checkpoint_fragment(
        populated_store((1, 2), 4),
        identity,
        CheckpointParticipant(0, 1),
    )
    first_storage.finalize_checkpoint(identity)
    locator = first_storage.commit_checkpoint(
        identity,
        (fragment,),
    )

    restarted = LocalLanceStorage(root)
    assert (
        restarted.commit_checkpoint(
            identity,
            (),
        )
        == locator
    )
    restarted.abort_checkpoint(identity)
    restarted.load_component_checkpoint(
        se.EmbeddingStore(config()),
        locator.handle,
        encode_restore_options(PhysicalRestore(CheckpointParticipant(0, 1))),
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
        fragment = storage.write_checkpoint_fragment(
            populated_store((index + 1,), index + 1),
            identity,
            CheckpointParticipant(0, 1),
        )
        prepared.append((identity, fragment))
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(storage.commit_checkpoint, identity, (fragment,))
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
            encode_restore_options(PhysicalRestore(CheckpointParticipant(0, 1))),
        )
    assert target.total_size() == 0


def test_commit_rejects_mixed_participant_specifications(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("schema", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragments = (
        storage.write_checkpoint_fragment(
            populated_store((1,), 1, dims=(4,)),
            identity,
            CheckpointParticipant(0, 2),
        ),
        storage.write_checkpoint_fragment(
            populated_store((2,), 2, dims=(8,)),
            identity,
            CheckpointParticipant(1, 2),
        ),
    )
    with pytest.raises(InvalidCheckpointFragment, match="specification"):
        storage.commit_checkpoint(identity, fragments)


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


def test_compact_checkpoint_locator_is_explicit_and_idempotent(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    locator = checkpoint.save_training_checkpoint(
        populated_store((1,), 1), "compact-me"
    )
    storage.compact_checkpoint(handle(locator))
    storage.compact_checkpoint(handle(locator))
    manifest = storage._resolve_manifest(_decode_checkpoint_handle(handle(locator))[1])
    assert manifest["compacted_version"] == 1
    assert len(manifest["fragment_ids"]) == 1


def test_inference_storage_requires_a_published_weight_path(tmp_path):
    assert (
        "bound_locator" not in inspect.signature(LocalLanceStorage.__init__).parameters
    )
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    store = se.EmbeddingStore(config())
    options = encode_restore_options(
        ElasticRestore(PartitionSpec.modulo(), 1, 0, key_dtype="uint64")
    )
    with pytest.raises(RuntimeError, match="from_weight_path"):
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
        populated_store((1,), 1), "series", weight_path=first_asset
    )
    checkpoint.save_training_checkpoint(
        populated_store((1,), 2), "series", weight_path=second_asset
    )

    for asset, expected in ((first_asset, 1), (second_asset, 2)):
        restored = se.EmbeddingStore(inference_config())
        LocalLanceStorage.from_weight_path(asset).load_inference_weights(
            restored,
            encode_restore_options(modulo_restore(1, 0)),
        )
        np.testing.assert_allclose(lookup(restored, (1,)), expected)

    storage.compact_checkpoint(handle(first))
    restored_after_compaction = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_weight_path(first_asset).load_inference_weights(
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
        fragment = storage.write_checkpoint_fragment(
            populated_store((1,), value),
            identity,
            CheckpointParticipant(0, 1),
        )
        locator = storage.commit_checkpoint(identity, (fragment,))
        storage.write_weight_reference(asset, locator.handle)
        storage.acknowledge_checkpoint(identity)
        return locator

    first = save(1, first_asset)
    second = save(2, second_asset)
    with pytest.raises(CheckpointConflictError, match="another identity"):
        storage.write_weight_reference(first_asset, second.handle)

    storage.reclaim(older_than=timedelta(0))
    restored = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_weight_path(first_asset).load_inference_weights(
        restored,
        encode_restore_options(modulo_restore(1, 0)),
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 1)

    storage.release_reference(first.handle)
    with pytest.raises(LookupError, match="missing or ambiguous"):
        LocalLanceStorage.from_weight_path(first_asset).load_inference_weights(
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
            storage.write_checkpoint_fragment(
                source,
                identity,
                CheckpointParticipant(participant_index, 4),
            )
        )
    locator = storage.commit_checkpoint(identity, fragments)
    storage.write_weight_reference(asset, locator.handle)
    storage.acknowledge_checkpoint(identity)

    for partition_id in range(7):
        restored = se.EmbeddingStore(inference_config())
        LocalLanceStorage.from_weight_path(asset).load_inference_weights(
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
        LocalLanceStorage.from_weight_path(missing)

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / ".sparse_embedding_reference").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="unsupported format"):
        LocalLanceStorage.from_weight_path(malformed)

    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    (ambiguous / ".sparse_embedding_reference").write_bytes(b"invalid")
    (ambiguous / ".sparse_embedding_reference.tmp").write_bytes(b"invalid")
    with pytest.raises(LookupError, match="ambiguous"):
        LocalLanceStorage.from_weight_path(ambiguous)


def test_inference_weight_load_is_read_only_and_does_not_create_leases(tmp_path):
    root = tmp_path / "checkpoint"
    asset = tmp_path / "asset"
    asset.mkdir()
    storage = LocalLanceStorage.from_checkpoint_root(root)
    checkpoint = SparseCheckpoint(storage)
    checkpoint.save_training_checkpoint(
        populated_store((1,), 3), "readonly", weight_path=asset
    )
    before = lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET)).version
    restored = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_weight_path(asset).load_inference_weights(
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
            populated_store((1,), 5), "reconcile", weight_path=asset
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
    fragment = storage.write_checkpoint_fragment(
        populated_store((1,), 7), identity, CheckpointParticipant(0, 1)
    )
    locator = storage.commit_checkpoint(identity, (fragment,))
    storage.write_weight_reference(asset, locator.handle)

    def fail_acknowledge(_identity):
        raise OSError("injected acknowledge failure")

    monkeypatch.setattr(storage, "acknowledge_checkpoint", fail_acknowledge)
    with pytest.raises(OSError, match="injected acknowledge failure"):
        storage.acknowledge_checkpoint(identity)

    restarted = LocalLanceStorage.from_checkpoint_root(root)
    restarted.acknowledge_checkpoint(identity)
    restored = se.EmbeddingStore(inference_config())
    LocalLanceStorage.from_weight_path(asset).load_inference_weights(
        restored,
        encode_restore_options(modulo_restore(1, 0)),
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 7)


def test_partition_spec_declarative_contracts():
    modulo_spec = PartitionSpec.modulo()
    assert modulo_spec.name == "sparse_embedding.modulo"
    assert modulo_spec.parameters == {}
    assert modulo_spec.sparse_assignment_name == modulo_spec.name
    assert modulo_spec.sparse_assignment_parameters == modulo_spec.parameters

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


def test_restore_plan_polymorphism_and_validation():
    strategy = se.ModuloPartition()
    plan = ElasticRestore(
        spec=strategy.spec,
        partition_count=4,
        partition_id=1,
        key_dtype="uint64",
    )
    assert plan.spec == PartitionSpec.modulo()
    assert plan.partition_count == 4
    assert plan.partition_id == 1
    assert plan.partition_count == 4
    assert plan.partition_id == 1
    assert plan.assignment_spec == PartitionSpec.modulo()

    encoded = plan.encode()
    assert isinstance(encoded, bytes)
    assert encoded == encode_restore_options(plan)

    decoded = RestorePlan.decode(encoded)
    assert isinstance(decoded, ElasticRestore)
    assert decoded == plan

    factory_plan = RestorePlan.elastic(strategy.spec, 4, 1, "uint64")
    assert factory_plan == plan

    # Strict bounds and dtype assertions without fallback
    with pytest.raises(ValueError):
        ElasticRestore(strategy.spec, 4, 4)

    with pytest.raises(ValueError):
        ElasticRestore(strategy.spec, 0, 0)

    with pytest.raises(TypeError):
        ElasticRestore(strategy.spec, 4.0, 0)  # type: ignore

    with pytest.raises(ValueError):
        ElasticRestore(strategy.spec, 4, 0, key_dtype="invalid_dtype")


def test_physical_restore_factory_and_properties():
    participant = CheckpointParticipant(0, 2)
    plan = PhysicalRestore(participant)
    assert plan.participant == participant
    assert plan.strategy == PhysicalRestoreStrategy.RECONSTRUCT_ENTRIES
    assert plan.key_dtype == "uint64"

    encoded = plan.encode()
    assert RestorePlan.decode(encoded) == plan

    factory_plan = RestorePlan.physical(participant)
    assert factory_plan == plan

    with pytest.raises(TypeError):
        PhysicalRestore(participant="invalid")  # type: ignore

    with pytest.raises(TypeError):
        PhysicalRestore()  # type: ignore


def test_sparse_checkpoint_load_with_typed_restore_plan(tmp_path):
    root = tmp_path / "typed_checkpoint"
    storage = LocalLanceStorage.from_checkpoint_root(root)
    checkpoint = SparseCheckpoint(storage)

    store = populated_store((1, 2, 3), 9.0)
    committed = checkpoint.save_training_checkpoint(store, "typed_ckpt")

    restore_store = se.EmbeddingStore(config())
    typed_plan = RestorePlan.elastic(PartitionSpec.modulo(), 1, 0, "uint64")
    checkpoint.load_training_checkpoint(restore_store, committed.handle, typed_plan)
    np.testing.assert_allclose(lookup(restore_store, (1, 2, 3)), 9.0)

    with pytest.raises(TypeError):
        checkpoint.load_training_checkpoint(
            restore_store,
            committed.handle,
            "unsupported_restore_type",  # type: ignore
        )
