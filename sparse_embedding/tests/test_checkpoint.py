from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import lance
import numpy as np
import pytest

import sparse_embedding as se
from sparse_embedding.checkpoint import (
    ElasticRestore,
    LocalLanceStorage,
    PhysicalRestore,
    SparseCheckpoint,
)
from sparse_embedding.checkpoint.spi import (
    CheckpointCommitAuthority,
    CheckpointConflictError,
    CheckpointFragment,
    CheckpointLeaseRequest,
    CheckpointParticipant,
    CheckpointSaveIdentity,
    InvalidCheckpointFragment,
    StorageBudgetExceeded,
)


def config(dims=(4,)):
    return se.EmbeddingConfig(dims=list(dims), thread_budget=1, value_dtype="float32")


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


def lookup(store, keys, dim=4):
    values, _ = store.lookup(np.asarray(keys, dtype=np.uint64), dim=dim)
    return np.from_dlpack(values).reshape(-1, dim)


def test_user_facade_roundtrip(tmp_path):
    checkpoint = SparseCheckpoint(LocalLanceStorage(tmp_path / "checkpoint"))
    source = populated_store((1, 2), 3)
    locator = checkpoint.save(source, "step-1")
    restored = se.EmbeddingStore(config())
    checkpoint.restore(restored, locator)
    np.testing.assert_allclose(lookup(restored, (1, 2)), 3)


def test_materialized_upsert_and_exact_version_restore(tmp_path):
    root = tmp_path / "checkpoint"
    checkpoint = SparseCheckpoint(LocalLanceStorage(root))
    first = checkpoint.save(populated_store((1, 2), 1), "first")
    second = checkpoint.save(populated_store((2, 3), 2), "second")
    current = lance.dataset(str(root / "entries")).to_table().to_pylist()
    assert {(row["key"], row["row_state"]) for row in current} == {
        (1, "deleted"),
        (2, "active"),
        (3, "active"),
    }
    old_store = se.EmbeddingStore(config())
    checkpoint.restore(old_store, first)
    np.testing.assert_allclose(lookup(old_store, (1, 2)), 1)
    new_store = se.EmbeddingStore(config())
    checkpoint.restore(new_store, second)
    np.testing.assert_allclose(lookup(new_store, (2, 3)), 2)
    elastic_store = se.EmbeddingStore(config())
    checkpoint.restore(
        elastic_store,
        second,
        ElasticRestore(lambda keys, count: keys % count, 1, 0),
    )
    np.testing.assert_allclose(lookup(elastic_store, (2, 3)), 2)


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

    partition_calls = 0

    def partition_fn(keys, partition_count):
        nonlocal partition_calls
        partition_calls += 1
        before = keys.copy()
        assert keys.dtype == np.uint64
        result = (keys * 7 + 3) % partition_count
        np.testing.assert_array_equal(keys, before)
        return result

    seen = set()
    all_keys = np.arange(100, dtype=np.uint64)
    for partition_id in range(7):
        restored = se.EmbeddingStore(config())
        storage.restore_checkpoint(
            restored,
            locator,
            ElasticRestore(partition_fn, 7, partition_id),
        )
        expected = all_keys[partition_fn(all_keys, 7) == partition_id]
        assert restored.total_size() == expected.size
        assert seen.isdisjoint(expected.tolist())
        seen.update(expected.tolist())
        if expected.size:
            np.testing.assert_allclose(
                lookup(restored, expected),
                np.repeat(expected[:, None].astype(np.float32), 4, axis=1),
            )
    assert seen == set(range(100))
    assert partition_calls >= 14


def test_duplicate_key_across_fragments_is_rejected(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    identity = CheckpointSaveIdentity("duplicate", "attempt", authority())
    storage.prepare_checkpoint(identity)
    fragments = [
        storage.write_checkpoint_fragment(
            populated_store((1,), value),
            identity,
            CheckpointParticipant(index, 2),
        )
        for index, value in enumerate((1, 2))
    ]
    with pytest.raises(InvalidCheckpointFragment, match="duplicate"):
        storage.commit_checkpoint(identity, fragments)


def test_partition_function_validation_precedes_store_mutation(tmp_path):
    checkpoint = SparseCheckpoint(LocalLanceStorage(tmp_path / "checkpoint"))
    locator = checkpoint.save(populated_store((1, 2), 1), "partition")
    restored = se.EmbeddingStore(config())

    def invalid(keys, partition_count):
        return np.zeros(len(keys) + 1, dtype=np.int64)

    with pytest.raises(ValueError, match="length"):
        checkpoint.restore(restored, locator, ElasticRestore(invalid, 2, 0))
    assert restored.total_size() == 0


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


def test_repeated_updates_keep_logical_rows_bounded(tmp_path):
    root = tmp_path / "checkpoint"
    checkpoint = SparseCheckpoint(LocalLanceStorage(root))
    for version in range(30):
        checkpoint.save(populated_store(range(64), version), f"v{version}")
    entries = lance.dataset(str(root / "entries")).to_table()
    pages = lance.dataset(str(root / "page_images")).to_table()
    assert entries.num_rows == 64
    assert pages.num_rows == 1
    names = {path.name for path in root.rglob("*")}
    assert not names.intersection(
        {"payload.pkl", "metadata.json", "fragment.json", "artifact.json"}
    )


def test_reclaim_bounds_physical_versions_after_repeated_updates(tmp_path):
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    checkpoint = SparseCheckpoint(storage)
    for version in range(20):
        checkpoint.save(populated_store(range(32), version), f"v{version}")
    assert len(lance.dataset(str(root / "entries")).versions()) > 10
    assert len(lance.dataset(str(root / "page_images")).versions()) > 10
    storage.reclaim(older_than=timedelta(0))
    assert len(lance.dataset(str(root / "entries")).versions()) == 1
    assert len(lance.dataset(str(root / "page_images")).versions()) == 1


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
    manifest = lance.dataset(str(root / "snapshot_manifest"))
    manifest.tags.create(storage._fence_tag(current), manifest.version)

    storage.prepare_checkpoint(current)

    tags = manifest.tags.list()
    assert storage._authority_tag(current) in tags
    assert storage._authority_tag(old) not in tags


def test_retention_lease_pins_exact_version(tmp_path):
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    checkpoint = SparseCheckpoint(storage)
    old = checkpoint.save(populated_store((1,), 1), "old")
    lease = storage.acquire_checkpoint_lease(
        old, CheckpointLeaseRequest("trainer", 60, "request-1")
    )
    checkpoint.save(populated_store((1,), 2), "new")
    manifest_rows = (
        lance.dataset(str(root / "snapshot_manifest")).to_table().to_pylist()
    )
    old_entries_version = next(
        row["entries_lance_version"]
        for row in manifest_rows
        if row["checkpoint_name"] == old.checkpoint_name
        and row["checkpoint_version"] == old.checkpoint_version
    )
    storage.reclaim(older_than=timedelta(0))
    lance.dataset(str(root / "entries"), version=old_entries_version)
    storage.release_checkpoint_lease(lease)
    storage.reclaim(older_than=timedelta(0))
    with pytest.raises(Exception):
        lance.dataset(str(root / "entries"), version=old_entries_version)


def test_reclaim_reservation_removes_only_released_reference(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")

    def save(name, value, reference_id):
        identity = CheckpointSaveIdentity(name, name, authority())
        storage.prepare_checkpoint(identity)
        fragment = storage.write_checkpoint_fragment(
            populated_store((1,), value),
            identity,
            CheckpointParticipant(0, 1),
        )
        return storage.commit_checkpoint(
            identity, (fragment,), reference_id=reference_id
        )

    old = save("old", 1, "reference-old")
    current = save("current", 2, "reference-current")
    reservation = storage.reserve_cleanup("reservation", (old,), 1)
    storage.commit_cleanup(reservation)
    storage.cleanup_checkpoint(reservation, old, "reference-old")
    storage.cleanup_checkpoint(reservation, old, "reference-old")

    restored = se.EmbeddingStore(config())
    storage.restore_checkpoint(
        restored,
        current,
        PhysicalRestore(CheckpointParticipant(0, 1)),
    )
    np.testing.assert_allclose(lookup(restored, (1,)), 2)
    with pytest.raises(Exception):
        storage.restore_checkpoint(
            se.EmbeddingStore(config()),
            old,
            PhysicalRestore(CheckpointParticipant(0, 1)),
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
    locator = storage.commit_checkpoint(identity, (fragment,), reference_id="reference")
    reservation = storage.reserve_cleanup("reservation", (locator,), 1)

    with pytest.raises(CheckpointConflictError, match="not pending"):
        storage.cleanup_checkpoint(reservation, locator, "reference")


def test_active_lease_blocks_reclaim_reservation(tmp_path):
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    locator = checkpoint.save(populated_store((1,), 1), "leased")
    lease = storage.acquire_checkpoint_lease(
        locator, CheckpointLeaseRequest("reader", 60, "request")
    )
    assert (
        storage.acquire_checkpoint_lease(
            locator, CheckpointLeaseRequest("reader", 60, "request")
        )
        == lease
    )
    with pytest.raises(CheckpointConflictError, match="unleased"):
        storage.reserve_cleanup("reservation", (locator,), 1)
    renewed = storage.renew_checkpoint_lease(lease, 60)
    assert renewed.generation == lease.generation + 1
    with pytest.raises(CheckpointConflictError, match="generation CAS"):
        storage.renew_checkpoint_lease(lease, 60)
    storage.release_checkpoint_lease(renewed)


def test_committed_attempt_is_idempotent_and_abortable_after_storage_restart(tmp_path):
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
        reference_id="coordinator:embedding:snapshot",
    )

    restarted = LocalLanceStorage(root)
    assert (
        restarted.commit_checkpoint(
            identity,
            (),
            reference_id="coordinator:embedding:snapshot",
        )
        == locator
    )
    restarted.abort_checkpoint(identity)
    with pytest.raises(LookupError):
        restarted.restore_checkpoint(
            se.EmbeddingStore(config()),
            locator,
            PhysicalRestore(CheckpointParticipant(0, 1)),
        )

    checkpoint = SparseCheckpoint(LocalLanceStorage(root))
    replacement = checkpoint.save(populated_store((3, 4), 8), "replacement")
    assert replacement.checkpoint_version == 1


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
    assert {locator.checkpoint_version for locator in locators} == {1, 2}


def test_manifest_records_table_specification_and_creation_time(tmp_path):
    root = tmp_path / "checkpoint"
    checkpoint = SparseCheckpoint(LocalLanceStorage(root))
    checkpoint.save(populated_store((1,), 1, dims=(4, 8)), "specification")
    manifest = lance.dataset(str(root / "snapshot_manifest")).to_table().to_pylist()
    assert len(manifest) == 1
    assert manifest[0]["dimensions"] == [4, 8]
    assert manifest[0]["embedding_dtype"] == "float32"
    assert manifest[0]["optimizer_dtype"] == "float32"
    assert manifest[0]["created_at"] is not None


def test_restore_rejects_incompatible_target_specification_before_mutation(tmp_path):
    checkpoint = SparseCheckpoint(LocalLanceStorage(tmp_path / "checkpoint"))
    locator = checkpoint.save(populated_store((1,), 1, dims=(4,)), "source")
    target = se.EmbeddingStore(config((8,)))
    with pytest.raises(ValueError, match="target does not match"):
        checkpoint.restore(target, locator)
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
    baseline = SparseCheckpoint(LocalLanceStorage(root)).save(source, "baseline")
    current_size = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    storage = LocalLanceStorage(root, max_storage_bytes=current_size + 1)
    checkpoint = SparseCheckpoint(storage)
    with pytest.raises(StorageBudgetExceeded):
        checkpoint.save(source, "over-budget")
    manifests = lance.dataset(str(root / "snapshot_manifest")).to_table().to_pylist()
    assert [row["checkpoint_name"] for row in manifests] == [baseline.checkpoint_name]
    current = lance.dataset(str(root / "entries")).to_table().to_pylist()
    assert {(row["key"], row["row_state"]) for row in current} == {(1, "active")}
