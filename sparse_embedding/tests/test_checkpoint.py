#!/usr/bin/env python3
"""
Tests for score policies, async lookup/assign, and distributed checkpoint
save/load with hash-based shard filtering via the LanceCheckpointBackend.
"""

import asyncio
import os
import shutil
import tempfile
from time import perf_counter

import lance
import numpy as np
import pytest

import sparse_embedding
import sparse_embedding as se
from sparse_embedding.checkpoint import (
    CheckpointBackend,
    get_checkpointer,
    register_backend,
)
from sparse_embedding.checkpoint.lance_backend import _FORMAT_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LANCE_URI = "/tmp/test_lance_distributed_ckpt"
NUM_ORIG_KEYS = 1000
# Two feature groups with distinct dims; group 0 uses dim=8, group 1 uses dim=16.
# Keys for group 1 are namespace-separated by embedding slot bits into high bits.
DIMS = {0: 8, 1: 16}  # slot_id -> dim, for test bookkeeping only
UNIQUE_DIMS = sorted(set(DIMS.values()))  # [8, 16] — what the store actually needs


def make_cfg(dims, threads: int = 4):
    return se.EmbeddingConfig(
        dims=dims if isinstance(dims, list) else sorted(set(dims.values())),
        num_threads=threads,
        value_dtype="float32",
    )


def lookup_np(store, keys: np.ndarray, dim: int) -> np.ndarray:
    vals_dlp, _ = store.lookup(keys, dim=dim)
    return np.from_dlpack(vals_dlp).reshape(-1, dim)


def slotted(keys: np.ndarray, slot_id: int) -> np.ndarray:
    return keys | (np.uint64(slot_id) << np.uint64(48))


def make_ground_truth(dims: dict = DIMS, num_keys: int = NUM_ORIG_KEYS, seed: int = 42):
    rng = np.random.default_rng(seed)
    truth = {}
    for slot_id, dim in dims.items():
        raw_keys = np.arange(num_keys, dtype=np.uint64)
        s_keys = slotted(raw_keys, slot_id)
        embeds = rng.standard_normal((num_keys, dim)).astype(np.float32)
        truth[slot_id] = (s_keys, embeds)
    return truth


def verify_shard_contents(store, truth, dims, partitioner, shard_id):
    for slot_id, (s_keys, embeds) in truth.items():
        dim = dims[slot_id]
        mask = np.from_dlpack(partitioner.filter(s_keys, shard_id))
        expected_keys = s_keys[mask]
        expected_embeds = embeds[mask]

        if len(expected_keys) == 0:
            continue

        vals_dlp, _ = store.lookup(expected_keys, dim=dim)
        got = np.from_dlpack(vals_dlp).reshape(-1, dim)
        np.testing.assert_allclose(
            got,
            expected_embeds,
            atol=1e-5,
            err_msg=f"slot={slot_id} shard={shard_id} mismatch",
        )


# ---------------------------------------------------------------------------
# Distributed save / load helpers
# ---------------------------------------------------------------------------


async def distributed_save(
    uri: str,
    num_shards: int,
    truth: dict,
    slot_id_to_dim: dict,  # {slot_id: dim}, used only for test bookkeeping
    backend,
    optimizer_config=None,
):
    partitioner = se.ModuloPartitioner(num_shards=num_shards)
    store_dims = sorted(set(slot_id_to_dim.values()))

    init_store = se.EmbeddingStore(make_cfg(store_dims))
    await asyncio.to_thread(backend.init_dataset, init_store, uri)

    async def worker_export(shard_id: int) -> list:
        store = se.EmbeddingStore(
            make_cfg(store_dims),
            **({"optimizer_config": optimizer_config} if optimizer_config else {}),
        )
        assigns = [
            store.async_assign(s_keys[mask], embeds[mask])
            for slot_id, (s_keys, embeds) in truth.items()
            if len(mask := np.from_dlpack(partitioner.filter(s_keys, shard_id))) > 0
            and mask.any()
        ]
        if assigns:
            await asyncio.gather(*assigns)
        return await asyncio.to_thread(
            backend.export_fragments,
            store,
            uri,
            snapshot_id=1,
            worker_id=shard_id,
            worker_count=num_shards,
            shard_count=num_shards,
        )

    all_metas = await asyncio.gather(*[worker_export(s) for s in range(num_shards)])
    await asyncio.to_thread(backend.commit_fragments, uri, list(all_metas))


async def load_shard(
    uri: str,
    shard_id: int,
    partitioner,
    backend,
    dims: list = UNIQUE_DIMS,
    optimizer_config=None,
) -> se.EmbeddingStore:
    store = se.EmbeddingStore(
        make_cfg(dims),
        **({"optimizer_config": optimizer_config} if optimizer_config else {}),
    )
    await asyncio.to_thread(
        backend.load,
        store,
        uri,
        filter_fn=partitioner,
        distributed=True,
        target_shard_id=shard_id,
    )
    return store


async def load_all_shards(uri, num_shards, partitioner, backend, **kwargs):
    return await asyncio.gather(
        *[load_shard(uri, s, partitioner, backend, **kwargs) for s in range(num_shards)]
    )


def verify_all_shards(stores, truth, dims, partitioner, num_keys=NUM_ORIG_KEYS):
    all_keys_seen = {slot_id: set() for slot_id in dims}
    for shard_id, store in enumerate(stores):
        verify_shard_contents(store, truth, dims, partitioner, shard_id)
        for slot_id in dims:
            s_keys_all = truth[slot_id][0]
            mask = np.from_dlpack(partitioner.filter(s_keys_all, shard_id))
            my_keys = s_keys_all[mask]
            if len(my_keys) > 0:
                overlap = all_keys_seen[slot_id].intersection(my_keys.tolist())
                assert not overlap, (
                    f"Duplicate keys in shard {shard_id} slot {slot_id}: {overlap}"
                )
                all_keys_seen[slot_id].update(my_keys.tolist())

    for slot_id in dims:
        assert len(all_keys_seen[slot_id]) == num_keys, (
            f"slot {slot_id}: expected {num_keys} unique keys, "
            f"got {len(all_keys_seen[slot_id])}"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lance_backend():
    pytest.importorskip("lance", reason="pylance not installed")
    if os.path.exists(LANCE_URI):
        shutil.rmtree(LANCE_URI)
    yield get_checkpointer("lance")
    if os.path.exists(LANCE_URI):
        shutil.rmtree(LANCE_URI)


@pytest.fixture
def lance_tmp_uri(tmp_path):
    pytest.importorskip("lance", reason="pylance not installed")
    uri = str(tmp_path / "lance_ckpt")
    yield uri, get_checkpointer("lance")


# ---------------------------------------------------------------------------
# Unit tests: score policies, basic assign/lookup
# ---------------------------------------------------------------------------


def test_score_policies():
    for policy in ["none", "lfu", "lru", "manual"]:
        cfg = make_cfg({0: 4})
        store = se.EmbeddingStore(cfg, score_policy=policy)
        assert store.total_size() == 0

    try:
        se.EmbeddingStore(make_cfg({0: 4}), score_policy="badname")
        assert False, "should throw"
    except Exception:
        pass


def test_assign_and_lookup_lfu():
    store = se.EmbeddingStore(make_cfg({0: 4}), score_policy="lfu")
    keys = np.array([0, 1, 2], dtype=np.uint64)
    vals = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    store.assign(keys, vals)
    assert store.total_size() == 3
    result = lookup_np(store, keys, 4)
    np.testing.assert_allclose(result, vals, atol=1e-5)
    result2 = lookup_np(store, keys, 4)
    np.testing.assert_allclose(result2, vals, atol=1e-5)


def test_assign_and_lookup_no_score():
    store = se.EmbeddingStore(make_cfg({0: 4}), score_policy="none")
    keys = np.array([0, 1, 2], dtype=np.uint64)
    vals = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    store.assign(keys, vals)
    result = lookup_np(store, keys, 4)
    np.testing.assert_allclose(result, vals, atol=1e-5)


def test_distributed_checkpoint_restores_optimizer_state_by_dimension(
    lance_tmp_uri,
):
    uri, backend = lance_tmp_uri
    dimensions = [4, 8]
    keys = np.array([17, 29], dtype=np.uint64)
    optimizer = se.Adam(lr=1e-3)

    def create_store():
        return se.EmbeddingStore(
            make_cfg(dimensions, threads=2),
            initializer_config=se.Zeros(),
            optimizer_config=optimizer,
        )

    source = create_store()
    for dimension in dimensions:
        source.lookup_or_insert(keys, dim=dimension)
    for step in range(2):
        for dimension in dimensions:
            gradients = np.arange(
                1,
                keys.size * dimension + 1,
                dtype=np.float32,
            ).reshape(keys.size, dimension)
            source.accumulate_gradients(
                keys,
                gradients * np.float32(step + 1),
            )
        source.optimizer_step()

    backend.init_dataset(source, uri)
    fragments = backend.export_fragments(
        source, uri, snapshot_id=1, worker_id=0, worker_count=1, shard_count=1
    )
    backend.commit_fragments(uri, [fragments])

    restored = create_store()
    backend.load(
        restored,
        uri,
        filter_fn=se.ModuloPartitioner(num_shards=1),
        distributed=True,
        target_shard_id=0,
    )

    for dimension in dimensions:
        gradients = np.zeros((keys.size, dimension), dtype=np.float32)
        source.accumulate_gradients(keys, gradients)
        restored.accumulate_gradients(keys, gradients)
    source.optimizer_step()
    restored.optimizer_step()

    for dimension in dimensions:
        expected = lookup_np(source, keys, dimension)
        actual = lookup_np(restored, keys, dimension)
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    with pytest.raises(ValueError, match="dimension has no physical table"):
        restored.assign_opt_state(
            np.empty((0,), dtype=np.uint64),
            np.empty((0,), dtype=np.float32),
            16,
            np.empty((0,), dtype=np.uint32),
        )


def test_checkpoint_load_with_resharding():
    ckpt_dir = "/tmp/test_resharding_ckpt_dual"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    config = sparse_embedding.EmbeddingConfig(
        dims=[8], num_threads=2, value_dtype="float32"
    )
    store = sparse_embedding.EmbeddingStore(config)
    try:
        store.set_checkpoint_factories("lance")
    except Exception:
        pytest.skip("Lance checkpoint backend not configured")
        return

    keys = np.arange(100, 200, dtype=np.uint64)
    keys_slotted = keys.copy()
    keys_slotted[:50] |= np.uint64(0) << 48
    keys_slotted[50:] |= np.uint64(1) << 48

    def save_cluster_checkpoints(num_shards, base_dir):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir)
        for shard_id in range(num_shards):
            store.clear()
            shard_keys = keys_slotted[(keys % num_shards) == shard_id]
            if len(shard_keys) > 0:
                store.lookup_or_insert(shard_keys, dim=8)
            shard_ckpt_dir = os.path.join(base_dir, f"shard_{shard_id}")
            store.save_checkpoint(shard_ckpt_dir)

    def load_cluster_checkpoints(
        from_shards, to_shards, target_shard_id, base_dir, filter_fn
    ):
        store.clear()
        for shard_id in range(from_shards):
            shard_ckpt_dir = os.path.join(base_dir, f"shard_{shard_id}")
            store.load_checkpoint(
                shard_ckpt_dir,
                filter_fn=filter_fn,
                target_shard_id=target_shard_id,
                clear=False,
            )
        return store.total_size()

    ckpt_2_shards = "/tmp/ckpt_2_shards"
    save_cluster_checkpoints(2, ckpt_2_shards)

    partitioner_5 = sparse_embedding.ModuloPartitioner(num_shards=5)
    loaded_size = load_cluster_checkpoints(2, 5, 2, ckpt_2_shards, partitioner_5)
    assert loaded_size == 20, f"2->5 expansion expected 20, got {loaded_size}"

    ckpt_7_shards = "/tmp/ckpt_7_shards"
    save_cluster_checkpoints(7, ckpt_7_shards)

    def python_mod_3_filter(k_arr_capsule):
        k_arr = np.from_dlpack(k_arr_capsule)
        orig_keys = k_arr & ((np.uint64(1) << 48) - np.uint64(1))
        return (orig_keys % np.uint64(3)).astype(np.uint32)

    loaded_size = load_cluster_checkpoints(7, 3, 1, ckpt_7_shards, python_mod_3_filter)
    assert loaded_size == 34, f"7->3 contraction expected 34, got {loaded_size}"

    ckpt_4_shards = "/tmp/ckpt_4_shards"
    save_cluster_checkpoints(4, ckpt_4_shards)
    hash_partitioner_6 = sparse_embedding.HashPartitioner(num_shards=6)
    load_cluster_checkpoints(4, 6, 3, ckpt_4_shards, hash_partitioner_6)


# ---------------------------------------------------------------------------
# Async assign/lookup
# ---------------------------------------------------------------------------


def test_async_assign_and_lookup_correctness():
    async def run():
        cfg = make_cfg([4, 8])
        store = se.EmbeddingStore(cfg)
        rng = np.random.default_rng(7)

        keys0 = np.arange(500, dtype=np.uint64)
        vals0 = rng.standard_normal((500, 4)).astype(np.float32)
        keys1 = slotted(np.arange(500, dtype=np.uint64), 1)
        vals1 = rng.standard_normal((500, 8)).astype(np.float32)

        await asyncio.gather(
            store.async_assign(keys0, vals0),
            store.async_assign(keys1, vals1),
        )

        assert store.total_size() == 1000

        res0_dlp, res1_dlp = await asyncio.gather(
            store.async_lookup(keys0, dim=4),
            store.async_lookup(keys1, dim=8),
        )
        got0 = np.from_dlpack(res0_dlp[0]).reshape(-1, 4)
        got1 = np.from_dlpack(res1_dlp[0]).reshape(-1, 8)

        np.testing.assert_allclose(got0, vals0, atol=1e-5)
        np.testing.assert_allclose(got1, vals1, atol=1e-5)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Distributed checkpoint: embedding save / load / reshard
# ---------------------------------------------------------------------------


def test_distributed_save_and_load_single_shard(lance_backend):
    truth = make_ground_truth()
    num_shards = 3

    asyncio.run(distributed_save(LANCE_URI, num_shards, truth, DIMS, lance_backend))

    partitioner = se.ModuloPartitioner(num_shards=num_shards)
    store = asyncio.run(load_shard(LANCE_URI, 0, partitioner, lance_backend))
    verify_shard_contents(store, truth, DIMS, partitioner, shard_id=0)


def test_distributed_save_and_load_all_shards_concurrent(lance_backend):
    truth = make_ground_truth()
    num_shards = 4

    asyncio.run(distributed_save(LANCE_URI, num_shards, truth, DIMS, lance_backend))

    partitioner = se.ModuloPartitioner(num_shards=num_shards)
    stores = asyncio.run(
        load_all_shards(LANCE_URI, num_shards, partitioner, lance_backend)
    )

    assert len(stores) == num_shards
    for shard_id, store in enumerate(stores):
        verify_shard_contents(store, truth, DIMS, partitioner, shard_id)


def test_partitioned_lance_manifest_roundtrip_with_parallel_workers(lance_tmp_uri):
    import lance

    uri, backend = lance_tmp_uri
    worker_count = 8
    snapshot_id = 19
    dim = 8
    keys = np.arange(4096, dtype=np.uint64)
    values = np.arange(keys.size * dim, dtype=np.float32).reshape(keys.size, dim)

    init_store = se.EmbeddingStore(make_cfg([dim]))
    backend.init_dataset(init_store, uri)

    async def export_worker(worker_id: int):
        worker_store = se.EmbeddingStore(make_cfg([dim]))
        mask = (keys % np.uint64(worker_count)) == worker_id
        worker_store.assign(keys[mask], values[mask])
        return await asyncio.to_thread(
            backend.export_fragments,
            worker_store,
            uri,
            snapshot_id=snapshot_id,
            worker_id=worker_id,
            worker_count=worker_count,
            shard_count=worker_count,
        )

    async def export_all_workers():
        return await asyncio.gather(
            *(export_worker(worker_id) for worker_id in range(worker_count))
        )

    artifacts = asyncio.run(export_all_workers())
    backend.commit_fragments(uri, artifacts)

    next_epoch = snapshot_id + 1

    async def export_next_epoch_worker(worker_id: int):
        worker_store = se.EmbeddingStore(make_cfg([dim]))
        mask = (keys % np.uint64(worker_count)) == worker_id
        worker_store.assign(keys[mask], values[mask] + np.float32(1))
        return await asyncio.to_thread(
            backend.export_fragments,
            worker_store,
            uri,
            snapshot_id=next_epoch,
            worker_id=worker_id,
            worker_count=worker_count,
            shard_count=worker_count,
        )

    async def export_next_epoch():
        return await asyncio.gather(
            *(export_next_epoch_worker(worker_id) for worker_id in range(worker_count))
        )

    backend.commit_fragments(uri, asyncio.run(export_next_epoch()))

    manifest = lance.dataset(f"{uri}/snapshot_manifest").to_table().to_pylist()
    assert len(manifest) == 2
    first_manifest = next(row for row in manifest if row["snapshot_id"] == snapshot_id)
    assert first_manifest["worker_count"] == worker_count
    assert first_manifest["entries_version"] > 0
    assert first_manifest["page_images_version"] > 0

    partitioner = se.ModuloPartitioner(num_shards=worker_count)

    async def restore_worker(worker_id: int):
        worker_store = se.EmbeddingStore(make_cfg([dim]))
        await asyncio.to_thread(
            backend.load,
            worker_store,
            uri,
            filter_fn=partitioner,
            distributed=True,
            target_shard_id=worker_id,
            snapshot_id=snapshot_id,
        )
        return worker_store

    async def restore_all_workers():
        return await asyncio.gather(
            *(restore_worker(worker_id) for worker_id in range(worker_count))
        )

    restored = asyncio.run(restore_all_workers())
    seen = set()
    for worker_id, worker_store in enumerate(restored):
        mask = (keys % np.uint64(worker_count)) == worker_id
        worker_keys = keys[mask]
        actual = lookup_np(worker_store, worker_keys, dim)
        np.testing.assert_allclose(actual, values[mask], atol=1e-5)
        assert not seen.intersection(worker_keys.tolist())
        seen.update(worker_keys.tolist())
    assert seen == set(keys.tolist())


def test_partitioned_lance_full_snapshot_isolated_by_snapshot_id(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    dim = 2
    keys = np.asarray([1, 2], dtype=np.uint64)
    first_values = np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    second_values = np.asarray([[3.0, 3.0]], dtype=np.float32)
    backend.init_dataset(se.EmbeddingStore(make_cfg([dim])), uri)

    first = se.EmbeddingStore(make_cfg([dim]))
    first.assign(keys, first_values)
    first_artifact = backend.export_fragments(
        first, uri, snapshot_id=1, worker_id=0, worker_count=1, shard_count=1
    )
    backend.commit_fragments(uri, [first_artifact])

    second = se.EmbeddingStore(make_cfg([dim]))
    second.assign(keys[:1], second_values)
    second_artifact = backend.export_fragments(
        second, uri, snapshot_id=2, worker_id=0, worker_count=1, shard_count=1
    )
    backend.commit_fragments(uri, [second_artifact])

    restored = se.EmbeddingStore(make_cfg([dim]))
    backend.load(restored, uri, snapshot_id=2)
    present = np.from_dlpack(restored.contains(keys))
    assert present.tolist() == [1, 0]
    np.testing.assert_array_equal(lookup_np(restored, keys[:1], dim), second_values)


def test_partitioned_lance_default_load_uses_latest_committed_manifest(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    dim = 2
    key = np.asarray([7], dtype=np.uint64)
    backend.init_dataset(se.EmbeddingStore(make_cfg([dim])), uri)

    first = se.EmbeddingStore(make_cfg([dim]))
    first.assign(key, np.asarray([[1.0, 1.0]], dtype=np.float32))
    backend.commit_fragments(
        uri,
        [
            backend.export_fragments(
                first,
                uri,
                snapshot_id=1,
                worker_id=0,
                worker_count=1,
                shard_count=1,
            )
        ],
    )

    uncommitted = se.EmbeddingStore(make_cfg([dim]))
    uncommitted.assign(key, np.asarray([[99.0, 99.0]], dtype=np.float32))
    backend.export_fragments(
        uncommitted,
        uri,
        snapshot_id=2,
        worker_id=0,
        worker_count=1,
        shard_count=1,
    )

    restored = se.EmbeddingStore(make_cfg([dim]))
    backend.load(restored, uri)
    np.testing.assert_array_equal(
        lookup_np(restored, key, dim),
        np.asarray([[1.0, 1.0]], dtype=np.float32),
    )


def test_partitioned_lance_init_dataset_is_pure_lance(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    backend.init_dataset(se.EmbeddingStore(make_cfg([2])), uri)

    assert not os.path.exists(os.path.join(uri, "metadata.json"))
    assert lance.dataset(f"{uri}/entries").version >= 1
    assert lance.dataset(f"{uri}/page_images").version >= 1
    assert lance.dataset(f"{uri}/snapshot_manifest").version >= 1


def test_partitioned_lance_single_node_save_is_pure_lance(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    dim = 2
    key = np.asarray([13], dtype=np.uint64)
    value = np.asarray([[4.0, 5.0]], dtype=np.float32)
    store = se.EmbeddingStore(make_cfg([dim]))
    store.assign(key, value)

    backend.save(store, uri)

    assert not os.path.exists(os.path.join(uri, "metadata.json"))
    manifest = lance.dataset(f"{uri}/snapshot_manifest").to_table().to_pylist()
    assert manifest == [
        {
            "schema_version": _FORMAT_VERSION,
            "snapshot_id": 1,
            "worker_count": 1,
            "shard_count": 1,
            "entries_version": lance.dataset(f"{uri}/entries").version,
            "page_images_version": lance.dataset(f"{uri}/page_images").version,
            "parent_snapshot_id": 0,
            "shard_high_watermarks": [1],
            "checkpoint_mode": "full",
            "parent_uri": "",
        }
    ]

    restored = se.EmbeddingStore(make_cfg([dim]))
    backend.load(restored, uri)
    np.testing.assert_array_equal(lookup_np(restored, key, dim), value)


def test_partitioned_lance_load_rejects_unsupported_manifest_schema(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    dim = 2
    key = np.asarray([17], dtype=np.uint64)
    value = np.asarray([[6.0, 7.0]], dtype=np.float32)
    store = se.EmbeddingStore(make_cfg([dim]))
    store.assign(key, value)
    backend.save(store, uri)

    manifest_uri = f"{uri}/snapshot_manifest"
    manifest_dataset = lance.dataset(manifest_uri)
    manifest_schema = manifest_dataset.schema
    rows = manifest_dataset.to_table().to_pylist()
    rows[0]["schema_version"] = 4
    shutil.rmtree(manifest_uri)
    lance.write_dataset(rows, manifest_uri, schema=manifest_schema)

    with pytest.raises(RuntimeError, match="manifest schema is unsupported"):
        backend.load(se.EmbeddingStore(make_cfg([dim])), uri)


def test_partitioned_lance_custom_save_partition_is_not_modulo_checked(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    dim = 2
    keys = np.asarray([0, 1, 2, 3], dtype=np.uint64)
    values = np.arange(keys.size * dim, dtype=np.float32).reshape(keys.size, dim)
    backend.init_dataset(se.EmbeddingStore(make_cfg([dim])), uri)

    artifacts = []
    for worker_id, worker_keys in enumerate((keys[[1, 3]], keys[[0, 2]])):
        store = se.EmbeddingStore(make_cfg([dim]))
        positions = np.isin(keys, worker_keys)
        store.assign(worker_keys, values[positions])
        artifacts.append(
            backend.export_fragments(
                store,
                uri,
                snapshot_id=5,
                worker_id=worker_id,
                worker_count=2,
                shard_count=2,
            )
        )
    backend.commit_fragments(uri, artifacts)

    restored = se.EmbeddingStore(make_cfg([dim]))
    backend.load(restored, uri, snapshot_id=5)
    np.testing.assert_array_equal(lookup_np(restored, keys, dim), values)


def test_partitioned_lance_commit_rejects_cross_worker_duplicate_key(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    dim = 2
    key = np.asarray([11], dtype=np.uint64)
    value = np.asarray([[1.0, 2.0]], dtype=np.float32)
    backend.init_dataset(se.EmbeddingStore(make_cfg([dim])), uri)
    artifacts = []
    for worker_id in range(2):
        store = se.EmbeddingStore(make_cfg([dim]))
        store.assign(key, value + np.float32(worker_id))
        artifacts.append(
            backend.export_fragments(
                store,
                uri,
                snapshot_id=9,
                worker_id=worker_id,
                worker_count=2,
                shard_count=2,
            )
        )
    with pytest.raises(ValueError, match="duplicate key across workers"):
        backend.commit_fragments(uri, artifacts)

    manifest_rows = lance.dataset(f"{uri}/snapshot_manifest").to_table().to_pylist()
    assert all(row["snapshot_id"] != 9 for row in manifest_rows)


def test_partitioned_lance_incremental_tombstone_and_page_dedup(
    lance_tmp_uri, tmp_path
):
    import lance

    base_uri, backend = lance_tmp_uri
    child_uri = str(tmp_path / "incremental")
    dim = 4
    keys = np.arange(64, dtype=np.uint64)
    values = np.arange(keys.size * dim, dtype=np.float32).reshape(-1, dim)
    backend.init_dataset(se.EmbeddingStore(make_cfg([dim])), base_uri)
    base = se.EmbeddingStore(make_cfg([dim]))
    base.assign(keys, values)
    base_artifact = backend.export_fragments(
        base,
        base_uri,
        snapshot_id=1,
        worker_id=0,
        worker_count=1,
        shard_count=1,
    )
    backend.commit_fragments(base_uri, [base_artifact])

    backend.init_dataset(se.EmbeddingStore(make_cfg([dim])), child_uri)
    child = se.EmbeddingStore(make_cfg([dim]))
    changed = keys[:4]
    changed_values = values[:4] + np.float32(1000)
    child.assign(keys, values)
    child.assign(changed, changed_values)
    child.erase(keys[4:8])
    child_artifact = backend.export_fragments(
        child,
        child_uri,
        snapshot_id=2,
        worker_id=0,
        worker_count=1,
        shard_count=1,
        checkpoint_mode="incremental",
        parent_snapshot_id=1,
        parent_uri=base_uri,
    )
    backend.commit_fragments(child_uri, [child_artifact])
    child_pages = lance.dataset(f"{child_uri}/page_images").to_table().to_pylist()
    child_entries = lance.dataset(f"{child_uri}/entries").to_table().to_pylist()
    assert any(row["operation"] == 2 for row in child_entries)
    assert all(row["snapshot_id"] == 2 for row in child_entries)
    assert len(child_pages) <= 1

    restored = se.EmbeddingStore(make_cfg([dim]))
    backend.load(restored, child_uri, snapshot_id=2)
    actual = lookup_np(restored, keys[:4], dim)
    np.testing.assert_array_equal(actual, changed_values)
    exists = np.from_dlpack(restored.contains(keys[4:8]))
    assert not exists.any()
    np.testing.assert_array_equal(lookup_np(restored, keys[8:], dim), values[8:])


def test_partitioned_lance_incremental_parent_lineage_reshards(lance_tmp_uri, tmp_path):
    uri, backend = lance_tmp_uri
    parent_uri = str(tmp_path / "parent")
    child_uri = str(tmp_path / "child")
    dim = 4
    keys = np.arange(96, dtype=np.uint64)
    values = np.arange(keys.size * dim, dtype=np.float32).reshape(-1, dim)
    backend.init_dataset(se.EmbeddingStore(make_cfg([dim])), parent_uri)

    async def save_parent(worker_id):
        store = se.EmbeddingStore(make_cfg([dim]))
        mask = (keys % np.uint64(4)) == worker_id
        store.assign(keys[mask], values[mask])
        return backend.export_fragments(
            store,
            parent_uri,
            snapshot_id=11,
            worker_id=worker_id,
            worker_count=4,
            shard_count=4,
        )

    async def gather_parent():
        return await asyncio.gather(*[save_parent(i) for i in range(4)])

    backend.commit_fragments(parent_uri, asyncio.run(gather_parent()))
    backend.init_dataset(se.EmbeddingStore(make_cfg([dim])), child_uri)
    changed = keys[:8]
    updated_values = values[:8] + np.float32(700)

    async def save_child(worker_id):
        store = se.EmbeddingStore(make_cfg([dim]))
        mask = (keys % np.uint64(4)) == worker_id
        store.assign(keys[mask], values[mask])
        owned_changed = changed[(changed % np.uint64(4)) == worker_id]
        if owned_changed.size:
            store.assign(
                owned_changed,
                updated_values[(changed % np.uint64(4)) == worker_id],
            )
        owned_deleted = keys[8:16][(keys[8:16] % np.uint64(4)) == worker_id]
        if owned_deleted.size:
            store.erase(owned_deleted)
        return backend.export_fragments(
            store,
            child_uri,
            snapshot_id=12,
            worker_id=worker_id,
            worker_count=4,
            shard_count=4,
            checkpoint_mode="incremental",
            parent_snapshot_id=11,
            parent_uri=parent_uri,
        )

    async def gather_child():
        return await asyncio.gather(*[save_child(i) for i in range(4)])

    backend.commit_fragments(child_uri, asyncio.run(gather_child()))
    router = se.ModuloPartitioner(num_shards=3)
    restored = []
    for worker_id in range(3):
        store = se.EmbeddingStore(make_cfg([dim]))
        backend.load(
            store,
            child_uri,
            snapshot_id=12,
            filter_fn=router,
            distributed=True,
            target_shard_id=worker_id,
        )
        restored.append(store)
    expected = values.copy()
    expected[:8] = updated_values
    present = np.ones(keys.size, dtype=bool)
    present[8:16] = False
    seen = set()
    for worker_id, store in enumerate(restored):
        mask = ((keys % np.uint64(3)) == worker_id) & present
        np.testing.assert_array_equal(lookup_np(store, keys[mask], dim), expected[mask])
        seen.update(keys[mask].tolist())
    assert seen == set(keys[present].tolist())


def test_partitioned_lance_page_images_preserve_physical_export(lance_tmp_uri):
    import hashlib
    import struct

    import lance

    uri, backend = lance_tmp_uri
    worker_count = 4
    snapshot_id = 37
    dim = 8
    all_keys = np.arange(8192, dtype=np.uint64)
    all_values = np.arange(all_keys.size * dim, dtype=np.float32).reshape(
        all_keys.size, dim
    )
    optimizer = se.Adam(lr=1e-3)
    backend.init_dataset(se.EmbeddingStore(make_cfg([dim])), uri)
    expected_pages = {}
    expected_values = {}

    def checksum(row):
        digest = hashlib.sha256()
        digest.update(
            struct.pack(
                "<IQIIII",
                row["dim"],
                row["physical_page_id"],
                row["vector_dim"],
                row["vector_stride"],
                row["optimizer_state_stride"],
                len(row["slot_ordinals"]),
            )
        )
        for value, dtype in (
            (row["slot_ordinals"], "<u4"),
            (row["keys"], "<u8"),
            (row["seq_counts"], "<u8"),
            (row["values"], "<f4"),
            (row["scores"], "<u8"),
            (row["optimizer_states"], "<f4"),
            (row["steps"], "<u4"),
        ):
            digest.update(np.asarray(value, dtype=dtype).tobytes(order="C"))
        return digest.digest()

    async def export_worker(worker_id: int):
        store = se.EmbeddingStore(
            make_cfg([dim]), score_policy="lfu", optimizer_config=optimizer
        )
        mask = (all_keys % np.uint64(worker_count)) == worker_id
        store.assign(all_keys[mask], all_values[mask])
        lookup_np(store, all_keys[mask], dim)
        store.accumulate_gradients(
            all_keys[mask], np.ones((mask.sum(), dim), dtype=np.float32)
        )
        store.optimizer_step()
        expected_values[worker_id] = lookup_np(store, all_keys[mask], dim).copy()
        for image in store.export_page_images():
            expected_pages[(worker_id, image["physical_page_id"])] = image
        return await asyncio.to_thread(
            backend.export_fragments,
            store,
            uri,
            snapshot_id=snapshot_id,
            worker_id=worker_id,
            worker_count=worker_count,
            shard_count=worker_count,
        )

    async def export_all_workers():
        return await asyncio.gather(
            *(export_worker(worker) for worker in range(worker_count))
        )

    artifacts = asyncio.run(export_all_workers())
    backend.commit_fragments(uri, artifacts)
    manifest = next(
        row
        for row in lance.dataset(f"{uri}/snapshot_manifest").to_table().to_pylist()
        if row["snapshot_id"] == snapshot_id
    )
    pages = (
        lance.dataset(f"{uri}/page_images", version=manifest["page_images_version"])
        .to_table(filter=f"snapshot_id = {snapshot_id}")
        .to_pylist()
    )
    assert len(pages) == len(expected_pages)
    for page in pages:
        expected = expected_pages[(page["worker_id"], page["physical_page_id"])]
        assert page["checksum"] == checksum(page)
        for name in (
            "dim",
            "physical_page_id",
            "vector_dim",
            "vector_stride",
            "slot_ordinals",
            "keys",
            "seq_counts",
            "values",
            "scores",
            "optimizer_states",
            "steps",
        ):
            assert page[name] == expected[name]

    partitioner = se.ModuloPartitioner(num_shards=worker_count)

    async def restore_worker(worker_id: int):
        store = se.EmbeddingStore(
            make_cfg([dim]), score_policy="lfu", optimizer_config=optimizer
        )
        await asyncio.to_thread(
            backend.load,
            store,
            uri,
            filter_fn=partitioner,
            distributed=True,
            target_shard_id=worker_id,
            snapshot_id=snapshot_id,
        )
        return store

    async def restore_all_workers():
        return await asyncio.gather(
            *(restore_worker(worker) for worker in range(worker_count))
        )

    restored = asyncio.run(restore_all_workers())
    for worker_id, store in enumerate(restored):
        mask = (all_keys % np.uint64(worker_count)) == worker_id
        np.testing.assert_array_equal(
            lookup_np(store, all_keys[mask], dim), expected_values[worker_id]
        )

    layout_restored = []
    for worker_id in range(worker_count):
        store = se.EmbeddingStore(
            make_cfg([dim]), score_policy="lfu", optimizer_config=optimizer
        )
        backend.load(
            store,
            uri,
            snapshot_id=snapshot_id,
            restore_layout=True,
            saved_worker_id=worker_id,
            worker_count=worker_count,
        )
        actual = {
            image["physical_page_id"]: image for image in store.export_page_images()
        }
        expected = {
            page_id: image
            for (owner, page_id), image in expected_pages.items()
            if owner == worker_id
        }
        assert actual == expected
        layout_restored.append(store)

    with pytest.raises(ValueError, match="checkpoint worker topology"):
        backend.load(
            se.EmbeddingStore(make_cfg([dim])),
            uri,
            snapshot_id=snapshot_id,
            restore_layout=True,
            saved_worker_id=0,
            worker_count=worker_count - 1,
        )

    with pytest.raises(ValueError, match="does not accept logical routing"):
        backend.load(
            se.EmbeddingStore(make_cfg([dim])),
            uri,
            filter_fn=partitioner,
            distributed=True,
            target_shard_id=1,
            snapshot_id=snapshot_id,
            restore_layout=True,
            saved_worker_id=0,
            worker_count=worker_count,
        )


def test_partitioned_lance_layout_restore_preserves_compaction_hole(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    worker_count = 2
    snapshot_id = 41
    dim = 8
    keys_per_worker = 8192
    backend.init_dataset(se.EmbeddingStore(make_cfg([dim])), uri)
    expected_images = {}
    expected_values = {}

    async def export_worker(worker_id: int):
        keys = np.arange(keys_per_worker, dtype=np.uint64) * np.uint64(
            worker_count
        ) + np.uint64(worker_id)
        values = np.arange(keys.size * dim, dtype=np.float32).reshape(keys.size, dim)
        values += np.float32(worker_id * 1_000_000)
        store = se.EmbeddingStore(make_cfg([dim]), score_policy="lfu")
        store.assign(keys, values)
        assert store.compact_page(dim, 0) == 4096
        actual = lookup_np(store, keys, dim)
        np.testing.assert_array_equal(actual, values)
        images = {
            image["physical_page_id"]: image for image in store.export_page_images()
        }
        assert set(images) == {1, 2}
        expected_images[worker_id] = images
        expected_values[worker_id] = (keys, values)
        return await asyncio.to_thread(
            backend.export_fragments,
            store,
            uri,
            snapshot_id=snapshot_id,
            worker_id=worker_id,
            worker_count=worker_count,
            shard_count=worker_count,
        )

    async def export_all_workers():
        return await asyncio.gather(
            *(export_worker(worker) for worker in range(worker_count))
        )

    backend.commit_fragments(uri, asyncio.run(export_all_workers()))
    for worker_id in range(worker_count):
        restored = se.EmbeddingStore(make_cfg([dim]), score_policy="lfu")
        backend.load(
            restored,
            uri,
            snapshot_id=snapshot_id,
            restore_layout=True,
            saved_worker_id=worker_id,
            worker_count=worker_count,
        )
        actual_images = {
            image["physical_page_id"]: image for image in restored.export_page_images()
        }
        assert actual_images == expected_images[worker_id]
        keys, values = expected_values[worker_id]
        np.testing.assert_array_equal(lookup_np(restored, keys, dim), values)


@pytest.mark.parametrize("saved_workers,restored_workers", [(8, 3), (3, 8)])
def test_partitioned_lance_restore_uses_current_fluxway_router(
    lance_tmp_uri, saved_workers, restored_workers
):
    uri, backend = lance_tmp_uri
    snapshot_id = 29
    dim = 8
    keys = np.arange(6144, dtype=np.uint64)
    values = np.arange(keys.size * dim, dtype=np.float32).reshape(keys.size, dim)
    backend.init_dataset(se.EmbeddingStore(make_cfg([dim])), uri)

    async def save_worker(worker_id: int):
        store = se.EmbeddingStore(make_cfg([dim]))
        saved_mask = (keys % np.uint64(saved_workers)) == worker_id
        store.assign(keys[saved_mask], values[saved_mask])
        return await asyncio.to_thread(
            backend.export_fragments,
            store,
            uri,
            snapshot_id=snapshot_id,
            worker_id=worker_id,
            worker_count=saved_workers,
            shard_count=saved_workers,
        )

    async def save_all_workers():
        return await asyncio.gather(
            *(save_worker(worker_id) for worker_id in range(saved_workers))
        )

    backend.commit_fragments(uri, asyncio.run(save_all_workers()))
    fluxway_router = se.ModuloPartitioner(num_shards=restored_workers)

    async def restore_worker(worker_id: int):
        store = se.EmbeddingStore(make_cfg([dim]))
        await asyncio.to_thread(
            backend.load,
            store,
            uri,
            filter_fn=fluxway_router,
            distributed=True,
            target_shard_id=worker_id,
            snapshot_id=snapshot_id,
        )
        return store

    async def restore_all_workers():
        return await asyncio.gather(
            *(restore_worker(worker_id) for worker_id in range(restored_workers))
        )

    restored = asyncio.run(restore_all_workers())
    seen = set()
    for worker_id, store in enumerate(restored):
        current_mask = (keys % np.uint64(restored_workers)) == worker_id
        worker_keys = keys[current_mask]
        actual = lookup_np(store, worker_keys, dim)
        np.testing.assert_array_equal(actual, values[current_mask])
        assert not seen.intersection(worker_keys.tolist())
        seen.update(worker_keys.tolist())
    assert seen == set(keys.tolist())


def test_distributed_load_requires_explicit_filter_fn(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    backend.init_dataset(se.EmbeddingStore(make_cfg([2])), uri)
    store = se.EmbeddingStore(make_cfg([2]))
    artifact = backend.export_fragments(
        store, uri, snapshot_id=1, worker_id=0, worker_count=1, shard_count=1
    )
    backend.commit_fragments(uri, [artifact])
    with pytest.raises(ValueError, match="requires filter_fn"):
        backend.load(store, uri, distributed=True)


def test_filter_fn_requires_explicit_distributed_load(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    backend.init_dataset(se.EmbeddingStore(make_cfg([2])), uri)
    store = se.EmbeddingStore(make_cfg([2]))
    artifact = backend.export_fragments(
        store, uri, snapshot_id=1, worker_id=0, worker_count=1, shard_count=1
    )
    backend.commit_fragments(uri, [artifact])
    with pytest.raises(ValueError, match="requires distributed=True"):
        backend.load(
            store,
            uri,
            filter_fn=se.ModuloPartitioner(num_shards=1),
            target_shard_id=0,
        )


def test_single_shard_roundtrip_value_correctness(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    truth = make_ground_truth()

    asyncio.run(distributed_save(uri, 1, truth, DIMS, backend))

    partitioner = se.ModuloPartitioner(num_shards=1)
    store = asyncio.run(load_shard(uri, 0, partitioner, backend))

    for slot_id, (s_keys, embeds) in truth.items():
        dim = DIMS[slot_id]
        vals_dlp, _ = store.lookup(s_keys, dim=dim)
        got = np.from_dlpack(vals_dlp).reshape(-1, dim)
        np.testing.assert_allclose(
            got, embeds, atol=1e-5, err_msg=f"slot {slot_id} value mismatch"
        )


def test_single_node_lance_save_persists_shard_watermark(lance_tmp_uri):
    import lance

    uri, backend = lance_tmp_uri
    dim = 4
    keys = np.arange(3, dtype=np.uint64)
    values = np.arange(keys.size * dim, dtype=np.float32).reshape(-1, dim)
    store = se.EmbeddingStore(make_cfg([dim]))
    store.assign(keys, values)

    backend.save(store, uri)

    manifest = lance.dataset(f"{uri}/snapshot_manifest").to_table().to_pylist()
    assert len(manifest) == 1
    assert manifest[0]["shard_high_watermarks"] == [keys.size]


def test_signed_int64_keys_checkpoint_and_reshard(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    keys = np.asarray(
        [np.iinfo(np.int64).min, -9, -5, -1, 0, 1, 7, np.iinfo(np.int64).max],
        dtype=np.int64,
    )
    values = np.arange(keys.size * 4, dtype=np.float32).reshape(keys.size, 4)
    config = se.EmbeddingConfig(
        dims=[4],
        num_threads=2,
        value_dtype="float32",
        key_dtype="int64",
    )
    source = se.EmbeddingStore(config)
    backend.init_dataset(source, uri)
    source.assign(keys, values)
    metadata = backend.export_fragments(
        source, uri, snapshot_id=1, worker_id=0, worker_count=1, shard_count=1
    )
    backend.commit_fragments(uri, [metadata])

    partitioner = se.ModuloPartitioner(num_shards=3)
    for shard_id in range(3):
        restored = se.EmbeddingStore(config)
        backend.load(
            restored,
            uri,
            filter_fn=partitioner,
            distributed=True,
            target_shard_id=shard_id,
        )
        mask = np.mod(keys, np.int64(3)) == shard_id
        shard_keys = keys[mask]
        if shard_keys.size == 0:
            continue
        actual = lookup_np(restored, shard_keys, 4)
        np.testing.assert_allclose(actual, values[mask], atol=1e-5)


@pytest.mark.parametrize(
    "save_shards,load_shards",
    [
        (3, 5),  # expansion
        (3, 7),  # larger expansion
        (8, 3),  # contraction
        (2, 16),  # large expansion
        (16, 5),  # large contraction
    ],
    ids=["3->5", "3->7", "8->3", "2->16", "16->5"],
)
def test_distributed_resharding(lance_backend, save_shards, load_shards):
    truth = make_ground_truth()

    asyncio.run(distributed_save(LANCE_URI, save_shards, truth, DIMS, lance_backend))

    new_partitioner = se.ModuloPartitioner(num_shards=load_shards)
    stores = asyncio.run(
        load_all_shards(LANCE_URI, load_shards, new_partitioner, lance_backend)
    )
    verify_all_shards(stores, truth, DIMS, new_partitioner)


# ---------------------------------------------------------------------------
# Optimizer state: save / load / reshard
# ---------------------------------------------------------------------------


def test_optimizer_state_save_and_load_single_node(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    rng = np.random.default_rng(99)

    adam = se.Adam(lr=1e-2)
    dim_map = {0: 4, 1: 8}
    store1 = se.EmbeddingStore(make_cfg([4, 8]), optimizer_config=adam)
    keys0 = np.arange(50, dtype=np.uint64)
    vals0 = rng.standard_normal((50, dim_map[0])).astype(np.float32)
    keys1 = slotted(np.arange(50, dtype=np.uint64), 1)
    vals1 = rng.standard_normal((50, dim_map[1])).astype(np.float32)

    store1.assign(keys0, vals0)
    store1.assign(keys1, vals1)
    assert store1.total_size() == 100

    backend.save(store1, uri)

    store2 = se.EmbeddingStore(make_cfg([4, 8]), optimizer_config=adam)
    backend.load(store2, uri)
    assert store2.total_size() == 100

    for slot_keys, dim, ref_vals in [
        (keys0, dim_map[0], vals0),
        (keys1, dim_map[1], vals1),
    ]:
        v1_dlp, _ = store1.lookup(slot_keys, dim=dim)
        v2_dlp, _ = store2.lookup(slot_keys, dim=dim)
        got1 = np.from_dlpack(v1_dlp).reshape(-1, dim)
        got2 = np.from_dlpack(v2_dlp).reshape(-1, dim)
        np.testing.assert_allclose(
            got1,
            got2,
            atol=1e-5,
            err_msg=f"Embedding values differ after opt_state restore (dim={dim})",
        )


@pytest.mark.parametrize(
    "save_shards,load_shards",
    [
        (2, 2),  # same topology
        (4, 6),  # expansion
        (6, 3),  # contraction
    ],
    ids=["2->2", "4->6", "6->3"],
)
def test_optimizer_state_distributed_resharding(
    lance_tmp_uri, save_shards, load_shards
):
    uri, backend = lance_tmp_uri
    rng = np.random.default_rng(42)

    adam = se.Adam(lr=1e-2)
    dim_map = {0: 4, 1: 8}

    num_keys = 500
    truth = {}
    for slot_id, dim in dim_map.items():
        raw_keys = np.arange(num_keys, dtype=np.uint64)
        s_keys = slotted(raw_keys, slot_id)
        embeds = rng.standard_normal((num_keys, dim)).astype(np.float32)
        truth[slot_id] = (s_keys, embeds)

    asyncio.run(
        distributed_save(
            uri, save_shards, truth, dim_map, backend, optimizer_config=adam
        )
    )

    new_partitioner = se.ModuloPartitioner(num_shards=load_shards)

    async def load_and_verify(shard_id):
        store = await load_shard(
            uri,
            shard_id,
            new_partitioner,
            backend,
            dims=sorted(set(dim_map.values())),
            optimizer_config=adam,
        )
        for slot_id, (s_keys, embeds) in truth.items():
            dim = dim_map[slot_id]
            mask = np.from_dlpack(new_partitioner.filter(s_keys, shard_id))
            my_keys = s_keys[mask]
            my_embeds = embeds[mask]
            if len(my_keys) == 0:
                continue
            vals_dlp, _ = store.lookup(my_keys, dim=dim)
            got = np.from_dlpack(vals_dlp).reshape(-1, dim)
            np.testing.assert_allclose(
                got,
                my_embeds,
                atol=1e-5,
                err_msg=(
                    f"shard {shard_id}, slot {slot_id}: embedding mismatch "
                    f"after {save_shards}->{load_shards} resharding"
                ),
            )

    async def run_all():
        await asyncio.gather(*[load_and_verify(s) for s in range(load_shards)])

    asyncio.run(run_all())


def test_checkpoint_schema_agnostic_dim_reorder(lance_tmp_uri):
    uri, backend = lance_tmp_uri
    # Save with dims [128, 64]
    store1 = se.EmbeddingStore(se.EmbeddingConfig(dims=[128, 64], num_threads=2))
    keys_128 = np.array([1, 2], dtype=np.uint64)
    vals_128 = np.ones((2, 128), dtype=np.float32)
    store1.assign(keys_128, vals_128)

    keys_64 = np.array([3, 4], dtype=np.uint64)
    vals_64 = np.ones((2, 64), dtype=np.float32) * 2
    store1.assign(keys_64, vals_64)

    backend.save(store1, uri)

    # Load with dims [64, 128]
    store2 = se.EmbeddingStore(se.EmbeddingConfig(dims=[64, 128], num_threads=2))
    backend.load(store2, uri)

    # Verify data is loaded into correct dimensions
    res_64, _ = store2.lookup(keys_64, dim=64)
    assert np.allclose(np.from_dlpack(res_64).reshape(-1, 64), vals_64)

    res_128, _ = store2.lookup(keys_128, dim=128)
    assert np.allclose(np.from_dlpack(res_128).reshape(-1, 128), vals_128)


# --- Large Checkpoint Tests ---


def make_cfg_large(dim=4, threads=4):
    return se.EmbeddingConfig(
        dims=[dim],
        num_threads=threads,
        value_dtype="float32",
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not hasattr(se.EmbeddingStore, "set_checkpoint_factories"),
    reason="Lance Checkpointer not enabled",
)
def test_large_checkpoint_lance():
    print("Preparing large scale data (50,000 entries)...")
    dim = 64
    num_entries = 50000
    store = se.EmbeddingStore(make_cfg_large(dim=dim), score_policy="lfu")

    # Configure Lance Backend
    store.set_checkpoint_factories("lance")

    keys = np.arange(1, num_entries + 1, dtype=np.uint64)
    # Generate somewhat unique float vectors
    vals = np.random.randn(num_entries, dim).astype(np.float32)

    print("Assigning data...")
    t0 = perf_counter()
    store.assign(keys, vals.ravel())
    print(f"Assigned in {perf_counter() - t0:.3f}s. Store size: {store.total_size()}")

    # Perform some lookups to generate LFU scores
    print("Simulating lookups to generate scores...")
    lookup_keys = keys[: num_entries // 2]
    store.lookup(lookup_keys, dim=dim)  # Look up half the keys

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_uri = os.path.join(tmpdir, "large_ckpt.lance")

        print(f"Saving checkpoint to {ckpt_uri}...")
        t0 = perf_counter()
        store.save_checkpoint(ckpt_uri)
        print(f"Save completed in {perf_counter() - t0:.3f}s.")

        # Verify it was actually created
        assert os.path.exists(ckpt_uri) or os.path.exists(
            os.path.join(ckpt_uri, "slot_0000")
        )

        # Now create a new store and load it
        print("Loading checkpoint into new store...")
        store2 = se.EmbeddingStore(make_cfg_large(dim=dim), score_policy="lfu")
        store2.set_checkpoint_factories("lance")

        t0 = perf_counter()
        store2.load_checkpoint(ckpt_uri)
        print(
            f"Load completed in {perf_counter() - t0:.3f}s. New store size: {store2.total_size()}"
        )

        assert store2.total_size() == num_entries, (
            "Restored store must have same number of entries"
        )

        print("Verifying values...")
        vals_dlp, _ = store2.lookup(keys, dim=dim)
        restored_vals = np.from_dlpack(vals_dlp).reshape(-1, dim)

        np.testing.assert_allclose(restored_vals, vals, atol=1e-5)
        print("All values verified correctly!")

        print("Verifying incremental save...")
        # Now do an incremental save to the SAME dataset to utilize Lance's versioning
        store2.save_checkpoint(ckpt_uri, mode="incremental", min_score=1)

        store3 = se.EmbeddingStore(make_cfg_large(dim=dim), score_policy="none")
        store3.set_checkpoint_factories("lance")
        # Loading should load the latest version (which includes all rows, with the new ones appended)
        store3.load_checkpoint(ckpt_uri)
        print(
            f"Incremental store size (expected ~{num_entries}): {store3.total_size()}"
        )

        assert store3.total_size() == num_entries, (
            "Lance append keeps all rows, latest version should still have all entries"
        )
        print("Large scale Lance checkpoint test PASSED.")


if __name__ == "__main__":
    test_large_checkpoint_lance()


# --- Checkpoint Backend Registry Tests ---


class MockCheckpointBackend(CheckpointBackend):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def save(self, store, uri, **kwargs):
        pass

    def load(self, store, uri, **kwargs):
        pass


def test_get_checkpointer_unknown():
    with pytest.raises(ValueError, match="Unknown checkpoint backend"):
        get_checkpointer("nonexistent_backend_123")


def test_register_and_get_checkpointer():
    register_backend("mock", MockCheckpointBackend)

    backend = get_checkpointer("mock", some_param="test")
    assert isinstance(backend, MockCheckpointBackend)
    assert backend.kwargs == {"some_param": "test"}

    # Test that lance backend exists if pylance is installed
    try:
        import lance

        _ = lance
        lance_backend = get_checkpointer("lance")
        assert lance_backend is not None
    except ImportError:
        pass
