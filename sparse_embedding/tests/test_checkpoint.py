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

import numpy as np
import pytest

import sparse_embedding
import sparse_embedding as se
from sparse_embedding.checkpoint import (
    CheckpointBackend,
    get_checkpointer,
    register_backend,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LANCE_URI = "/tmp/test_lance_distributed_ckpt"
NUM_ORIG_KEYS = 1000
# Two feature groups with distinct dims; group 0 uses dim=8, group 1 uses dim=16.
# Keys for group 1 are namespace-separated by embedding slot bits into high bits.
DIMS = {0: 8, 1: 16}  # slot_id -> dim, for test bookkeeping only
UNIQUE_DIMS = sorted(set(DIMS.values()))  # [8, 16] — what the store actually needs


def make_cfg(dims, threads: int = 4, max_keys: int = 65536):
    return se.EmbeddingConfig(
        dims=dims if isinstance(dims, list) else sorted(set(dims.values())),
        num_threads=threads,
        value_dtype="float32",
        max_active_keys=max_keys,
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
        return await asyncio.to_thread(backend.export_fragments, store, uri)

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
        backend.load, store, uri, filter_fn=partitioner, target_shard_id=shard_id
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


def test_checkpoint_load_with_resharding():
    ckpt_dir = "/tmp/test_resharding_ckpt_dual"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    config = sparse_embedding.EmbeddingConfig(
        dims=[8], num_threads=2, value_dtype="float32", max_active_keys=1000
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


def make_cfg_large(dim=4, threads=4, max_keys=100000):
    return se.EmbeddingConfig(
        dims=[dim],
        num_threads=threads,
        value_dtype="float32",
        max_active_keys=max_keys,
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
    store = se.EmbeddingStore(
        make_cfg_large(dim=dim, max_keys=num_entries * 2), score_policy="lfu"
    )

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
    store.lookup(lookup_keys)  # Look up half the keys

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
        store2 = se.EmbeddingStore(
            make_cfg_large(dim=dim, max_keys=num_entries * 2), score_policy="lfu"
        )
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
        vals_dlp, _ = store2.lookup(keys)
        restored_vals = np.from_dlpack(vals_dlp).reshape(-1, dim)

        np.testing.assert_allclose(restored_vals, vals, atol=1e-5)
        print("All values verified correctly!")

        print("Verifying incremental save...")
        # Now do an incremental save to the SAME dataset to utilize Lance's versioning
        store2.save_checkpoint(ckpt_uri, mode="incremental", min_score=1)

        store3 = se.EmbeddingStore(
            make_cfg_large(dim=dim, max_keys=num_entries * 2), score_policy="none"
        )
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
