#!/usr/bin/env python3
"""
Tests for score policies, async lookup/assign, and distributed checkpoint
save/load with hash-based shard filtering via the LanceCheckpointBackend.
"""

import asyncio
import os
import shutil

import numpy as np
import pytest

import sparse_embedding
import sparse_embedding._core as se
from sparse_embedding.checkpoint import get_backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LANCE_URI = "/tmp/test_lance_distributed_ckpt"
NUM_ORIG_KEYS = 1000
DIMS = {0: 8, 1: 16}


def make_cfg(slot_dims: dict, threads: int = 4, max_keys: int = 65536):
    return se.EmbeddingConfig(
        slot_dims=slot_dims,
        num_threads=threads,
        dtype="float32",
        max_active_keys=max_keys,
    )


def lookup_np(store, keys: np.ndarray, dim: int) -> np.ndarray:
    vals_dlp, _ = store.lookup(keys)
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

        vals_dlp, _ = store.lookup(expected_keys)
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
    dims: dict,
    backend,
    optimizer_config=None,
):
    partitioner = se.ModuloPartitioner(num_shards=num_shards)

    init_store = se.EmbeddingStore(make_cfg(dims))
    await asyncio.to_thread(backend.init_dataset, init_store, uri)

    async def worker_export(shard_id: int) -> list:
        cfg = make_cfg(dims)
        store = se.EmbeddingStore(
            cfg, **({"optimizer_config": optimizer_config} if optimizer_config else {})
        )

        assigns = []
        for slot_id, (s_keys, embeds) in truth.items():
            mask = np.from_dlpack(partitioner.filter(s_keys, shard_id))
            my_keys = s_keys[mask]
            my_embeds = embeds[mask]
            if len(my_keys) > 0:
                assigns.append(store.async_assign(my_keys, my_embeds.ravel()))

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
    dims: dict = DIMS,
    optimizer_config=None,
) -> se.EmbeddingStore:
    cfg = make_cfg(dims)
    store = se.EmbeddingStore(
        cfg, **({"optimizer_config": optimizer_config} if optimizer_config else {})
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
    yield get_backend("lance")
    if os.path.exists(LANCE_URI):
        shutil.rmtree(LANCE_URI)


@pytest.fixture
def lance_tmp_uri(tmp_path):
    pytest.importorskip("lance", reason="pylance not installed")
    uri = str(tmp_path / "lance_ckpt")
    yield uri, get_backend("lance")


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
    store.assign(keys, vals.ravel())
    assert store.total_size() == 3
    result = lookup_np(store, keys, 4)
    np.testing.assert_allclose(result, vals, atol=1e-5)
    result2 = lookup_np(store, keys, 4)
    np.testing.assert_allclose(result2, vals, atol=1e-5)


def test_assign_and_lookup_no_score():
    store = se.EmbeddingStore(make_cfg({0: 4}), score_policy="none")
    keys = np.array([0, 1, 2], dtype=np.uint64)
    vals = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    store.assign(keys, vals.ravel())
    result = lookup_np(store, keys, 4)
    np.testing.assert_allclose(result, vals, atol=1e-5)


def test_checkpoint_load_with_resharding():
    ckpt_dir = "/tmp/test_resharding_ckpt_dual"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    config = sparse_embedding.EmbeddingConfig(
        slot_dims={0: 8, 1: 8}, num_threads=2, dtype="float32", max_active_keys=1000
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
                store.lookup_or_insert(shard_keys)
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
        cfg = make_cfg({0: 4, 1: 8})
        store = se.EmbeddingStore(cfg)
        rng = np.random.default_rng(7)

        keys0 = np.arange(500, dtype=np.uint64)
        vals0 = rng.standard_normal((500, 4)).astype(np.float32)
        keys1 = slotted(np.arange(500, dtype=np.uint64), 1)
        vals1 = rng.standard_normal((500, 8)).astype(np.float32)

        await asyncio.gather(
            store.async_assign(keys0, vals0.ravel()),
            store.async_assign(keys1, vals1.ravel()),
        )

        assert store.total_size() == 1000

        res0_dlp, res1_dlp = await asyncio.gather(
            store.async_lookup(keys0),
            store.async_lookup(keys1),
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
        vals_dlp, _ = store.lookup(s_keys)
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
    slot_dims = {0: 4, 1: 8}
    cfg = make_cfg(slot_dims)

    store1 = se.EmbeddingStore(cfg, optimizer_config=adam)
    keys0 = np.arange(50, dtype=np.uint64)
    vals0 = rng.standard_normal((50, slot_dims[0])).astype(np.float32)
    keys1 = slotted(np.arange(50, dtype=np.uint64), 1)
    vals1 = rng.standard_normal((50, slot_dims[1])).astype(np.float32)

    store1.assign(keys0, vals0.ravel())
    store1.assign(keys1, vals1.ravel())
    assert store1.total_size() == 100

    backend.save(store1, uri)

    store2 = se.EmbeddingStore(cfg, optimizer_config=adam)
    backend.load(store2, uri)
    assert store2.total_size() == 100

    for slot_keys, dim, ref_vals in [
        (keys0, slot_dims[0], vals0),
        (keys1, slot_dims[1], vals1),
    ]:
        v1_dlp, _ = store1.lookup(slot_keys)
        v2_dlp, _ = store2.lookup(slot_keys)
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
    slot_dims = {0: 4, 1: 8}

    num_keys = 500
    truth = {}
    for slot_id, dim in slot_dims.items():
        raw_keys = np.arange(num_keys, dtype=np.uint64)
        s_keys = slotted(raw_keys, slot_id)
        embeds = rng.standard_normal((num_keys, dim)).astype(np.float32)
        truth[slot_id] = (s_keys, embeds)

    asyncio.run(
        distributed_save(
            uri, save_shards, truth, slot_dims, backend, optimizer_config=adam
        )
    )

    new_partitioner = se.ModuloPartitioner(num_shards=load_shards)

    async def load_and_verify(shard_id):
        store = await load_shard(
            uri,
            shard_id,
            new_partitioner,
            backend,
            dims=slot_dims,
            optimizer_config=adam,
        )
        for slot_id, (s_keys, embeds) in truth.items():
            dim = slot_dims[slot_id]
            mask = np.from_dlpack(new_partitioner.filter(s_keys, shard_id))
            my_keys = s_keys[mask]
            my_embeds = embeds[mask]
            if len(my_keys) == 0:
                continue
            vals_dlp, _ = store.lookup(my_keys)
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
