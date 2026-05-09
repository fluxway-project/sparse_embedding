#!/usr/bin/env python3
"""Verify score_policy construction + assign/lookup with score policies."""

import numpy as np

import sparse_embedding._core as se


def make_cfg(dim=4, threads=2):
    return se.EmbeddingConfig(
        slot_dims={0: dim}, num_threads=threads, dtype="float32", max_active_keys=1024
    )


def lookup_np(store, keys, dim=4):
    vals_dlp, lens_dlp = store.lookup(keys)
    vals = np.from_dlpack(vals_dlp).reshape(-1, dim)
    return vals


def test_score_policies():
    for policy in ["none", "lfu", "lru", "manual"]:
        cfg = make_cfg()
        store = se.EmbeddingStore(cfg, score_policy=policy)
        assert store.total_size() == 0

    try:
        se.EmbeddingStore(make_cfg(), score_policy="badname")
        assert False, "should throw"
    except Exception:
        pass


def test_assign_and_lookup_lfu():
    store = se.EmbeddingStore(make_cfg(), score_policy="lfu")
    keys = np.array([0, 1, 2], dtype=np.uint64)
    vals = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    store.assign(keys, vals.ravel())
    assert store.total_size() == 3
    result = lookup_np(store, keys)
    np.testing.assert_allclose(result, vals, atol=1e-5)

    result2 = lookup_np(store, keys)
    np.testing.assert_allclose(result2, vals, atol=1e-5)


def test_assign_and_lookup_no_score():
    store = se.EmbeddingStore(make_cfg(), score_policy="none")
    keys = np.array([0, 1, 2], dtype=np.uint64)
    vals = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    store.assign(keys, vals.ravel())
    result = lookup_np(store, keys)
    np.testing.assert_allclose(result, vals, atol=1e-5)


def test_checkpoint_load_with_resharding():
    import os
    import shutil
    import sparse_embedding

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
        import pytest

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
        import numpy as np

        k_arr = np.from_dlpack(k_arr_capsule)
        orig_keys = k_arr & ((np.uint64(1) << 48) - np.uint64(1))
        return (orig_keys % np.uint64(3)).astype(np.uint32)

    loaded_size = load_cluster_checkpoints(7, 3, 1, ckpt_7_shards, python_mod_3_filter)
    assert loaded_size == 34, f"7->3 contraction expected 34, got {loaded_size}"

    ckpt_4_shards = "/tmp/ckpt_4_shards"
    save_cluster_checkpoints(4, ckpt_4_shards)

    hash_partitioner_6 = sparse_embedding.HashPartitioner(num_shards=6)
    loaded_size = load_cluster_checkpoints(4, 6, 3, ckpt_4_shards, hash_partitioner_6)
