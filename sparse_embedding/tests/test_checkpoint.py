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
        print(f"  [OK] score_policy={policy} constructs fine")

    try:
        se.EmbeddingStore(make_cfg(), score_policy="badname")
        assert False, "should throw"
    except Exception as e:
        print(f"  [OK] invalid policy throws: {type(e).__name__}: {e}")


def test_assign_and_lookup_lfu():
    store = se.EmbeddingStore(make_cfg(), score_policy="lfu")
    keys = np.array([0, 1, 2], dtype=np.uint64)
    vals = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    store.assign(keys, vals.ravel())
    assert store.total_size() == 3
    result = lookup_np(store, keys)
    np.testing.assert_allclose(result, vals, atol=1e-5)
    print("  [OK] assign + lookup with lfu score_policy")

    result2 = lookup_np(store, keys)
    np.testing.assert_allclose(result2, vals, atol=1e-5)
    print("  [OK] second lookup (score incremented internally)")


def test_assign_and_lookup_no_score():
    store = se.EmbeddingStore(make_cfg(), score_policy="none")
    keys = np.array([0, 1, 2], dtype=np.uint64)
    vals = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    store.assign(keys, vals.ravel())
    result = lookup_np(store, keys)
    np.testing.assert_allclose(result, vals, atol=1e-5)
    print("  [OK] assign + lookup with no score (saves memory)")


def test_checkpoint_backend_not_configured():
    store = se.EmbeddingStore(make_cfg(), score_policy="none")

    try:
        store.save_checkpoint("/tmp/nonexistent_ckpt")
        assert False, "should throw"
    except RuntimeError as e:
        assert "not configured" in str(e)
        print(f"  [OK] save throws: {e}")

    try:
        store.load_checkpoint("/tmp/nonexistent_ckpt")
        assert False, "should throw"
    except RuntimeError as e:
        assert "not configured" in str(e)
        print(f"  [OK] load throws: {e}")


if __name__ == "__main__":
    print("=== test_score_policies ===")
    test_score_policies()
    print("\n=== test_assign_and_lookup_lfu ===")
    test_assign_and_lookup_lfu()
    print("\n=== test_assign_and_lookup_no_score ===")
    test_assign_and_lookup_no_score()
    print("\n=== test_checkpoint_backend_not_configured ===")
    test_checkpoint_backend_not_configured()
    print("\n=== ALL TESTS PASSED ===")
