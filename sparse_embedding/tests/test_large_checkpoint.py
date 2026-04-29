#!/usr/bin/env python3
"""Verify large-scale checkpoint save and load using the Lance backend."""

import os
import numpy as np
import tempfile
from time import perf_counter

import sparse_embedding._core as se


def make_cfg(dim=4, threads=4, max_keys=100000):
    return se.EmbeddingConfig(
        slot_dims={0: dim},
        num_threads=threads,
        dtype="float32",
        max_active_keys=max_keys,
    )


def test_large_checkpoint_lance():
    print("Preparing large scale data (50,000 entries)...")
    dim = 64
    num_entries = 50000
    store = se.EmbeddingStore(
        make_cfg(dim=dim, max_keys=num_entries * 2), score_policy="lfu"
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
        assert os.path.exists(ckpt_uri) or os.path.exists(os.path.join(ckpt_uri, "0"))

        # Now create a new store and load it
        print("Loading checkpoint into new store...")
        store2 = se.EmbeddingStore(
            make_cfg(dim=dim, max_keys=num_entries * 2), score_policy="lfu"
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
            make_cfg(dim=dim, max_keys=num_entries * 2), score_policy="none"
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
