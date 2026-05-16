import numpy as np
import pytest

import sparse_embedding


def test_embedding_table_basic():
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], num_threads=4, dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
    )

    assert store.num_slots() == 1

    keys = np.array([1, 2, 3], dtype=np.int64)
    vals, lens, mask = store.lookup_or_insert(keys, dim=128)

    assert vals is not None
    assert lens is not None
    assert mask is not None

    mask_np = np.from_dlpack(mask)
    assert mask_np.shape[0] == 3

    vals_lookup, lens_lookup = store.lookup(keys, dim=128)
    assert vals_lookup is not None


def test_embedding_table_assign():
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], num_threads=4, dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
    )

    keys = np.array([10, 20], dtype=np.int64)
    values = np.ones((2, 128), dtype=np.float32)

    # Pre-insert using assign with lookup_or_insert implicitly or explicitly
    store.assign(keys, values)

    vals, lens = store.lookup(keys, dim=128)
    vals_np = np.from_dlpack(vals)
    assert np.allclose(vals_np, 1.0)


def test_embedding_table_32bit_keys():
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], num_threads=4, dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
    )

    keys_int32 = np.array([1, 2, 3], dtype=np.int32)
    keys_uint32 = np.array([4, 5, 6], dtype=np.uint32)

    # Test int32
    vals, lens, mask = store.lookup_or_insert(keys_int32, dim=128)
    assert vals is not None
    assert lens is not None
    assert mask is not None
    mask_np = np.from_dlpack(mask)
    assert mask_np.shape[0] == 3

    # Test uint32
    vals, lens, mask = store.lookup_or_insert(keys_uint32, dim=128)
    assert vals is not None
    assert lens is not None
    assert mask is not None
    mask_np = np.from_dlpack(mask)
    assert mask_np.shape[0] == 3


@pytest.mark.asyncio
async def test_embedding_table_async():
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], num_threads=4, dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
        score_policy="lru",
    )

    keys = np.array([1, 2, 3], dtype=np.int64)
    values = np.random.rand(3, 128).astype(np.float32)
    scores = np.array([10, 20, 30], dtype=np.uint64)

    # test async_assign
    await store.async_assign(keys, values)

    # test async_lookup
    vals, lens = await store.async_lookup(keys, dim=128)
    vals_np = np.from_dlpack(vals)
    assert vals_np.shape[0] == 3 * 128

    vals2, lens2, mask2 = await store.async_lookup_or_insert(keys, dim=128)
    mask_np = np.from_dlpack(mask2)
    assert mask_np.shape[0] == 3

    # test async_accumulate_gradients
    grads = np.random.rand(3, 128).astype(np.float32)
    await store.async_accumulate_gradients(keys, grads)

    # test async_optimizer_step
    await store.async_optimizer_step()

    # test async_assign_scores
    await store.async_assign_scores(keys, scores)

    # test async_contains
    contains_mask = await store.async_contains(keys)
    contains_np = np.from_dlpack(contains_mask)
    assert np.all(contains_np == 1)

    # test async_erase
    await store.async_erase(keys)
    contains_mask = await store.async_contains(keys)
    contains_np = np.from_dlpack(contains_mask)
    assert np.all(contains_np == 0)

    # test async_clear
    await store.async_assign(keys, values)
    await store.async_clear()
    contains_mask = await store.async_contains(keys)
    contains_np = np.from_dlpack(contains_mask)
    assert np.all(contains_np == 0)


def test_optimizer_per_id_step():
    """Verify that optimizer tracks steps per-ID for bias correction."""
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], num_threads=1, dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
        optimizer_config=sparse_embedding.Adam(
            lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0
        ),
    )

    # 1. Update ID 0 at step 1
    keys_0 = np.array([0], dtype=np.uint64)
    grads_0 = np.ones((1, 128), dtype=np.float32)
    store.lookup_or_insert(keys_0, dim=128)
    store.accumulate_gradients(keys_0, grads_0)
    store.optimizer_step()

    val_0_step1 = np.from_dlpack(store.lookup(keys_0, dim=128)[0]).copy()

    # 2. Step the optimizer globally 100 times using ID 0
    for _ in range(100):
        store.accumulate_gradients(keys_0, grads_0)
        store.optimizer_step()

    # 3. Update ID 1 for the very first time, but at global step > 100
    keys_1 = np.array([1], dtype=np.uint64)
    grads_1 = np.ones((1, 128), dtype=np.float32)
    store.lookup_or_insert(keys_1, dim=128)
    store.accumulate_gradients(keys_1, grads_1)
    store.optimizer_step()

    val_1_step1 = np.from_dlpack(store.lookup(keys_1, dim=128)[0]).copy()

    # 4. Check that the first update size is identical for ID 0 and ID 1,
    # meaning bias correction correctly used step=1 for ID 1.
    np.testing.assert_allclose(val_0_step1, val_1_step1, atol=1e-6)
