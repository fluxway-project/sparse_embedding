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
