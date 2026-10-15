import time

import numpy as np
import pytest

import sparse_embedding


def test_embedding_table_basic():
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], thread_budget=4, value_dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
    )

    assert store.num_slots() == 1

    keys = np.array([1, 2, 3], dtype=np.int64)
    vals, lens = store.lookup_or_insert(keys, dim=128)

    assert vals is not None
    assert lens is not None

    vals_lookup, lens_lookup = store.lookup(keys, dim=128)
    assert vals_lookup is not None


def test_embedding_table_reports_spill(tmp_path, capfd):
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[8],
            thread_budget=1,
            storage=sparse_embedding.AutoStorage(
                memory_limit_bytes=0,
                spill_directory=tmp_path,
                max_spill_bytes=4 << 20,
            ),
        ),
        initializer_config=sparse_embedding.Zeros(),
    )

    keys = np.array([1, 2, 3], dtype=np.int64)
    store.lookup_or_insert(keys, dim=8)
    with pytest.warns(RuntimeWarning, match="SparseEmbeddingSpillWarning"):
        stats = store.storage_stats()

    assert stats["spill_bytes"] > 0
    assert stats["memory_bytes"] == 0
    assert stats["spill_count"] > 0
    assert stats["eviction_selection_count"] == 0
    assert stats["eviction_evaluated_pages"] == 0
    assert stats["eviction_selection_ns"] == 0
    assert stats["eviction_candidate_invalidated_count"] == 0
    assert "SparseEmbeddingSpillWarning" in capfd.readouterr().err


def test_embedding_table_reports_eviction_telemetry(tmp_path):
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128],
            thread_budget=2,
            storage=sparse_embedding.AutoStorage(
                memory_limit_bytes=5 << 20,
                spill_directory=tmp_path,
                max_spill_bytes=1 << 30,
            ),
        ),
        initializer_config=sparse_embedding.Zeros(),
        score_policy="lfu",
    )

    keys = np.arange(16_384, dtype=np.int64)
    store.lookup_or_insert(keys, dim=128)
    with pytest.warns(RuntimeWarning, match="SparseEmbeddingSpillWarning"):
        stats = store.storage_stats()
    for _ in range(100):
        if stats["swap_out_count"] > 0:
            break
        time.sleep(0.01)
        stats = store.storage_stats()

    assert stats["swap_out_count"] > 0
    assert stats["eviction_selection_count"] > 0
    assert stats["eviction_evaluated_pages"] > 0
    assert stats["eviction_selection_ns"] > 0


def test_embedding_table_assign():
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], thread_budget=4, value_dtype="float32"
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
    int_store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], thread_budget=4, value_dtype="float32", key_dtype="int32"
        ),
        initializer_config=sparse_embedding.Zeros(),
    )

    keys_int32 = np.array([1, 2, 3], dtype=np.int32)
    vals, lens = int_store.lookup_or_insert(keys_int32, dim=128)
    assert vals is not None
    assert lens is not None

    uint_store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], thread_budget=4, value_dtype="float32", key_dtype="uint32"
        ),
        initializer_config=sparse_embedding.Zeros(),
    )
    keys_uint32 = np.array([4, 5, 6], dtype=np.uint32)
    vals, lens = uint_store.lookup_or_insert(keys_uint32, dim=128)
    assert vals is not None
    assert lens is not None


@pytest.mark.parametrize("key_dtype", ["int32", "uint32", "int64", "uint64"])
def test_embedding_store_exposes_and_enforces_key_dtype(key_dtype):
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[4], thread_budget=1, key_dtype=key_dtype
        ),
        initializer_config=sparse_embedding.Zeros(),
    )
    assert store.key_dtype() == key_dtype
    keys = np.asarray([1, 2], dtype=np.dtype(key_dtype))
    store.lookup_or_insert(keys, dim=4)
    wrong_dtype = np.asarray(
        [1, 2], dtype=np.int64 if key_dtype != "int64" else np.int32
    )
    with pytest.raises(ValueError, match="key dtype does not match store"):
        store.lookup(wrong_dtype, dim=4)


def test_embedding_table_reserve_api():
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[8, 16], thread_budget=4, value_dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
    )

    shard_count = store.internal_shard_count()
    assert shard_count == 4

    store.reserve(128, dim=8)
    store.reserve_per_shard([32] * shard_count, dim=8)

    keys = np.arange(64, dtype=np.int64)
    vals, _ = store.lookup_or_insert(keys, dim=8)
    assert np.from_dlpack(vals).shape[0] == 64 * 8
    assert store.total_size() == 64

    vals_lookup, _ = store.lookup(keys, dim=8)
    assert np.allclose(np.from_dlpack(vals_lookup), 0.0)


def test_embedding_table_reserve_preserves_existing_keys():
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[8], thread_budget=4, value_dtype="float32"
        )
    )

    keys = np.arange(16, dtype=np.int64)
    values = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
    store.assign(keys, values)

    store.reserve(4096, dim=8)

    vals, _ = store.lookup(keys, dim=8)
    np.testing.assert_array_equal(np.from_dlpack(vals).reshape(16, 8), values)
    assert store.total_size() == 16


def test_embedding_table_reserve_errors():
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[8], thread_budget=4, value_dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
    )

    with pytest.raises(ValueError, match="counts size mismatch"):
        store.reserve_per_shard([1, 2], dim=8)

    with pytest.raises(RuntimeError, match="invalid dim access"):
        store.reserve(1, dim=16)


def test_embedding_table_grows_beyond_initial_shard_capacity():
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[8],
            thread_budget=1,
            storage=sparse_embedding.MemoryOnly(),
        ),
        initializer_config=sparse_embedding.Zeros(),
        optimizer_config=sparse_embedding.Sgd(),
    )

    keys = np.arange(8193, dtype=np.int64)
    store.lookup_or_insert(keys, dim=8)
    assert store.total_size() == len(keys)
    store.erase(np.array([1], dtype=np.int64))
    store.lookup_or_insert(np.array([8193], dtype=np.int64), dim=8)
    assert store.total_size() == len(keys)


@pytest.mark.asyncio
async def test_embedding_table_async():
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], thread_budget=4, value_dtype="float32"
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

    vals2, lens2 = await store.async_lookup_or_insert(keys, dim=128)

    # test async_accumulate_gradients
    grads = np.random.rand(3, 128).astype(np.float32)
    await store.async_accumulate_gradients(keys, grads)

    # test async_optimizer_step
    await store.async_optimizer_step(1)

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
            dims=[128], thread_budget=1, value_dtype="float32", key_dtype="uint64"
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
    store.optimizer_step(1)

    val_0_step1 = np.from_dlpack(store.lookup(keys_0, dim=128)[0]).copy()

    # 2. Step the optimizer globally 100 times using ID 0
    for _ in range(100):
        store.accumulate_gradients(keys_0, grads_0)
        store.optimizer_step(1)

    # 3. Update ID 1 for the very first time, but at global step > 100
    keys_1 = np.array([1], dtype=np.uint64)
    grads_1 = np.ones((1, 128), dtype=np.float32)
    store.lookup_or_insert(keys_1, dim=128)
    store.accumulate_gradients(keys_1, grads_1)
    store.optimizer_step(1)

    val_1_step1 = np.from_dlpack(store.lookup(keys_1, dim=128)[0]).copy()

    # 4. Check that the first update size is identical for ID 0 and ID 1,
    # meaning bias correction correctly used step=1 for ID 1.
    np.testing.assert_allclose(val_0_step1, val_1_step1, atol=1e-6)
