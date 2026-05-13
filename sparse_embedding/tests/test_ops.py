import asyncio

import numpy as np

import sparse_embedding


def test_async_unique() -> None:
    keys = np.array([10, 20, 10, 30, 20, 40], dtype=np.int64)

    async def run_unique():
        fut = sparse_embedding.async_unique(keys)
        res = await fut
        return res

    u_dlm, inv_dlm = asyncio.run(run_unique())
    assert u_dlm is not None
    assert inv_dlm is not None


def test_async_unique_all_flags() -> None:
    keys = np.array([10, 20, 10, 30, 20, 40], dtype=np.int64)

    async def run_unique():
        fut = sparse_embedding.async_unique(
            keys, return_inverse=True, return_first_indices=True, return_counts=True
        )
        res = await fut
        return res

    res = asyncio.run(run_unique())
    assert len(res) == 4
    u_dlm, inv_dlm, first_dlm, counts_dlm = res
    assert u_dlm is not None
    assert inv_dlm is not None
    assert first_dlm is not None
    assert counts_dlm is not None


def test_async_unique_int32() -> None:
    keys = np.array([10, 20, 10, 30], dtype=np.int32)

    async def run_unique():
        fut = sparse_embedding.async_unique(
            keys, return_inverse=False, return_first_indices=True, return_counts=True
        )
        res = await fut
        return res

    res = asyncio.run(run_unique())
    assert len(res) == 3
    u_dlm, first_dlm, counts_dlm = res
    assert u_dlm is not None
    assert first_dlm is not None
    assert counts_dlm is not None


def test_partitioners() -> None:
    keys = np.array([10, 11, 12, 13, 14, 15], dtype=np.uint64)

    # Modulo Partitioner
    modulo_part = sparse_embedding.ModuloPartitioner(num_shards=2)
    shard_ids_dlm = modulo_part.partition(keys)
    shard_ids = np.from_dlpack(shard_ids_dlm)
    np.testing.assert_array_equal(
        shard_ids, np.array([0, 1, 0, 1, 0, 1], dtype=np.uint32)
    )

    mask_dlm = modulo_part.filter(keys, target_shard=0)
    mask = np.from_dlpack(mask_dlm)
    np.testing.assert_array_equal(
        mask, np.array([True, False, True, False, True, False])
    )

    # Hash Partitioner
    hash_part = sparse_embedding.HashPartitioner(num_shards=4)
    shard_ids_dlm2 = hash_part.partition(keys)
    shard_ids2 = np.from_dlpack(shard_ids_dlm2)
    assert len(shard_ids2) == 6
    assert all(0 <= x < 4 for x in shard_ids2)

    mask_dlm2 = hash_part.filter(keys, target_shard=1)
    mask2 = np.from_dlpack(mask_dlm2)
    assert len(mask2) == 6
    # The mask should exactly correspond to the shard ids that are 1
    np.testing.assert_array_equal(mask2, shard_ids2 == 1)


# =============================================================================
# encode_keys / decode_keys
# =============================================================================


def test_encode_keys_slot_prefix() -> None:
    """slot_id must appear verbatim in the top slot_bits bits of every encoded key."""
    slot_bits = 16
    raw = np.arange(1000, dtype=np.uint64)

    for slot_id in (0, 1, 3, 255, (1 << slot_bits) - 1):
        enc = np.from_dlpack(
            sparse_embedding.encode_keys(raw, slot_id=slot_id, slot_bits=slot_bits)
        )
        top = enc >> np.uint64(64 - slot_bits)
        np.testing.assert_array_equal(top, slot_id, err_msg=f"slot_id={slot_id}")


def test_encode_keys_raw_preserved() -> None:
    """The raw key value must survive encoding in the low (64 - slot_bits) bits."""
    slot_bits = 16
    raw = np.arange(1000, dtype=np.uint64)
    mask = (np.uint64(1) << np.uint64(64 - slot_bits)) - np.uint64(1)

    enc = np.from_dlpack(
        sparse_embedding.encode_keys(raw, slot_id=7, slot_bits=slot_bits)
    )
    np.testing.assert_array_equal(enc & mask, raw)


def test_encode_keys_slot_isolation() -> None:
    """Two different slot_ids on the same raw key range must produce disjoint key spaces."""
    slot_bits = 16
    raw = np.arange(1000, dtype=np.uint64)

    enc0 = np.from_dlpack(
        sparse_embedding.encode_keys(raw, slot_id=0, slot_bits=slot_bits)
    )
    enc1 = np.from_dlpack(
        sparse_embedding.encode_keys(raw, slot_id=1, slot_bits=slot_bits)
    )

    assert not np.any(np.isin(enc0, enc1)), "slot0 and slot1 key spaces overlap"


def test_decode_keys_roundtrip() -> None:
    """decode(encode(raw, slot_id)) == (raw, slot_id) — zero intermediate copy."""
    slot_bits = 16
    raw = np.arange(2048, dtype=np.uint64)

    # Zero-copy chain: encode output pipes directly into decode.
    raw_dec_dlp, slot_dec_dlp = sparse_embedding.decode_keys(
        sparse_embedding.encode_keys(raw, slot_id=5, slot_bits=slot_bits),
        slot_bits=slot_bits,
    )

    raw_dec = np.from_dlpack(raw_dec_dlp)
    slot_dec = np.from_dlpack(slot_dec_dlp)

    np.testing.assert_array_equal(raw_dec, raw)
    np.testing.assert_array_equal(slot_dec, 5)


def test_decode_keys_slot_boundary() -> None:
    """slot_bits=1 should split the 64-bit space in half."""
    raw = np.array([0, 1, 2, 3], dtype=np.uint64)
    enc = np.from_dlpack(sparse_embedding.encode_keys(raw, slot_id=1, slot_bits=1))

    raw_dec_dlp, slot_dec_dlp = sparse_embedding.decode_keys(enc.copy(), slot_bits=1)
    np.testing.assert_array_equal(np.from_dlpack(raw_dec_dlp), raw)
    np.testing.assert_array_equal(np.from_dlpack(slot_dec_dlp), 1)


def test_encode_keys_invalid_slot_bits() -> None:
    raw = np.array([1, 2, 3], dtype=np.uint64)
    try:
        sparse_embedding.encode_keys(raw, slot_id=0, slot_bits=0)
        assert False, "should raise"
    except (ValueError, RuntimeError):
        pass
    try:
        sparse_embedding.encode_keys(raw, slot_id=0, slot_bits=64)
        assert False, "should raise"
    except (ValueError, RuntimeError):
        pass


def test_encode_keys_slot_id_too_large() -> None:
    raw = np.array([1, 2, 3], dtype=np.uint64)
    try:
        sparse_embedding.encode_keys(raw, slot_id=256, slot_bits=8)
        assert False, "should raise"
    except (ValueError, RuntimeError):
        pass


def test_assign_shape_mismatch() -> None:
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], num_threads=1, dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
        optimizer_config=sparse_embedding.Adam(lr=0.01),
    )
    keys = np.array([1, 2, 3], dtype=np.int64)
    values = np.random.rand(1, 128).astype(np.float32)

    # Sync version
    try:
        store.assign(keys, values)
        assert False, "sync assign should have raised an exception for shape mismatch"
    except (ValueError, RuntimeError) as e:
        assert "must match keys length" in str(e) or "invalid" in str(e).lower()

    # Async version
    async def run_async():
        try:
            await store.async_assign(keys, values)
            assert False, (
                "async assign should have raised an exception for shape mismatch"
            )
        except (ValueError, RuntimeError) as e:
            assert "must match keys length" in str(e) or "invalid" in str(e).lower()

    asyncio.run(run_async())


def test_accumulate_gradients_shape_mismatch() -> None:
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], num_threads=1, dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
        optimizer_config=sparse_embedding.Adam(lr=0.01),
    )
    keys = np.array([1, 2, 3], dtype=np.int64)
    grads = np.random.rand(4, 128).astype(np.float32)

    # Sync version
    try:
        store.accumulate_gradients(keys, grads)
        assert False, (
            "sync accumulate_gradients should have raised an exception for shape mismatch"
        )
    except (ValueError, RuntimeError) as e:
        assert "must match keys length" in str(e) or "invalid" in str(e).lower()

    # Async version
    async def run_async():
        try:
            await store.async_accumulate_gradients(keys, grads)
            assert False, (
                "async accumulate_gradients should have raised an exception for shape mismatch"
            )
        except (ValueError, RuntimeError) as e:
            assert "must match keys length" in str(e) or "invalid" in str(e).lower()

    asyncio.run(run_async())


def test_1d_tensor_support() -> None:
    """Test 1D flat tensor input to support PyArrow DLPack."""
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            dims=[128], num_threads=1, dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
        optimizer_config=sparse_embedding.Adam(lr=0.01),
    )
    keys = np.array([1, 2, 3], dtype=np.int64)

    # 3 * 128 = 384
    flat_values = np.random.rand(384).astype(np.float32)

    # Should succeed without shape mismatch error
    store.assign(keys, flat_values)

    # And grads
    flat_grads = np.random.rand(384).astype(np.float32)
    store.accumulate_gradients(keys, flat_grads)

    # Test failure on indivisible sizes
    bad_flat = np.random.rand(385).astype(np.float32)
    try:
        store.assign(keys, bad_flat)
        assert False, "should fail on indivisible size"
    except (ValueError, RuntimeError):
        pass


def test_embedding_store_huge_dim() -> None:
    try:
        _ = sparse_embedding.EmbeddingStore(
            embedding_config=sparse_embedding.EmbeddingConfig(
                dims=[2000000000], num_threads=1, dtype="float32"
            ),
            initializer_config=sparse_embedding.Zeros(),
            optimizer_config=sparse_embedding.Adam(lr=0.01),
        )
        assert False, "Constructor should refuse huge dimensions to prevent OOM"
    except (ValueError, RuntimeError) as e:
        assert "dim is too large" in str(e)
