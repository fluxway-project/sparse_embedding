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
