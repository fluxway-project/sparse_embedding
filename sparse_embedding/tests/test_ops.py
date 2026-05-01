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
