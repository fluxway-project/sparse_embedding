import asyncio
import numpy as np

import sparse_embedding


def add_wake_handler(loop):
    fd = sparse_embedding.wake_fd()
    loop.add_reader(fd, sparse_embedding.drain_wake_queue)


def test_async_unique() -> None:
    keys = np.array([10, 20, 10, 30, 20, 40], dtype=np.int64)

    async def run_unique():
        add_wake_handler(asyncio.get_running_loop())
        fut = sparse_embedding.async_unique(keys)
        res = await fut
        asyncio.get_running_loop().remove_reader(sparse_embedding.wake_fd())
        return res

    u_dlm, inv_dlm = asyncio.run(run_unique())
    assert u_dlm is not None
    assert inv_dlm is not None
