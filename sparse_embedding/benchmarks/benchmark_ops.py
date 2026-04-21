import asyncio
import numpy as np
import time

import sparse_embedding


def add_wake_handler(loop):
    fd = sparse_embedding.wake_fd()
    loop.add_reader(fd, sparse_embedding.drain_wake_queue)


def benchmark_async_unique() -> None:
    keys = np.random.randint(0, 10000, size=100000, dtype=np.int64)

    async def run():
        add_wake_handler(asyncio.get_running_loop())
        fut = sparse_embedding.async_unique(keys)
        res = await fut
        asyncio.get_running_loop().remove_reader(sparse_embedding.wake_fd())
        return res

    start = time.time()
    for _ in range(10):
        asyncio.run(run())
    end = time.time()

    print(f"Unique Op 100k elements (10 iters) Time: {(end - start) * 1000:.2f} ms")


if __name__ == "__main__":
    benchmark_async_unique()
