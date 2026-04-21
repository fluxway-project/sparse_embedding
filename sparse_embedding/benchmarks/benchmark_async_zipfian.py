import asyncio
import numpy as np
import time

import sparse_embedding as se


async def worker(store, keys, values, batch_size):
    """
    Simulate an async worker (like in a parameter server or distributed training node)
    that processes batches concurrently and utilizes the async `async_assign`.
    """
    num_keys = len(keys)
    tasks = []

    for start in range(0, num_keys, batch_size):
        end = min(start + batch_size, num_keys)
        batch_keys = keys[start:end]
        batch_vals = values[start:end]

        # The C++ extension returns an asyncio.Future directly
        future = store.async_assign(batch_keys, batch_vals)
        tasks.append(future)

    await asyncio.gather(*tasks)


async def worker_lookup(store, keys, batch_size):
    num_keys = len(keys)
    tasks = []

    for start in range(0, num_keys, batch_size):
        end = min(start + batch_size, num_keys)
        batch_keys = keys[start:end]

        # The C++ extension returns an asyncio.Future directly
        future = store.async_lookup(batch_keys)
        tasks.append(future)

    await asyncio.gather(*tasks)


async def run_benchmark():
    # 1. Zipfian Distribution for Recsys representation
    num_queries = 2000000  # 2 million queries
    vocab_size = 500000  # 500k vocabulary
    a = 1.2  # Zipfian parameter (the smaller, the heavier the tail/collision)
    slot_dims = {0: 128}

    print("Generating Zipfian keys...")
    raw_keys = np.random.zipf(a, size=num_queries).astype(np.int64)
    # Map to vocab space
    keys = raw_keys % vocab_size
    values = np.random.rand(num_queries, 128).astype(np.float32)

    store = se.EmbeddingStore(
        embedding_config=se.EmbeddingConfig(
            slot_dims=slot_dims, num_threads=8, dtype="float32", grad_accum_steps=100
        ),
        initializer_config=se.Zeros(),
        optimizer_config=se.Adam(lr=0.01),
    )

    batch_size = 10000

    print("Running Async Assignment (Zipfian Distributed Keys)...")
    start = time.time()
    await worker(store, keys, values, batch_size)
    end = time.time()

    speed = num_queries / (end - start)
    print(
        f"Async Assign: {num_queries} queries in {end - start:.2f}s ({speed:.2f} queries/sec)"
    )

    print("Running Async Lookup (Zipfian Distributed Keys)...")
    start_lookup = time.time()
    await worker_lookup(store, keys, batch_size)
    end_lookup = time.time()

    speed_lookup = num_queries / (end_lookup - start_lookup)
    print(
        f"Async Lookup: {num_queries} queries in {end_lookup - start_lookup:.2f}s ({speed_lookup:.2f} queries/sec)"
    )


if __name__ == "__main__":
    asyncio.run(run_benchmark())
