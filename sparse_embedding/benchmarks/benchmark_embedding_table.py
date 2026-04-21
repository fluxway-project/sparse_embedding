import asyncio
import numpy as np
import time

import sparse_embedding


def benchmark_embedding_table_lookup():
    slot_dims = {0: 128}
    store = sparse_embedding.EmbeddingStore(
        slot_dims, num_threads=4, init_type=0, dtype="float32"
    )

    # Pre-insert 1M keys
    keys = np.arange(1000000, dtype=np.int64)
    values = np.random.rand(1000000, 128).astype(np.float32)
    store.assign(keys, values)

    # Benchmark Lookup
    lookup_keys = np.random.randint(0, 1000000, size=100000, dtype=np.int64)

    start = time.time()
    for _ in range(10):
        # lookup 100k keys
        store.lookup(lookup_keys)
    end = time.time()

    print(
        f"EmbeddingTable Lookup 100k elements (10 iters) Time: {(end - start) * 1000:.2f} ms"
    )


def benchmark_embedding_table_lookup_or_insert():
    slot_dims = {0: 128}
    store = sparse_embedding.EmbeddingStore(
        slot_dims, num_threads=4, init_type=0, dtype="float32"
    )

    # Benchmark Lookup or Insert
    lookup_keys = np.random.randint(0, 500000, size=100000, dtype=np.int64)

    start = time.time()
    for _ in range(10):
        store.lookup_or_insert(lookup_keys)
    end = time.time()

    print(
        f"EmbeddingTable LookupOrInsert 100k elements (10 iters) Time: {(end - start) * 1000:.2f} ms"
    )


if __name__ == "__main__":
    benchmark_embedding_table_lookup()
    benchmark_embedding_table_lookup_or_insert()
