import argparse
import concurrent.futures
import os
import shutil
import tempfile
import time

import numpy as np

import sparse_embedding


def benchmark(
    backend, num_keys, batch_size, dim, concurrency, num_iterations, reserve=True
):
    tmp_dir = None
    if backend == "disk":
        tmp_dir = tempfile.mkdtemp()
        # Allocate a large enough disk space for spill (e.g. 100GB)
        storage = sparse_embedding.AutoStorage(
            memory_limit_bytes=0,
            spill_directory=tmp_dir,
            max_spill_bytes=1024 * 1024 * 1024 * 100,
        )
    elif backend == "memory":
        storage = sparse_embedding.MemoryOnly()
    else:
        raise ValueError(f"Unknown backend {backend}")

    # Set the local thread budget to concurrency as well, to allow the C++ backend
    # to utilize parallel processing. We can also just use os.cpu_count()
    thread_budget = min(128, os.cpu_count() or 16)

    config = sparse_embedding.EmbeddingConfig(
        dims=[dim], thread_budget=thread_budget, storage=storage
    )

    store = sparse_embedding.EmbeddingStore(
        embedding_config=config,
        initializer_config=sparse_embedding.Zeros(),
    )

    if reserve:
        print(f"Reserving capacity for {num_keys} keys (dim={dim})...")
        store.reserve(num_keys, dim)
        print("Reservation complete.")

    print(f"Generating data: {num_keys} keys, batch_size {batch_size}")
    all_keys = np.arange(num_keys, dtype=np.int64)
    # Shuffle keys to simulate random access which is typical in embedding lookups
    np.random.shuffle(all_keys)

    batches = [all_keys[i : i + batch_size] for i in range(0, num_keys, batch_size)]

    def worker_insert(batch):
        store.lookup_or_insert(batch, dim=dim)

    def worker_lookup(batch):
        store.lookup(batch, dim=dim)

    # --- INSERT BENCHMARK ---
    print(f"Starting Insert Benchmark ({backend}, concurrency={concurrency})...")
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        list(executor.map(worker_insert, batches))
    insert_time = time.time() - start_time
    insert_qps = num_keys / insert_time
    print(f"Insert Time: {insert_time:.4f} s")
    print(f"Insert QPS: {insert_qps:.2f} keys/s")

    # --- LOOKUP BENCHMARK ---
    print(f"Starting Lookup Benchmark ({backend}, concurrency={concurrency})...")

    start_time = time.time()
    for _ in range(num_iterations):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            list(executor.map(worker_lookup, batches))
    lookup_time = time.time() - start_time
    total_lookups = num_keys * num_iterations
    lookup_qps = total_lookups / lookup_time
    print(f"Lookup Time: {lookup_time:.4f} s (for {num_iterations} iterations)")
    print(f"Lookup QPS: {lookup_qps:.2f} keys/s")

    if tmp_dir:
        shutil.rmtree(tmp_dir)

    return insert_qps, lookup_qps


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sparse_embedding disk vs memory"
    )
    parser.add_argument(
        "--num_keys", type=int, default=10_000_000, help="Total number of keys"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8192, help="Batch size per request"
    )
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument(
        "--concurrency", type=int, default=64, help="Thread pool concurrency"
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of lookup iterations"
    )
    parser.add_argument(
        "--no_reserve", action="store_true", help="Do not reserve capacity upfront"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Configuration:")
    print(f"Keys: {args.num_keys:,}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Dimension: {args.dim}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Reserve Capacity: {not args.no_reserve}")
    print("=" * 60)

    print("\n[ RUNNING MEMORY BENCHMARK ]")
    mem_iqps, mem_lqps = benchmark(
        "memory",
        args.num_keys,
        args.batch_size,
        args.dim,
        args.concurrency,
        args.iterations,
        not args.no_reserve,
    )

    print("\n[ RUNNING DISK BENCHMARK ]")
    disk_iqps, disk_lqps = benchmark(
        "disk",
        args.num_keys,
        args.batch_size,
        args.dim,
        args.concurrency,
        args.iterations,
        not args.no_reserve,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} | {'Memory':<15} | {'Disk':<15} | {'Disk/Mem Ratio':<15}")
    print("-" * 60)
    print(
        f"{'Insert QPS':<20} | {mem_iqps:<15.2f} | {disk_iqps:<15.2f} | {disk_iqps / mem_iqps:<15.4f}"
    )
    print(
        f"{'Lookup QPS':<20} | {mem_lqps:<15.2f} | {disk_lqps:<15.2f} | {disk_lqps / mem_lqps:<15.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
