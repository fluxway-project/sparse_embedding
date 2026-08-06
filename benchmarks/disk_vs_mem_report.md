# Sparse Embedding Disk vs Memory Benchmark

This report documents the performance evaluation of the `sparse_embedding` package when comparing pure memory storage (`MemoryOnly`) against Auto spill storage with no memory cache.

## Benchmark Configuration

The benchmark simulates a high-concurrency, multi-core environment accessing a massive embedding table.

- **Total Keys:** 10,000,000 (10M)
- **Batch Size:** 8,192
- **Embedding Dimension:** 128
- **Concurrency (Threads):** 128
- **Memory Storage:** `sparse_embedding.MemoryOnly()`
- **Spill Storage:** `sparse_embedding.AutoStorage(memory_limit_bytes=0, spill_directory=tmp_dir, max_spill_bytes=100GB)`

*Note on Reserve Capacity: Before inserting data, `store.reserve()` was explicitly called to pre-allocate capacity. This prevents the benchmark from measuring the overhead of page allocation and hash table rehashing, focusing purely on the raw insert/lookup memory vs disk overhead.*

## Performance Results

The following metrics show the Queries Per Second (QPS) for bulk `lookup_or_insert` (Insert) and `lookup` (Lookup) operations across 128 concurrent Python threads.

| Metric               | Memory (keys/s) | Disk (keys/s) | Disk/Mem Ratio |
|----------------------|----------------:|--------------:|---------------:|
| **Insert QPS**       | 682,642.36      | 336,386.05    | **49.28%**     |
| **Lookup QPS**       | 21,348,252.61   | 17,415,414.31 | **81.58%**     |

## Analysis and Conclusion

### 1. Lookup Performance (ReadOnly)
Under extreme concurrency (128 threads), the pure memory lookup reaches ~21.3M keys/s. Remarkably, the pure disk lookup retains ~17.4M keys/s, which is **81.58% of the pure memory performance**.
This indicates that the C++ backend and OS-level memory mapping / page caching handles concurrent read-only access exceptionally well. Even when the backing store is on disk, the read throughput does not suffer a catastrophic drop, demonstrating the robustness of the zero-copy and multi-core dispatch architecture (`py::gil_scoped_release`).

### 2. Insert Performance (Write + Allocate)
The `lookup_or_insert` operation on disk drops to **49.28%** of the memory baseline.
Unlike lookups, new insertions onto disk require dirtying pages and triggering physical disk IO operations (e.g., ext4/xfs block allocation, fsync constraints, and OS writeback). With 128 threads competing for disk-backed memory blocks, the IO contention and dirty page write-out become the primary bottleneck, resulting in approximately half the performance of pure memory allocations.

### 3. Effectiveness of Pre-reserving Capacity
By using `store.reserve(10000000, 128)`, the benchmark cleanly separates data-path performance from control-path (resizing/warmup) overhead. The `SparseEmbeddingSpillWarning` during the initialization phase confirms that the disk space was successfully pre-reserved before any timing started.

---

**Script Location:** The benchmark script used to generate this report can be found at [`python/benchmark_disk_vs_mem.py`](python/benchmark_disk_vs_mem.py).
**To reproduce:**
```bash
python benchmarks/python/benchmark_disk_vs_mem.py --num_keys 10000000 --batch_size 8192 --concurrency 128 --iterations 1
```
