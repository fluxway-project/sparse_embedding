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

## Interpretation

### 1. Lookup Performance (ReadOnly)
Under extreme concurrency (128 threads), the pure memory lookup reaches ~21.3M keys/s. Remarkably, the pure disk lookup retains ~17.4M keys/s, which is **81.58% of the pure memory performance**.
The result is specific to this workload and warm-up state. Disk-backed pages may be
served by the operating-system cache; it must not be interpreted as a storage
device latency or durability guarantee.

### 2. Insert Performance (Write + Allocate)
The `lookup_or_insert` operation on disk drops to **49.28%** of the memory baseline.
Insertions modify disk-backed pages and can contend on allocation and writeback.
The measured ratio is therefore workload- and device-dependent.

### 3. Effectiveness of Pre-reserving Capacity
`store.reserve(10000000, 128)` moves capacity preparation out of the timed phase,
so the table compares data-path behavior rather than resizing overhead.

---

**Script Location:** The benchmark script used to generate this report can be found at [`python/benchmark_disk_vs_mem.py`](python/benchmark_disk_vs_mem.py).
**To reproduce:**
```bash
python benchmarks/python/benchmark_disk_vs_mem.py --num_keys 10000000 --batch_size 8192 --concurrency 128 --iterations 1
```
