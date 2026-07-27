# sparse_embedding

[中文](README_zh.md)

High-performance Sparse Embedding Table library for large-scale recommendation system training, using Hogwild-style relaxed updates, shard-local hash-map synchronization, and paged storage.

## Core Features

- **Shard-Local Concurrency**: Uses `parlayhash` for expected O(1) Key→Index access, with shared/exclusive locks scoped to one shard so independent shards continue in parallel during insert, erase, rebuild, and expansion.
- **SIMD Acceleration**: AVX-512 vectorized hash computation and gradient Scatter operations.
- **Hogwild! Gradient Accumulation**: Sharded TransientArena for lock-free gradient aggregation, avoiding cross-thread synchronization.
- **C++23 Standard Concurrency**: Sender/Receiver task scheduling based on NVIDIA `stdexec` + `exec::static_thread_pool`.
- **Epoch Memory Safety**: Batch protection for concurrent hash map read/write using `epoch::with_epoch`, avoiding ABA problems.
- **Distributed Lance Checkpointing**: Native distributed two-phase commit Lance backend with zero-copy PyArrow exports, independently storing embeddings and optimizer states for ultra-fast distributed storage and retrieval.
- **Dynamic Partitioner OPs**: High-performance C++ Partitioners (`ModuloPartitioner`, `HashPartitioner`, `FastRangePartitioner`) supporting efficient feature routing and on-the-fly cluster resharding during checkpoint loading.
- **Eviction Management**: Memory-efficient scoring policies (e.g., LFU) using thread-safe `std::atomic_ref` for concurrent state updates without locking.
- **Page-Level Hot/Cold Swapping**: Background eviction of cold data pages through address-stable `MAP_FIXED` remapping to file-backed storage based on configurable memory watermarks.

## System Dependencies

| Component | Version |
|------|------|
| GCC | ≥ 13.4 (validated) |
| stdexec | NVIDIA main branch `931f256c` |
| parlayhash | main branch (including 56-bit pointer mask patch) |
| lance-c | main branch (Rust-based C shared library) |
| Rust | 1.70+ (2024 Edition) |
| Bazel | ≥ 7.0 |

The Bazel configuration compiles C++ sources with `-std=c++2b`. The current acceptance build uses GCC 13.4 and Bazel 7.7.

---

## Performance Benchmarks

### Benchmark Machine

| Item | Measured configuration |
|---|---|
| CPU | 2 x Intel Xeon Silver 4116, nominal 2.10 GHz |
| Topology | 12 cores/socket, 2 threads/core, 24 physical cores, 48 online logical CPUs |
| NUMA | 2 nodes; node 0 CPUs `0-11,24-35`, node 1 CPUs `12-23,36-47` |
| Cache | 768 KiB L1d, 768 KiB L1i, 24 MiB L2, 33 MiB L3 across reported instances |
| Memory | 125 GiB visible to the OS; swap disabled |
| OS | Ubuntu 24.04.4 LTS, x86-64 |
| Toolchain | GCC 13.4, Bazel 7.7, C++23 (`-std=c++2b`), `-c opt` |
| Optimization flags | `-O3 -march=native -mtune=native -mavx512bf16 -fno-strict-aliasing -DNDEBUG`; link with `-flto` |
| CPU affinity | Not pinned for this snapshot |

### C++ Production Scenario

| Parameter | Value |
|---|---|
| Embedding dimension | 128 `float` values (512 bytes) |
| Vocabulary | 500,000 `uint64_t` keys; one slot |
| Batch | 65,536 keys |
| Access distribution | Generated operation batches use 80% traffic to the hottest 10%; erase keys are uniform |
| Random seed | 42 |
| Thread counts | 1, 4, 8, 16, 32, 64 |
| Capacity preparation | `reserve(500000, 128)` before timing |
| Optimizer | Benchmark SGD, learning rate 0.01 |
| Timed operations | lookup/insert, forward lookup, assign, gradient accumulation, optimizer step, contains, erase |

The LFU pressure scenario uses dimension 128, 16,384 possible keys, a 4,096-key batch, two workers, seed 42, the same 80/20 hot/cold distribution, benchmark SGD at 0.01, a 5 MiB memory limit, a 1 GiB spill limit, and 0.8/0.6 watermarks. It calls `reserve(16384, 128)` before insertion, then executes four timed repetitions of lookup, assign, gradient accumulation, and optimizer step with background eviction enabled. Telemetry collected after a 100 ms wait proves eviction eventually triggered; it does not prove overlap with the timed interval.

The latest reproducibility snapshot was collected on 2026-07-21. This C++ snapshot is one benchmark execution with internal repetitions; it is not the five-run, affinity-pinned baseline/optimized protocol.

| Reported benchmark section (Mops/s) | 1T | 4T | 8T | 16T | 32T | 64T |
|---|---:|---:|---:|---:|---:|---:|
| `lookup_or_insert` | 1.84 | 1.26 | 1.88 | 3.18 | 4.35 | 4.20 |
| `forward_lookup` | 2.90 | 5.34 | 7.91 | 10.18 | 12.10 | 11.61 |
| `assign` | 1.98 | 1.55 | 2.46 | 3.91 | 5.46 | 6.02 |
| `accumulate_gradients` | 2.35 | 6.30 | 8.60 | 10.37 | 11.54 | 10.80 |
| `optimizer_step` | 5.53 | 12.99 | 20.66 | 27.77 | 32.54 | 30.30 |
| `contains` | 6.25 | 5.90 | 8.62 | 11.13 | 15.59 | 16.86 |
| `erase` (10% batch) | 8.05 | 4.06 | 5.53 | 8.41 | 13.08 | 16.99 |

These are not isolated API timings. Random batch generation is inside the timer for lookup/insert, forward lookup, assign, gradient accumulation, and contains. The reported optimizer section also includes generation and accumulation of the next gradient batch while using current store size as its throughput numerator, so that row must not be interpreted as standalone optimizer throughput. Erase repeats one key set ten times without reinsertion, so later repetitions primarily measure the not-found path.

The LFU storage-pressure path reported 2.75 Mops/s, two eventual swap-outs, four evaluated pages, and 9,616 ns cumulative selection time. The Python pressure benchmark measured 4.69 Mops/s for a synchronous training step; asynchronous lookup, assign, and optimizer step measured 12.38, 13.45, and 17.16 Mops/s respectively.

These are reproducibility snapshots, not portable performance guarantees. See [Storage, Concurrency, and Performance](docs/storage_and_performance.md) for workload definitions, telemetry semantics, and verification commands.

### Automatic Storage Management

No storage configuration is required. `EmbeddingStore` detects system and cgroup memory limits, process resource limits, filesystem capacity, and local device characteristics. Cache-line-aligned pages stay in RAM while the safe memory budget has capacity.

A condition-variable-driven background thread tracks configurable watermarks (`high_watermark` and `low_watermark`, defaulting to 0.8 and 0.6). It does not poll periodically or probabilistically skip required eviction. At the high watermark, a score-enabled eviction proxy evaluates candidates and swaps cold pages through `MAP_FIXED` to an unlinked file-backed mapping until memory reaches the low watermark.

Selection is exact for at most 1,024 evictable pages. Larger sets evaluate at most 256 unique sampled pages per pass and continue making passes until the low watermark is reached. Candidate snapshots and proxy generations prevent stale candidates from being applied, and scoring runs without holding the storage mutex.

The first disk-backed page emits `SparseEmbeddingSpillWarning` to stderr. `storage_stats()` also raises a standard Python warning once and returns current memory and spill usage:

```python
import sparse_embedding

config = sparse_embedding.EmbeddingConfig(dims=[128])
store = sparse_embedding.EmbeddingStore(config)
print(store.storage_stats())
```

`spill_count` counts disk-backed page allocations, including reused file
extents. It is not a count of distinct memory-pressure episodes.

`storage_stats()` also reports `swap_out_count`, `eviction_selection_count`,
`eviction_evaluated_pages`, `eviction_selection_ns`, and
`eviction_candidate_invalidated_count`. These counters are cumulative
diagnostics; they do not represent current resident page counts. In particular,
`swap_out_count` is distinct from `spill_count`: the former counts background
RAM-to-file remaps, while the latter counts disk-backed page allocations.

Advanced users can override automatic selection without changing store operations:

```python
storage = sparse_embedding.DiskSpill("/mnt/nvme/sparse-embedding", 1 << 40)
config = sparse_embedding.EmbeddingConfig(dims=[128], storage=storage)
```

`MemoryOnly()` disables disk spill and raises when the detected safe memory budget is exhausted. Key hash metadata stays RAM-resident and grows dynamically per shard. The structural embedding-index limit and actual storage allocations report their own resource errors; disk spill does not make hash metadata disk-resident.

For known large workloads, call `store.reserve(keys_count, dim)` or
`store.reserve_per_shard(counts, dim)` before the timed phase. This prepares
ParlayHash capacity outside the hot path without globally serializing Upsert.

---

## Architecture Overview

```
HogwildEmbeddingStore
├── parlayhash (KeyMap)          # Sharded Key→uint32_t mapping
├── EmbeddingArena               # Cache-line aligned contiguous embedding storage
│   └── PageTable[8192]          # Atomic page pointers, lazy allocated
├── PageStorage                   # Auto-detected RAM budget + file-backed spill
├── TransientArena[num_threads]  # Per-shard gradient accumulation (lock-free)
├── exec::static_thread_pool     # C++23 stdexec thread pool
└── pro::proxy<OptimizerFacade<ParamT, OptT>> # Type-erased optimizer
```

### Key Design Decisions

1. **Decoupled Key-Value Storage**: The traditional `Key -> Vector` hash map is split into a two-level structure: `Key -> Index` (`parlayhash`) and `Index -> Vector` (`EmbeddingArena`). The hash table only stores lightweight integer indices, keeping the embedding vectors compactly stored, contiguous, and aligned. This significantly improves CPU cache hit rates and enables efficient SIMD vectorization.

2. **Shard-Local Synchronization**: ParlayHash lookup uses a shared lock; Upsert, erase, rebuild, and internal expansion use an exclusive lock for only the affected shard. Independent shards remain concurrent, avoiding the correctness failure of unsynchronized growth and the throughput collapse of a global Upsert lock.

3. **Paged Arena Allocation**: The underlying `EmbeddingArena` uses a PageTable-like structure similar to operating system virtual memory. By lazily allocating large memory pages, it avoids massive upfront memory allocation overhead and prevents memory fragmentation caused by high-frequency, small-scale allocations.

4. **Conflict-Free Gradient Accumulation**: During the backward pass, each thread is assigned an independent shard of a `TransientArena`. Threads accumulate gradients locally within their shards, which are then aggregated and applied using SIMD-accelerated optimizer steps. This prevents false sharing and atomic contention when multiple threads update the same embedding vector.

5. **Storage as a Deep Module**: Embedding, optimizer-state, score, and transient-gradient arenas share one page-storage interface. Allocation policy, resource detection, disk selection, warning emission, and extent reuse stay off the lookup and optimizer hot paths.

---

## Build & Install

This library now integrates with the standard Python packaging workflow, automatically invoking Bazel underneath:

```bash
# Compile from source and install the wheel
pip install .

# Or install in editable mode for development
pip install -e .
```

If you only need to run the C++ binaries or test directly:

```bash
bazel build -c opt //sparse_embedding/...
```

## Running Benchmarks

```bash
# Full API benchmark (recommendation system scenario, power-law access distribution)
bazel run -c opt //benchmarks:embedding_store_benchmark

# Hot-path microbenchmarks (isolating epoch / parlayhash / memcpy overheads)
bazel run -c opt //benchmarks:hotpath_microbench

# Python synchronous/asynchronous training under eviction pressure
PYTHONPATH="$PWD" python3 benchmarks/python/benchmark_e2e.py

# Five-run storage-pressure telemetry benchmark
PYTHONPATH="$PWD" python3 benchmarks/python/benchmark_storage_pressure.py
```
