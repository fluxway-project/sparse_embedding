# sparse_embedding

[中文](README_zh.md)

High-performance Sparse Embedding Table library for large-scale recommendation system training, using Hogwild-style relaxed updates, a slot-directory read path, x86 SeqLock publication, and paged storage.

## Core Features

- **Slot Directory Read Path**: Uses `parlayhash` for stable Key→slot lookup and a segmented locator table for Slot→arena publication, keeping warmed lookup/forward readers off shard shared locks and lifecycle locks.
- **SIMD Acceleration**: AVX-512 vectorized hash computation and gradient Scatter operations.
- **Hogwild! Gradient Accumulation**: Sharded TransientArena for lock-free gradient aggregation, avoiding cross-thread synchronization.
- **Batch Optimizer Extension**: `OptimizerExtensionFacade<ParamT, OptT>` is the single optimizer type-erasure seam. The store resolves locator rows under shard-local SeqLock write guards and dispatches one `update(OptimizerBatchView, global_step)` per key shard instead of calling a proxy for every row.
- **C++23 Standard Concurrency**: Sender/Receiver task scheduling based on NVIDIA `stdexec` + `exec::static_thread_pool`.
- **Epoch Memory Safety**: Batch protection for concurrent hash map read/write using `epoch::with_epoch`, avoiding ABA problems.
- **Distributed Lance Checkpointing**: A framework-neutral prepare/write/commit protocol writes manifest-pinned `entries` and physical `page_images`, including slot layout, sequence values, payload/state, and SHA-256 validation. Exact-version restore supports physical participants or caller-owned elastic partition functions. Monotonic commit authority rejects stale coordinators without depending on a specific distributed framework.
- **Selective Optimizer Loading**: `EmbeddingConfig(enable_optimizer=False)` creates a weight-only inference store without optimizer-state allocation. Training restores load optimizer state by default, while `restore_weight_snapshot` always performs a weight-only restore.
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
| libnuma | numactl 2.0.19 (Bazel module `2.0.19.bcr.1`) |

The Bazel configuration compiles C++ sources with `-std=c++23`; GCC 13+ and Bazel 7+ are supported.

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
| Toolchain | GCC 13.4, Bazel 7.7, C++23 (`-std=c++23`), `-c opt` |
| Optimization flags | `-O3 -march=native -mtune=native -mavx512bf16 -fno-strict-aliasing -DNDEBUG`; link with `-flto` |
| CPU affinity | Slot-directory forward proof binds workers through the benchmark affinity helper; 48 CPUs were allowed |

### Forward scaling

The benchmark uses a warmed memory-only store, fixed batches, and physical-core affinity for implementation comparisons. Detailed workload definitions and reference tables are maintained in the performance report.

#### Slot-directory forward scaling

The reference workload uses a dimension-32 memory-only table, 4,096-key fixed
power-law batches, 400 timed `forward_lookup` repetitions, and physical-core
affinity on 48 allowed CPUs.

| Threads | Forward throughput |
|---:|---:|
| 1 | 13.40 Mops/s |
| 2 | 20.25 Mops/s |
| 4 | 34.37 Mops/s |
| 8 | **44.46 Mops/s** |
| 16 | 35.69 Mops/s |
| 24 | 34.21 Mops/s |
| 32 | 36.56 Mops/s |
| 64 | 31.38 Mops/s |

The performance report contains the complete workload definition, concurrency
tables, compaction scaling, and reliability validation commands.

### NUMA-aware execution and pressure validation

`--numa-aware-sharding` enables execution and memory NUMA locality: workers bind
to physical CPUs and arena pages are placed by shard on the corresponding node.
The public pressure contract and validation command are documented in the
[Performance and Reliability Report](docs/performance_and_reliability_report.md).

### Lock-free migration and expansion

Performance references are collected in the [Performance and Reliability Report](docs/performance_and_reliability_report.md):
The current eight-worker `training_path_profile` reaches 20.00 Mops/s for `assign`,
5.68 Mops/s for `accumulate_gradients`, and 11.80 Mops/s for `optimizer_step`.
Expansion uses precise per-shard
budgets instead of equal power-of-two rounding: directory capacity uses a 2.5x probe
budget rounded to 16, while runtime `MapBundle` capacity grows by 2x below 64K,
1.5x for `64K <= capacity < 1M`, and 1.25x at or above 1M.

Mutation measurements deduplicate power-law candidates before calling the
unique-mutation APIs and report rows actually processed. Use
`training_path_profile` for pure assign/accumulate/optimizer measurements. The
profile reuses one hash across gradient routing, transient accumulation, and
optimizer lookup; the workload definition and statistics are documented in the
performance report.
NUMA-aware sharding is opt-in with `--numa-aware-sharding`; it places arena pages
and workers by node, rebinds pooled threads to their assigned CPU, and caps the
benchmark profile at physical cores so SMT siblings are not treated as extra
memory bandwidth. Lookup-only forward uses the shard `ReadView` and reuses the
route hash. For estimated payloads below 512 KiB it skips NUMA re-bucketing,
because dispatch overhead exceeds locality benefit for small batches; larger
payloads retain node-local routing. High worker counts remain bounded by socket
bandwidth, while data path and API semantics stay unchanged.

The long-lived `exec::static_thread_pool` removes thread creation/destruction,
but each `bulk + sync_wait` still pays task enqueue, wakeup, steal, completion,
and barrier costs. An empty-bulk proof measured about 3.47/21.7/41.6/87.4 µs
at 1/2/8/24 workers.

For a reproducible disk-swap pressure check covering every public operation,
run `small_memory_all_paths_pressure_test`. Its 64 MiB Auto-storage profile
completed with real swap-outs and verified post-swap payloads:

```bash
bazel test -c opt //sparse_embedding/src/backend/hogwild:small_memory_all_paths_pressure_test \
  --test_output=errors
```

### Automatic Storage Management

Storage topology is calibrated once during store construction. The tuner uses
the embedding dimension, worker budget, workload, and NUMA mode to compare the
small candidate set `4096/8192/16384` with robust median samples. Results are
cached by request shape; explicit `entries_per_page` remains the reproducible
override and avoids calibration.

No storage configuration is required. `EmbeddingStore` detects system and cgroup memory limits, process resource limits, filesystem capacity, and local device characteristics. Cache-line-aligned pages stay in RAM while the safe memory budget has capacity.

A condition-variable-driven background thread tracks configurable watermarks (`high_watermark` and `low_watermark`, defaulting to 0.8 and 0.6). It does not poll periodically or probabilistically skip required eviction. At the high watermark, a score-enabled eviction proxy evaluates candidates and swaps cold pages through `MAP_FIXED` to an unlinked file-backed mapping until memory reaches the low watermark.

Selection is exact for at most 1,024 evictable pages. Larger sets evaluate at most 256 unique sampled pages per pass and continue making passes until the low watermark is reached. Candidate snapshots and proxy generations prevent stale candidates from being applied, and scoring runs without holding the storage mutex.

The first disk-backed page emits `SparseEmbeddingSpillWarning` to stderr. `storage_stats()` also raises a standard Python warning once and returns current memory and spill usage:

```python
import sparse_embedding

config = sparse_embedding.EmbeddingConfig(
    dims=[128],
    storage_tuning=sparse_embedding.StorageTuningOptions(
        target_capacity_keys=10_000_000,
        memory_budget_bytes=4 << 30,
    ),
)
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
storage = sparse_embedding.AutoStorage(
    memory_limit_bytes=0,
    spill_directory="/mnt/nvme/sparse-embedding",
    max_spill_bytes=1 << 40,
)
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
├── EmbeddingArena               # Cache-line aligned embedding storage
│   └── StablePageDirectory      # Stable-address segmented page metadata
├── PageStorage                   # Auto-detected RAM budget + file-backed spill
├── TransientArena[thread_budget]  # Per-shard gradient accumulation (lock-free)
├── exec::static_thread_pool     # C++23 stdexec thread pool
└── pro::proxy<OptimizerExtensionFacade<ParamT, OptT>> # Batch optimizer seam
```

### Key Design Decisions

1. **Decoupled Key-Value Storage**: The traditional `Key -> Vector` hash map is split into a two-level structure: `Key -> Index` (`parlayhash`) and `Index -> Vector` (`EmbeddingArena`). The hash table only stores lightweight integer indices, keeping the embedding vectors compactly stored, contiguous, and aligned. This significantly improves CPU cache hit rates and enables efficient SIMD vectorization.

2. **Shard-Local Synchronization**: ParlayHash lookup uses a shared lock; Upsert, erase, rebuild, and internal expansion use an exclusive lock for only the affected shard. Independent shards remain concurrent, avoiding the correctness failure of unsynchronized growth and the throughput collapse of a global Upsert lock.

3. **Paged Arena Allocation**: The underlying `EmbeddingArena` uses a PageTable-like structure similar to operating system virtual memory. By lazily allocating large memory pages, it avoids massive upfront memory allocation overhead and prevents memory fragmentation caused by high-frequency, small-scale allocations.

4. **Conflict-Free Gradient Accumulation**: During the backward pass, each thread is assigned an independent shard of a `TransientArena`. Threads accumulate gradients locally within their shards, which are then aggregated and applied using SIMD-accelerated optimizer steps. This prevents false sharing and atomic contention when multiple threads update the same embedding vector.

5. **Batch-Scoped Optimizer Dispatch**: Locator resolution and parameter mutation stay inside the same packed SeqLock guard callback. The optimizer proxy is entered once per resolved shard batch; parameter spans are never retained for deferred updates.
6. **Storage as a Deep Module**: Embedding, optimizer-state, score, and transient-gradient arenas share one page-storage interface. Allocation policy, resource detection, disk selection, warning emission, and extent reuse stay off the lookup and optimizer hot paths.

### Training Update Path

The training path (`accumulate_gradients()` followed by `optimizer_step()`)
keeps gradient accumulation in the per-shard `TransientArena`.
`optimizer_step()` resolves a shard batch, enters `OptimizerExtensionFacade`
once, and consumes all parameter spans before the locator
`PackedWriteGuardBatch` is released.

Optimizer extension dispatch is deliberately batch-scoped. Static dispatch is
an implementation option for the same contract, and `std::variant` remains a
cold configuration mechanism rather than a per-row runtime path.

Mutation APIs require unique `(key, dimension)` rows. `lookup_or_insert` may
coalesce duplicate misses, but `assign`, `accumulate_gradients`, and
`optimizer_step` consume batches produced from published keys. A concurrent
erase/recycle race is sampled in release builds: the producer increments a
relaxed counter and publishes a fixed-size diagnostic event to the bounded
telemetry ring. Its single consumer may write the event to `std::clog`; the
producer performs no I/O, and a full ring drops the event without blocking.
Debug builds throw immediately.

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
