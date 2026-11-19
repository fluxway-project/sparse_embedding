# sparse_embedding

[中文](README_zh.md)

High-performance sparse storage and computation library purpose-built for **PB-scale (Petabyte-scale)** sparse Embeddings (industrial recommender systems), sparse Engram (conditional memory and knowledge retrieval), sparse MoE (expert routing and parameter storage), and sparse KV cache (long-context tail caching). Combining user-friendly Python APIs with a high-concurrency C++23 engine, it leverages Hogwild-style relaxed updates, a slot-directory read path, x86 SeqLock publication, and paged virtual memory architecture to break through memory wall, hash collision, and I/O bottlenecks under massive parameter scales.

## Core Features

- **Slot Directory Read Path**: A flat key directory resolves slot IDs; the segmented locator table resolves arena rows. Lookup read paths bypass the shard writer mutex entirely.
- **SIMD Acceleration**: AVX-512 vectorized hash computation and gradient Scatter operations.
- **Hogwild! Gradient Accumulation**: Sharded TransientArena for lock-free gradient aggregation, avoiding cross-thread synchronization.
- **Batch Optimizer Extension**: `OptimizerExtensionFacade<ParamT, OptT>` is the single optimizer type-erasure seam. The store resolves locator rows under shard-local SeqLock write guards and dispatches one `update(OptimizerBatchView, global_step)` per key shard instead of calling a proxy for every row.
- **C++23 Standard Concurrency**: Sender/Receiver task scheduling based on NVIDIA `stdexec` + `exec::static_thread_pool`.
- **Epoch Memory Safety**: Batch protection for concurrent hash map read/write using `epoch::with_epoch`, avoiding ABA problems.
- **Distributed Lance Checkpointing**: Two-phase commit protocol separating `snapshot_manifest.lance` (control ledger) and `entries.lance` (worker slices). Supports 4 native Arrow integer key types (`int32`, `uint32`, `int64`, `uint64`) with DuckDB pushdown and optional physical page cache.
- **Selective Optimizer Loading**: `EmbeddingConfig(enable_optimizer=False)` creates a weight-only inference store. `checkpoint.load_inference_checkpoint` (or storage `load_inference_weights`) restores weights only.
- **Segregated Optimizer Memory Alignment**: Adam optimizer decouples vector and scalar alignments, maintaining 64-byte AVX-512 SIMD alignment for momentum vectors (`m/v`) while packing scalar step counters with 4-byte natural alignment (`OptimizerRowLayout::kNaturalAligned`) into 16 KiB pages to reduce the step working set.
- **Configurable Shard Mutation Journaling**: Shard-local Change Data Capture (CDC) can be enabled or disabled via `EmbeddingConfig.enable_mutation_journal`. When disabled for pure training or FULL-only checkpoints, hot-path sequence accounting and record memory overhead are eliminated; when enabled, it tracks dirty keys and deletion tombstones for incremental checkpointing.
- **Partitioner Registry and Routing (PartitionerRegistry)**: Supports `modulo`, `hash`, `fast_range`, and dynamic algorithmic extensions. Offline restore uses AVX-512 `filter_span`; online routing (`parallel_partitioner.hpp`) integrates with Fluxway `FunctionPartition`.
- **Reference Vector Validation**: Canonical `generate_reference_vectors` validates bit-identical partition mappings between distributed runtimes and offline restore engines across all key types.
- **Eviction Management**: Memory-efficient scoring policies (e.g., LFU) using atomic in-place updates for concurrent state tracking without locking.
- **Page-Level Hot/Cold Swapping**: Background eviction of cold data pages through address-stable `MAP_FIXED` remapping to file-backed storage based on configurable memory watermarks.

## Documentation

- [Storage Architecture and API Contracts](docs/storage_architecture.md): Concurrency model, SeqLock locator, hardware auto-tuning, and disk spilling.
- [Lance Storage and Gravitino Integration](docs/lance_storage_and_gravitino.md): Dual-Lance 2PC protocol, Arrow native pushdown, and Lakehouse catalog.
- [Incremental Checkpointing and Streaming Compaction](docs/incremental_checkpoint.md): Mutation journaling, non-blocking cuts, and streaming compaction.
- [Dynamic Learning Rate](docs/dynamic_learning_rate.md): Score-based frequency-adaptive learning rate policies.
- [Performance and Reliability Report](docs/performance_and_reliability_report.md): Concurrency scaling benchmarks, memory watermarks, and verification gates.

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
| CPU affinity | Slot-directory forward benchmark binds workers through the affinity helper; 48 CPUs were allowed |

### Performance Summary Across Operations

Validated benchmark results across hot-path C++ and Python workloads on the reference platform (see the [performance report](docs/performance_and_reliability_report.md) for multi-run medians):

| Operation Category | Target Benchmark / Kernel | Concurrency / Configuration | Throughput | Bandwidth or measurement notes |
|---|---|---|---:|---:|
| **Directory Lookup** | C++ pure find microbench (`hotpath_microbench`) | 32 threads, 100K keys | **232.25 Mops/s** | CPU cache bound |
| **Forward Lookup** | C++ slot-directory forward (`forward_scaling_benchmark`) | 8 threads, dim=32, batch=4K | **52.164 Mops/s** | 4,096 shuffled unique keys |
| **Forward Lookup (Wide)** | C++ full store forward (`embedding_store_benchmark`) | 16 threads, NUMA sharding off, dim=128, batch=65K | **26.26 Mops/s** | **13.45 GB/s** |
| **In-Place Assign** | Python multi-threaded table assign (`benchmark_embedding_table`) | 32 threads, dim=128, batch=65K | **31.37 Mops/s** | 16.06 GB/s (FP32 payload) |
| **In-Place Assign** | C++ full store assign (`embedding_store_benchmark`) | 16 threads, NUMA, dim=128, batch=65K | **33.43 Mops/s** | **17.11 GB/s** |
| **Optimizer Step** | C++ Adam hot-set profile (`training_path_profile`) | 8 workers, dim=128, 4K keys | **17.89 Mops/s** | 8.54 GiB/s |
| **Optimizer Step** | C++ full store SGD step (`embedding_store_benchmark`) | 16 threads, NUMA, dim=128, batch=65K | **20.065 Mops/s** | **20.545 GB/s** |
| **Adam Training Step** | Python gradient accumulation + Adam commit ([benchmark](benchmarks/python/benchmark_adam_training_step.py)) | 16 threads, dim=16, batch=65K (p50: 6.00 ms, p99: 6.43 ms) | **10.68 Mops/s** | Not measured |
| **Gradient Accumulation** | C++ gradient accumulation (`embedding_store_benchmark`) | 16 threads, NUMA, dim=128, batch=65K | **4.28 Mops/s** | **2.19 GB/s** |
| **Key Membership** | C++ contains check (`embedding_store_benchmark`) | 16 threads, NUMA, batch=65K | **22.295 Mops/s** | Metadata bound |
| **Batch Erase** | C++ batch erase (`embedding_store_benchmark`) | 1 thread, 10% batch | **9.70 Mops/s** | Requested keys/s across repeated erases |
| **End-to-End Step** | Python full training step (`benchmark_embedding_table`) | 32 threads, lookup + accum + step | **2.02 Mops/s** | E2E compute |
| **Async LFU lookup** | Python storage pressure (`benchmark_e2e`) | 4 threads, dim=16, batch=4,096 | **14.95 Mops/s** | p95 0.2825 ms |
| **Async Zipfian Access** | Python async Zipfian lookup (`benchmark_async_zipfian`) | 8-thread budget, dim=128, Zipf exponent 1.2 | **13.36 Mops/s** | Async scheduler |
| **Distributed Routing** | Parallel partitioner Lemire fast_range (`parallel_partitioner`) | 8 workers, 10M keys | **1.03 B keys/s** | Vectorized routing |
| **SIMD Slice Filter** | AVX-512 slice mask filter (`filter_span`) | 8 workers, Arrow memory slices | **1.45 B keys/s** | AVX-512 SIMD |

Reproduce the gradient accumulation and Adam measurement with `python benchmarks/python/benchmark_adam_training_step.py --threads 16 --keys 65536 --dim 16 --steps 30`.

### Forward scaling

The benchmark uses a warmed memory-only store, fixed batches, and physical-core affinity for implementation comparisons. Detailed workload definitions and reference tables are maintained in the performance report.

#### Slot-directory forward scaling

The reference workload uses a dimension-32 memory-only table, 4,096-key fixed
power-law batches, 400 timed `forward_lookup` repetitions, and physical-core
affinity on 48 allowed CPUs.

| Threads | Forward throughput (five-run median) |
|---:|---:|
| 1 | 14.61 Mops/s |
| 2 | 27.04 Mops/s |
| 4 | 43.11 Mops/s |
| 8 | **50.39 Mops/s** |
| 16 | 45.74 Mops/s |
| 24 | 40.11 Mops/s |
| 32 | 39.57 Mops/s |
| 64 | 37.42 Mops/s |

The performance report contains the complete workload definition, concurrency
tables, compaction scaling, and reliability validation commands.

### NUMA-aware execution and pressure validation

`--numa-aware-sharding` enables execution and memory NUMA locality: workers bind
to physical CPUs and arena pages are placed by shard on the corresponding node.
The public pressure contract and validation command are documented in the
[Performance and Reliability Report](docs/performance_and_reliability_report.md).

### Lock-free migration and expansion

Performance references are collected in the [Performance and Reliability Report](docs/performance_and_reliability_report.md):
The current eight-worker `training_path_profile` reaches 13.43 Mops/s for `assign`,
2.92 Mops/s for `accumulate_gradients`, and 14.63 Mops/s for `optimizer_step`.
Each shard reserves its own logical capacity. The runtime `MapBundle` capacity grows by 2x below 64K,
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

The long-lived `exec::static_thread_pool` avoids thread creation and destruction overhead across batch invocations.

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
    enable_mutation_journal=True,  # Set False to disable CDC tracking for pure training or FULL-only checkpoints
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
directory capacity outside the hot path.

---

## Architecture Overview

```
HogwildEmbeddingStore
├── StorageTopology (SSOT)       # Immutable topology & branchless bit-shift addressing
├── StorageResourceTuner         # 2MB HugePage & hardware auto-tuning
├── FlatKeyDirectory             # Sharded key-to-slot index
├── EmbeddingArena               # Cache-line aligned embedding storage
│   └── StablePageDirectory      # Stable-address segmented page metadata
├── PageStorage                  # Auto-detected RAM budget + file-backed spill
├── TransientArena[thread_budget]  # Per-shard gradient accumulation (lock-free)
├── exec::static_thread_pool     # C++23 stdexec thread pool
├── PartitionerRegistry (ops)    # Extensible partitioner registry (modulo/hash/fast_range)
├── ParallelPartitioner          # 1.03B keys/s multi-threaded router & AVX-512 mask filter
└── OptimizerExtensionFacade<ParamT, OptT> # Batch optimizer seam
```

### Key Design Decisions

1. **Decoupled Key-Value Storage**: Directory buckets store 32-bit locator slot IDs. Embedding vectors remain contiguous and aligned in the arena.

2. **Shard-Local Synchronization**: See the [shard publication contract](docs/storage_architecture.md#31-hash-routing-and-shard-publication) for insertion visibility, directory replacement, and reader lifetime.

3. **Paged Arena Allocation**: The underlying `EmbeddingArena` uses a PageTable-like structure similar to operating system virtual memory. By lazily allocating large memory pages, it avoids massive upfront memory allocation overhead and prevents memory fragmentation caused by high-frequency, small-scale allocations.

4. **Conflict-Free Gradient Accumulation**: During the backward pass, each thread is assigned an independent shard of a `TransientArena`. Threads accumulate gradients locally within their shards, which are then aggregated and applied using SIMD-accelerated optimizer steps. This prevents false sharing and atomic contention when multiple threads update the same embedding vector.

5. **Batch-Scoped Optimizer Dispatch**: Locator resolution and parameter mutation stay inside the same packed SeqLock guard callback. The optimizer proxy is entered once per resolved shard batch; parameter spans are never retained for deferred updates.
6. **Storage as a Deep Module**: Embedding, optimizer-state, score, and transient-gradient arenas share one page-storage interface. Allocation policy, resource detection, disk selection, warning emission, and extent reuse stay off the lookup and optimizer hot paths.

### Training Update Path

The training path (`accumulate_gradients()` followed by
`optimizer_step(normalization_steps=...)`) keeps gradient accumulation in the
per-shard `TransientArena`. `gradient_dtype` controls scaled accumulation
storage independently of `value_dtype`; after normalization, the optimizer
receives FP32 gradients, while `optimizer_dtype` controls its state.
`optimizer_step(normalization_steps=...)` resolves a shard batch, enters `OptimizerExtensionFacade`
once, and consumes all parameter spans before releasing the locator batch locks.

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

This library integrates standard Python packaging with Bazel, compiling the C++ native extension under the hood:

```bash
# Compile and install the package
pip install .

# Or install in editable mode for local development
pip install -e .

# Build redistributable binary wheel
pip wheel . --no-build-isolation
```

Run Python test suite:

```bash
# Install test extras and run pytest
pip install -e ".[test]"
pytest
```

Lance integration is enabled by default. Set `SE_BUILD_WITH_LANCE=0` for a standalone build without Lance:

```bash
SE_BUILD_WITH_LANCE=0 pip install -e .
```

To build and test C++ targets directly via Bazel:

```bash
bazel build -c opt //sparse_embedding/...
bazel test -c opt //sparse_embedding/...
```

See [Building and Testing Guide](docs/building_and_testing.md) for full architecture and configuration details.

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
