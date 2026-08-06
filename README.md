# sparse_embedding

[中文](README_zh.md)

High-performance Sparse Embedding Table library for large-scale recommendation system training, using Hogwild-style relaxed updates, a slot-directory read path, x86 SeqLock publication, and paged storage.

## Core Features

- **Slot Directory Read Path**: Uses `parlayhash` for stable Key→slot lookup and a segmented locator table for Slot→arena publication, keeping warmed lookup/forward readers off shard shared locks and lifecycle locks.
- **SIMD Acceleration**: AVX-512 vectorized hash computation and gradient Scatter operations.
- **Hogwild! Gradient Accumulation**: Sharded TransientArena for lock-free gradient aggregation, avoiding cross-thread synchronization.
- **C++23 Standard Concurrency**: Sender/Receiver task scheduling based on NVIDIA `stdexec` + `exec::static_thread_pool`.
- **Epoch Memory Safety**: Batch protection for concurrent hash map read/write using `epoch::with_epoch`, avoiding ABA problems.
- **Distributed Lance Checkpointing**: Fluxway-coordinated two-phase commit writes manifest-pinned `entries` and physical `page_images`, including slot layout, `seq_count` values, payload/state, and SHA-256 validation for elastic restore. The Lance manifest is the only checkpoint metadata record; distributed restore requires an explicit `filter_fn`, and routing is owned exclusively by the caller.
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

The Bazel configuration compiles C++ sources with `-std=c++23`. The current acceptance build uses GCC 13.4 and Bazel 7.7.

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

### Accepted C++ Proofs (2026-08-04)

The accepted refactor performance data comes from two purpose-built proof binaries. The broader `embedding_store_benchmark --bind-affinity --skip-pressure` run violated the new caller-dedup contract and threw `key shard batch contains duplicate key`, so it is recorded only as a failed attempt.

#### Slot Directory Forward Scaling

The isolated forward proof uses a dimension-32 memory-only store with 128 already-published keys, a 4,096-key fixed power-law batch, seed `0x5EED5CA1E`, 400 timed `forward_lookup` repetitions, 48 allowed CPUs, and thread counts 1/2/4/8/16/64.

| Threads | Forward throughput |
|---:|---:|
| 1 | 9.632 Mops/s |
| 2 | 16.436 Mops/s |
| 4 | 27.914 Mops/s |
| 8 | 40.707 Mops/s |
| 16 | 42.049 Mops/s |
| 64 | 12.591 Mops/s |

The scaling ratios versus one thread are 1.706x at two threads, 2.898x at four threads, and 4.226x at eight threads, passing the current 1.6x/2.5x/3.5x forward-scaling gate.

#### Sustained Retirement Pressure and Sink

The sustained-pressure proof uses eight producers, 1,048,576 total keys, dimension 8, Auto storage, an 8 MiB resident-memory limit, a 512 MiB spill limit, and 0.75/0.50 watermarks. It validates retained payloads after the pressure phase and reports the current retirement/Sink contract, not disk spill/swap throughput.

| Metric | Result |
|---|---:|
| Pressure pages (`spill_count + swap_out_count`) | 0 |
| `spill_count` | 0 |
| `swap_out_count` | 0 |
| `memory_bytes` | 294,912 |
| `spill_bytes` | 0 |
| `retired_bytes` | 2,109,456 |
| `retirement_sink_hit_count` | 1,047,040 |
| Allocation p95 | 1.163 ms |
| Allocation p99 | 2.599 ms |
| Backlog drain | 0.003 ms |
| Producer elapsed | 129.397 ms |

This run intentionally opened the retirement-pressure circuit before page spill/swap became necessary. The accepted observation is that Sink admission handled pressure without payload corruption.

These are reproducibility snapshots, not portable performance guarantees. See [Storage Architecture and API](docs/storage_architecture.md) for architecture and telemetry semantics, and [Performance and Reliability Report](docs/performance_and_reliability_report.md) for workload definitions and verification commands.

Mutation measurements deduplicate power-law candidates before calling the
unique-mutation APIs and report rows actually processed. Use
`training_path_profile` for pure assign/accumulate/optimizer measurements. The
profile reuses one hash across gradient routing, transient accumulation, and
optimizer lookup; current affinity-enabled measurements are documented in the
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
├── EmbeddingArena               # Cache-line aligned contiguous embedding storage
│   └── PageTable[8192]          # Atomic page pointers, lazy allocated
├── PageStorage                   # Auto-detected RAM budget + file-backed spill
├── TransientArena[thread_budget]  # Per-shard gradient accumulation (lock-free)
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
