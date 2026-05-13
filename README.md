# sparse_embedding

High-performance Sparse Embedding Table library—designed for large-scale recommendation system training, based on the Hogwild! lock-free concurrent architecture.

## Core Features

- **Lock-free Concurrency**: Implements Key→Index O(1) lock-free concurrent read/write based on `parlayhash` (Parlay Parallel Hash Map).
- **SIMD Acceleration**: AVX-512 vectorized hash computation and gradient Scatter operations.
- **Hogwild! Gradient Accumulation**: Sharded TransientArena for lock-free gradient aggregation, avoiding cross-thread synchronization.
- **C++26 Standard Concurrency**: Standard Sender/Receiver task scheduling based on NVIDIA `stdexec` + `exec::static_thread_pool`.
- **Epoch Memory Safety**: Batch protection for concurrent hash map read/write using `epoch::with_epoch`, avoiding ABA problems.
- **Distributed Lance Checkpointing**: Native distributed two-phase commit Lance backend with zero-copy PyArrow exports, independently storing embeddings and optimizer states for ultra-fast distributed storage and retrieval.
- **Dynamic Partitioner OPs**: High-performance C++ Partitioners (`ModuloPartitioner`, `HashPartitioner`, `FastRangePartitioner`) supporting efficient feature routing and on-the-fly cluster resharding during checkpoint loading.
- **Eviction Management**: Memory-efficient scoring policies (e.g., LFU) using thread-safe `std::atomic_ref` for concurrent state updates without locking.

## System Dependencies

| Component | Version |
|------|------|
| GCC | ≥ 15.0 (Requires C++26 constexpr exceptions support) |
| stdexec | NVIDIA main branch `931f256c` |
| parlayhash | main branch (including 56-bit pointer mask patch) |
| lance-c | main branch (Rust-based C shared library) |
| Rust | 1.70+ (2024 Edition) |
| Bazel | ≥ 7.0 |

> **Note**: GCC 13 and below do not support `__cpp_constexpr_exceptions`, causing `stdexec`'s `sender_in` concept checks to fail, rendering `sync_wait` / `start_detached` unusable. GCC 15+ is required.

---

## Performance Benchmarks

### Test Environment

| Item | Specification |
|------|------|
| **CPU** | Intel® Xeon® Silver 4116 × 2 Socket |
| **Physical Cores** | 12 cores/socket × 2 = **24 cores**, **48 vCPUs** with Hyper-Threading |
| **CPU Frequency** | 2.10 GHz (Turbo 2.40 GHz) |
| **L1d Cache** | 768 KiB |
| **L2 Cache** | 24 MiB |
| **L3 Cache** | 33 MiB (per socket) |
| **NUMA** | 2 nodes, inter-node latency ratio 2.1x (node 0: 64 GB, node 1: 64 GB) |
| **Memory** | 128 GiB DDR4 |
| **OS** | Ubuntu 20.04 |
| **Compiler** | GCC 15.2.0, `-O2 -march=native` |

### Test Scenarios

Simulating realistic recommendation system training scenarios:

| Parameter | Value |
|------|-----|
| Embedding dimension (dim) | 128 floats (512 bytes) |
| Vocabulary size (vocab) | 500,000 unique keys |
| Batch size | 65,536 |
| Access distribution | Power-law: top-10% keys account for 80% traffic |
| Slot count | 1 |

### Throughput Results

> Unit: **Mops/s** (Million operations / second); `forward_lookup` / `assign` also mark memory bandwidth (GB/s)

#### `forward_lookup` (Inference Forward, Hot Path)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| forward_lookup | 3.24 Mops/s (1.66 GB/s) | 7.37 Mops/s (3.77 GB/s) | 11.23 Mops/s (5.75 GB/s) | 12.54 Mops/s (6.42 GB/s) | 15.08 Mops/s (7.72 GB/s) | 12.91 Mops/s (6.61 GB/s) |

`forward_lookup` saturates around 16-32T, with an observed peak of 6.94 GB/s.

**Note**: This is NOT DRAM bandwidth saturation (theoretical peak is 230.4 GB/s, actual utilization is only 3%), but rather a **random access latency bottleneck**:
- Each embedding lookup requires random access to 512 bytes (8 cache lines) in the arena, DRAM access latency ~90 ns
- Single-thread random read bandwidth limit: 64B / 90ns ≈ **0.71 GB/s**
- Reaching theoretical bandwidth requires maintaining ~324 outstanding misses simultaneously, whereas Xeon Silver 4116 has only ~12 MSHRs per core
- Adding more threads cannot increase throughput because the DRAM controller's outstanding request queue is full, not because the bus bandwidth is insufficient

#### `lookup_or_insert` (Training Insert, Write Hot Path)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| lookup_or_insert | 1.48 Mops/s | 2.60 Mops/s | 4.02 Mops/s | 4.88 Mops/s | 5.06 Mops/s | 5.17 Mops/s |

1→8T shows a 3.6x linear speedup; post-8T, it saturates due to parlayhash CAS contention and DRAM write bandwidth limits.

#### `assign` (Parameter Assignment)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| assign | 1.49 Mops/s (0.76 GB/s) | 3.19 Mops/s (1.63 GB/s) | 4.93 Mops/s (2.52 GB/s) | 5.90 Mops/s (3.02 GB/s) | 6.15 Mops/s (3.15 GB/s) | 6.28 Mops/s (3.21 GB/s) |

#### `accumulate_gradients` (Gradient Accumulation, Backward Pass)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| accumulate_gradients | 2.11 Mops/s | 5.76 Mops/s | 8.81 Mops/s | 11.33 Mops/s | 11.06 Mops/s | 10.66 Mops/s |

Note: `accumulate_gradients` uses AVX-512 SIMD sharded Scatter-Reduce. With the fixed thread-pool threshold and thread-local buffers, the throughput scales efficiently up to 16-32 threads.

#### `optimizer_step` (Parameter Update)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| optimizer_step | 5.67 Mops/s (5.80 GB/s) | 12.99 Mops/s (13.30 GB/s) | 19.82 Mops/s (20.30 GB/s) | 30.73 Mops/s (31.47 GB/s) | **31.15 Mops/s (31.90 GB/s)** | 28.95 Mops/s (29.65 GB/s) |

Note: `optimizer_step` processes the accumulated unique key set (approx. 252K). With proper parallel dispatch and removal of data races, the update scales impressively to 25.06 GB/s at 32 threads.

#### `contains` (Existence Check)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| contains | 9.40 Mops/s | 16.57 Mops/s | 20.69 Mops/s | 22.48 Mops/s | 23.64 Mops/s | 22.65 Mops/s |

Pure hash lookup without embedding copies, limited by CPU and hash table bucket access latency, nears saturation after 8T.

#### `erase` (Entry Deletion, 10% Batch)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| erase (10% batch) | 12.39 Mops/s | 34.61 Mops/s | 63.85 Mops/s | 61.76 Mops/s | 49.25 Mops/s | 27.92 Mops/s |

`erase` reaches its peak at 8T (64.49 Mops/s), and begins to decline due to NUMA cross-socket contention under Hyper-Threading.

### Python API Performance

The following throughputs were measured by calling the corresponding interfaces in Python (via `nanobind` and DLPack zero-copy wrappers). These numbers include Python/C++ context switching and asyncio scheduling overheads:

| API | Throughput (Mops/s) | Description |
|-----|---------------------|-------------|
| `async_unique` | ~25.58 Mops/s | Async fetch of unique keys, ultra-fast hashing |
| `async_lookup_or_insert` | ~22.72 Mops/s | Async insert and lookup, returning zero-copy tensors |
| `async_lookup` | ~27.40 Mops/s | Async forward lookup without blocking the event loop |
| `async_assign` (Zipfian) | ~2.85 Mops/s | Async assignment under heavy power-law distribution |
| **End-to-End Step** | **~3.87 Mops/s** | Full training step using full asyncio ops |

All `EmbeddingStore` methods (e.g. `lookup_or_insert`, `assign`, `contains`, `erase`, `clear`, `assign_scores`, `accumulate_gradients`, `optimizer_step`) have high-performance, edge-triggered `async_*` variants. Thanks to DLPack and lock-free `eventfd` asyncio integration, the Python invocation overhead is minimized, easily sustaining millions of operations per second.

---

## Architecture Overview

```
HogwildEmbeddingStore
├── parlayhash (KeyMap)          # Lock-free concurrent Key→uint32_t mapping
├── EmbeddingArena               # Cache-line aligned contiguous embedding storage
│   └── PageTable[8192]          # Atomic page pointers, lazy allocated
├── TransientArena[num_threads]  # Per-shard gradient accumulation (lock-free)
├── exec::static_thread_pool     # C++26 stdexec thread pool
└── pro::proxy<OptimizerFacade>  # Type-erased optimizer (SGD, Adam, etc.)
```

### Key Design Decisions

1. **Decoupled Key-Value Storage**: The traditional `Key -> Vector` hash map is split into a two-level structure: `Key -> Index` (`parlayhash`) and `Index -> Vector` (`EmbeddingArena`). The hash table only stores lightweight integer indices, keeping the embedding vectors compactly stored, contiguous, and aligned. This significantly improves CPU cache hit rates and enables efficient SIMD vectorization.

2. **Lock-Free Everywhere**: Mutexes are entirely eliminated across the entire data path, from forward lookup to backward propagation. The hash table uses CAS primitives for concurrent insertions, memory allocation relies on atomic pointers, and gradient updates use the Hogwild! relaxed consistency model. This eliminates scaling bottlenecks on high-core-count NUMA machines.

3. **Paged Arena Allocation**: The underlying `EmbeddingArena` uses a PageTable-like structure similar to operating system virtual memory. By lazily allocating large memory pages, it avoids massive upfront memory allocation overhead and prevents memory fragmentation caused by high-frequency, small-scale allocations.

4. **Conflict-Free Gradient Accumulation**: During the backward pass, each thread is assigned an independent shard of a `TransientArena`. Threads accumulate gradients locally within their shards, which are then aggregated and applied using SIMD-accelerated optimizer steps. This prevents false sharing and atomic contention when multiple threads update the same embedding vector.

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
```
