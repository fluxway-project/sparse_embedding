# sparse_embedding

High-performance Sparse Embedding Table library—designed for large-scale recommendation system training, based on the Hogwild! lock-free concurrent architecture.

## Core Features

- **Lock-free Concurrency**: Implements Key→Index O(1) lock-free concurrent read/write based on `parlayhash` (Parlay Parallel Hash Map).
- **SIMD Acceleration**: AVX-512 vectorized hash computation and gradient Scatter operations.
- **Hogwild! Gradient Accumulation**: Sharded TransientArena for lock-free gradient aggregation, avoiding cross-thread synchronization.
- **C++26 Standard Concurrency**: Standard Sender/Receiver task scheduling based on NVIDIA `stdexec` + `exec::static_thread_pool`.
- **Epoch Memory Safety**: Batch protection for concurrent hash map read/write using `epoch::with_epoch`, avoiding ABA problems.

## System Dependencies

| Component | Version |
|------|------|
| GCC | ≥ 15.0 (Requires C++26 constexpr exceptions support) |
| stdexec | NVIDIA main branch `931f256c` |
| parlayhash | main branch (including 56-bit pointer mask patch) |
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
| forward_lookup | 3.26 Mops/s (1.67 GB/s) | 8.26 Mops/s (4.23 GB/s) | 11.21 Mops/s (5.74 GB/s) | 11.70 Mops/s (5.99 GB/s) | 13.56 Mops/s (6.94 GB/s) | 12.41 Mops/s (6.36 GB/s) |

`forward_lookup` saturates around 16-32T, with an observed peak of 6.94 GB/s.

**Note**: This is NOT DRAM bandwidth saturation (theoretical peak is 230.4 GB/s, actual utilization is only 3%), but rather a **random access latency bottleneck**:
- Each embedding lookup requires random access to 512 bytes (8 cache lines) in the arena, DRAM access latency ~90 ns
- Single-thread random read bandwidth limit: 64B / 90ns ≈ **0.71 GB/s**
- Reaching theoretical bandwidth requires maintaining ~324 outstanding misses simultaneously, whereas Xeon Silver 4116 has only ~12 MSHRs per core
- Adding more threads cannot increase throughput because the DRAM controller's outstanding request queue is full, not because the bus bandwidth is insufficient

#### `lookup_or_insert` (Training Insert, Write Hot Path)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| lookup_or_insert | 1.64 Mops/s | 3.39 Mops/s | 5.85 Mops/s | 6.03 Mops/s | 6.29 Mops/s | 6.03 Mops/s |

1→8T shows a 3.6x linear speedup; post-8T, it saturates due to parlayhash CAS contention and DRAM write bandwidth limits.

#### `assign` (Parameter Assignment)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| assign | 1.58 Mops/s (0.81 GB/s) | 3.39 Mops/s (1.73 GB/s) | 6.42 Mops/s (3.29 GB/s) | 7.68 Mops/s (3.93 GB/s) | 7.65 Mops/s (3.92 GB/s) | 6.98 Mops/s (3.57 GB/s) |

#### `accumulate_gradients` (Gradient Accumulation, Backward Pass)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| accumulate_gradients | 3.02 Mops/s | 2.95 Mops/s | 2.96 Mops/s | 2.96 Mops/s | 3.03 Mops/s | **9.15 Mops/s** |

Note: `accumulate_gradients` uses AVX-512 SIMD sharded Scatter-Reduce; the bottleneck for 1-32T lies in write conflicts during the Scatter phase. At 64T, the TransientArena sharding granularity fully decouples, showing breakthrough improvements.

#### `optimizer_step` (Parameter Update)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| optimizer_step | 1.95 Mops/s (2.00 GB/s) | 1.93 Mops/s (1.97 GB/s) | 1.90 Mops/s (1.95 GB/s) | 1.91 Mops/s (1.95 GB/s) | 1.89 Mops/s (1.94 GB/s) | **4.50 Mops/s (4.61 GB/s)** |

Note: `optimizer_step` processes the accumulated unique key set (approx. 252K), limited by the overhead of the single-slot optimizer `pro::proxy` invocation. At 64T, Find + SGD update parallel efficiency is optimal.

#### `contains` (Existence Check)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| contains | 8.83 Mops/s | 15.95 Mops/s | 20.58 Mops/s | 22.31 Mops/s | 23.52 Mops/s | 22.84 Mops/s |

Pure hash lookup without embedding copies, limited by CPU and hash table bucket access latency, nears saturation after 8T.

#### `erase` (Entry Deletion, 10% Batch)

| Operation | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| erase (6553 keys) | 11.58 Mops/s | 36.27 Mops/s | 64.49 Mops/s | 58.06 Mops/s | 39.33 Mops/s | 24.53 Mops/s |

`erase` reaches its peak at 8T (64.49 Mops/s), and begins to decline due to NUMA cross-socket contention under Hyper-Threading.

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

1. **EmbeddingArena Pre-zeroing**: `ensure_capacity` performs a `memset(0)` on the entire page (4096 × dim × sizeof(float)) when allocating a new page, so `allocate()` no longer repeats zeroing for single entries.

2. **Epoch Batching**: `epoch::with_epoch` is hoisted to the outer layer of each dispatch chunk rather than entering/exiting for every single key, eliminating global epoch atomic counter contention under high concurrency.

3. **parlayhash Pre-allocation**: Hash table capacity is initialized with `max_active_keys * 2`, avoiding concurrent deadlocks caused by runtime rehashes (`copy_if_needed` race conditions).

4. **Zero-redundancy Find in `lookup_or_insert`**: The new insertion path captures the arena_idx directly from the `upsert_fn` closure, eliminating the extra Find call after Upsert.

---

## Build

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
