# Storage Architecture and API Contracts

[中文](storage_architecture_zh.md) | [Incremental Checkpointing](incremental_checkpoint.md) | [Lance Storage and Gravitino Integration](lance_storage_and_gravitino.md) | [Performance and Reliability Report](performance_and_reliability_report.md) | [Dynamic Learning Rate](dynamic_learning_rate.md)

This document provides a comprehensive technical overview of the Sparse Embedding storage engine, detailing its core architecture, concurrency and memory model, tiered storage topology, eviction scheduling, and distributed interface contracts.

---

## 1. Overview and Design Philosophy

The Sparse Embedding storage engine is designed for modern deep recommendation engines, search ranking systems, and large language model sparse retrieval workflows, efficiently managing massive sparse tables scaling to hundreds of millions or billions of parameters.

To handle high-throughput irregular memory access, extreme parameter skew, and massive memory capacity requirements, the engine is built on the following core design principles:

- **Lock-Free Read Hot Paths**: Forward lookup pipelines completely bypass global locks and shard mutexes. Slot lookups utilize optimistic SeqLocks, eliminating locks and atomic counter write overhead on the read path.
- **Branchless Single-Cycle Addressing**: Physical page tables use flat, compact pointer directories. Addressing is resolved in a single step using bitwise shifts and masks, eliminating virtual functions and multi-level pointer indirection.
- **Hardware-Aware Auto-Tuning**: Probes physical system memory (`/proc/meminfo`) and NUMA topology, running adaptive micro-benchmarks at startup to select optimal page sizes.
- **Tiered Storage and Seamless Spill-to-Disk**: Supports pure in-memory operation (`MemoryOnly`) as well as adaptive disk spilling (`AutoStorage`), transparently remapping anonymous memory pages to file-backed pages via `MAP_FIXED` when memory pressure reaches high watermarks.
- **Defensive Engineering and Fail-Fast Contracts**: Rejects implicit fallbacks that could cause silent corruption or data loss. Invalid parameters, exhausted capacities, and operational errors strictly raise explicit exceptions.

---

## 2. System Architecture

The Sparse Embedding engine is structured into five distinct layers:

```text
┌─────────────────────────────────────────────────────────────┐
│                      Python API Layer                       │
│       EmbeddingStore / EmbeddingConfig / Partitioner        │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│             Concurrent Hash Routing & Locator Layer         │
│  ParlayHash Shards ──► MapBundle (RCU) ──► SegmentedLocator │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                 Physical Page Management Layer              │
│  StablePageDirectory ──► StoragePage (RAM / Spill SSD)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│             Eviction & Adaptive Locality Scheduling         │
│  Eviction Thread ──► Boltzmann Policy ──► Coaccess Sketch   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│            Distributed Partitioning & Persistence           │
│  PartitionerRegistry (SIMD)  │  Dual-Lance 2PC Checkpoint   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Concurrency Model and Lock-Free Queries

### 3.1 Hash Routing and Immutable MapBundle (RCU Publication)

The core query resolution path follows: `Key Hash -> Shard ID -> slot_id -> SegmentedLocatorTable -> arena_index`.

- **Read/Write Decoupling**: Each shard manages an immutable `MapBundle`. Readers acquire-load the current `MapBundle` pointer without taking shared mutexes.
- **Copy-on-Write and Atomic Publication**: Structural mutations (insert, erase, reserve, slot rotation) clone the active bundle under the affected shard's writer mutex, applying changes and publishing the new bundle via a single atomic exchange. Individual shards grow independently without blocking concurrent operations on other shards.
- **Zero Secondary Lookups**: Hot/cold tiers and physical pages do not participate in routing decisions. Each lookup requires exactly one directory lookup and one locator read, eliminating secondary hash probes.

### 3.2 Locator Slots and Optimistic SeqLocks

Slot allocation is managed by `SegmentedLocatorTable`, structured in fixed 32 KiB segments with 8-byte aligned slots storing `{sequence, arena_index}`:

- **Optimistic Reading and In-Place Updates**: Readers and in-place weight updaters (e.g., `optimizer_step`, `assign`) use optimistic `try_read` to load the sequence and `arena_index`, passing compiler barriers to access physical memory in-place before validating sequence consistency. Hot paths are entirely lock-free with zero atomic writes.
- **Relocation Awareness and ABA Defense**: When background compaction migrates physical pages and mutates `arena_index`, it flips the sequence from even to odd via CAS, copies payloads, and atomically publishes the new even sequence and address. Concurrent readers detect the version change and retry automatically. As sequences approach wraparound, keys are rotated to fresh slots while permanently retiring old slots.

### 3.3 Flat Two-Level Page Directory (`StablePageDirectory`)

Physical page management employs a compact two-level pointer directory `StablePageDirectory<T>` (managing data pages, state pages, owners, and locks) resident in CPU L1 cache:

$$\text{page\_id} = \text{index} \gg \text{entries\_shift}, \quad \text{slot\_offset} = \text{index} \ \& \ \text{slot\_mask}$$

Addressing is resolved directly through single-cycle branchless bitwise operations, eliminating heap pointer indirection.

### 3.4 Four-Stage Software Pipelined Prefetching

Because sparse feature lookups exhibit irregular and non-contiguous memory access patterns, hardware stream prefetchers cannot anticipate cachelines. The engine implements a four-stage software pipelined prefetch architecture across forward and backward hot paths:

1. **Stage 1 (Input Features)**: Prefetches batch `keys[i + distance]`, row lengths, and offsets into L1/L2 cache ahead of current compute.
2. **Stage 2 (Output Destination Buffer)**: Issues write-intent prefetching for destination memory (`out_flat_values + offset`).
3. **Stage 3 (Directory Hash Buckets)**: Computes future key hashes and prefetches directory hash buckets into CPU cache prior to probing.
4. **Stage 4 (Locator Slots & Optimizer State)**: Prefetches 8-byte aligned `LocatorSlot` entries and optimizer first/second moment vectors, hiding LLC cache miss latency.

---

## 4. Tiered Storage and Hardware Auto-Tuning

### 4.1 Hardware Probing and Dynamic Tuning

The storage layer includes dedicated auto-tuning modules:

- **Resource Probing (`SystemResourceProbe`)**: Parses high-precision `MemAvailable` from `/proc/meminfo` and detects NUMA topology and physical core counts via `libnuma`.
- **Startup Calibration (`StorageResourceTuner`)**: Evaluates candidate page sizes `{4096, 8192, 16384}` across embedding dimensions and workload types at startup, selecting the median duration winner. Explicit `entries_per_page` bypasses calibration.
- **Dynamic Max Pages (`max_pages`)**: Automatically calculates page count limits based on a 75% available RAM budget, aligned to NUMA cohorts.

### 4.2 Storage Modes and Disk Spilling (Spill-to-Disk)

- **`MemoryOnly`**: Pure memory mode. Raises an exception when memory limits are exceeded.
- **`AutoStorage`**: Spills cold pages to disk automatically. Background threads monitor resident memory watermarks (High and Low):
  - When memory reaches the high watermark, the eviction thread wakes up and writes cold pages to the spill directory;
  - Utilizes `mmap(MAP_FIXED)` to transparently remap virtual memory addresses from anonymous physical RAM to file-backed disk pages;
  - Once memory drains to the low watermark, the eviction thread returns to sleep.

---

## 5. Eviction Scheduling and Adaptive Locality

### 5.1 Score-Threshold Eviction (`evict_by_score`)

- Collects live entries with scores at or below `max_score` using bounded per-worker heaps. Temporary memory overhead scales with the eviction budget rather than the total table size.
- Deletions are re-validated under the affected shard's writer mutex without acquiring global table locks.

### 5.2 Boltzmann Utility Model and Co-Access Clustering

To maximize cache hits, the engine implements adaptive utility estimation:

- **Adaptive Boltzmann Policy (`AdaptiveBoltzmannPolicy`)**:
  $$\text{weight}_i = \frac{\exp(-(\text{utility}_i - \min(\text{utility})) / T)}{\sum_j \exp(-(\text{utility}_j - \min(\text{utility})) / T)}$$
  Page utility combines decayed access frequency, co-access density, and residency duration. Higher utility monotonically decreases the probability of eviction.
- **Co-Access Sketch (`CoaccessSketch`)**: Forward lookups non-blockingly sample co-occurring feature keys to update co-access counters. High-frequency co-accessed features are clustered into contiguous physical pages during compaction, improving CPU cacheline locality.

---

## 6. Lifecycle Management and Epoch Backpressure

All resources potentially visible to concurrent readers (including `MapBundle`, `StoreSnapshot`, locator slots, and physical pages) are managed within the epoch-backed `RetirementPressure` domain:

- **Three-Tier Backpressure Control**:
  1. **Normal Phase** (below `backpressure_fraction = 0.15`): Full-speed execution.
  2. **Backpressure Phase** (between 0.15 and 0.25): Writers and compactor threads execute brief CPU pauses to pace mutations.
  3. **Circuit Phase** (at or above `circuit_fraction = 0.25`): Rejects control-plane mutations that create new retirement records by raising exceptions, preventing memory blowup.
- **Safe Reclamation**: Once all concurrent readers exit the corresponding epoch, retired memory and metadata are freed, draining `retired_bytes` to zero.

---

## 7. Distributed Partitioner Subsystem

### 7.1 Partitioner Registry (`PartitionerRegistry`)

The C++ core provides a centralized registry for algorithm factories:
- `sparse_embedding.modulo`: Normalized integer modulo (supports negative key normalization).
- `sparse_embedding.hash`: MurmurHash3 avalanche hash with optional seed.
- `sparse_embedding.fast_range`: Lemire FastRange division-free range partitioner.
- Custom algorithms can be dynamically registered via `register_factory`.

### 7.2 Vectorized Slice Filtering (`filter_span`)

Executes native AVX-512 SIMD kernels over contiguous Arrow key slices to produce bitmasks and matching row indices, accelerating offline checkpoint restore and communication planning.

### 7.3 Python Elastic Strategy (`PartitionStrategy`)

Decouples algorithm declaration from physical cluster topology:

```python
import numpy as np
import sparse_embedding as se

keys = np.array([10, 20, 30, 40], dtype=np.uint64)

# 1. Declarative partition strategy (late-binding topology)
strategy = se.HashPartition(seed=42)

# 2. Compute partition ranks given target worker count
ranks = strategy(keys, partition_count=4)

# 3. Vectorized filtering for a specific shard
filtered_keys = strategy.filter(keys, target_shard=0, partition_count=4)

# 4. Functional convenience API
ranks_func = se.partition(keys, partition_count=4, algorithm="sparse_embedding.hash")
```

---

## 8. Checkpoint Subsystem Integration Contracts

The storage engine coordinates with the persistence subsystem under the following contracts:

- **`(dimension, key)` Identity Tuple**: Embedding and optimizer state rows are uniquely identified by `(dimension, key)`, allowing distinct dimension tables to hold identical key values.
- **Training vs. Inference State Isolation**:
  - Training instances allocate and manage optimizer state pages (e.g., Adam first/second moments).
  - Inference instances configure `EmbeddingConfig(enable_optimizer=False)` to create weight-only stores, skipping optimizer allocation.
- **Partition-Target Restoration**: `RestorePlan.from_partition` explicitly describes the current assignment. The same plan type serves identical and changed target topologies; canonical entry routing remains in the native partitioner, while page-image loading is selected only through `IngestionEngine.FAST_PAGE_CACHE`.
- **Two-Phase Commit and Lance Lakehouse Persistence**: Underlying storage persists state via `snapshot_manifest.lance` (control-plane ledger) and `entries.lance` (data-plane physical slices).
- **Incremental Checkpointing & Streaming Compaction**: Mutation journaling tracks dirty keys and deletion tombstones during training, enabling non-blocking cuts and native streaming compaction.

> For detailed checkpoint architecture, Two-Phase Commit protocols, incremental restoration, and Gravitino integration, refer to [Lance Storage and Gravitino Integration](lance_storage_and_gravitino.md) and [Incremental Checkpointing and Streaming Compaction](incremental_checkpoint.md).

---

## 9. Python Configuration Guide

```python
import sparse_embedding as se

# 1. Configure tiered storage (AutoStorage spills cold pages to NVMe/SSD)
storage = se.AutoStorage(
    memory_limit_bytes=4 << 30,          # Resident memory limit: 4 GiB
    spill_directory="/mnt/nvme/spill",    # Spill storage directory
    max_spill_bytes=100 << 30,           # Max disk spill capacity: 100 GiB
    high_watermark=0.8,                  # Trigger background swap-out at 80%
    low_watermark=0.6,                   # Sleep swap-out thread at 60%
)

# 2. Configure EmbeddingStore parameters
config = se.EmbeddingConfig(
    dims=[128],                          # Embedding vector dimension
    thread_budget=8,                     # Internal thread budget
    storage=storage,                     # Storage backend
    key_dtype="uint64",                  # Key dtype (int32/uint32/int64/uint64)
    value_dtype="float32",               # Value dtype
    optimizer_dtype="float32",           # Optimizer moment dtype (supports bf16/fp16/fp32)
    gradient_dtype="bfloat16",           # Gradient accumulation dtype (optional)
    gradient_scale=1.0,                  # Gradient scaling factor
    enable_optimizer=True,               # False for weight-only inference stores
    storage_tuning=se.StorageTuningOptions(
        target_capacity_keys=10_000_000, # Expected key capacity
        memory_budget_bytes=4 << 30,     # Memory tuning budget
        entries_per_page=4096,           # Entries per page (4096/8192/16384)
    ),
)

# 3. Instantiate store
store = se.EmbeddingStore(config, score_policy="lfu")
```

---

## 10. Telemetry Reference and Diagnostics

Call `store.storage_stats()` to inspect cumulative runtime counters and memory usage:

| Metric Name | Type | Description and Operational Guidance |
|---|---|---|
| `memory_bytes` | bytes | Resident physical memory currently accounted to storage |
| `spill_bytes` | bytes | Active file-backed disk memory in bytes |
| `memory_limit_bytes` | bytes | Configured resident memory limit |
| `spill_limit_bytes` | bytes | Configured disk spill limit |
| `spill_count` | counter | Disk-backed page allocations (including reused extents) |
| `swap_out_count` | counter | Background anonymous-to-file remapping count |
| `eviction_selection_count` | counter | Completed background eviction selection passes |
| `eviction_evaluated_pages` | counter | Cumulative pages evaluated during eviction passes |
| `eviction_selection_ns` | nanoseconds | Cumulative time spent in eviction candidate selection |
| `eviction_candidate_invalidated_count` | counter | Candidates invalidated during final verification (normal concurrency) |
| `retired_bytes` | bytes | Bytes retired into epoch domains awaiting reclamation |
| `retirement_pressure_state` | state | Current epoch pressure: `normal`, `backpressure`, or `circuit` |
| `retirement_backpressure_count` | counter | Writer calls that executed CPU pause due to epoch pressure |
| `oldest_unreclaimed_age_ns` | nanoseconds | Age of the oldest unreclaimed retirement record |
| `epoch_stall_count` | counter | Critical epoch stalls that generated diagnostic logs |
| `spill_directory` | string | Active disk spill directory path |
