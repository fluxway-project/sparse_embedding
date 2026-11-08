# Sparse Embedding Performance and Reliability

[中文](performance_and_reliability_report_zh.md) | [Storage Architecture](storage_architecture.md) | [Lance Storage and Gravitino Integration](lance_storage_and_gravitino.md) | [Incremental Checkpointing](incremental_checkpoint.md) | [Dynamic Learning Rate](dynamic_learning_rate.md)

This document provides public performance references, recommended configuration,
and reliability guarantees for Sparse Embedding.

## Capabilities

Sparse Embedding provides shard-local concurrent storage, lock-free directory
lookup, miss-only batch coalescing, score-based eviction, adaptive page
selection, full-page compaction, and optional hot-key promotion.

`lookup_or_insert` performs a directory lookup for existing keys. Missing keys
enter a lightweight insertion slow path. Duplicate misses in one batch share a
single arena slot, and every duplicate output row reuses that embedding.

Mutation batches for `assign`, `accumulate_gradients`, and optimizer updates use
unique `(key, dimension)` rows and published keys. A concurrent erase/recycle
race invalidates only the affected gradient batch: debug builds throw for
diagnosis; release builds increment a relaxed counter and publish a fixed-size
diagnostic event to the bounded telemetry ring. The single consumer may write
the event to `std::clog`; the producer never performs I/O, and a full ring drops
the diagnostic without blocking the training path.

Lance checkpoints maintain canonical `entries` as the SSOT, optional `page_images` as import cache, and versioned manifests as visibility boundaries. Persistence splits into `snapshot_manifest.lance` (control ledger) and `entries.lance` (worker data slices), supporting 4 native Arrow integer key types (`int32`, `uint32`, `int64`, `uint64`) with DuckDB pushdown. Offline restore applies `PartitionerRegistry` and AVX-512 `filter_span`. Online routing via `parallel_partitioner.hpp` reaches 1.03B keys/s across 8 threads. Writing and compaction stream through bounded channels. Local publication uses POSIX atomic `rename(2)`; remote object storage uses immutable staging and pointer publication with cascading cleanup.
Page images store slot layout, `seq_count`, parameters, score, optimizer state/step, and
SHA-256 checksums. Incremental checkpoints append only changed entries and tombstones; an external coordinator
performs global commit.
Inference uses `EmbeddingConfig(enable_optimizer=False)` and `checkpoint.load_inference_checkpoint` (or storage `load_inference_weights`),
which avoids optimizer-state allocation and loading.

Compaction, hot-key promotion, and page eviction update one authoritative
directory. Source pages remain readable until epoch safety and are never reused
as destinations while draining or retired.

## Testing Standard Machine and Build Configuration

| Item | Configuration |
|---|---|
| CPU | 2 × Intel Xeon Silver 4116, 2.10 GHz |
| Topology | 24 physical cores, 48 logical CPUs, 2 NUMA nodes |
| NUMA CPUs | node 0: `0-11,24-35`; node 1: `12-23,36-47` |
| Cache | 768 KiB L1d, 768 KiB L1i, 24 MiB L2, 33 MiB L3 |
| Memory | 125 GiB visible, swap disabled |
| System | Ubuntu 24.04.4 LTS, x86-64 |
| Toolchain | GCC 13.4, Bazel 7.7, C++23 |
| Optimization | `-O3 -march=native -mtune=native -mavx512bf16 -DNDEBUG -flto` |

Use physical cores as the execution-worker limit. For NUMA locality, enable both
worker affinity and arena-page placement:

```bash
bazel run -c opt //benchmarks:embedding_store_benchmark -- \
  --threads=16 --numa-aware-sharding --skip-pressure
```

`--numa-aware-sharding` enables both execution and memory NUMA locality. Workers
bind to physical CPUs, and arena pages are placed by shard on the corresponding
NUMA node. Small payloads bypass the extra NUMA bucketing pass when dispatch cost
would exceed locality benefit.

The combined default `embedding_store_benchmark` pre-generates power-law batches
before timing `forward_lookup`. Its scaling gate binds memory to the NUMA node
of the measured workers and runs the coordinator on a separate local CPU during
forward timing. It fails explicitly if the allowed CPUs cannot support this
placement; `--threads=N` runs do not execute the aggregate scaling gate.

## Representative performance

The numbers below are reference measurements on the standard machine above.
High-concurrency and compaction results are medians of five samples in one
benchmark invocation. Mixed lookup/insert results are medians of five
independent process runs. Throughput is reported in Mops/s or M keys/s; these
figures are workload references, not cross-machine capacity guarantees.

### Concurrent memory lookup

Configuration: dimension 128, reserved memory-only table, 10,000,000 `int64` keys,
8,192-key batches, 128 concurrent callers, and up to 128 internal workers.

| Metric | Result |
|---|---:|
| Lookup throughput | **22.13 M keys/s** |
| Telemetry relaxed-atomic reduction | **93.75%** |

### Slot-directory forward scaling

Configuration: dimension 32, a 4,096-key memory-only table, 4,096-key fixed
power-law batches, 400 timed `forward_lookup` repetitions, process affinity
`taskset -c 0-47`, and the default automatic storage tuner.

| Threads | Forward throughput |
|---:|---:|
| 1 | 14.61 Mops/s |
| 2 | 27.04 Mops/s |
| 4 | 43.11 Mops/s |
| 8 | **50.39 Mops/s** |
| 16 | 45.74 Mops/s |
| 24 | 40.11 Mops/s |
| 32 | 39.57 Mops/s |
| 64 | 37.42 Mops/s |

The default memory-work and row-grain limits bound effective task count, keeping
enqueue, steal, completion, and barrier costs from dominating the timed path.

The storage resource tuner (`StorageResourceTuner`) evaluates candidate page geometries `{4096, 8192, 16384}` once at store construction for the requested workload shape and caches the optimal winner, while explicit `entries_per_page` remains available for deterministic reproducibility. The hot path utilizes cached bit-shifts and masks without runtime dispatch.

The calibration gate requires a candidate to outperform the baseline 4096 profile by at least 50% on median calibration latency before switching; otherwise 4096 is retained. This conservative floor guards against system noise while ensuring optimal geometries are selected under high-density or wide-dimension configurations.

The forward, training, and optimizer hot-set tables report medians of five independent process runs with `taskset -c 0-47` (48 allowed logical CPUs, 24 physical cores). High-concurrency and compaction tables use five in-process samples; the mixed lookup/insert table uses five independent process runs.

### Full-store forward gate

The separate `embedding_store_benchmark` workload uses dimension 128, a 500,000-key
vocabulary, 65,536-key power-law batches, and 20 timed repetitions. Five full
default runs completed, including the storage-pressure phase. The table shows
five-run median throughput and the lowest same-run scaling ratio.

| Native workers | Forward median | Minimum ratio to 1 worker | Required ratio |
|---:|---:|---:|---:|
| 1 | 4.23 Mops/s | — | — |
| 2 | 7.86 Mops/s | 1.85× | 1.6× |
| 4 | 14.51 Mops/s | 3.40× | 2.5× |
| 8 | **20.06 Mops/s** | **4.72×** | 3.5× |

### Training paths

Dimension 128, 65,536 existing unique rows, eight workers, process affinity
`taskset -c 0-47`, and 20 timed repetitions. Throughput is the five-run median:

| Operation | Throughput | Bandwidth |
|---|---:|---:|
| `assign` | **13.43 Mops/s** | 6.40 GiB/s |
| `accumulate_gradients` (including required step) | **2.92 Mops/s** | 1.39 GiB/s |
| `optimizer_step` | **14.63 Mops/s** | 6.98 GiB/s |

The optimizer-only hot-set profile uses the same dimension and process affinity,
200 timed repetitions, and one native `optimizer_step()` caller. Each value is
the median of five independent runs; the 65,536-key set is a separate control
workload, not a replacement for the 20-repetition training-path table.

| Unique keys | Native workers | Optimizer throughput |
|---:|---:|---:|
| 4,096 | 1 | 3.99 Mops/s |
| 4,096 | 8 | **17.89 Mops/s** |
| 4,096 | 16 | 16.46 Mops/s |
| 65,536 | 1 | 2.32 Mops/s |
| 65,536 | 8 | **13.43 Mops/s** |
| 65,536 | 16 | 10.13 Mops/s |

The batch optimizer profile is a separate workload from the tables below. Use
`training_path_profile` when comparing assign, gradient accumulation, and
optimizer-step implementations; use the benchmark commands in this document
for storage and compaction comparisons.

### High-concurrency pressure and reliability

Pressure profile: eight producers, 1,048,576 keys, dimension 8, Auto storage, an
8 MiB resident limit, a 512 MiB spill limit, and 0.75/0.50 watermarks.

| Metric | Result |
|---|---:|
| `full_lookup_verified` | `1` |
| Pressure pages | `0` |
| Spill / swap-out | `0 / 0` |
| Retirement sink hits | `1,044,480` |
| Allocation p99 | `1.55 ms` |

The small-memory all-path pressure test covers `lookup_or_insert`,
`forward_lookup`, `assign`, `contains`, `erase`, duplicate gradient accumulation,
and `optimizer_step`, and verifies payload integrity after paging.

## Runtime and performance reference

The current path is `key -> shard MapBundle -> locator_slot -> arena_index ->
payload`. The logical directory remains a COW snapshot and the physical page
directory uses stable segments (`StablePageDirectory`); lookup, optimizer, and ordinary compaction
relocation take no mutex. Expansion runs in the writer/control plane. When a
runtime insert exceeds the reserved budget, growth uses these exact factors:

| Layer | Expansion rule |
|---|---|
| Directory bucket budget | `ceil(2.5 * key_capacity)`, rounded up to 16; use the smaller `ceil_pow2(4 * key_capacity)` when it fits |
| `MapBundle` runtime capacity | 2x below 64K; 1.5x for `64K <= capacity < 1M`; 1.25x at or above 1M |
| Shard budget | `reserve(keys_count, dim)` distributes exactly; `reserve_per_shard(counts, dim)` uses caller-provided per-shard budgets |

Expansion clones the current `MapBundle` and publishes it with one atomic
exchange. Epoch retirement reclaims the old bundle; locator-slot and arena-index
addresses remain stable. Bucket/slot/arena limits and allocation failures throw
explicit errors; no fallback capacity path is used.

### Performance results

High-concurrency lookup/insert: dimension 32, `1<<20` warm keys, 1,024-key
batches, 64 lookup rounds, 8 insert rounds, memory-only 1 GiB, and five
repetitions per configuration reporting median/p95/p99.

| Threads | Lookup median/p95/p99 (Mops/s) | Insert median/p95/p99 (Mops/s) |
|---:|---:|---:|
| 1 | 15.38 / 15.52 / 15.52 | 0.092 / 0.093 / 0.093 |
| 16 | 60.74 / 68.57 / 68.57 | 0.339 / 0.355 / 0.355 |
| 64 | **80.42 / 86.77 / 86.77** | **0.528 / 0.559 / 0.559** |

Mixed lookup/insert uses the same dimension, warm keys, and batch for 64 rounds;
16/64 total threads are split evenly, with Auto storage at 1 GiB memory and
2 GiB spill limits.

| Total | Lookup/insert threads | Lookup / insert (Mops/s) | Lookup/insert p99 (ns) |
|---:|---:|---:|---:|
| 16 | 8 / 8 | 48.44 / 0.338 | 191,635 / 61,571,904 |
| 64 | 32 / 32 | **57.71 / 0.412** | 695,113 / 141,594,266 |

Compaction strong scaling uses dimension 128, 64 shards, four pages per source
worker, the calibrated page geometry (4096 entries per page on this host), 64
store threads, fixed work, 1/2/4/8/16 workers, reporting the five-sample median.

| Workers | Relocated keys/s |
|---:|---:|
| 1 | 1.94 M |
| 2 | 3.83 M |
| 4 | 7.12 M |
| 8 | 10.59 M |
| 16 | **12.04 M** |

Without preallocated destinations, the timed phase also includes destination-page
allocation, giving slightly lower throughput. Corresponding binaries are
`high_concurrency_lookup_insert_benchmark`, `mixed_lookup_insert_concurrency_benchmark`,
and `page_compaction_benchmark`.

### Partitioner and Filter Benchmarks

Benchmark configuration: 8 worker threads, $10^7$ test keys, covering all 4 supported `key_dtype` variants (`int32`, `uint32`, `int64`, `uint64`).

| Algorithm / Kernel | Single-core C++ Kernel (keys/s) | 8-Thread Throughput (keys/s) |
|---|---:|---:|
| `sparse_embedding.modulo` | 2.33 B | **789.2 M** |
| `sparse_embedding.hash` (MurmurHash3) | 2.41 B | **893.5 M** |
| `sparse_embedding.fast_range` (Lemire) | 2.41 B | **1.03 B** |
| AVX-512 SIMD Continuous Slice Mask Filter (`filter_span`) | 1.82 B | **1.45 B** |

- **Online Routing**: The parallel dispatch kernel (`parallel_partitioner.hpp`) utilizes chunked execution over `std::execution`, sustaining 789M ~ 1.03B keys/s across algorithms (and 2.4B keys/s on single-core microbenchmarks), delivering microsecond-level latency for distributed SPMD communication planning.
- **Offline Restore Filtering**: The native `filter_span` kernel evaluates contiguous Arrow physical memory slices using AVX-512 vectorized byte masks, processing 8 (64-bit) or 16 (32-bit) keys per chunk simultaneously. Single-core filter throughput reaches 1.82B keys/s, completely bypassing row-by-row iteration and deserialization.

Paging tests should verify `swap_out_count > 0` and `spill_count > 0`. Safety
regression must cover dimension 1, Adam, concurrent `optimizer_step()` and
`compact_page()`, and inspect native subprocess termination codes. Full native,
TSAN, and ASAN/UBSAN suites are release validation requirements; exception
handling must not replace signal detection.

## Public validation

```bash
bazel test -c opt --test_output=errors //sparse_embedding/...
bazel build -c opt //:package_core_extension
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py
PYTHONPATH="$PWD" pytest -q
```

Pressure validation:

```bash
bazel test -c opt //sparse_embedding/src/backend/hogwild:small_memory_all_paths_pressure_test \
  --test_output=errors
```

Benchmark commands:

```bash
bazel run -c opt //benchmarks:high_concurrency_lookup_insert_benchmark
bazel run -c opt //benchmarks:mixed_lookup_insert_concurrency_benchmark
bazel run -c opt //benchmarks:page_compaction_benchmark -- \
  --strong-scaling --preallocate-destinations
```

## Design boundaries

Hot-key promotion is control-plane work and is not performed on every lookup.
The authoritative directory maps each key directly to its current arena locator;
independent HotMap and ColdMap directories are not maintained.

Checkpoints materialize stable native page images through the SeqLock reader
protocol. They do not clone spill files or require filesystem copy-on-write.
Compaction moves every live key before releasing a source page, preserving one
lookup path and explicit physical-page reclamation.
