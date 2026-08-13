# Sparse Embedding Performance and Reliability

[中文](performance_and_reliability_report_zh.md)

This document provides public performance references, recommended configuration,
and reliability guarantees for Sparse Embedding.

## Capabilities

Sparse Embedding provides shard-local concurrent storage, lock-free directory
lookup, miss-only batch coalescing, score-based eviction, adaptive page
selection, full-page compaction, and optional hot-key promotion.

`lookup_or_insert` performs a directory lookup for existing keys. Missing keys
enter a lightweight insertion slow path. Duplicate misses in one batch share a
single arena slot, and every duplicate output row reuses that embedding.

Lance checkpoints contain `entries`, physical `page_images`, and a version-pinned
manifest. Page images store slot layout, `seq_count`, parameters, score,
optimizer state/step, and SHA-256 checksums. Incremental checkpoints append only
changed entries and tombstones; an external coordinator performs global commit.

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

Dispatch can be configured explicitly:

```bash
bazel run -c opt //benchmarks:embedding_store_benchmark -- \
  --threads=16 --operation-physical-workers=16 \
  --operation-memory-parallelism=16 --forward-prefetch-distance=16 \
  --numa-aware-sharding --skip-pressure
```

`--numa-aware-sharding` enables both execution and memory NUMA locality. Workers
bind to physical CPUs, and arena pages are placed by shard on the corresponding
NUMA node. Small payloads bypass the extra NUMA bucketing pass when dispatch cost
would exceed locality benefit.

## Representative performance

### Concurrent memory lookup

Configuration: dimension 128, reserved memory-only table, 10,000,000 `int64` keys,
8,192-key batches, 128 concurrent callers, and up to 128 internal workers.

| Metric | Result |
|---|---:|
| Lookup throughput | **21.54 M keys/s** |
| Telemetry relaxed-atomic reduction | **93.75%** |

### Forward scaling

Configuration: dimension 32, fixed 4,096-key batches, 128 published keys, 48
allowed CPUs, slot-directory reads, and contiguous partition dispatch.

| Threads | Forward throughput |
|---:|---:|
| 1 | 14.988 Mops/s |
| 2 | 21.852 Mops/s |
| 4 | 35.165 Mops/s |
| 8 | **45.945 Mops/s** |
| 16 | 42.049 Mops/s |
| 64 | 33.446 Mops/s |

The default 256 KiB memory-work grain and 64 KiB minimum row grain bound the
number of effective tasks, preventing additional enqueue, steal, completion, and
barrier cost from cancelling the benefit of more threads.

### Training paths

Dimension 128, 65,536 existing unique rows, eight workers, and affinity enabled:

| Operation | Throughput |
|---|---:|
| `assign` | **21.82 Mops/s** |
| `accumulate_gradients` (including required step) | **7.33 Mops/s** |
| `optimizer_step` | **15.67 Mops/s** |

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
directory uses stable segments; lookup, optimizer, and ordinary compaction
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
| 1 | 6.31 / 6.80 / 6.80 | 0.0456 / 0.0519 / 0.0519 |
| 16 | 50.38 / 51.65 / 51.65 | 0.282 / 0.313 / 0.313 |
| 64 | **64.76 / 72.30 / 72.30** | **0.583 / 0.608 / 0.608** |

Mixed lookup/insert uses the same dimension, warm keys, and batch for 64 rounds;
16/64 total threads are split evenly, with Auto storage at 1 GiB memory and
2 GiB spill limits.

| Total | Lookup/insert threads | Lookup / insert (Mops/s) | Lookup/insert p99 (ns) |
|---:|---:|---:|---:|
| 16 | 8 / 8 | 26.67 / 0.298 | 376,041 / 56,071,999 |
| 64 | 32 / 32 | **52.63 / 0.365** | 1,212,543 / 153,709,741 |

Compaction strong scaling uses dimension 128, 64 shards, four pages per source
worker, 4,096 entries per page, 64 store threads, fixed work, 1/2/4/8/16
compaction workers, five repetitions, and preallocated destinations.

| Workers | Relocated keys/s |
|---:|---:|
| 1 | 1.19 M |
| 2 | 2.10 M |
| 4 | 3.83 M |
| 8 | 10.17 M |
| 16 | **13.59 M** |

Without destination preallocation the values are 1.16/2.04/3.79/7.54/9.56 M
keys/s. The executable targets are `high_concurrency_lookup_insert_benchmark`,
`mixed_lookup_insert_concurrency_benchmark`, and `page_compaction_benchmark`.

Paging tests should verify `swap_out_count > 0` and `spill_count > 0`. Safety
regression must cover dimension 1, Adam, concurrent `optimizer_step()` and
`compact_page()`, and inspect native subprocess termination codes. Full native,
TSAN, and ASAN/UBSAN suites are release validation requirements; exception
handling must not replace signal detection.

## Public validation

```bash
bazel test -c opt --test_output=errors //sparse_embedding/...
bazel build -c opt //:copy_core_so
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py
PYTHONPATH="$PWD" pytest -q
```

Pressure validation:

```bash
bazel test -c opt //sparse_embedding/src/backend/hogwild:small_memory_all_paths_pressure_test \
  --test_output=errors
```

Latest benchmark commands:

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
