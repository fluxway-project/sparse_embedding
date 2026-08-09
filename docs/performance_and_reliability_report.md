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

## Standard machine and build configuration

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

## Design boundaries

Hot-key promotion is control-plane work and is not performed on every lookup.
The authoritative directory maps each key directly to its current arena locator;
independent HotMap and ColdMap directories are not maintained.

Checkpoints materialize stable native page images through the SeqLock reader
protocol. They do not clone spill files or require filesystem copy-on-write.
Compaction moves every live key before releasing a source page, preserving one
lookup path and explicit physical-page reclamation.
