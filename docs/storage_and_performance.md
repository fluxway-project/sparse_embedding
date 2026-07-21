# Storage, Concurrency, and Performance

[中文](storage_and_performance_zh.md)

This document describes the storage and concurrency contract validated on
2026-07-21. The implementation is the source of truth for constants; the names
below make those policy points searchable.

## C++ API Contract

`PageStorage` and `PageEvictionProxy` are `pro::proxy` facades. New code creates
storage with `create_page_storage(StorageOptions)` and passes the facade by
value. Copying or moving the facade is shallow and shares the underlying
storage lifetime. `StoragePage` is move-only RAII ownership for one allocation.

`set_page_eviction_proxy()` returns a `PageEvictionRegistration`. A
registration must be cleared exactly once. Empty, unknown, and already-cleared
tokens throw; there is no silent fallback. The proxy generation changes when
the active proxy identity or lifetime transitions, so a candidate scored by an
older active-proxy generation cannot be swapped out.

The legacy virtual storage hierarchy and `PageStorage::create()` are not part
of the contract.

The embedding and optimizer templates use the following canonical shapes:

```cpp
using Store = se::HogwildEmbeddingStore<KeyT, ParamT, OptT, ScorePolicy>;
using Optimizer = pro::proxy<se::OptimizerFacade<ParamT, OptT>>;
using OptimizerFactory = se::Function<Optimizer(size_t)>;

auto optimizer =
    pro::make_proxy<se::OptimizerFacade<ParamT, OptT>, OptimizerImpl>();
auto optimizer_factory = pro::make_proxy<
    se::FunctionFacade<Optimizer(size_t)>>(
    [](size_t) -> Optimizer {
      return pro::make_proxy<
          se::OptimizerFacade<ParamT, OptT>, OptimizerImpl>();
    });
```

Facade definitions live in their interface headers. Callers use these exact
template parameters rather than redeclaring a legacy facade or store alias.

## Python Configuration

```python
import sparse_embedding as se

storage = se.AutoStorage(
    memory_limit_bytes=5 << 20,
    spill_directory="/mnt/nvme/sparse-embedding",
    max_spill_bytes=1 << 40,
    high_watermark=0.8,
    low_watermark=0.6,
)
config = se.EmbeddingConfig(
    dims=[128],
    num_threads=8,
    max_active_keys=1 << 20,
    storage=storage,
)
store = se.EmbeddingStore(config, score_policy="lfu")
```

`MemoryOnly()` disables disk-backed pages and raises when the memory budget is
exhausted. `DiskSpill(directory, max_bytes)` uses the specified spill device.
Invalid storage limits, alignments, watermarks, capacities, or registration
tokens raise instead of selecting a fallback policy.

## ParlayHash Concurrency

Each embedding dimension owns multiple key shards. A shard uses
`std::shared_mutex`: lookup takes a shared lock, while Upsert, erase, rebuild,
and internal expansion take an exclusive lock for that shard. There is no
global Upsert mutex, so unrelated shards continue in parallel while one shard
grows.

The initial-capacity policy is centralized in the store:

- `kParlayHashCapacityMultiplier = 4`
- `kInitialKeysPerShard = 2048`
- `kEagerPreallocationKeyLimit = 131072`

Tables at or below the eager limit preallocate their expected per-shard key
count. Larger tables start from the bounded per-shard target to control startup
RSS and cache footprint. Known production workloads should call
`reserve(keys_count, dim)` or `reserve_per_shard(counts, dim)` before the timed
phase. Reserve rebuilds capacity outside the hot path and rejects requests that
exceed `max_active_keys`.

The regression test inserts 65,536 unique keys through eight workers for both
unscored and scored stores and requires an exact key count. This covers the
dynamic-growth race that previously caused lost keys and SIGSEGV in
ParlayHash.

## Background Eviction

Eviction is event-driven. The background thread waits on a condition variable
until resident evictable memory reaches the high watermark, then continues
until it reaches the low watermark. It does not use periodic polling or a
probability gate.

The storage maintains an incremental index of anonymous evictable pages, so a
selection pass does not scan the complete page registry.

- `kExactEvictionPageThreshold = 1024`: score every candidate.
- `kSampledEvictionPageCount = 256`: above the threshold, score at most 256
  unique candidates per pass.
- `kMaxEvictionNoProgressAttempts = 8`: terminate repeated invalidation rather
  than spin indefinitely.

Selection snapshots page identity and proxy generation while holding the
storage mutex, releases that mutex for score evaluation, and validates both
again before `MAP_FIXED` remapping. Data-page evaluators aggregate entry
scores. Score metadata pages use a constant cold score so selection does not
scan the same score array twice.

## Telemetry

`EmbeddingStore.storage_stats()` returns cumulative counters and current byte
usage:

| Field | Meaning |
|---|---|
| `memory_bytes` | Current resident bytes accounted to storage |
| `spill_bytes` | Current live file-backed bytes |
| `memory_limit_bytes` | Effective resident-memory limit |
| `spill_limit_bytes` | Effective spill limit |
| `spill_count` | Disk-backed page allocations, including reused extents |
| `swap_out_count` | Background anonymous-to-file remaps |
| `eviction_selection_count` | Completed selection passes |
| `eviction_evaluated_pages` | Candidate pages scored across all passes |
| `eviction_selection_ns` | Cumulative selection time in nanoseconds |
| `eviction_candidate_invalidated_count` | Candidates rejected during final validation |
| `spill_directory` | Selected or configured spill directory |

`spill_count` and `swap_out_count` are intentionally different. Direct
file-backed allocation increments the former; background eviction increments
the latter.

## Benchmark Environment and Scenarios

### Machine

| Item | Measured configuration |
|---|---|
| CPU | 2 x Intel Xeon Silver 4116, nominal 2.10 GHz |
| CPU topology | 12 cores/socket, 2 threads/core, 24 physical cores, 48 online logical CPUs |
| NUMA node 0 | CPUs `0-11,24-35` |
| NUMA node 1 | CPUs `12-23,36-47` |
| Cache reported by `lscpu` | L1d 768 KiB, L1i 768 KiB, L2 24 MiB, L3 33 MiB |
| Memory | 125 GiB visible to the OS; no swap |
| OS | Ubuntu 24.04.4 LTS, x86-64 |
| Compiler/build | GCC 13.4, Bazel 7.7, C++23 (`-std=c++2b`), `-c opt` |
| Compile flags | `-O3 -march=native -mtune=native -mavx512bf16 -fno-strict-aliasing -DNDEBUG` |
| Link flags | `-flto` |
| CPU affinity | Not set for the recorded snapshot |

### C++ Production Workload

The production workload uses one dimension-128 slot, 500,000 possible
`uint64_t` keys, a 65,536-key batch, `max_active_keys=1,000,000`, and seed 42.
The hottest 10% of the vocabulary receives 80% of generated operation-batch
accesses; erase keys are sampled uniformly. Each of the 1, 4, 8, 16, 32, and
64-thread stores calls `reserve(500000, 128)` before warmup.
Parameter and optimizer types are `float`; the benchmark SGD uses learning
rate 0.01.

| Reported section | Precondition / dedicated warmup | Repetitions | Work inside timer | Throughput numerator |
|---|---:|---:|---|---|
| `lookup_or_insert` | 1 batch | 10 | Random batch generation + operation | 65,536 keys |
| `forward_lookup` | Store partially populated; no dedicated warmup | 20 | Random batch generation + operation | 65,536 keys |
| `assign` | Store partially populated; no dedicated warmup | 10 | Random batch generation + operation | 65,536 keys |
| `accumulate_gradients` | Store partially populated; no dedicated warmup | 10 | Random batch generation + operation | 65,536 keys |
| `optimizer_step` | Ten prior gradient repetitions remain, then one extra batch is accumulated | 10 | Step + next batch generation + next gradient accumulation | Current store size |
| `contains` | Store partially populated; no dedicated warmup | 20 | Random batch generation + operation | 65,536 keys |
| `erase` | None | 10 | Repeated erase of the same key set, without reinsertion | 6,553 keys |

The table documents the benchmark as implemented, not isolated API cost. In
particular, the optimizer numerator does not match only the unique keys in its
accumulated batch and therefore overstates standalone optimizer throughput.
Repeated erase iterations after the first are primarily not-found lookups.

The C++ LFU pressure workload uses dimension 128, 16,384 possible keys, a
4,096-key batch, two workers, `max_active_keys=20,000`, seed 42, the same 80/20
hot/cold distribution, benchmark SGD with learning rate 0.01, a 5 MiB
resident-memory limit, a 1 GiB spill limit, and high/low watermarks 0.8/0.6.
After insertion it performs four timed repetitions; each repetition executes
lookup, assign, gradient accumulation, and optimizer step. It waits 100 ms
before reading eviction telemetry. The final counters prove eviction was
enabled and eventually triggered, but do not prove it overlapped the timer.

### Python Workloads

The Python E2E workload uses dimension 16, 4,096 sequential `int64` keys, four
workers, `max_active_keys=8,192`, SGD with learning rate 0.01, LFU scoring, a
256 KiB resident-memory limit, and a 1 GiB spill limit. Synchronous and
asynchronous paths each run two warmups and five timed iterations. Every async
lookup, assign, gradient accumulation, and optimizer Future is awaited.

The separate Python storage-pressure workload runs five independent stores.
Each uses dimension 128, 16,384 sequential keys, two workers,
`max_active_keys=20,000`, LFU scoring, a 5 MiB resident-memory limit, and a
1 GiB spill limit. It reports median and p95 throughput/selection time plus
min/max eviction counters.

## Verification Workloads

```bash
bazel test -c opt \
  //sparse_embedding/src/storage:page_storage_test \
  //sparse_embedding/src/backend/hogwild:hogwild_storage_test
bazel build -c opt //:copy_core_so
PYTHONPATH="$PWD" pytest -q
bazel run -c opt //benchmarks:embedding_store_benchmark
PYTHONPATH="$PWD" python3 benchmarks/python/benchmark_e2e.py
PYTHONPATH="$PWD" python3 benchmarks/python/benchmark_storage_pressure.py
```

The correctness run completed 69 Python tests with 11 skips. The C++ pressure
workload executed lookup, assign, gradient accumulation, and optimizer step
with eviction enabled and eventually triggered: 2.75 Mops/s, two swap-outs,
four evaluated pages, and 9,616 ns cumulative selection time. The Python pressure workload measured
4.69 Mops/s for a synchronous training step and 12.38, 13.45, and 17.16 Mops/s
for asynchronous lookup, assign, and optimizer step.

The optimized extension build took 543.8 seconds on the documented machine.
Build time is tracked separately from runtime throughput. These measurements
are provisional reproducibility snapshots. The C++ result is one benchmark
execution with internal repetitions, while the Python storage-pressure result
uses five independent runs. A relative 95% throughput or 2x selection gate
requires baseline and optimized runs produced with the same CPU affinity,
seed, batch, and toolchain.

### Measurement Status

| Item | Current status |
|---|---|
| C++ production workload | One execution; fixed seed 42; internal warmup/repetitions; no CPU affinity |
| C++ pressure workload | One execution; full lookup/assign/gradient/optimizer path; telemetry captured |
| Python E2E workload | Two warmups, five timed runs; median throughput and p95 latency reported |
| Python pressure workload | Five independent runs; median/p95 selection time and throughput reported |
| Async completion | Every lookup, assign, gradient, and optimizer Future is awaited and returned |
| Leak accounting | Process exits cleanly, but no explicit wake-queue or thread-leak counter is recorded |
| Relative gates | Pending fixed baseline/optimized/delta runs with identical CPU affinity |

The final Python p95 latencies were 0.921 ms for the synchronous training
step, 0.355 ms for async lookup, 0.322 ms for async assign, and 0.279 ms for
async optimizer step. Async gradient accumulation is exercised and awaited but
is not timed as a separate metric.
