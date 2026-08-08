# Sparse Embedding Performance and Reliability

[中文](performance_and_reliability_report_zh.md)

This document describes the public performance methodology, implementation
guarantees, and regression coverage for Sparse Embedding.

## Delivered Capabilities

Sparse Embedding provides shard-local concurrent storage, score-based entry
eviction, adaptive page selection, full-page compaction, and optional hot-key
promotion.

Full-page compaction drains a selected source page. Each live key is relocated
through the authoritative sharded directory, then the empty source page is
released after epoch safety. A draining or retired source page is never reused
as a destination.

The Lance checkpoint format stores `entries`, physical `page_images`, and a
manifest that pins both dataset versions and stores the checkpoint schema
version. The manifest is the only checkpoint metadata record; there is no
sidecar `metadata.json`. Page images include slot layout,
per-slot `seq_count`, parameters, optional score, optimizer state and step, and
 a SHA-256 checksum. Distributed restore requires an explicit caller-supplied
`filter_fn` and `target_shard_id`; `distributed=True` is only a temporary
compatibility assertion, not a routing or synchronization mechanism. The backend
does not create or persist a routing plan. Thermal
tier, heat, coaccess, and residency scheduling metadata are intentionally not
checkpointed.

Incremental checkpoints append only changed entry rows and delete tombstones.
Each shard carries a monotonic sequence watermark. Unchanged page checksums are
not appended again; a manifest links the delta to its committed parent
snapshot. An external coordinator is the global transaction coordinator across all sparse
components. Workers export fragments only; after the cross-component barrier,
each component commit owner invokes `commit_fragments` once with all of that
component's worker artifacts. The global checkpoint is published only after
every component commit succeeds.

Score-enabled stores can promote a bounded top-K set of keys from a physical
page. Promotion copies the selected payload and metadata into fresh arena
slots, updates the single authoritative directory, and leaves every unselected
cold key in the original page. The source page is marked draining so it cannot
be reused as a destination, while remaining readable until its cold entries are
explicitly evicted or compacted. This is control-plane work and does not add a
second hot/cold lookup map.

Elastic restore reads routed `entries` using the caller-provided current
topology router. Same-topology layout restore imports `page_images` into their
saved native page and slot positions. It rejects logical routing parameters and
worker-topology mismatches.

## Performance Methodology

Reported measurements are workload-specific and are intended for regression
tracking and like-for-like comparisons. They are not portable throughput
guarantees.

The C++ workload uses one dimension-128 table, 500,000 possible `uint64_t`
keys, 65,536-key batches, and a fixed seed. Eighty percent of accesses target
the hottest ten percent of keys. Stores reserve capacity before warmup.

| Operation | Repetitions | Timed work |
|---|---:|---|
| `lookup_or_insert` | 10 | Batch generation and operation |
| `forward_lookup` | 20 | Batch generation and operation |
| `assign` | 10 | Batch generation and operation |
| `accumulate_gradients` | 10 | Batch generation and operation |
| `optimizer_step` | 10 | Step and next gradient preparation |
| `contains` | 20 | Batch generation and operation |
| `erase` | 10 | Repeated erase without reinsertion |

The storage-pressure workload uses dimension 128, 16,384 possible keys,
4,096-key batches, two workers, an 80/20 hot/cold access distribution, and
configured resident-memory and spill limits.

Mutation benchmarks deduplicate the 65,536 power-law candidates before calling
the unique-mutation APIs and report the rows actually submitted. The dedicated
`training_path_profile` binary uses 65,536 already-existing unique rows to
measure pure write, gradient, and optimizer paths separately from directory
growth.

## Measured Results

### Concurrent Memory Lookup Regression Gate

The 2026-08-03 lookup regression gate uses a reserved memory-only,
dimension-128 table with 10,000,000 shuffled `int64` keys, 8,192-key batches,
128 concurrent Python callers, up to 128 internal workers, one timed lookup
iteration, and seed 42. Each reported sample runs in a fresh Python process.

| Measurement | Lookup throughput |
|---|---:|
| Pre-fix source baseline | 20,842,691 keys/s |
| Fixed sample 1 | 21,567,284 keys/s |
| Fixed sample 2 | 21,500,684 keys/s |
| Fixed sample 3 | 21,539,993 keys/s |
| Fixed median | **21,539,993 keys/s** |
| Historical regression threshold | 21,348,000 keys/s |

The fixed median is 3.3455% above the pre-fix source baseline and 0.8993%
above the historical threshold. All three independent fixed samples pass the
threshold. The structural change separates the logical memory-work width from
the number of lookup tasks: the 4 MiB batch retains 16 logical telemetry
partitions but exposes up to 128 tasks for random-memory latency hiding, while
active concurrent invocations limit their execution share. This result does
not come from changing the 256 KiB work-grain configuration.

Weighted telemetry sampling reduces relaxed atomic updates for this batch from
15,872 to 992, a 93.75% reduction. One logical partition is sampled per
invocation with its partition count as weight, preserving page heat and
co-access totals over a complete partition rotation.

### Slot Directory Forward Scaling

The 2026-08-04 isolated forward-scaling proof uses the new slot-directory read
path after warmup. The workload is a dimension-32 memory-only store with 128
already-published keys, a 4,096-key fixed power-law batch, seed
`0x5EED5CA1E`, 400 timed `forward_lookup` repetitions, 48 allowed CPUs, and
thread counts 1/2/4/8/16/64.

| Threads | Forward throughput |
|---:|---:|
| 1 | 9.632 Mops/s |
| 2 | 16.436 Mops/s |
| 4 | 27.914 Mops/s |
| 8 | 40.707 Mops/s |
| 16 | 42.049 Mops/s |
| 64 | 12.591 Mops/s |

The scaling ratios versus one thread were 1.706x at two threads, 2.898x at
four threads, and 4.226x at eight threads, passing the current forward scaling
gate of 1.6x/2.5x/3.5x. The same audit also preserved the failed full
`embedding_store_benchmark --bind-affinity --skip-pressure` run: it threw
`key shard batch contains duplicate key`, so it is not an accepted performance
result. The single-thread full benchmark attempt only reached an epoch-stall
critical log and likewise is not an accepted throughput result.

### Sustained Retirement Pressure

The 2026-08-04 sustained-pressure proof uses eight producers, 1,048,576 total
keys, dimension 8, Auto storage, an 8 MiB resident-memory limit, a 512 MiB spill
limit, and 0.75/0.50 watermarks. It validates retained payloads after the
pressure phase and reports the current retirement/Sink contract rather than a
spill-throughput claim.

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

This run intentionally opened the retirement-pressure circuit before page
spill/swap became necessary. The accepted observation is that Sink admission
handled pressure without payload corruption; it is not evidence of disk
spill/swap throughput.

The threshold-eviction comparison uses a dimension-128 table, 512-key batches,
an 8,192-key resident budget, two workers, and five independent runs.

| Metric | Baseline | Threshold eviction |
|---|---:|---:|
| Throughput | 329,593 ops/s | 304,496 ops/s |
| Direct spill allocations | 10 | 1 |
| Paired throughput ratio | — | 92.125% |

Threshold eviction reduces direct spill allocations but adds bookkeeping cost.
Use it to bound growth, not as a free lookup-throughput optimization.

The adaptive Boltzmann exponent and normalization loops are compiler-verified
as vectorized with 32-byte vectors. Numerical tests keep the approximation
finite, normalize weights to one, and bound relative exponential error below
`0.001`.
The distribution is weight_i = exp(-(utility_i - min_utility) / temperature)
normalized across candidates; higher utility therefore receives lower eviction
probability. approximate_exp is a bounded base-2 polynomial approximation used
in the candidate loop, and target workload benchmarks validate locality benefit.

Checkpoint regressions cover concurrent worker export, manifest-pinned
recovery, checksums, optimizer state, score state, elastic worker-count
changes, exact-layout recovery, and a drained-page hole.

### NUMA-aware sharding and training paths

NUMA-aware sharding is opt-in and uses `numactl@2.0.19.bcr.1` through Bazel.
Shard ranges map to NUMA nodes, arena pages use page-aligned anonymous mappings
and explicit `numa_tonode_memory` placement, and dispatch workers are assigned
to node-local CPU groups. The affinity path caches the topology, rebinds pooled
worker threads when their logical target CPU changes, and the benchmark's
NUMA profile caps physical workers at the 24 physical cores on this host rather
than using the 48 SMT contexts. The isolated forward proof reported
14.055/17.992/20.858/21.141/22.713/21.888 Mops/s at 2/4/8/16/32/64 logical
workers; 32/16 is 1.074x and 64/32 is 0.964x. This is an intentional locality
and bandwidth boundary, not a correctness limitation: the 64-worker run does
not transmit work through SMT siblings.

Lookup-only NUMA forward uses the shard `ReadView` and reuses the route hash.
For estimated payloads below 512 KiB it bypasses NUMA re-bucketing because the
extra pass and dispatch cost exceed locality benefit for small batches. In the
dim=32/4,096-row proof this raised throughput from roughly 23–24 Mops/s with
forced routing to 44.66–45.25 Mops/s with contiguous partitioning. Larger
payloads retain node-local routing. The long-lived `exec::static_thread_pool`
still pays per-dispatch enqueue, wakeup, steal, completion and barrier costs;
an empty `bulk + sync_wait` measured 3.47/21.7/41.6/87.4 microseconds at
1/2/8/24 workers.

The deterministic `small_memory_all_paths_pressure_test` exercises
`lookup_or_insert`, `forward_lookup`, `assign`, `contains`, `erase`, duplicate
`accumulate_gradients`, and `optimizer_step` concurrently while Auto storage
uses a 64 MiB resident budget, 0.75/0.50 watermarks, a temporary spill file,
and 64 evictable 1 MiB pages. It completed with 45 swap-outs, 11 direct spill
allocations, 2 eviction-selection passes, and preserved every sampled swapped
page payload. Run it with:

```bash
bazel test -c opt //sparse_embedding/src/backend/hogwild:small_memory_all_paths_pressure_test \
  --test_output=errors
```

The latest profiling pass also measured assign at 26.5–27.0 Mops/s and
accumulate at 7.89–8.82 Mops/s with 24 workers; these are host-load-sensitive
snapshots and are separate from the 8-worker regression series below.

The training profile uses 8 workers, dimension 128, 65,536 existing unique
rows, and 100 repetitions. The latest affinity-enabled profile measured
21.67–21.82 Mops/s for pure assign, 7.24–7.33 Mops/s for accumulate plus its
required step, and 13.62–15.67 Mops/s for optimizer step alone across three
repetitions. The pure assign path excludes missing-key directory rebuilds, so
these figures are intentionally separate from the mixed full benchmark. Hash
calculation is performed once per input row in accumulation and carried into
the transient slot and optimizer lookup. A no-affinity control was measured;
it was not consistently faster and was slower for accumulation/optimizer on
this NUMA host.

```bash
bazel run -c opt //benchmarks:embedding_store_benchmark -- \
  --threads=8 --numa-aware-sharding --skip-pressure
bazel run -c opt //benchmarks:training_path_profile -- assign
```

## Public Validation

Run the optimized C++ regression suite:

```bash
bazel test -c opt --test_output=errors \
  //sparse_embedding/...
```

Build the Python extension and run checkpoint regressions:

```bash
bazel build -c opt //:copy_core_so
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py
```

Run the general Python suite and benchmarks appropriate to the target machine:

```bash
PYTHONPATH="$PWD" pytest -q
PYTHONPATH="$PWD" python3 benchmarks/python/benchmark_e2e.py
PYTHONPATH="$PWD" python3 benchmarks/python/benchmark_storage_pressure.py
bazel run -c opt //benchmarks:embedding_store_benchmark
```

## Design Boundary

Hot-key promotion is explicit control-plane work; it is not performed on every
lookup. The authoritative directory maps each key directly to its current
arena locator, so lookup never needs a second tier lookup. Independent HotMap
and ColdMap directories are intentionally not maintained.

Lance checkpoints materialize stable native page images through the SeqLock
reader protocol. Checkpointing does not clone spill files and does not require a
filesystem copy-on-write primitive. Entries and page images remain ordinary
Lance datasets pinned by manifest versions and checksums.

The current compaction policy moves every live key from a selected page before
releasing that page. This preserves a single lookup path and makes physical
page reclamation explicit.
