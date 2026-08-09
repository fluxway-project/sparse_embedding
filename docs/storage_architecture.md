# Storage Architecture and API

[中文](storage_architecture_zh.md)

[Performance and Reliability Report](performance_and_reliability_report.md)

[Lance Storage and Gravitino Validation](lance_storage_and_gravitino.md)

This document describes the storage, concurrency, and core mechanism contracts. The implementation is the source of truth for constants; the names
below make those policy points searchable.

## C++ API Contract

`PageStorage` and `PageEvictionProxy` are the only storage `pro::proxy`
facades. New code creates storage with `create_page_storage(StorageOptions)`
and passes the facade by value. Copying or moving the facade is shallow and
shares the underlying storage lifetime. `StoragePage` is move-only RAII
ownership for one allocation.

`set_page_eviction_proxy()` returns a `PageEvictionRegistration`. A
registration must be cleared exactly once. Empty, unknown, and already-cleared
tokens throw; there is no silent fallback. The proxy interface exposes only
score evaluation and evaluator registration. It does not expose lifecycle
locks; the concrete Hogwild adapter waits for active evaluator callbacks during
unregistration. The proxy generation changes when the active proxy identity or
lifetime transitions, so a candidate scored by an older active-proxy generation
cannot be swapped out.

The legacy virtual storage hierarchy and `PageStorage::create()` are not part
of the contract.

Runtime replacement remains confined to the `PageStorage` and
`PageEvictionProxy` facades. `SwapScheduler`, `SpillAllocator`, and
`PageRegistry` are private concrete classes rather than facades: the scheduler
owns wake, stop, join, and exception propagation; the spill allocator owns the
file descriptor, directory, capacity, and extent free-list; the registry owns
page identity, address indexing, eviction snapshots, and evaluator records.
Their locks are independent, and none of these cold-path modules adds a virtual
or type-erased call to memory allocation or lookup.

## Checkpoint Identity

Embedding and optimizer checkpoint rows are identified by `(dimension, key)`.
`key` alone is not unique because one store can contain the same `uint64` key
in multiple dimension tables. Checkpoint loaders must call
`assign_opt_state(keys, optimizer_states, dimension, steps)` with the physical
dimension from the persisted row group.

The store selects that dimension once, validates the optimizer-state width and
step count, and then resolves every key only inside the selected table. An
unknown dimension, missing optimizer, malformed state width, mismatched step
count, or key absent from that dimension raises immediately. No loader may
search other dimensions as a fallback.

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
    spill_directory="/path/to/local-spill-device",
    max_spill_bytes=1 << 40,
    high_watermark=0.8,
    low_watermark=0.6,
)
config = se.EmbeddingConfig(
    dims=[128],
    thread_budget=8,
    storage=storage,
)
store = se.EmbeddingStore(config, score_policy="lfu")
```

`MemoryOnly()` disables disk-backed pages and raises when the memory budget is
exhausted. `AutoStorage(memory_limit_bytes=0, spill_directory=..., max_spill_bytes=...)`
uses the specified spill device for trainable pages.
Invalid storage limits, alignments, watermarks, capacities, or registration
tokens raise instead of selecting a fallback policy.

## Runtime Architecture and Concurrency

### Lookup and Storage

The authoritative lookup path is `ParlayHash key -> slot_id ->
SegmentedLocatorTable -> arena_index`. Shard selection depends only on the key
hash; hot/cold tier and physical page do not participate in lookup routing. A
lookup performs one directory lookup and one locator read, avoiding a
HotMap-to-ColdMap fallback lookup.

Each shard publishes an immutable `MapBundle*`. A map value is a nonzero
`slot_id`, never an arena index. Readers acquire-load the current bundle and
do not take a shard shared mutex. Structural writers hold that shard's writer
mutex, clone the published bundle, apply insert/erase/reserve changes, then
publish the replacement with one atomic exchange. The retired bundle is returned
through the shared epoch-retirement pressure domain.

`SegmentedLocatorTable` owns fixed 32 KiB segments of 4,096 eight-byte locator
slots. A slot stores `{sequence, arena_index}` and is eight-byte aligned. Slot
IDs, locator slots, arena indices, pages, and store snapshots are reusable only
after an epoch-safe retirement. If a locator sequence approaches wraparound, the
directory rotates that key to a fresh slot and permanently retires the old slot
rather than risking ABA.

Payload reads and writes use the Linux x86_64 GCC/Clang SeqLock contract.
Writers CAS an even sequence to odd, update the value, score, optimizer state,
or compaction relocation payload, then release-publish an even sequence. Readers
load the sequence with acquire semantics, pass a compiler barrier, run the SIMD
copy, pass a second compiler barrier, and validate the sequence again. Readers
are lock-free and may retry under writer pressure; they do not mutate reader
counters or telemetry synchronously.

Store lifetime is protected by an atomic `StoreSnapshot*` and an invocation
gate. Forward, lookup, mutation, export, and metadata calls register once at
entry, acquire-load the snapshot, and leave through the gate. `clear`, reserve,
page-image import, and other reconfiguration work publish a new snapshot under
a private control-plane mutex after waiting for active invocations. Destruction
stops telemetry, closes the invocation gate, unregisters page-eviction
callbacks, retires the snapshot, and only then destroys slot state.

Lookup dispatch separates three quantities. The logical memory-work width
decides whether a batch benefits from parallel execution and defines telemetry
partitions. The task count provides enough independent random-memory work to
hide latency. The execution width is then limited by the number of active
lookup invocations so nested dispatch does not assign the full pool to every
concurrent caller. For an 8,192-key, dimension-128 batch on a 128-worker
topology, these quantities are 16 logical partitions, up to 128 tasks, and one
execution worker for an invocation that enters when 128 invocations are active.
The share is fixed when each invocation enters; earlier invocations are not
retroactively resized. Explicit `memory_parallelism` remains the upper bound
for both task decomposition and execution.

### ParlayHash Concurrency

Each embedding dimension owns multiple key shards. Lookup never takes a shard
mutex. Insert, erase, reserve, locator rotation, checkpoint import, and
compaction publication take only the affected shard writer mutex while they
rebuild and atomically publish immutable directory state. There is no global
Upsert mutex, so unrelated shards continue in parallel while one shard grows.

The initial-capacity policy is centralized in the store:

- `kParlayHashCapacityMultiplier = 4`
- `kInitialKeysPerShard = 1`
Each shard starts with one expected key and grows independently. This keeps
startup RSS and cache footprint bounded without imposing a logical key quota.
Known production workloads should call `reserve(keys_count, dim)` or
`reserve_per_shard(counts, dim)` before the timed phase. Reserve rebuilds
capacity outside the hot path; structural index overflow and storage allocation
failures are reported precisely.

## Eviction Mechanism and Background Threads

### Score-Threshold Eviction

`evict_by_score(max_score, max_evict_count, ...)` selects the globally coldest
live entries whose current score is at most `max_score`. Candidate collection
uses a bounded per-worker heap, so memory remains proportional to the eviction
budget rather than the complete table. 

Final deletion revalidates the directory entry, locator slot, score, and live
bit while holding only the affected shard writer mutex. It does not use a store
lifecycle shared lock or a global exclusive scan lock.

The live bitmap is cleared before index retirement. Page heat evaluation skips
non-live slots, preventing deleted entries from distorting page heat.
This is entry-level garbage collection. It controls arena growth but does not
release a partially occupied storage page.

The output spans represent one physical dimension. If a store has candidates
from multiple dimensions, the operation throws before copying any value; callers
must evict each dimension separately.

### Background Eviction Thread

Eviction is event-driven. The background thread waits on a condition variable
until resident evictable memory reaches the high watermark, then continues
until it reaches the low watermark. It does not use periodic polling or a
probability gate.

The storage maintains an incremental index of anonymous evictable pages, so a
background swap selection pass does not scan the complete page registry.

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

The storage layer does not receive a store lifecycle lock through its facade.
When a score-enabled store unregisters its proxy, the concrete proxy removes
page evaluators and waits for active evaluator callbacks to exit before the
store frees slot state.

### Retirement Pressure and Sink

All objects that may remain visible to old readers are retired through the same
epoch-backed `RetirementPressure` domain. The domain accounts retired bundle,
snapshot, page, arena-index, locator-slot, and page-image bytes. The default
configuration uses `backpressure_fraction = 0.15`,
`circuit_fraction = 0.25`, and `epoch_stall_timeout = 30s`; construction
rejects invalid fractions, and Auto storage requires the circuit threshold to
be below the spill high watermark.

Below backpressure the store runs normally. Between backpressure and circuit,
each writer or compactor yields at most once per call. At circuit pressure,
control-plane mutations that would create new retirement throw unless already
admitted. Existing keys continue to read and update normally. Missing
lookup-or-insert keys return the dimension-sized, 64-byte-aligned zero
`SinkImage`, set `inserted_mask = 0`, report `contains = false`, do not
grow capacity, and do not enter checkpoints. Writes to sink keys are no-ops;
after pressure falls below the backpressure threshold, new insertion resumes.

### Adaptive Placement Signals

`AdaptiveBoltzmannPolicy` combines decayed heat, co-access density, and
residency age into a page utility. Temperature adapts to memory pressure and
utility dispersion. The utility loop accepts caller-owned structure-of-arrays
spans and has no policy-side allocation.

`CoaccessSketch` records bounded co-access telemetry. Forward lookup samples at
most 16 keys into stack storage and attempts one bounded ring publication per
sampled table. A single store-owned consumer performs page-heat and co-access
counter mutations in batches. Publication never blocks or expands the ring;
full rings increment `dropped_batches` and discard the sample without affecting
embedding values. Forward lookup rotates one sampled logical partition per
invocation and weights its observations by the logical partition count.
Telemetry partitioning is therefore independent of task decomposition and
concurrency limiting. Store destruction stops and joins the consumer before
destroying any slot referenced by queued events.

The adaptive policy uses the Boltzmann distribution
`weight_i = exp(-(utility_i - min_utility) / temperature) / sum_j exp(...)`.
Higher utility monotonically lowers eviction probability. The minimum-utility
shift preserves the normalized distribution while keeping exponents non-positive.
Temperature adapts to memory pressure and utility dispersion within configured
bounds. The candidate loop uses approximate_exp, which is a bounded base-2
polynomial approximation to exp; tests bound relative error below 0.001 and
verify finite weights normalized to one. Co-access density enters utility, so
frequently co-used keys receive lower eviction probability and can be clustered
by full-page relocation. Expected locality benefit is validated by target
workload benchmarks.

`compact_adaptive_pages()` applies the selected physical-page budget through
the same full-page relocation transaction as explicit compaction. A source page
first enters a draining state, then every live key is revalidated and moved
under its own shard's exclusive access; only after its live bitmap is empty are
its data and score pages epoch-retired. Draining and retired source slots are
never reused as destinations. The directory remains the sole lookup authority,
so page movement never introduces a second HotMap/ColdMap search.

Score-enabled stores also expose `promote_hot_keys(dim, page_id, maximum_keys,
minimum_score)`. This operation selects a deterministic top-K from one source
page, relocates only those keys through the authoritative shard directory, and
publishes an immutable Hot Page snapshot. The source page remains allocated and
continues to serve unselected cold keys; it is marked draining so its slots are
never reused as relocation destinations. Callers can inspect the published
snapshot count with `promoted_hot_page_size(dim)`.

`seal_adaptive_pages()` produces immutable, 64-byte-aligned page snapshots for
checkpoint/layout control work. A sealed page is not a lookup index and does
not change routing. Selector and planner APIs additionally accept reusable
scratch objects, so a scheduler can retain vector capacities across control
cycles rather than allocating temporary selection buffers each time.

## Framework-neutral Lance Checkpoint Protocol

Distributed integrations use `SparseCheckpointProtocol` and supply a
`CheckpointSaveIdentity`, one `CheckpointParticipant` per writer, and one final
commit call containing all fragments. `CheckpointCommitAuthority` carries an
opaque scope, monotonic generation, and authority identity; it contains no type
or identifier from a particular orchestration framework.

`LocalLanceStorage` and `GravitinoLanceStorage` share one Lance checkpoint
transaction engine. It stores one materialized `(dim, key)` entry state,
physical page images, and a snapshot manifest. Storage adapters only provide
dataset locations, staging prefixes, durable records, and commit locking. The
manifest is the visibility boundary and pins exact entries and page-image Lance
versions. Duplicate keys across participants, incomplete participant coverage,
schema disagreement, and stale commit authority fail explicitly.

Restore selects an exact `SparseCheckpointLocator`. `PhysicalRestore` requires
the saved participant topology and rebuilds native page layout.
`ElasticRestore` accepts the caller's stable `partition_fn(keys,
partition_count)` and target partition. It first scans only keys and row state,
then reads payload columns only for rows assigned to the target partition; no
topology or routing fallback exists inside Sparse Embedding.

The native `ModuloPartitioner` follows the language-independent normalized
modulo contract used by Python and NumPy: every owner is in
`[0, partition_count)`, including for negative signed keys. Custom partition
functions remain responsible for their own routing semantics.

Thermal tier, heat, coaccess, and residency scheduling metadata are intentionally
not checkpointed. They are re-observed after restore.

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
| `retired_bytes` | Epoch-retired bytes not yet reclaimed |
| `retirement_pressure_state` | Current pressure state: normal, backpressure, or circuit |
| `retirement_backpressure_count` | Writer calls that yielded for retirement pressure |
| `retirement_sink_hit_count` | Missing-key admissions redirected to Sink |
| `oldest_unreclaimed_age_ns` | Age of the oldest unreclaimed retirement record |
| `epoch_stall_count` | Count of epoch-stall events that emitted a CRITICAL JSON log |
| `spill_directory` | Selected or configured spill directory |

`spill_count` and `swap_out_count` are intentionally different. Direct
file-backed allocation increments the former; background eviction increments
the latter.
