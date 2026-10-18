# Incremental Checkpointing and Streaming Compaction

[中文](incremental_checkpoint_zh.md) | [Lance Storage and Gravitino Integration](lance_storage_and_gravitino.md) | [Storage Architecture](storage_architecture.md) | [Performance and Reliability Report](performance_and_reliability_report.md) | [Dynamic Learning Rate](dynamic_learning_rate.md)

This document provides a comprehensive architecture specification and user guide for the **Incremental Checkpointing** and **Streaming Compaction** subsystem in Sparse Embedding.

---

## 1. Overview and Core Capabilities

In large-scale recommendation training, embedding tables frequently contain hundreds of millions of sparse parameters. Performing periodic full snapshots consumes excessive I/O bandwidth, introduces prolonged training stalls, and produces redundant storage data.

Sparse Embedding provides native **Incremental Checkpointing** and **Streaming Compaction**:

- **Consistent Cuts**: Capture quiesces competing store invocations and uses epoch protection. Bounded native writer channels provide backpressure; slow IO can delay capture.
- **Native In-Memory Mutation Journaling**: Shard-local journals record key-level mutations and monotonic sequence numbers with zero mutex contention on forward lookup hot paths.
- **Delta-Chain Schema & Tombstone Tracking**: Physical Lance slices persist `mutation_sequence`, `row_state` (`"active"` vs. `"deleted"`), and optional eviction `score` (LFU/LRU).
- **Native Multi-Version Reconciliation**: Bounded external sorting validates historical identities and selects surviving records across the delta chain (`FULL` → `INCREMENTAL` → ...). Requested payload is staged during one Lance source pass and replayed in source order; only surviving keys allocate target slots.
- **Offline / Online Streaming Compaction**: Compaction shares reconciliation and payload decoding with entry restore, while bounded writer channels produce a self-contained `FULL` checkpoint. Temporary disk grows with staged history; sorter and batch RAM follow `CheckpointStreamPolicy`.

- **Page-Image Restoration Boundaries**: Page-image restoration consumes bounded page scan batches in one store-owned empty-target lifecycle. It is eligible only when native validation proves unchanged source ownership and the target page geometry matches. A changed assignment that moves historical identities requires a `FULL` save after entries restoration, including when the partition count stays the same (see Section 2.4).
- **Lifecycle & Operational Invariants**: Generation is published after all page imports and checks succeed; a failed import attempt leaves the target terminal before access reopens. Training restore requires the target optimizer's actual state layout; weight-only loading is an explicit scope. Final store memory, sparse physical optimizer capacity, and Lance runtime caches are separate from bounded temporary buffers. Restore includes control-plane and I/O synchronization and is not an entirely lock-free operation.

---

## 2. Architecture and Data Flow

```text
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                               Online Training Pipeline                               │
│   Forward / Backward ──► Shard Mutation ──► CheckpointMutationJournal                │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │ capture_checkpoint() (non-blocking)
┌──────────────────────────────────────────▼───────────────────────────────────────────┐
│                          Consistent Cut & Fragment Staging                           │
│   Dirty Entries + Tombstones ──► _staging/<attempt>/<participant_id>/entries.lance   │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │ 2PC Atomic Commit (Manifest Append)
┌──────────────────────────────────────────▼───────────────────────────────────────────┐
│                           Lance Lakehouse Metadata Ledger                            │
│   snapshot_manifest.lance (parent_reference, watermarks, mode)                       │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┴────────────────────────┐
                    ▼                                               ▼
┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
│      Native Restore Resolution       │        │         Streaming Compaction         │
│  Multi-version delta chain           │        │  Consolidates delta chain into       │
│  reconciled in C++ streaming         │        │  a clean FULL snapshot via           │
│  RestorePlan.from_partition()        │        │  storage.compact_checkpoint()        │
└──────────────────────────────────────┘        └──────────────────────────────────────┘
```

### 2.1 In-Memory Mutation Journaling

Each shard maintains a lightweight `CheckpointMutationJournal`:
- **Dirty Key Tracking**: When embedding vectors or optimizer states are modified during `assign`, `accumulate_gradients`, or `optimizer_step`, their slot and monotonic sequence numbers are recorded in the active journal.
- **Tombstone Recording**: Explicit deletions (`erase`, `erase_dimension`) or score-based evictions emit deletion tombstones, preserving deletion intent across delta boundaries.
- **Zero Reader Contention**: Read hot paths (`forward_lookup`, `lookup_or_insert` hits) never acquire journal mutexes or increment sequence counters.

### 2.2 Two-Phase Consistent Cut Protocol

1. **Capture Cut (`capture_checkpoint`)**: Quiesces the active journal generation, captures the current mutation watermark, and initializes a clean delta generation for incoming training operations.
2. **Asynchronous Streaming**: Worker threads stream dirty rows and deletion tombstones into isolated staging directories (`_staging/<attempt>/<participant_id>/entries.lance`).
3. **Commit & Acknowledge (`acknowledge_store_mutations`)**: Upon successful 2PC commit to `snapshot_manifest.lance`, the store advances its committed mutation watermark and safely reclaims acknowledged journal entries.

### 2.3 Lance Layout Schema

Physical entry slices (`entries.lance`) store canonical rows adhering to `kEntryColumns`:

| Column | Type | Description |
|---|---|---|
| `dim` | `uint32` | Table embedding vector dimension |
| `key` | `int32` / `uint32` / `int64` / `uint64` | Native Arrow fixed-width integer key |
| `embedding_values` | `binary` | Parameter embedding vector bytes |
| `embedding_dtype` | `utf8` | Value data type (`float16`, `float32`, `bfloat16`) |
| `optimizer_state_values` | `binary` | First/second moment momentum vectors (if enabled) |
| `optimizer_state_dtype` | `utf8` | Optimizer state data type |
| `optimizer_state_stride` | `uint32` | Optimizer state row stride in bytes |
| `step` | `uint32` | Optimizer step count |
| `score` | `uint64` | Persisted eviction score (e.g., LFU count), if enabled |
| `sequence` | `uint64` | Internal SeqLock slot version sequence |
| `mutation_sequence` | `uint64` | Monotonically increasing sequence number per participant |
| `row_state` | `utf8` | Entry state: `"active"` (live) or `"tombstone"` (deleted) |
| `source_participant_index` | `uint32` | Originating participant rank index |

### 2.4 Physical Source Shards and Cross-Topology Checkpoint Invariants

Incremental checkpointing represents a delta over a base snapshot: $f(\text{Base}) + \Delta$. When cluster topology changes (such as worker count scaling or partition specification updates), key ownership shifts across participants. In this scenario, the next checkpoint following topology reconfiguration must be a `FULL` checkpoint rather than an `INCREMENTAL` checkpoint.

#### 1. Physical Architecture: Independent Participant Shards
On distributed training paths, achieving zero mutex contention and non-blocking cuts requires that distributed workers write into isolated physical directories without concurrent table lock contention. Each training worker independently tracks mutations for its local shards and stages entries directly into its dedicated physical sub-directory:
```
_staging/<attempt>/<participant_id>/entries.lance
```
The global manifest `snapshot_manifest.lance` references these discrete Lance fragments as lightweight pointers. As a result, physical Lance storage is inherently partitioned by participant (source shards).

#### 2. Cross-Topology Invariants and Snapshot Consolidation
Because deltas rely on a deterministic baseline and monotonic sequence numbers within each participant journal, shifting key ownership across topology changes introduces key requirements:

```text
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ Topology 1 (2 Workers Training):                                                     │
│  Worker 0 ──► Fragment 0 [Key 100 @ seq=10] ─┐                                       │
│  Worker 1 ──► Fragment 1 [Key 200 @ seq=12]  │                                       │
│                                              │ Resharding (2 -> 4 Workers)           │
│                                              ▼ Restored to: Worker 3                 │
│ Topology 2 (4 Workers Training) ── Post-Reshard Checkpoint Invariant:                │
│                                                                                      │
│ [Disallowed Cross-Topology INCREMENTAL]      [Enforced Invariant: FULL Baseline]     │
│   • Worker 3 logs only newly dirtied keys      • Worker 3 materializes full state    │
│   • Unmodified Key 100 missing from delta      • Key 100 captured in Fragment 3      │
│   • Future restore must scan all old shards    • Truncates old topological chain     │
│   • Read explosion + tombstone resurrection    • Enables GC of obsolete fragments    │
│   (❌ Violates consistency & bounds)           (✅ Subsequent saves resume INC)       │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

The invariant enforcing `FULL` checkpointing after topology reconfiguration addresses four core issues:
1. **Base Slice Continuity**: An incremental fragment contains only entries mutated in the current step. If a worker saves an incremental delta immediately after adopting new keys, historical keys that were not mutated in the subsequent step would be absent from the delta. Reconstructing state would require recursively backtracking into fragments across all predecessor workers.
2. **Bounded Read Amplification**: Cascading cross-topology deltas would require scanning historical fragments across all prior topologies during recovery, degrading restore time from $O(N_{\text{checkpoints}})$ to $O(N_{\text{participants}} \times N_{\text{checkpoints}})$.
3. **Sequence Domain Isolation & Tombstone Elimination**: Sequence numbers are local to each worker journal without a global cluster clock. Cross-node merges across different sequence domains risk misordering deletion tombstones against active keys.
4. **Lineage Truncation & Safe Garbage Collection**: If incremental chains traversed topology boundaries, storage files from older topologies would remain pinned indefinitely by downstream references. A `FULL` checkpoint serves as a **snapshot consolidation and lineage truncation boundary**, establishing a self-contained baseline for each participant and allowing historical fragments to be reclaimed by storage garbage collection. Subsequent iterations resume lightweight incremental checkpoints against the new baseline.

---

## 3. Quickstart and Usage Guide

### 3.1 Saving Full and Incremental Checkpoints

```python
import numpy as np
import sparse_embedding as se
from sparse_embedding.checkpoint import (
    CheckpointSaveMode,
    LocalLanceStorage,
    SparseCheckpoint,
)

# 1. Initialize storage and facade
storage = LocalLanceStorage.from_checkpoint_root("/checkpoints/user_model")
checkpoint = SparseCheckpoint(storage)

# 2. Save baseline FULL checkpoint
committed_full = checkpoint.save_training_checkpoint(
    store,
    "user_model",
    mode=CheckpointSaveMode.FULL,
)
print("Committed FULL version:", committed_full.version_id.decode())

# 3. Training continues... modifications and deletions are tracked automatically
# Save INCREMENTAL checkpoint referencing parent handle
committed_inc = checkpoint.save_training_checkpoint(
    store,
    "user_model",
    mode=CheckpointSaveMode.INCREMENTAL,
    parent_handle=committed_full.handle,
)
print("Committed INCREMENTAL version:", committed_inc.version_id.decode())
```

> **Note**: An incremental checkpoint must specify the same lineage name (`"user_model"`) and provide a valid `parent_handle`. Failing to provide a parent handle for incremental mode strictly raises a `ValueError`.

### 3.2 Restoring from Incremental Chains

The restore pipeline validates parent continuity and reconciles multi-version records in C++:

```python
from sparse_embedding.checkpoint import (
    IngestionEngine,
    PartitionSpec,
    RestorePlan,
)

# Scenario A: Same target topology. The plan identifies the current partition.
plan = RestorePlan.from_partition(
    spec=PartitionSpec.modulo(),
    partition_count=1,
    partition_id=0,
    key_dtype="uint64",
)
restore_store = se.EmbeddingStore(config)
checkpoint.load_training_checkpoint(
    restore_store,
    committed_inc.handle,
    plan,
)

# Scenario B: The same plan type also handles a changed target topology.
reshard_plan = RestorePlan.from_partition(
    PartitionSpec.hash(seed=42),
    partition_count=8,
    partition_id=0,
    engine=IngestionEngine.RECONSTRUCT_ENTRIES,
    key_dtype="uint64",
)
elastic_store = se.EmbeddingStore(config)
checkpoint.load_training_checkpoint(elastic_store, committed_inc.handle, reshard_plan)
```

### 3.3 Streaming Compaction

As delta chains grow, periodic compaction merges fragments into a consolidated `FULL` checkpoint:

```python
# 1. High-level managed compaction: merges delta chain into a new FULL checkpoint
compacted_handle = storage.compact_checkpoint(
    committed_inc.handle,
    include_page_images=True,  # Optional: generate page cache for instant cold-start
)

# Future restores can load directly from the compacted FULL handle
compacted_store = se.EmbeddingStore(config)
checkpoint.load_training_checkpoint(
    compacted_store,
    compacted_handle,
    RestorePlan.from_partition(
        PartitionSpec.modulo(),
        partition_count=1,
        partition_id=0,
        key_dtype="uint64",
    ),
)

# 2. Standalone directory compaction: compact raw Lance fragments directly
from sparse_embedding.checkpoint import compact_fragments

compact_fragments(
    source_path="/checkpoints/user_model/_staging",
    target_path="/exports/user_model",
)
```

### 3.4 Inference Loading from Incremental Checkpoints

Inference instances skip optimizer allocation by setting `enable_optimizer=False`:

```python
infer_config = se.EmbeddingConfig(
    dims=[128],
    key_dtype="uint64",
    value_dtype="float32",
    enable_optimizer=False,  # Weight-only inference store
)
infer_store = se.EmbeddingStore(infer_config)

checkpoint.load_inference_checkpoint(
    infer_store,
    committed_inc.handle,
    RestorePlan.from_partition(
        PartitionSpec.modulo(),
        partition_count=1,
        partition_id=0,
        key_dtype="uint64",
    ),
)
```

---

## 4. Reliability and Integrity Guarantees

- **Strict Lineage Fencing & Head Resolution**: Incremental checkpoints require monotonic sequence increments and matching lineage names. Cross-lineage parent references raise `CheckpointConflictError`. Unacknowledged checkpoints (retaining active `seintent` intent tags or `sereclaiming` exclusion markers) are strictly excluded from lineage head resolution.
- **Atomic Abort & Reclaim Protocol**: Aborting an unacknowledged save or reclaiming an orphaned committed row adheres to a strict fail-safe lifecycle order:
  1. Persist the `sereclaiming.<locator>` exclusion marker to ensure exclusion from lineage resolution even if subsequent steps fail;
  2. Physically delete fragment data roots;
  3. Purge parent chain references, reference tags, and page attempt/write intents;
  4. Delete the row from `snapshot_manifest.lance` and invalidate manifest caches;
  5. Delete the `seintent` intent tag, pending outcome, and committed tag;
  6. Finally delete the `sereclaiming` marker.
  This invariant ensures that an unacknowledged or aborted checkpoint can never be exposed as the lineage head across crashes or partial deletion failures.
- **Parent Reference Pinning**: Active parent datasets are pinned by reference tags. Storage cleanup tasks will never prune datasets that are still required by downstream delta chains.
- **Empty Delta Optimization**: When no mutations occur within an interval, the checkpoint engine appends a lightweight commit record to `snapshot_manifest.lance` without creating empty payload datasets.
- **Deterministic Resolution**: Conflicting key updates across multi-version deltas are resolved strictly by the highest `(version, mutation_sequence)` tuple, guaranteeing bit-identical consistency regardless of chunk read order.

---

## 5. Verification Commands

Run the test suites covering incremental checkpointing, chain reconciliation, and streaming compaction:

```bash
# Core incremental checkpoint contracts
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py -k "incremental"

# Multi-version chain compaction and page image materialization
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py -k "compact"

# Key type matrix verification across all 4 integer dtypes
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_key_checkpoint_matrix.py
```
