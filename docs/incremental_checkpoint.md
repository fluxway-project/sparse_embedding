# Incremental Checkpointing and Streaming Compaction

[中文](incremental_checkpoint_zh.md) | [Lance Storage and Gravitino Integration](lance_storage_and_gravitino.md) | [Storage Architecture](storage_architecture.md) | [Performance and Reliability Report](performance_and_reliability_report.md) | [Dynamic Learning Rate](dynamic_learning_rate.md)

This document provides a comprehensive architecture specification and user guide for the **Incremental Checkpointing** and **Streaming Compaction** subsystem in Sparse Embedding.

---

## 1. Overview and Core Capabilities

In large-scale recommendation training, embedding tables frequently contain hundreds of millions of sparse parameters. Performing periodic full snapshots consumes excessive I/O bandwidth, introduces prolonged training stalls, and produces redundant storage data.

Sparse Embedding provides native, production-grade **Incremental Checkpointing** and **Streaming Compaction**:

- **Non-Blocking Consistent Cuts**: Snapshots dirty entries and deletion tombstones under epoch protection, decoupling memory capture from disk I/O and eliminating training stalls.
- **Native In-Memory Mutation Journaling**: Shard-local journals record key-level mutations and monotonic sequence numbers with zero mutex contention on forward lookup hot paths.
- **Delta-Chain Schema & Tombstone Tracking**: Physical Lance slices persist `mutation_sequence`, `row_state` (`"active"` vs. `"tombstone"`), and optional eviction `score` (LFU/LRU).
- **Streaming Multi-Version Reconciliation**: The restore engine deterministically resolves surviving keys across the delta chain (`FULL` → `INCREMENTAL` → ...) in native C++, skipping tombstoned historical entries without inflating peak RAM.
- **Offline / Online Streaming Compaction**: Bounded-channel streaming compaction consolidates multi-version delta chains into a new, self-contained `FULL` checkpoint with bounded memory footprint.

---

## 2. Architecture and Data Flow

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Online Training Pipeline                        │
│   Forward / Backward ──► Shard Mutation ──► CheckpointMutationJournal  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ capture_checkpoint() (non-blocking)
┌───────────────────────────────────▼────────────────────────────────────┐
│                  Consistent Cut & Fragment Staging                     │
│   Dirty Entries + Tombstones ──► _staging/<attempt>/<participant_id>/entries.lance
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 2PC Atomic Commit (Manifest Append)
┌───────────────────────────────────▼────────────────────────────────────┐
│                    Lance Lakehouse Metadata Ledger                     │
│   snapshot_manifest.lance (parent_reference, watermarks, mode)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────────────┐       ┌────────────────────────────────┐
│   Native Restore Resolution   │       │      Streaming Compaction      │
│  Multi-version delta chain    │       │  Consolidates delta chain into │
│  reconciled in C++ streaming  │       │  a clean FULL snapshot via     │
│  RestorePlan.from_topology()  │       │  storage.compact_checkpoint()  │
└───────────────────────────────┘       └────────────────────────────────┘
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
    CheckpointParticipant,
    IngestionEngine,
    PartitionSpec,
    RankRestore,
    ReshardRestore,
    RestorePlan,
)

# Scenario A: Automated topology-aware restore (Recommended)
# Automatically selects RankRestore when saved and current participant counts match,
# or ReshardRestore when the cluster scales or re-shards.
plan = RestorePlan.from_topology(
    saved_participant_count=1,
    participant=CheckpointParticipant(0, 1),
    spec=PartitionSpec.modulo(),
    key_dtype="uint64",
)
restore_store = se.EmbeddingStore(config)
checkpoint.load_training_checkpoint(
    restore_store,
    committed_inc.handle,
    plan,
)

# Scenario B: Explicit rank-preserving restore (identical cluster topology)
rank_plan = RestorePlan.from_participant(
    CheckpointParticipant(0, 1),
    engine=IngestionEngine.RECONSTRUCT_ENTRIES,
    key_dtype="uint64",
)
checkpoint.load_training_checkpoint(restore_store, committed_inc.handle, rank_plan)

# Scenario C: Explicit reshard restore (dynamic cluster re-sharding)
reshard_plan = RestorePlan.from_partition(
    PartitionSpec.hash(seed=42),
    partition_count=8,
    partition_id=0,
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
    RestorePlan.from_participant(
        CheckpointParticipant(0, 1),
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
    RankRestore(
        participant=CheckpointParticipant(0, 1),
        key_dtype="uint64",
    ),
)
```

---

## 4. Reliability and Integrity Guarantees

- **Strict Lineage Fencing**: Incremental checkpoints require monotonic sequence increments and matching lineage names. Cross-lineage parent references raise `CheckpointConflictError`.
- **Parent Reference Pinning**: Active parent datasets are pinned by reference tags. Storage cleanup tasks will never prune datasets that are still required by downstream delta chains.
- **Empty Delta Optimization**: When no mutations occur within an interval, the checkpoint engine appends a lightweight commit record to `snapshot_manifest.lance` without creating empty payload datasets.
- **Zero Orphan Cleanup**: Failed or aborted checkpoint attempts automatically trigger staged prefix cleanup with preserved primary exception diagnostics.
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
