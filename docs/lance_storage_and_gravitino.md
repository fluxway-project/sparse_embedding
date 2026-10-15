# Lance Storage and Gravitino Checkpoint Integration Guide

[中文](lance_storage_and_gravitino_zh.md) | [Architecture and API](storage_architecture.md) | [Incremental Checkpointing](incremental_checkpoint.md) | [Performance and Reliability Report](performance_and_reliability_report.md) | [Dynamic Learning Rate](dynamic_learning_rate.md)

This document provides a comprehensive guide for engineers and architects integrating the Sparse Embedding checkpoint engine with the Lance columnar format, Apache Gravitino Lakehouse catalog, and Lance REST operations.

---

## 1. Overview and Core Benefits

In massive-scale recommendation and sparse embedding workloads, traditional checkpoint solutions struggle with three key bottlenecks: gigantic monolithic files that cannot be partially queried, severe multi-worker write conflicts, and a lack of unified Lakehouse metadata governance.

Sparse Embedding adopts **Lance columnar storage** combined with **Apache Gravitino** to deliver:

- **Decoupled Data and Metadata via Dual-Lance Architecture**: Control-plane metadata records transaction ledgers while data-plane slices are streamed independently, eliminating multi-node write contention and manifest explosion.
- **Zero-Copy Analytical Pushdown**: Key columns are stored as native Arrow fixed-width integers, enabling zero-copy projection and predicate pushdown in analytical engines such as DuckDB, DataFusion, and Lance.
- **Instant Restore and Elastic Re-sharding**: Deterministically reconstructs native memory page tables or applies AVX-512 SIMD slice mask filtering (`filter_span`) for instant topological re-sharding across cluster scale-in/scale-out.
- **Lakehouse Metadata Governance**: Integrates natively with Gravitino and Lance REST for multi-tenant namespaces, lease-based concurrency locking, and centralized credential management.

---

## 2. Storage Hierarchy and Dual-Lance 2PC Architecture

Persistence is cleanly partitioned into two physical Lance datasets governed by a Two-Phase Commit (2PC) protocol:

```text
SparseCheckpoint
    │
    ├── LocalLanceStorage / GravitinoLanceStorage (Shared Transaction Engine)
    │
    ├── [Control-Plane Ledger] snapshot_manifest.lance
    │       └── Global visibility boundary, transaction state (PREPARED → COMMITTED / ABORT),
    │           leases, fencing markers, and fragment mapping lists (1 compact row appended per commit).
    │
    └── [Data-Plane Physical Slices] _staging/<attempt>/<fragment_id>/
            ├── entries.lance      (Canonical keys, values, and optimizer state slices - SSOT)
            └── page_images.lance  (Optional physical page image cache for accelerated load)
```

### Transaction Workflow and Write Isolation

1. **Concurrent Worker Staging**: Each worker streams its physical slice into an isolated directory (`_staging/<attempt>/<fragment_id>/entries.lance`), completely isolated from peers without write lock contention.
2. **Two-Phase Atomic Commit (2PC)**:
   - **Prepare Phase**: Validates fragment integrity, schema conformance, and monotonic commit generation across all participants.
   - **Commit Phase**: Appends a single commit record to `snapshot_manifest.lance` to advance the global visibility boundary. Incomplete or aborted attempts are never exposed to readers.
3. **Atomic Publication Guarantees**:
   - **Local Filesystem**: Utilizes POSIX `rename(2)` atomic directory rename with zero intermediate coordinating files.
   - **Object Storage (S3 / S3A)**: Mitigates the absence of atomic directory renames by writing immutable staged prefixes paired with a single-object strongly consistent pointer (`.manifest_pointer.json`) for linearizable atomic publication.
   - **Zero Orphan Cleanup**: Overwriting, rollback, and prefix deletion cascade strictly; failures raise explicit exceptions without silent degradation.

---

## 3. Quickstart: Checkpoint Save and Restore

The following example demonstrates initializing local Lance storage, saving training state, and restoring via automated topology routing or explicit plans:

```python
import numpy as np
import sparse_embedding as se
from sparse_embedding.checkpoint import (
    CheckpointParticipant,
    CheckpointSaveMode,
    IngestionEngine,
    LocalLanceStorage,
    PartitionSpec,
    RankRestore,
    ReshardRestore,
    RestorePlan,
    SparseCheckpoint,
    compact_fragments,
)

# 1. Initialize storage and checkpoint facade
storage = LocalLanceStorage.from_checkpoint_root("/checkpoints/user_embedding")
checkpoint = SparseCheckpoint(storage)

# 2. Save training checkpoint (default persists canonical entries, saving 50% write bandwidth)
committed = checkpoint.save_training_checkpoint(
    store,
    "user_embedding",
    mode=CheckpointSaveMode.FULL,
)

# Optional: persist physical page images for fastest restore cold-start
# committed = checkpoint.save_training_checkpoint(
#     store, "user_embedding", include_page_images=True
# )

# 3. Scenario A: Automated Topology-Aware Restore (Recommended)
# Auto-selects RankRestore when cluster topology matches, or ReshardRestore when resized
restored_store = se.EmbeddingStore(config)
plan = RestorePlan.from_topology(
    saved_participant_count=1,
    participant=CheckpointParticipant(0, 1),
    spec=PartitionSpec.hash(seed=42),
    key_dtype="uint64",
)
checkpoint.load_training_checkpoint(restored_store, committed.handle, plan)

# 4. Scenario B: Explicit Rank Restore (RankRestore)
# Used when worker count and partition topology match; reconstructs pages deterministically
checkpoint.load_training_checkpoint(
    restored_store,
    committed.handle,
    RestorePlan.from_participant(
        CheckpointParticipant(0, 1),
        engine=IngestionEngine.RECONSTRUCT_ENTRIES,
        key_dtype="uint64",
    ),
)

# 5. Scenario C: Explicit Reshard Restore (ReshardRestore)
# Used when cluster size changes; dynamically filters keys for the current partition
restored_store_elastic = se.EmbeddingStore(config)
checkpoint.load_training_checkpoint(
    restored_store_elastic,
    committed.handle,
    RestorePlan.from_partition(
        PartitionSpec.hash(seed=42),
        partition_count=8,
        partition_id=0,
        key_dtype="uint64",
    ),
)
```

---

## 4. Restore Strategies and Data Type Contracts

### 4.1 Restore Strategy Matrix

| Restore Strategy | Primary Use Case | Underlying Mechanism | Error & Defense Policy |
|---|---|---|---|
| **`RestorePlan.from_topology`** | **Automated topology routing (Recommended)** | Inspects saved participant count: auto-routes to `RankRestore` if matching, or `ReshardRestore` if changed | Validates `saved_participant_count > 0`, zero silent fallback |
| **`RankRestore (RECONSTRUCT_ENTRIES)`** | Identical topology, post-compaction restore | Reads canonical `entries.lance`, reconstructs native page layouts deterministically | Single Source of Truth (SSOT), fully deterministic |
| **`RankRestore (FAST_PAGE_CACHE)`** | Ultra-fast cold-start (requires `include_page_images=True`) | Direct bounded-batch mapping of `page_images.lance` binary pages, bypassing row builds | If cache was not saved, **strictly raises `FileNotFoundError`** with zero silent fallback |
| **`ReshardRestore`** | Cluster resize, topological re-sharding, dynamic loading | Evaluates partition spec (`hash`, `modulo`, `fast_range`) via native AVX-512 SIMD slice filtering (`filter_span`) | Missing or mismatched partitioner algorithm raises exception directly |

### 4.2 Training vs. Inference Weight Isolation and Orthogonal Restore Matrix

Restoration operates across two strictly orthogonal dimensions ($2 \times 2$ Cartesian product architecture):
- **Payload Scope**:
  - **Training Restore (`load_training_checkpoint`)**: Loads embedding weights and optimizer momentum states (`load_optimizer_state=True`).
  - **Inference Restore (`load_inference_checkpoint`)**: Loads embedding weights only (`load_optimizer_state=False`), requiring the target store to disable optimizer state (`enable_optimizer=False`) to avoid memory waste. Requesting optimizer loading into an optimizer-disabled store raises an exception directly.
- **Topology Routing Strategy (`RestorePlan`)**:
  - Both training and inference **uniformly support** `RankRestore` and `ReshardRestore`.
  - Inference supports `RankRestore(..., engine=IngestionEngine.FAST_PAGE_CACHE)` for ultra-fast direct binary page image mapping cold-start, as well as `ReshardRestore` for heterogeneous inference sharding.

### 4.3 Native Arrow Key DTypes and Pushdown

The physical storage layer natively supports four fixed-width Arrow integer key types:
- `int32` (`pa.int32()`)
- `uint32` (`pa.uint32()`)
- `int64` (`pa.int64()`)
- `uint64` (`pa.uint64()`)

Key columns remain in their native binary integer layout without serialization overhead, enabling native pushdown queries in modern OLAP engines:

```sql
-- Query checkpoint slices directly in DuckDB
INSTALL lance;
LOAD lance;
SELECT key, values FROM '/checkpoints/user_embedding/_staging/0/entries.lance'
WHERE key = 42;
```

### 4.4 Bit-Identical Reference Vectors

To guarantee that offline C++ restore filtering produces identical results to online distributed runtimes (e.g., Fluxway/JAX) across languages and compilers, a canonical reference vector generator is provided:

```python
from sparse_embedding.checkpoint import generate_reference_vectors

sample_keys, expected_ranks = generate_reference_vectors(
    assignment_name="sparse_embedding.hash",
    key_dtype="uint64",
    partition_count=8,
)
```

---

## 5. Incremental Checkpoints and Streaming Compaction

### 5.1 Incremental Workflow

- **FULL Snapshot**: Captures all live keys and optimizer state (`CheckpointSaveMode.FULL`).
- **INCREMENTAL Snapshot**: Captures only dirty keys and deletion tombstones since the parent checkpoint (`CheckpointSaveMode.INCREMENTAL`, specifying `parent_handle`). Empty deltas append only a manifest record without creating payload datasets.
- **Restore Resolution**: The restore pipeline validates the parent chain and executes streaming multi-version reconciliation in native C++, eliminating deleted historical keys without inflating peak memory.

### 5.2 Streaming Compaction

As incremental checkpoints accumulate, multi-version delta chains can be compacted into a consolidated `FULL` version:

```python
# 1. Managed compaction: merges delta chain into a new self-contained FULL checkpoint
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

# 2. Standalone directory compaction: merge raw Lance fragment folders directly
from sparse_embedding.checkpoint import compact_fragments

compact_fragments(
    source_path="/checkpoints/user_embedding/_staging",
    target_path="/exports/user_embedding",
)
```

Compaction utilizes native streaming readers and bounded channels configured via `checkpoint_stream_policy.hpp`, bounding memory usage and leaving source versions immutable.

> For in-depth architecture, mutation journaling, and schema details, refer to [Incremental Checkpointing and Streaming Compaction](incremental_checkpoint.md).

---

## 6. Gravitino and Lance REST Integration

In enterprise Lakehouse deployments, checkpoint metadata is governed by Apache Gravitino while data files reside in object storage accessed through Lance REST operations.

```text
               ┌─────────────────────────────────────┐
               │    Gravitino Control Plane (API)    │
               │   (Metalake / Catalog / Namespace)  │
               └──────────────────┬──────────────────┘
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │       Lance REST Operation API      │
               │      (Table Handles / Metadata)     │
               └──────────────────┬──────────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      [S3 / Object Storage]             [Training Cluster / Workers]
       snapshot_manifest.lance           SparseCheckpoint Client
       entries.lance                     Direct I/O via S3 Credentials
```

### 6.1 Catalog Configuration

Example catalog creation payload in Gravitino configured for Lance:

```json
{
  "name": "sparse_checkpoint_catalog",
  "type": "RELATIONAL",
  "provider": "lakehouse-generic",
  "properties": {
    "location": "s3://my-bucket/training/checkpoints",
    "lance.schema-refresh-mode": "VERSION_CHECK",
    "lance.storage.endpoint": "http://s3.us-east-1.amazonaws.com",
    "lance.storage.region": "us-east-1",
    "lance.storage.access_key_id": "<ACCESS_KEY>",
    "lance.storage.secret_access_key": "<SECRET_KEY>",
    "lance.storage.allow_http": "true"
  }
}
```

> **Security Note**: Credentials should be injected via a runtime Secret Provider rather than hardcoded in public configuration files.

### 6.2 Python Client Integration

```python
from sparse_embedding.checkpoint import SparseCheckpoint
from sparse_embedding.checkpoint.gravitino_lance import (
    GravitinoLanceStorage,
    LanceRestCatalog,
    SdkConfig,
)

# 1. Initialize REST catalog and create namespace
config = SdkConfig.for_rest(
    lance_rest_api_base_uri="http://127.0.0.1:9101/lance/v1",
    timeout=30,
)
catalog = LanceRestCatalog(config)
catalog.create_namespace(["my_catalog", "checkpoint"], mode="create")

# 2. Initialize Gravitino checkpoint storage
storage = GravitinoLanceStorage(
    config,
    namespace=("my_catalog", "checkpoint"),
    component_id="user_embedding",
    location="s3://my-bucket/training/checkpoints/user_embedding",
)

checkpoint = SparseCheckpoint(storage)
```

---

## 7. Local Development: Docker Sidecar

For local development and unit testing, run the official Gravitino Lance REST sidecar container:

```bash
# Launch Lance REST container
docker run -d --name sparse-embedding-lance-rest \
  -p 9101:9101 \
  -e LANCE_REST_GRAVITINO_METALAKE_NAME=default \
  -e LANCE_REST_GRAVITINO_URI=http://<gravitino-host>:8090 \
  apache/gravitino-lance-rest:1.3.0
```

Once running, the operation API root `http://127.0.0.1:9101/lance/v1` becomes available.

### Conformance and Verification Commands

Execute the following test suites to verify integration contracts:

```bash
# Local Checkpoint core contracts
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py

# Gravitino and Lance REST integration contracts
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_gravitino_lance.py

# Selective optimizer loading and mixed precision contracts
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_lance_optimizer_loading.py
```
