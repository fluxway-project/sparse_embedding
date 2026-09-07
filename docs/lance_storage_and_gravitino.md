# Lance Storage and Gravitino Checkpoint Integration

This document details the boundary conditions and specifications for the Sparse Embedding checkpoint system when integrating with Lance, Gravitino, and Lance REST. All endpoints, metalakes, catalogs, and credentials depend on the deployment configuration and are omitted from this documentation.

## Requirements and validation

Local checkpoint use requires the Python package and a writable checkpoint
directory. Gravitino-backed storage additionally requires a reachable control
plane, Lance REST operation API, object storage with conditional writes, and
credentials supplied by the deployment secret provider. Validate the local
contract with:

```bash
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_gravitino_lance.py
```

## 1. Storage Hierarchies

```text
SparseCheckpoint
    |
    +-- LocalLanceStorage
    |       +-- snapshot_manifest (sole visibility boundary)
    |       +-- _staging/<fragment_id>/entries (Canonical SSOT dataset)
    |       +-- _staging/<fragment_id>/page_images (Optional page image cache, generated on-demand)
    |
    +-- GravitinoLanceStorage
            +-- Gravitino control-plane API base URI
            +-- Lance REST operation API base URI
            +-- Lance datasets in S3/local storage
```

`LocalLanceStorage` and `GravitinoLanceStorage` use one shared Lance checkpoint
transaction engine. Schema validation, Fragment staging, manifest publication,
Fragment restore, commit authority, leases, retention, reclaim,
and completion records are implemented once. The storage adapters only supply
dataset locations, prefix operations, durable records, and commit locking.

Local storage uses filesystem directories, atomic files, and `flock`. Gravitino
storage uses canonical table handles returned by Lance REST and S3 conditional
writes. Missing conditional-write support is an initialization/runtime error; no
local or process-memory fallback exists.

## 2. Local Lance Integration

```python
from sparse_embedding.checkpoint import (
    LocalLanceStorage,
    PhysicalRestore,
    PhysicalRestoreStrategy,
    SparseCheckpoint,
)

checkpoint = SparseCheckpoint(
    LocalLanceStorage("/checkpoints/user_embedding")
)

# Default: persist only canonical entries (saves 50% write bandwidth and storage space)
locator = checkpoint.save(store, "step-1000")

# Optional: persist physical page_images cache for fastest restore speed in fixed topologies:
# locator = checkpoint.save(store, "step-1000", include_page_images=True)

# Restore with explicit strategy (zero implicit fallback)
checkpoint.restore(
    restored_store,
    locator,
    mode=PhysicalRestore(participant, strategy=PhysicalRestoreStrategy.RECONSTRUCT_ENTRIES),
)
```

Training and inference stores use the same checkpoint format but have explicit
optimizer-state policies. `EmbeddingConfig(enable_optimizer=False)` constructs
a weight-only store without allocating optimizer-state pages. Ordinary
`checkpoint.restore(..., load_optimizer_state=True)` preserves the training
round trip and can be set to `False` for weight-only restore. A disabled target
rejects optimizer-state loading. `restore_weight_snapshot` is the inference
entry point and always loads embeddings without optimizer state.

The selective-loading contract is covered by
`sparse_embedding/tests/test_lance_optimizer_loading.py`; run it together with
the checkpoint/config/mixed-precision focused tests when changing this path.

Checkpoints no longer use shared materialized tables. Each participant writes a
snapshot into an independent Fragment via a true streaming pipeline based on
`BoundedChannel`, maintaining $O(N_{live})$ compact page-index metadata plus
bounded single-page and writer buffers, and writing in a single commit (data
version 1, followed by metadata publish version 2).
This completely eliminates Lance multi-transaction append loops and manifest explosion.
The Manifest records `fragment_ids`, digests, `commit_state`, `compacted_version`,
and protocol version. The sole visibility metadata table is:

- `snapshot_manifest`: The unique visibility boundary for checkpoints.

### Restore Strategies and Single Source of Truth (SSOT)

- **Canonical `entries` as Sole Source of Truth**:
  Default writes persist only the `entries` dataset containing all keys, embedding vectors,
  and optimizer states.
  - `ElasticRestore`: Elastic resharding with new `partition_fn`, completely decoupled
    from historical participant/page topologies.
  - `PhysicalRestore(..., strategy=PhysicalRestoreStrategy.RECONSTRUCT_ENTRIES)`: Rebuilds
    physical pages deterministically from canonical `entries`. Even after multi-participant
    checkpoint compaction, it precisely filters by `source_participant_index` to restore only
    the targeted worker's parameters.
- **Optional `page_images` Cache and Zero Fallback**:
  Only persisted when explicitly enabled with `include_page_images=True`.
  - `PhysicalRestore(..., strategy=PhysicalRestoreStrategy.FAST_PAGE_CACHE)`: bounded-batch
    import of physical page images (avoids retaining the complete cache in RAM).
  - **Zero Fallback Contract**: If `FAST_PAGE_CACHE` is requested but the checkpoint does not
    contain physical page images, it **strictly raises `FileNotFoundError`** with zero implicit fallback.

### Standalone Compaction and Remote Storage Support

Export jobs can compact persisted Fragments by path alone without instantiating storage:

```python
from sparse_embedding.checkpoint import compact_fragments

# Local directory compaction
compact_fragments("/checkpoints/user_embedding/_staging", "/exports/user_embedding")

# Remote object storage compaction (supports s3:// and s3a:// URIs with storage_options)
compact_fragments(
    "s3://my-bucket/checkpoints/user_embedding/_staging",
    "s3://my-bucket/exports/user_embedding",
    storage_options={"endpoint": "http://s3.us-east-1.amazonaws.com", "region": "us-east-1"},
)
```

Native Lance scanning and streaming writing use the single C++ policy in `checkpoint_stream_policy.hpp` for scanner batch rows, channel capacity, and page-import batch pages. This avoids duplicating batch literals; resident compaction memory is bounded by those fixed capacities.
Publication strictly enforces atomicity and zero-orphan cleanup rules:
- **Local File System (Local)**: Uses POSIX atomic `rename(2)` to publish directories with **zero auxiliary JSON files**.
- **Remote Object Storage (S3 / S3A)**: Since object storage lacks atomic directory rename, a single, strongly-consistent pointer object (`.manifest_pointer.json`) is written over an immutable staged prefix to provide linearized atomic publication without partial visibility.
- **Full Lifecycle Cleanup (Zero Orphans)**: Auxiliary JSON files are used exclusively on remote object stores as an atomic pointer. Successful overwrite, explicit `delete_prefix`, recursive parent deletion, and rollback paths strictly clean pointer objects and staged directories; cleanup failures are propagated and preserve the published root for retry instead of reporting success.
- **Strict Zero Fallback Deletion**: `delete_prefix(..., missing_ok=False)` strictly raises `FileNotFoundError` when the target prefix does not exist; idempotent exit is permitted only when `missing_ok=True` is explicitly passed.
- **Protocol Version Stability**: Protocol version remains `_PROTOCOL_VERSION = 1`. Multi-participant compaction records the complete `participant_indices` array in the compacted fragment metadata.

The distributed framework provides framework-agnostic commit authority via the SPI:

```python
CheckpointSaveIdentity(
    checkpoint_name=...,
    save_attempt=...,
    commit_authority=CheckpointCommitAuthority(
        scope=...,
        generation=...,
        authority_id=...,
    ),
)
```

`LocalLanceStorage` maintains a monotonic generation high-watermark and an attempt authority within the same Lance tag authority for a given scope. Delayed prepare, fragment write, and commit operations with lower generations will be rejected. Aborts only remove the attempt authority without lowering the high-watermark. The `authority_id` should provide an irreversible identity digest from the caller; replayable coordinator tokens must not be passed. This protocol is agnostic to every orchestration framework.

Distributed integrations can provide a durable `reference_id` upon commit. Locators that remain referenced will pin their corresponding entries, page-images, and exact manifest versions. `reserve_reclaim` checks active leases and writes reservations under the same storage lock; `publish_reclaim` converts these to pending. Subsequently, `reclaim_checkpoint` only releases references of selected locators and executes GC. Successful reclaim writes a sparse-owned durable completion record, making repair retries idempotent without treating a missing pending reservation as success. Lease acquisition uses idempotent request IDs, while renew/release leverages generation CAS.

### Physical DType

The physical column in the current Lance checkpoint schema is `float32`; logical `embedding_dtype` and `optimizer_dtype` are recorded in fragment metadata and manifests, allowing the restoration process to strictly validate target specifications.

Current Lance/PyArrow implementations support reading and writing Arrow `float16` natively; however, PyArrow does not yet support the `bfloat16` type. Sparse Embedding checkpoints do not claim support for dynamic physical dtypes or lossless `bfloat16` storage at this time. All non-`float32` logical dtypes will be cast through a canonical `float32` physical column.


## 3. Gravitino Control Plane

Gravitino control-plane API root URL:

```text
https://<gravitino-host>/api
```

Typical Control APIs:

```text
GET /api/metalakes                         -> 200
GET /api/metalakes/<metalake_name>                -> 200
GET /api/metalakes/<metalake_name>/catalogs       -> 200
```

The catalog is provisioned by the deployment configuration:

```text
metalake: <metalake_name>
catalog:  <catalog_name>
provider: lakehouse-generic
```

Catalog creation requests must configure Lance storage properties:

```json
{
  "name": "<unique-name>",
  "type": "RELATIONAL",
  "provider": "lakehouse-generic",
  "properties": {
    "location": "s3://<bucket>/<prefix>",
    "lance.schema-refresh-mode": "VERSION_CHECK",
    "lance.storage.endpoint": "http://<s3-endpoint>",
    "lance.storage.region": "us-east-1",
    "lance.storage.access_key_id": "<access-key>",
    "lance.storage.secret_access_key": "<secret-key>",
    "lance.storage.allow_http": "true"
  }
}
```

The `in-use` property is server-managed by Gravitino and must not be included in CREATE requests. Credentials are only passed through the runtime secret provider and are not persisted in checkpoint manifests, fragments, or logs.

## 4. Lance REST

`LANCE_REST_API_BASE_URI` specifies the full root address for directly invoking Lance REST operations:

```text
https://<gravitino-host>/lance/v1
```

Official Gravitino Docker deployments generally expose the operation API under `/lance/v1`, though this is subject to the deployment setup. The SDK does not append `/lance`, `/v1`, or the metalake path automatically. It merely appends `namespace/...` and `table/...` resources to the provided full API base URI.

Server configuration:

```text
gravitino.lance-rest.gravitino-metalake = <metalake_name>
gravitino.lance-rest.gravitino-uri = https://<gravitino-host>
```

Python client:

```python
from sparse_embedding.checkpoint.gravitino_lance import LanceRestCatalog
from sparse_embedding.checkpoint.gravitino_lance.config import SdkConfig

catalog = LanceRestCatalog(
    SdkConfig.for_rest(
        lance_rest_api_base_uri="http://127.0.0.1:9101/lance/v1",
        timeout=30,
    )
)
catalog.create_namespace([catalog_name, "checkpoint"], mode="create")
```

`LanceRestCatalog` is a metadata adapter. Its public operations are namespace
management plus table register, describe, exists, list, and drop. It returns an
immutable `TableHandle(table_id, location, storage_options)`. It does not transport
checkpoint data and does not publish or restore a Lance latest version.

```python
from sparse_embedding.checkpoint import GravitinoLanceStorage, SparseCheckpoint
from sparse_embedding.checkpoint.gravitino_lance import SdkConfig

storage = GravitinoLanceStorage(
    SdkConfig.for_rest(
        lance_rest_api_base_uri="http://127.0.0.1:9101/lance/v1",
    ),
    namespace=(catalog_name, "checkpoint"),
    component_id="embedding",
    location="s3://bucket/training/checkpoints/embedding",
)
checkpoint = SparseCheckpoint(storage)
locator = checkpoint.save(store, "step-1000")
checkpoint.restore(restored_store, locator)
```

The adapter internally registers `<component_id>_entries`,
`<component_id>_page_images`, and `<component_id>_snapshot_manifest`. Those names
are physical metadata, not additional checkpoint identities. The locator remains
the only restore identity exposed to algorithm code.

## 5. Docker Sidecar

The local sidecar utilizes the official image and connects to the deployment's Gravitino control plane:

```shell
docker run -d --name sparse-embedding-lance-rest \
  -p 9101:9101 \
  -e LANCE_REST_GRAVITINO_METALAKE_NAME=<metalake_name> \
  -e LANCE_REST_GRAVITINO_URI=https://<gravitino-host> \
  apache/gravitino-lance-rest:1.3.0
```

Once the container starts, the default official operation API root `http://127.0.0.1:9101/lance/v1` should be accessible. The following validations are recommended:

- Creating and listing catalog-level namespaces should succeed.
- Table register/describe/drop operations must adhere to the actual API contract of the target Gravitino version.
- Native Lance writes, returned storage options, conditional object writes, and exact version reads must pass the conformance suite.

To stop the sidecar:

```shell
docker rm -f sparse-embedding-lance-rest
```

## 6. Validation Status

The local `apache/gravitino-lance-rest:1.3.0` sidecar has validated namespace
operations, table register/describe/drop, native Lance writes using returned
storage options, exact old-version reads, lease blocking, retention reclaim, and
object cleanup. Its Arrow IPC table-create route still returns HTTP 500 because
the server image lacks an Arrow allocation implementation. The checkpoint path
does not call that route: it registers metadata first and creates datasets through
the native Lance client.

The Gravitino control API and catalog lifecycles of remote deployments can be verified via the configured `GRAVITINO_API_BASE_URI`. The official Gravitino Lance REST operation API is typically hosted at `/lance/v1/...`, but clients rely on the fully-qualified `LANCE_REST_API_BASE_URI` provided by the deployment. If this API root is invoked and reports a missing metalake, the deployment must configure `gravitino.lance-rest.gravitino-metalake=<metalake_name>` before proceeding with full namespace, table, read/write, and exact-version conformance tests. Until these checks pass, neither the documentation nor the client should claim that the Gravitino-backed checkpoint data plane is available.

`LANCE_REST_API_BASE_URI` must be the complete operation API root. For example, if the deployment exposes `https://<host>/gateway/lance`, this exact URI should be configured. If it exposes `https://<host>/gateway/lance/v1`, then configure the latter. The SDK does not infer versions or proxy prefixes.

## 7. Current Limitations

The following capabilities are not yet declared as complete:

- Strict consistency between Lance REST base paths, metalake configurations, and reverse proxy rewrites.
- Gravitino Lance REST Arrow `create` route conformance; checkpoint storage does
  not depend on that route.
- Lossless storage for `bfloat16` or dynamic physical dtypes.

Failure logs, credentials, and internal endpoints from the deployment environment must not be committed to open-source repositories; they should be securely stored in the deployment's own logging systems.
