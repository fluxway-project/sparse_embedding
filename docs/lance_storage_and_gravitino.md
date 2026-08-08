# Lance Storage and Gravitino Checkpoint Integration

This document details the boundary conditions and specifications for the Sparse Embedding checkpoint system when integrating with Lance, Gravitino, and Lance REST. All endpoints, metalakes, catalogs, and credentials depend on the deployment configuration and are omitted from this documentation.

## 1. Storage Hierarchies

```text
SparseCheckpoint
    |
    +-- LocalLanceStorage
    |       +-- entries
    |       +-- page_images
    |       +-- snapshot_manifest
    |
    +-- GravitinoLanceStorage
            +-- Gravitino control-plane API base URI
            +-- Lance REST operation API base URI
            +-- Lance datasets in S3/local storage
```

`LocalLanceStorage` and `GravitinoLanceStorage` use one shared Lance checkpoint
transaction engine. Schema validation, staging, materialized merge, manifest
publication, exact-version restore, commit authority, leases, retention, reclaim,
and completion records are implemented once. The storage adapters only supply
dataset locations, prefix operations, durable records, and commit locking.

Local storage uses filesystem directories, atomic files, and `flock`. Gravitino
storage uses canonical table handles returned by Lance REST and S3 conditional
writes. Missing conditional-write support is an initialization/runtime error; no
local or process-memory fallback exists.

## 2. Local Lance Integration

```python
from sparse_embedding.checkpoint import LocalLanceStorage, SparseCheckpoint

checkpoint = SparseCheckpoint(
    LocalLanceStorage("/checkpoints/user_embedding")
)
locator = checkpoint.save(store, "step-1000")
checkpoint.restore(restored_store, locator)
```

The fixed Lance table triplet consists of:

- `entries`: The materialized logical state `(dim, key)`.
- `page_images`: Physical page images and their checksums.
- `snapshot_manifest`: The unique visibility boundary for checkpoints.

The manifest records two exact Lance versions; the restoration process does not read the latest version. During write failures, merged versions are rolled back, and staging fragments are cleaned up via abort/reconciliation.

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
