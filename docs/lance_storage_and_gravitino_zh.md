# Lance 存储与 Gravitino 检查点集成

本文档记录了 Sparse Embedding 检查点系统在集成 Lance、Gravitino 和 Lance REST 时的边界条件和规范。所有的端点 (endpoint)、metalake、catalog 以及凭据 (credentials) 均依赖于部署方配置，不在本文档中记录。

## 1. 存储分层

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

`LocalLanceStorage` 与 `GravitinoLanceStorage` 使用唯一的 Lance checkpoint 事务引擎。
schema 校验、staging、物化合并、manifest 发布、精确版本恢复、提交权、lease、retention、
reclaim 与完成记录只实现一次。storage adapter 只提供 dataset 定位、prefix 操作、持久小记录
与 commit lock。

Local storage 使用文件系统目录、原子文件与 `flock`；Gravitino storage 使用 Lance REST 返回
的 canonical table handle 与 S3 条件写。对象存储不支持条件写时直接失败，不使用本地或进程
内 fallback。

## 2. 本地 Lance 集成

```python
from sparse_embedding.checkpoint import LocalLanceStorage, SparseCheckpoint

checkpoint = SparseCheckpoint(
    LocalLanceStorage("/checkpoints/user_embedding")
)
locator = checkpoint.save(store, "step-1000")
checkpoint.restore(restored_store, locator)
```

固定的 Lance 表三元组 (triplet) 包含：

- `entries`: 物化的逻辑状态 `(dim, key)`。
- `page_images`: 物理页镜像及校验和。
- `snapshot_manifest`: 检查点唯一的可见性边界。

manifest 记录两个精确的 Lance 版本；恢复操作 (restore) 不会读取最新版本。写入失败时，合并版本会被回滚，暂存分片 (staging fragment) 将通过中止或协调 (abort/reconciliation) 机制被清理。

分布式框架通过 SPI 提供框架无关的提交权限 (commit authority)：

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

`LocalLanceStorage` 在同一个 Lance 标签权限 (tag authority) 中维护该作用域 (scope) 的单调代次高水位线 (monotonic generation high-watermark) 以及尝试权限 (attempt authority)。任何低于该代次的延迟准备 (delayed prepare)、分片写入 (fragment write) 和提交 (commit) 都会失败；中止 (abort) 仅移除尝试权限，不会降低高水位线。`authority_id` 应当由调用方提供不可逆的身份摘要，不得传递可重放的协调器令牌 (coordinator token)。该协议与所有编排框架类型解耦。

分布式集成可以在提交时提供持久化的 `reference_id`。仍被引用的定位器 (locator) 会将其对应的 entries、page-images 与 manifest 的精确版本固定 (pin)。`reserve_reclaim` 会在同一个存储锁内检查活跃租约 (active lease) 并写入预留 (reservation)；`publish_reclaim` 将其转为待定 (pending) 状态，随后 `reclaim_checkpoint` 仅释放被选 locator 的引用并执行 GC。成功回收会写入由 sparse 自己管理的持久 completion record，使 repair 重试保持幂等，同时不会把缺失 pending reservation 误判为成功。租约获取使用幂等的请求 ID (request ID)，而更新和释放 (renew/release) 使用代次 CAS (generation CAS)。

### 物理数据类型

当前 Lance 检查点 schema 的物理列为 `float32`；逻辑上的 `embedding_dtype` 和 `optimizer_dtype` 记录在分片元数据 (fragment metadata) 与 manifest 中，恢复流程会严格校验目标规范 (target specification)。

经实测，当前的 Lance/PyArrow 支持原样读写 Arrow 的 `float16` 类型，但 PyArrow 目前暂不支持 `bfloat16` 类型。Sparse Embedding 检查点尚未宣称支持动态物理 dtype 或 bfloat16 无损存储；现阶段，非 `float32` 的逻辑 dtype 将通过规范的 float32 物理列进行转换。


## 3. Gravitino 控制平面

Gravitino control-plane API 根地址:

```text
https://<gravitino-host>/api
```

典型的控制 API:

```text
GET /api/metalakes                         -> 200
GET /api/metalakes/<metalake_name>                -> 200
GET /api/metalakes/<metalake_name>/catalogs       -> 200
```

catalog 由部署方配置：

```text
metalake: <metalake_name>
catalog:  <catalog_name>
provider: lakehouse-generic
```

catalog 创建请求必须配置 Lance 存储属性 (storage properties)：

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

`in-use` 是 Gravitino 服务端管理的保留属性，不能复制到 CREATE 请求中。凭据仅通过运行时密钥提供者 (secret provider) 传递，绝不写入检查点 manifest、分片或日志中。

## 4. Lance REST

`LANCE_REST_API_BASE_URI` 是可以直接调用 Lance REST operation 的完整根地址：

```text
https://<gravitino-host>/lance/v1
```

官方 Gravitino Docker 部署通常在 `/lance/v1` 暴露 operation API，但具体值由部署方决定。SDK 不会自动补充 `/lance`、`/v1` 或 metalake 路径；它仅在完整的 API 根地址后追加 `namespace/...` 与 `table/...` 资源。

服务端配置:

```text
gravitino.lance-rest.gravitino-metalake = <metalake_name>
gravitino.lance-rest.gravitino-uri = https://<gravitino-host>
```

Python 客户端:

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

`LanceRestCatalog` 只负责 metadata：namespace 管理以及 table 的
register、describe、exists、list 和 drop。它返回不可变的
`TableHandle(table_id, location, storage_options)`，不传输 checkpoint 数据，也不修改
Lance latest version。

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

adapter 内部注册 `<component_id>_entries`、`<component_id>_page_images` 和
`<component_id>_snapshot_manifest`。这些名称只是物理 metadata，不是新的 checkpoint identity。

## 5. Docker Sidecar

本地 sidecar 使用官方镜像，并连接到部署方的 Gravitino 控制平面：

```shell
docker run -d --name sparse-embedding-lance-rest \
  -p 9101:9101 \
  -e LANCE_REST_GRAVITINO_METALAKE_NAME=<metalake_name> \
  -e LANCE_REST_GRAVITINO_URI=https://<gravitino-host> \
  apache/gravitino-lance-rest:1.3.0
```

容器启动后，官方默认的 operation API 根地址 `http://127.0.0.1:9101/lance/v1` 应当可以访问。建议进行以下验证：

- catalog 级别的 namespace 创建和列表 (list) 操作应当成功。
- 表的 register/describe/drop 需符合目标 Gravitino 版本的实际 API 契约。
- Lance 原生写入、返回的 storage options、对象条件写与精确版本读取必须通过 conformance suite。

停止 sidecar:

```shell
docker rm -f sparse-embedding-lance-rest
```

## 6. 验证状态

本地 `apache/gravitino-lance-rest:1.3.0` sidecar 已验证 namespace、table register/describe/drop、
Gravitino 返回 storage options 后的 Lance 原生写入、精确旧版本读取、lease 阻塞、retention reclaim
和对象清理。该镜像的 Arrow IPC table create 路由仍因服务端 Arrow allocator 打包问题返回
HTTP 500，但 checkpoint 路径不调用该路由，而是先注册 table metadata，再由 Lance 原生客户端
创建 dataset。

远端部署的 Gravitino 控制 API 和 catalog 生命周期可以通过已配置的 `GRAVITINO_API_BASE_URI` 进行验证。官方 Gravitino 的 Lance REST operation API 通常位于 `/lance/v1/...`，但客户端使用部署方提供的完整 `LANCE_REST_API_BASE_URI`。如果访问该 API root 时服务端报告 metalake 不存在，部署方必须配置 `gravitino.lance-rest.gravitino-metalake=<metalake_name>`，随后再执行完整的 namespace、table、读写和 exact-version 一致性测试。在完成这些检查之前，文档和客户端都不应宣称 Gravitino 支撑的检查点数据平面可用。

`LANCE_REST_API_BASE_URI` 必须是完整的 operation API 根地址。例如，若部署方公开了 `https://<host>/gateway/lance`，则直接配置该值；若公开了 `https://<host>/gateway/lance/v1`，则配置后者。SDK 不会去猜测版本或代理前缀。

## 7. 当前限制

以下能力尚未宣称完成：

- Lance REST base path、metalake 配置和反向代理重写 (reverse proxy rewrite) 之间的严格一致性。
- Gravitino Lance REST Arrow `create` 路由的一致性；checkpoint 实现不依赖该路由。
- `bfloat16` 或动态物理 dtype 的无损存储。

部署环境的失败日志、凭据和内部端点不得提交至开源仓库；请保存在部署方自己的受控日志系统中。
