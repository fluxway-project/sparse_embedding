# Lance 存储与 Gravitino 检查点集成

本文档记录了 Sparse Embedding 检查点系统在集成 Lance、Gravitino 和 Lance REST 时的边界条件和规范。所有的端点 (endpoint)、metalake、catalog 以及凭据 (credentials) 均依赖于部署方配置，不在本文档中记录。

## 使用前提与验证

本地 checkpoint 需要 Python package 和可写的 checkpoint 目录。Gravitino storage 还需要
可访问的 control plane、Lance REST operation API、支持条件写的对象存储，以及由部署方
secret provider 提供的凭据。可用以下命令验证本地契约：

```bash
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_gravitino_lance.py
```

## 1. 存储分层

```text
SparseCheckpoint
    |
    +-- LocalLanceStorage
    |       +-- snapshot_manifest (唯一可见性边界)
    |       +-- _staging/<fragment_id>/entries (Canonical SSOT 规范事实来源)
    |       +-- _staging/<fragment_id>/page_images (可选物理页高速缓存，按需生成)
    |
    +-- GravitinoLanceStorage
            +-- Gravitino control-plane API base URI
            +-- Lance REST operation API base URI
            +-- Lance datasets in S3/local storage
```

`LocalLanceStorage` 与 `GravitinoLanceStorage` 使用唯一的 Lance checkpoint 事务引擎。
schema 校验、Fragment staging、manifest 发布、Fragment 恢复、提交权、lease、retention、
reclaim 与完成记录只实现一次。storage adapter 只提供 dataset 定位、prefix 操作、持久小记录
与 commit lock。

Local storage 使用文件系统目录、原子文件与 `flock`；Gravitino storage 使用 Lance REST 返回
的 canonical table handle 与 S3 条件写。对象存储不支持条件写时直接失败，不使用本地或进程
内 fallback。

## 2. 本地 Lance 集成

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

# 默认仅持久化 canonical entries (节约 50% 写入带宽与磁盘空间)
locator = checkpoint.save(store, "step-1000")

# 可选生成物理 page_images 缓存以获得最快恢复导入速度：
# locator = checkpoint.save(store, "step-1000", include_page_images=True)

# 恢复：显式选择策略 (零隐式 Fallback)
checkpoint.restore(
    restored_store,
    locator,
    mode=PhysicalRestore(participant, strategy=PhysicalRestoreStrategy.RECONSTRUCT_ENTRIES),
)
```

训练与推理使用相同的 checkpoint 格式，但优化器状态策略必须显式区分。
`EmbeddingConfig(enable_optimizer=False)` 会构造不分配优化器状态页的权重-only
store。普通 `checkpoint.restore(..., load_optimizer_state=True)` 保持训练恢复语义，
也可以设置为 `False` 执行权重-only 恢复；目标 store 禁用 optimizer 时请求加载状态会
直接抛异常。`restore_weight_snapshot` 是推理入口，始终只加载 embedding，不加载优化器状态。

选择性加载契约由 `sparse_embedding/tests/test_lance_optimizer_loading.py` 独立覆盖；修改该
链路时应与 checkpoint/config/mixed-precision focused tests 一起运行。

检查点不再依赖共享的物化表；每个 participant 将快照写入独立 Fragment，
数据写入采用基于 `BoundedChannel` 的真流式单事务管道，使用 $O(N_{live})$ 的紧凑页索引元数据，
并将单页与 writer 缓冲限制在有界范围内，
单次生成版本 1（随后元数据发布追加版本 2），彻底消除 Lance Append 多事务与版本爆炸。
Manifest 记录其 `fragment_ids`、digest、`commit_state`、`compacted_version` 与协议版本。
固定的可见性元数据表为：

- `snapshot_manifest`: 检查点唯一的可见性边界。

### 恢复策略与单事实来源 (SSOT)

- **Canonical `entries` 作为唯一事实来源**：
  默认仅写入 `entries` 表，承载所有键值与优化器状态。
  - `ElasticRestore`: 弹性恢复，按新集群的 `partition_fn` 重新分片，与历史物理布局完全解耦。
  - `PhysicalRestore(..., strategy=PhysicalRestoreStrategy.RECONSTRUCT_ENTRIES)`: 从规范 `entries`
    确定性重建物理页表结构，支持在经过 Checkpoint Compaction 合流后，按 `source_participant_index`
    精确还原目标 worker 的参数。
- **可选 `page_images` 缓存与零 Fallback**：
  仅当保存时显式设置 `include_page_images=True` 时落盘。
  - `PhysicalRestore(..., strategy=PhysicalRestoreStrategy.FAST_PAGE_CACHE)`：以有界批次导入物理页缓存，避免将完整缓存长期保留在 RAM 中。
  - **零 Fallback 规范**：若请求 `FAST_PAGE_CACHE` 但检查点未保存物理缓存，**严格抛出 `FileNotFoundError`**，
    绝不进行隐式回退。

### 独立 Compaction 与远端支持

训练结束后的导出阶段可仅凭落盘路径执行流式合并，无需打开 `LocalLanceStorage` 或读取完整数据：

```python
from sparse_embedding.checkpoint import compact_fragments

# 本地路径压缩
compact_fragments("/checkpoints/user_embedding/_staging", "/exports/user_embedding")

# 远端对象存储压缩 (支持 s3:// 与 s3a:// 协议与 storage_options)
compact_fragments(
    "s3://my-bucket/checkpoints/user_embedding/_staging",
    "s3://my-bucket/exports/user_embedding",
    storage_options={"endpoint": "http://s3.us-east-1.amazonaws.com", "region": "us-east-1"},
)
```

该操作使用 Native Lance scanner/streaming writer；scanner batch、`BoundedChannel` 容量和 page-import 批次统一由 `checkpoint_stream_policy.hpp` 中的 C++ policy 定义，避免重复硬编码，驻留内存受这些固定容量约束，不将全表加载到 Python 内存。
在发布机制上严格遵循原子性与资源清理规范：
- **本地文件系统 (Local)**: 直接利用 POSIX `rename(2)` 原子系统调用发布目录，**零额外 JSON 文件产生**；
- **远端对象存储 (S3 / S3A)**: 由于对象存储无目录原子重命名，为杜绝读者中途读取到部分写出的碎片，采用不可变 Staged 前缀写入 + 单对象强一致性 Pointer (`.manifest_pointer.json`) 进行原子线性化发布；
- **全生命周期清理保证 (Zero Orphans)**: 额外 JSON 仅在远端对象存储中作为必要原子载体使用。成功覆盖更新、显式 `delete_prefix`、递归级联父目录删除与回滚路径都会严格清理指针及 staged 数据；清理失败会直接抛出并保留已发布 root 以便重试，绝不静默报告成功；
- **严格零 Fallback 目录删除**: `delete_prefix(..., missing_ok=False)` 默认严格抛出 `FileNotFoundError`，禁止吞异常返回成功；仅在显式传入 `missing_ok=True` 时允许幂等安全退出；
- **协议版本一致性**: 协议版本严格保持为 `_PROTOCOL_VERSION = 1`。多 Participant 合流后元数据记录包含完整的 `participant_indices` 映射列表。

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
