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

## 1. 存储分层与双 Lance 2PC 架构

```text
SparseCheckpoint
    |
    +-- LocalLanceStorage
    |       +-- snapshot_manifest.lance (2PC 事务控制面账本与全局唯一可见性边界)
    |       +-- _staging/<attempt>/<fragment_id>/entries.lance (Worker 本地物理切片事实来源)
    |       +-- _staging/<attempt>/<fragment_id>/page_images.lance (可选物理页高速缓存，按需生成)
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

### 双 Lance 2PC 存储分层

落盘存储分为控制面与数据面两个物理 Lance 数据集：
1. **`snapshot_manifest.lance`**（控制面账本与可见性边界）：
   - 记录事务状态（`PREPARED` $\to$ `COMMITTED` / `ABORT`）、Lease 租约、Fence 标记与 `fragment_ids`。单次提交追加 1 行，隔离数据 I/O。
2. **`entries.lance`**（数据面物理切片）：
   - 各 Worker 并发写入专属路径 `_staging/<attempt>/<fragment_id>/entries.lance`，多节点写隔离且支持离线流式 Compaction。

## 2. 本地 Lance 集成

```python
from sparse_embedding.checkpoint import (
    ElasticRestore,
    PhysicalRestore,
    PhysicalRestoreStrategy,
    SparseCheckpoint,
    SparsePartitionAssignmentSpec,
    encode_restore_options,
)
from sparse_embedding.checkpoint.lance_backend import LocalLanceStorage

checkpoint = SparseCheckpoint(
    LocalLanceStorage.from_checkpoint_root("/checkpoints/user_embedding")
)

# 默认仅持久化 canonical entries (节约 50% 写入带宽与磁盘空间)
committed = checkpoint.save_training_checkpoint(store, "step-1000")

# 可选生成物理 page_images 缓存以获得最快恢复导入速度：
# committed = checkpoint.save_training_checkpoint(store, "step-1000", include_page_images=True)

# 1. 物理对等恢复 (PhysicalRestore)
checkpoint.load_training_checkpoint(
    restored_store,
    committed.handle,
    encode_restore_options(
        PhysicalRestore(
            participant,
            strategy=PhysicalRestoreStrategy.RECONSTRUCT_ENTRIES,
            key_dtype="uint64",
        )
    ),
)

# 2. 弹性拓扑恢复 (ElasticRestore + 规范化分片算法)
# 支持 modulo()、hash()、fast_range() 或动态注册算法
assignment_spec = SparsePartitionAssignmentSpec.hash()
checkpoint.load_training_checkpoint(
    restored_store,
    committed.handle,
    encode_restore_options(
        ElasticRestore(
            spec=assignment_spec,
            partition_count=8,
            partition_count
            key_dtype="uint64",
        )
    ),
)
```

训练与推理使用相同的 checkpoint 格式，但优化器状态策略必须显式区分。
`EmbeddingConfig(enable_optimizer=False)` 会构造不分配优化器状态页的权重-only
store。普通 `checkpoint.load_training_checkpoint(..., load_optimizer_state=True)` 保持训练恢复语义，
也可以设置为 `False` 执行权重-only 恢复；目标 store 禁用 optimizer 时请求加载状态会
直接抛异常。`load_inference_weights` 是推理入口，始终只加载 embedding，不加载优化器状态。

发布 sparse 推理引用时，在同一次保存事务中传入组件拥有的物理资产目录。该目录只包含
sparse_embedding 的 opaque 引用，不复制完整 Lance 表：

```python
asset = "/published/steps/1000/sparse/group"
checkpoint.save_training_checkpoint(store, "step-1000", weight_path=asset)
inference_storage = LocalLanceStorage.from_weight_path(asset)
# owner_restore_options 由 owner-specific router adapter 生成。
inference_storage.load_inference_weights(restored_store, owner_restore_options)
```

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
  - `ElasticRestore`: 支持 `SparsePartitionAssignmentSpec`（`modulo()`、`hash()`、`fast_range()` 及动态注册算法），底层 C++ 恢复引擎结合 `PartitionerRegistry` 与 AVX-512 SIMD 连续切片掩码过滤（`filter_span`）极速筛选目标逻辑分区键。若算法未注册或参数不匹配直接抛出异常，零隐式 fallback。
  - `PhysicalRestore(..., strategy=PhysicalRestoreStrategy.RECONSTRUCT_ENTRIES)`: 从规范 `entries`
    确定性重建物理页表结构，支持在经过 Checkpoint Compaction 合流后，按 `source_participant_index`
    精确还原目标 worker 的参数。
- **可选 `page_images` 缓存与零 Fallback**：
  仅当保存时显式设置 `include_page_images=True` 时落盘。
  - `PhysicalRestore(..., strategy=PhysicalRestoreStrategy.FAST_PAGE_CACHE)`：以有界批次导入物理页缓存，避免将完整缓存长期保留在 RAM 中。
  - **零 Fallback 规范**：若请求 `FAST_PAGE_CACHE` 但检查点未保存物理缓存，**严格抛出 `FileNotFoundError`**，
    绝不进行隐式回退。

### 跨仓黄金参考向量校验 (Reference Vectors)

为确保离线 C++ 恢复引擎的分区计算与在线运行时（如 Fluxway/JAX）的通信规划在跨语言、跨编译器条件下产生比特级完全一致（Bit-identical）的结果，系统提供官方黄金参考向量生成工具：

```python
from sparse_embedding.checkpoint import generate_reference_vectors

keys, ranks = generate_reference_vectors(
    assignment_name="sparse_embedding.hash",
    key_dtype="uint64",
    partition_count=8,
)
```

此工具针对 4 种原生 `key_dtype` 提供全覆盖测试样本（包括极值边界、负数等），跨框架接入时可通过单测严格校验。

### 独立 Compaction 与远端支持

训练结束后的导出阶段可仅凭落盘路径执行流式合并，无需打开 `LocalLanceStorage` 或读取完整数据：

```python
from sparse_embedding.checkpoint.lance_backend import compact_fragments

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
- **元数据一致性**: 多 Participant 合流后元数据记录包含完整的 `participant_indices` 映射列表。

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

#### 1. 键列多态类型 (Key DType SSOT)
系统支持 4 种 Arrow 原生固定宽度整数键类型：
- `int32` (`pa.int32()`)
- `uint32` (`pa.uint32()`)
- `int64` (`pa.int64()`)
- `uint64` (`pa.uint64()`)

键列在物理 Lance 数据集中直接以原生 Arrow 定长整型存储，零编解码转换开销，并天然支持 DuckDB 与 Lance 存储算子的谓词下推（Predicate Pushdown，例如 `SELECT key FROM 'entries.lance' WHERE key = -1`）。

#### 2. 载荷与优化器状态数值类型 (Values & Optimizer DTypes)
当前 Lance 检查点 schema 的物理值列采用规范的 `float32` 格式存储（支持原样读写 `float16`）。逻辑上的 `embedding_dtype` 和 `optimizer_dtype` 记录在分片元数据 (fragment metadata) 与 manifest 中，恢复流程会严格校验目标规范 (target specification)；其它逻辑浮点表示统一经由标准 `float32` 物理列进行规范化映射，以保证跨异构计算引擎与存储驱动的严格二进制定型一致性。


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
from sparse_embedding.checkpoint import SparseCheckpoint
from sparse_embedding.checkpoint.gravitino_lance import (
    GravitinoLanceStorage,
    SdkConfig,
)

storage = GravitinoLanceStorage(
    SdkConfig.for_rest(
        lance_rest_api_base_uri="http://127.0.0.1:9101/lance/v1",
    ),
    namespace=(catalog_name, "checkpoint"),
    component_id="embedding",
    location="s3://bucket/training/checkpoints/embedding",
)
checkpoint = SparseCheckpoint(storage)
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

## 6. 验证与运维规范

系统针对本地与分布式环境提供标准化的生命周期验证契约：
1. **控制面与 Catalog 生命周期**：支持完整的 namespace 维护、table 注册/描述/注销、以及基于租约 (Lease) 的并发锁机制与保留期清理 (Retention Reclaim)。
2. **数据面读写分离**：检查点数据流直接通过 Lance 原生引擎管理高性能二进制分片数据集，将元数据生命周期治理与数据 I/O 彻底解耦。
3. **版本化快照读取**：原生支持基于版本标记的确定性历史快照点读取 (Point-in-Time Snapshot Read)。

## 7. 部署与接入规范 (Deployment & Integration Specifications)

- **REST 端点声明**：`LANCE_REST_API_BASE_URI` 必须显式声明为部署方提供的确切 operation API 根地址（例如 `https://<host>/gateway/lance/v1`），客户端严格按声明路径访问，不执行隐式路径推测或版本重写。
- **物理类型标准**：数据面严格遵循 Arrow 原生定长整型键列与 `float32`/`float16` 物理值列，保障跨异构计算引擎与存储驱动的严格二进制定型一致性。
- **凭据与网络安全**：生产环境中的访问令牌、内部网络端点与敏感密钥统一通过标准环境变量（如 `GRAVITINO_AUTH_TOKEN`）注入，严禁持久化至公共配置或代码仓库。
