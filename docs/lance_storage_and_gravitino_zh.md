# Lance 存储与 Gravitino 检查点集成指南

[English](lance_storage_and_gravitino.md) | [架构与 API 契约](storage_architecture_zh.md) | [增量检查点与流式 Compaction](incremental_checkpoint_zh.md) | [性能与可靠性报告](performance_and_reliability_report_zh.md) | [动态学习率](dynamic_learning_rate_zh.md)

本文档面向使用 Sparse Embedding 存储引擎的工程师与架构师，系统性阐述基于 Lance 列式格式与 Apache Gravitino Lakehouse 元数据目录的检查点（Checkpoint）架构设计、事务协议与集成规范。

---

## 1. 概述与核心价值

在大规模推荐系统与稀疏嵌入场景中，传统检查点方案通常面临三大瓶颈：单体文件庞大无法部分读取、分布式 Worker 写冲突严重，以及缺乏统一的 Lakehouse 元数据生命周期治理。

Sparse Embedding 采用 **Lance 列式存储** 与 **Apache Gravitino** 协同方案，提供以下核心能力：

- **数据与元数据解耦的双 Lance 架构**：控制面记录事务账本，数据面并发流式落盘，消除多节点写锁竞争与版本爆炸。
- **计算引擎零拷贝下推**：键列直接采用 Arrow 原生定长整型存储，原生支持 DuckDB、DataFusion 及 Lance 计算算子的谓词下推（Predicate Pushdown）。
- **极速恢复与弹性拓扑**：支持从规范数据精确恢复物理页表；支持结合 SIMD 掩码过滤在集群扩缩容时直接执行逻辑分片重组（Reshard Restore）。
- **Lakehouse 标准元数据治理**：无缝对接 Gravitino 与 Lance REST 服务，实现多租户 Namespace、租约锁机制与统一凭据管理。

---

## 2. 存储分层与双 Lance 2PC 架构

检查点持久化分为**控制面元数据**与**数据面物理切片**两个物理 Lance 数据集，由两阶段提交（2PC）协议保证全局原子性：

```text
SparseCheckpoint
    │
    ├── LocalLanceStorage / GravitinoLanceStorage (共享统一事务引擎)
    │
    ├── [控制面账本] snapshot_manifest.lance
    │       └── 记录全局唯一可见性边界、事务状态 (PREPARED → COMMITTED / ABORT)、
    │           租约 (Lease)、Fencing 标记与 Fragment 映射列表。单次提交追加单行。
    │
    └── [数据面物理切片] _staging/<attempt>/<fragment_id>/
            ├── entries.lance      (规范键值与优化器状态切片，唯一真实事实来源)
            └── page_images.lance  (可选的物理页缓存切片，用于极速导入)
```

### 事务提交流程与写入隔离

1. **Worker 并发暂存 (Staging)**：各 Worker 将切片流式写入独立的暂存目录（`_staging/<attempt>/<fragment_id>/entries.lance`），节点之间完全隔离，零锁竞争。
2. **两阶段原子提交 (2PC Commit)**：
   - **Prepare 阶段**：校验所有 Participant 的 Fragment 完整性、数据 Schema 与提交权限（Generation 高水位线）。
   - **Commit 阶段**：向 `snapshot_manifest.lance` 追加一行提交记录，切换全局可见性。未完成提交或异常中止的事务绝不会被 Reader 观察到。
3. **原子发布机制**：
   - **本地文件系统 (Local)**：直接利用 POSIX `rename(2)` 系统调用执行目录原子重命名，无冗余中间文件。
   - **对象存储 (S3 / S3A)**：针对对象存储缺乏目录原子重命名的特性，采用不可变前缀写入配合单对象强一致性指针文件（`.manifest_pointer.json`）完成线性化原子发布。
   - **无孤儿资源清理 (Zero Orphans)**：覆盖更新、回滚与删除均执行级联资源清理；操作失败时严格抛出异常并保留现场供重试排查，杜绝静默吞异常。

---

## 3. 快速上手：训练快照保存与加载

以下示例演示如何在本地文件系统上初始化检查点、保存训练状态，以及执行自适应拓扑恢复、物理恢复与弹性分片恢复：

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

# 1. 初始化存储与 Checkpoint 客户端
storage = LocalLanceStorage.from_checkpoint_root("/checkpoints/user_embedding")
checkpoint = SparseCheckpoint(storage)

# 2. 保存训练检查点 (默认保存规范 entries，节约 50% 写入带宽与磁盘空间)
committed = checkpoint.save_training_checkpoint(
    store,
    "user_embedding",
    mode=CheckpointSaveMode.FULL,
)

# 可选：显式生成物理页缓存以获取最快恢复速度
# committed = checkpoint.save_training_checkpoint(
#     store, "user_embedding", include_page_images=True
# )

# 3. 场景 A：推荐 - 拓扑自适应恢复 (RestorePlan.from_topology)
# 自动比对检查点保存卡数与当前卡数：卡数一致时走 RankRestore，卡数变更时自动切换为 ReshardRestore
restored_store = se.EmbeddingStore(config)
plan = RestorePlan.from_topology(
    saved_participant_count=1,
    participant=CheckpointParticipant(0, 1),
    spec=PartitionSpec.hash(seed=42),
    key_dtype="uint64",
)
checkpoint.load_training_checkpoint(restored_store, committed.handle, plan)

# 4. 场景 B：显式物理对等恢复 (RankRestore)
# 适用于卡数与拓扑完全相同的场景，从规范 entries 精确还原物理页表
checkpoint.load_training_checkpoint(
    restored_store,
    committed.handle,
    RestorePlan.from_participant(
        CheckpointParticipant(0, 1),
        engine=IngestionEngine.RECONSTRUCT_ENTRIES,
        key_dtype="uint64",
    ),
)

# 5. 场景 C：显式弹性拓扑重组恢复 (ReshardRestore)
# 适用于训练卡数扩缩容场景，通过分片算法极速筛选当前节点所属的键
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

## 4. 恢复策略与数据类型契约

### 4.1 恢复策略对比

| 恢复策略 | 适用场景 | 底层机制 | 异常与防御策略 |
|---|---|---|---|
| **`RestorePlan.from_topology`** | **拓扑自适应路由（推荐）** | 自动校验已保存卡数与当前卡数：匹配则走 `RankRestore`，扩缩容则走 `ReshardRestore` | 严格校验 `saved_participant_count > 0`，零静默降级 |
| **`RankRestore (RECONSTRUCT_ENTRIES)`** | 拓扑不变恢复、Compaction 合流后恢复 | 读取规范 `entries.lance`，在 C++ 层确定性重建物理页布局与定位槽 | 唯一事实来源（SSOT），完全确定性 |
| **`RankRestore (FAST_PAGE_CACHE)`** | 极速冷启动（需保存时开启 `include_page_images=True`） | 按有界批次直接映射 `page_images.lance` 物理页镜像，绕过逐行构建 | 若检查点未持久化物理页缓存，**严格抛出 `FileNotFoundError`**，拒绝隐式降级 |
| **`ReshardRestore`** | 训练集群扩缩容、拓扑重组、弹性分片加载 | 结合注册的分片算法（`hash`、`modulo`、`fast_range`），利用 AVX-512 SIMD 连续切片掩码（`filter_span`）极速过滤 | 分片算法未注册或参数不匹配直接抛出异常，无隐式回退 |

### 4.2 训练与推理权重隔离及正交恢复矩阵

恢复操作由两个完全正交的维度构成（$2 \times 2$ 笛卡尔积架构）：
- **状态载荷范围（Scope / Payload）**：
  - **训练恢复 (`load_training_checkpoint`)**：完整加载嵌入表权重与优化器动量状态（`load_optimizer_state=True`）。
  - **推理恢复 (`load_inference_checkpoint`)**：仅拉取 Embedding 权重（`load_optimizer_state=False`），要求目标 Store 禁用优化器（`enable_optimizer=False`），避免浪费内存。若对未启用优化器的 Store 请求加载优化器状态，系统直接抛出异常。
- **拓扑路由策略（RestorePlan）**：
  - 训练与推理**均统一支持** `RankRestore` 与 `ReshardRestore`。
  - 推理场景支持 `RankRestore(..., engine=IngestionEngine.FAST_PAGE_CACHE)` 实现物理页二进制镜像直接映射的极致冷启动，也支持 `ReshardRestore` 实现异构分片加载。

### 4.3 Arrow 原生键类型与算子下推

系统在物理存储层原生支持 4 种 Arrow 定长整型键：
- `int32` (`pa.int32()`)
- `uint32` (`pa.uint32()`)
- `int64` (`pa.int64()`)
- `uint64` (`pa.uint64()`)

在 Lance 数据集中，键列保持原始定长整型二进制形态，避免任何编解码转换开销，并原生支持 OLAP 引擎下推查询：

```sql
-- 使用 DuckDB 直接查询切片数据
INSTALL lance;
LOAD lance;
SELECT key, values FROM '/checkpoints/user_embedding/_staging/0/entries.lance'
WHERE key = 42;
```

### 4.4 跨语言黄金参考向量验证 (Reference Vectors)

为确保离线 C++ 恢复引擎的分区计算与在线运行时（如 Fluxway/JAX）在跨语言、跨编译器条件下产生比特级完全一致（Bit-identical）的结果，系统提供跨框架校验工具：

```python
from sparse_embedding.checkpoint import generate_reference_vectors

sample_keys, expected_ranks = generate_reference_vectors(
    assignment_name="sparse_embedding.hash",
    key_dtype="uint64",
    partition_count=8,
)
```

---

## 5. 增量快照与流式压实 (Compaction)

### 5.1 增量检查点工作机制

- **全量保存 (FULL)**：保存当前表中所有存活键值与状态（`CheckpointSaveMode.FULL`）。
- **增量保存 (INCREMENTAL)**：仅记录上一版本以来的增量变更（Dirty Keys）与删除标记（Tombstone），需指定 `parent_handle`。空增量仅追加控制面 Manifest，不创建冗余数据表。
- **恢复链处理**：恢复时先校验 Parent 链的连续性与完整性，在 C++ 恢复引擎内流式执行状态合并与删除记录消除，被删除的历史键不会占用目标内存容量。

### 5.2 流式压实 (Compaction)

随着增量保存次数增加，历史版本切片增多。系统支持通过流式 Compaction 将增量链条合并为自包含的全量版本：

```python
# 1. 托管式压实：将增量链合并为全新的自包含 FULL 检查点
compacted_handle = storage.compact_checkpoint(
    committed_inc.handle,
    include_page_images=True,  # 可选：同步物化物理页镜像缓存，加速冷启动
)

# 后续恢复可直接挂载合流后的新句柄
compacted_store = se.EmbeddingStore(config)
checkpoint.load_training_checkpoint(
    compacted_store,
    compacted_handle,
    RestorePlan.from_participant(
        CheckpointParticipant(0, 1),
        key_dtype="uint64",
    ),
)

# 2. 独立目录压实：直接对底层 Lance Fragment 目录执行合流
from sparse_embedding.checkpoint import compact_fragments

compact_fragments(
    source_path="/checkpoints/user_embedding/_staging",
    target_path="/exports/user_embedding",
)
```

流式合并由 C++ `checkpoint_stream_policy.hpp` 控制扫描批次与有界通道（`BoundedChannel`），严格控制峰值驻留内存，压实过程不修改输入版本。

> 关于增量检查点的底层变动日志（Mutation Journaling）、物理 Schema 与一致性切片深度原理，请参见 [增量检查点与流式 Compaction](incremental_checkpoint_zh.md)。

---

## 6. Gravitino 与 Lance REST 集成

在企业级分布式湖仓架构中，检查点元数据接入 Apache Gravitino 控制面，数据文件存放于对象存储中，通过 Lance REST 服务提供标准目录管理。

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

### 6.1 Catalog 创建与配置

在 Gravitino 中注册支持 Lance 的 Catalog 属性配置示例：

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

> **安全提示**：敏感凭据应统一通过运行时 Secret Provider 注入，严禁硬编码在客户端代码或公共配置文件中。

### 6.2 Python 客户端接入

```python
from sparse_embedding.checkpoint import SparseCheckpoint
from sparse_embedding.checkpoint.gravitino_lance import (
    GravitinoLanceStorage,
    LanceRestCatalog,
    SdkConfig,
)

# 1. 初始化 REST Catalog 并创建命名空间
config = SdkConfig.for_rest(
    lance_rest_api_base_uri="http://127.0.0.1:9101/lance/v1",
    timeout=30,
)
catalog = LanceRestCatalog(config)
catalog.create_namespace(["my_catalog", "checkpoint"], mode="create")

# 2. 初始化 Gravitino Checkpoint Storage
storage = GravitinoLanceStorage(
    config,
    namespace=("my_catalog", "checkpoint"),
    component_id="user_embedding",
    location="s3://my-bucket/training/checkpoints/user_embedding",
)

checkpoint = SparseCheckpoint(storage)
```

---

## 7. 本地调试：Docker Sidecar 部署指南

在本地开发与测试环境中，可通过官方 Gravitino Lance REST 镜像快速拉起 Sidecar 容器：

```bash
# 启动 Lance REST 容器
docker run -d --name sparse-embedding-lance-rest \
  -p 9101:9101 \
  -e LANCE_REST_GRAVITINO_METALAKE_NAME=default \
  -e LANCE_REST_GRAVITINO_URI=http://<gravitino-host>:8090 \
  apache/gravitino-lance-rest:1.3.0
```

容器启动后，Operation API 根地址 `http://127.0.0.1:9101/lance/v1` 即可正常访问。

### 验收与单测验证命令

在修改或验证集成逻辑时，运行以下单元测试套件：

```bash
# 本地 Checkpoint 核心契约验证
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py

# Gravitino 与 Lance REST 集成验证
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_gravitino_lance.py

# 优化器选择性加载与混合精度契约验证
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_lance_optimizer_loading.py
```
