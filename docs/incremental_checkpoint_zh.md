# 增量检查点与流式 Compaction

[English](incremental_checkpoint.md) | [Lance 存储与 Gravitino 集成指南](lance_storage_and_gravitino_zh.md) | [架构与 API 契约](storage_architecture_zh.md) | [性能与可靠性报告](performance_and_reliability_report_zh.md) | [动态学习率](dynamic_learning_rate_zh.md)

本文档面向工程架构师与开发者，详细介绍 Sparse Embedding 的**增量检查点（Incremental Checkpoint）**与**流式合流压缩（Streaming Compaction）**子系统的设计架构、实现机制及使用规范。

---

## 1. 概述与核心能力

在大规模推荐系统与稀疏嵌入表训练场景中，嵌入表通常包含数亿到数十亿参数。周期性生成全量快照会带来巨大的 I/O 吞吐开销、引起训练停顿，并在持久化层造成极大的数据冗余。

Sparse Embedding 提供了生产级原生**增量检查点**与**流式 Compaction** 能力：

- **无阻塞一致性切片（Consistent Cut）**：在 Epoch 内存安全保护下秒级完成脏数据与删除墓碑的快照捕获，内存切片与磁盘 I/O 彻底解耦，训练过程零等待、无停顿。
- **分片原生内存变动日志（Mutation Journaling）**：Shard 本地管理变动条目与单调递增序列号，Forward 前向热路径无任何互斥锁开销与原子写争用。
- **Delta 增量物理模式与墓碑追踪**：底层 Lance 列存切片原生持久化 `mutation_sequence`、`row_state`（`"active"` vs. `"tombstone"`）以及可选的逐特征淘汰评分 `score`（LFU/LRU）。
- **流式多版本合流解析**：恢复引擎在 C++ 原生层高效遍历 Delta 增量链（`FULL` → `INCREMENTAL` → ...），确定性合并最新版本并自动剔除墓碑数据，峰值内存受严格控制。
- **离线/在线流式 Compaction**：利用原生有界通道（`BoundedChannel`）将冗长增量链合并为全新的自包含 `FULL` 检查点，消除历史冗余并支持物理页缓存重建。

---

## 2. 系统架构与数据流

```text
┌────────────────────────────────────────────────────────────────────────┐
│                              在线训练引擎                              │
│   Forward / Backward ──► 分片本地变更 ──► CheckpointMutationJournal    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ capture_checkpoint() (无锁快速捕获)
┌───────────────────────────────────▼────────────────────────────────────┐
│                       一致性切片与分片临时落盘                         │
│   脏数据条目 + 删除墓碑 ──► _staging/<attempt>/<participant_id>/entries.lance
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 2PC 两阶段原子提交 (Manifest 追加)
┌───────────────────────────────────▼────────────────────────────────────┐
│                    Lance Lakehouse 元数据事务总线                      │
│   snapshot_manifest.lance (记录父版本引用、水位线、检查点模式)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────────────┐       ┌────────────────────────────────┐
│       原生高效合流恢复        │       │       流式 Compaction 合流     │
│  在 C++ 原生流式扫描中遍历    │       │  将增量链条合并为自包含的全量  │
│  Delta 增量链，消除墓碑与冗余 │       │  快照: storage.compact_checkpoint()
│  RestorePlan.from_topology()  │       │                                │
└───────────────────────────────┘       └────────────────────────────────┘
```

### 2.1 分片内存变动日志 (Mutation Journaling)

每个 Hash 分片维护独立的轻量级 `CheckpointMutationJournal`：
- **脏键追踪**：当特征权重或优化器状态在 `assign`、`accumulate_gradients` 或 `optimizer_step` 中被修改时，其定位槽与递增序号即时记录到当前活跃 Journal 中。
- **删除墓碑记录**：显式删除（`erase`、`erase_dimension`）或基于分数的淘汰清理会生成对应的删除墓碑（Tombstone），确保跨版本删除语义严格保真。
- **只读热路径零开销**：`forward_lookup` 和 `lookup_or_insert` 命中路径不经过 Journal 互斥锁，也不执行任何原子写计数。

### 2.2 两阶段一致性切片协议

1. **快照切片（`capture_checkpoint`）**：在极短的停顿内封存当前 Journal 代际，读取当前变动水位线，并为后续到来的训练流量初始化全新一代 Journal。
2. **异步落盘**：后台工作线程将脏数据与删除墓碑按 Arrow/Lance 格式异步写入独立的暂存目录（`_staging/<attempt>/<participant_id>/entries.lance`）。
3. **提交与确认（`acknowledge_store_mutations`）**：2PC 提交成功并在 `snapshot_manifest.lance` 登记后，Store 推进已提交变动水位线，并安全回收已确认的 Journal 记录。

### 2.3 Lance 物理列存模式

数据面物理切片（`entries.lance`）遵循 `kEntryColumns` 规范定义：

| 列名 | Arrow 类型 | 说明 |
|---|---|---|
| `dim` | `uint32` | 嵌入维度标识 |
| `key` | `int32` / `uint32` / `int64` / `uint64` | 原生定宽整型键，支持算子零拷贝下推 |
| `embedding_values` | `binary` | 嵌入向量二进制字节 |
| `embedding_dtype` | `utf8` | 向量数据类型（`float16`、`float32`、`bfloat16`） |
| `optimizer_state_values` | `binary` | 优化器动量向量字节（若启用） |
| `optimizer_state_dtype` | `utf8` | 优化器状态数据类型 |
| `optimizer_state_stride` | `uint32` | 单行优化器状态字节跨度 |
| `step` | `uint32` | 优化器全局步数 |
| `score` | `uint64` | 特征淘汰评分（LFU/LRU） |
| `sequence` | `uint64` | 内部 SeqLock 槽版本序号 |
| `mutation_sequence` | `uint64` | 分片单调递增变动序号 |
| `row_state` | `utf8` | 记录状态：`"active"`（存活）或 `"tombstone"`（删除墓碑） |
| `source_participant_index` | `uint32` | 写入节点 Rank 序号 |

---

## 3. 快速上手与 API 规范

### 3.1 保存全量与增量检查点

```python
import numpy as np
import sparse_embedding as se
from sparse_embedding.checkpoint import (
    CheckpointSaveMode,
    LocalLanceStorage,
    SparseCheckpoint,
)

# 1. 初始化底层存储与门面接口
storage = LocalLanceStorage.from_checkpoint_root("/checkpoints/user_model")
checkpoint = SparseCheckpoint(storage)

# 2. 保存基础全量检查点 (FULL)
committed_full = checkpoint.save_training_checkpoint(
    store,
    "user_model",
    mode=CheckpointSaveMode.FULL,
)
print("已提交全量版本:", committed_full.version_id.decode())

# 3. 训练继续进行... 增删改操作由底层日志自动捕获
# 保存增量检查点 (INCREMENTAL)，需显式传入 parent_handle
committed_inc = checkpoint.save_training_checkpoint(
    store,
    "user_model",
    mode=CheckpointSaveMode.INCREMENTAL,
    parent_handle=committed_full.handle,
)
print("已提交增量版本:", committed_inc.version_id.decode())
```

> **注意**：增量检查点必须与父检查点保持相同的 Lineage 名称（如 `"user_model"`），并且必须指定有效的 `parent_handle`。未提供父句柄时系统将严格抛出 `ValueError`，杜绝悬空版本。

### 3.2 从增量链条恢复状态

恢复流水线会自动验证父链完整性，并在 C++ 原生层流式合流多版本记录：

```python
from sparse_embedding.checkpoint import (
    CheckpointParticipant,
    IngestionEngine,
    PartitionSpec,
    RankRestore,
    ReshardRestore,
    RestorePlan,
)

# 场景 A：推荐 - 拓扑自适应恢复 (RestorePlan.from_topology)
# 自动比对检查点保存卡数与当前卡数：卡数一致时走极致性能的 RankRestore，
# 扩缩容变更时自动切换为带 SIMD 分区掩码过滤的 ReshardRestore。
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

# 场景 B：显式物理对等恢复 (拓扑完全对等)
rank_plan = RestorePlan.from_participant(
    CheckpointParticipant(0, 1),
    engine=IngestionEngine.RECONSTRUCT_ENTRIES,
    key_dtype="uint64",
)
checkpoint.load_training_checkpoint(restore_store, committed_inc.handle, rank_plan)

# 场景 C：显式弹性拓扑重组恢复 (集群扩缩容分片过滤)
reshard_plan = RestorePlan.from_partition(
    PartitionSpec.hash(seed=42),
    partition_count=8,
    partition_id=0,
    key_dtype="uint64",
)
elastic_store = se.EmbeddingStore(config)
checkpoint.load_training_checkpoint(elastic_store, committed_inc.handle, reshard_plan)
```

### 3.3 流式 Compaction 合流压缩

随着增量链不断增长，可通过 Compaction 将链条合并为自包含的全量版本：

```python
# 1. 高阶托管 Compaction：将增量链合并为全新的 FULL 检查点
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

# 2. 独立目录流式压实：直接压实底层 Lance 片段目录 (支持 POSIX 与 s3:// / s3a://)
from sparse_embedding.checkpoint import compact_fragments

compact_fragments(
    source_path="/checkpoints/user_model/_staging",
    target_path="/exports/user_model",
)
```

### 3.4 推理场景从增量检查点加载

推理实例配置 `enable_optimizer=False` 实现显存最小化：

```python
infer_config = se.EmbeddingConfig(
    dims=[128],
    key_dtype="uint64",
    value_dtype="float32",
    enable_optimizer=False,  # 仅加载权重，不分配优化器动量状态
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

## 4. 可靠性与一致性保证

- **严格 Lineage 隔离保护**：增量检查点强制校验变动序号的单调递增性与 Lineage 名称一致性，跨血缘父引用直接触发 `CheckpointConflictError`。
- **父版本生命周期锚定（Parent Pinning）**：活跃的父数据集由内部引用标签加锁保护。存储垃圾回收任务绝不会清理仍被下游增量链依赖的数据文件。
- **空变动开销消除（Empty Delta Optimization）**：若训练间隔内无任何权重更新，系统仅向 `snapshot_manifest.lance` 追加紧凑的元数据记录，跳过空数据集的无谓创建。
- **零孤儿文件级联清理**：异常中断或中止的检查点保存任务将自动清理其暂存目录前缀，主异常信息严格保真传递，拒绝静默失败。
- **多版本确定性决胜规则**：跨多版本的数据冲突严格按照最高 `(version, mutation_sequence)` 胜出，保证跨环境、跨语言还原出的数据绝对位一致（Bit-Identical）。

---

## 5. 验证命令

运行以下测试套件以验证增量检查点与合流功能：

```bash
# 运行增量检查点核心契约测试
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py -k "incremental"

# 运行多版本链条 Compaction 与物理页物化测试
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py -k "compact"

# 验证 4 种整型主键矩阵
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_key_checkpoint_matrix.py
```
