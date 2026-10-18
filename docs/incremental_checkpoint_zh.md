# 增量检查点与流式 Compaction

[English](incremental_checkpoint.md) | [Lance 存储与 Gravitino 集成指南](lance_storage_and_gravitino_zh.md) | [架构与 API 契约](storage_architecture_zh.md) | [性能与可靠性报告](performance_and_reliability_report_zh.md) | [动态学习率](dynamic_learning_rate_zh.md)

本文档面向工程架构师与开发者，详细介绍 Sparse Embedding 的**增量检查点（Incremental Checkpoint）**与**流式合流压缩（Streaming Compaction）**子系统的设计架构、实现机制及使用规范。

---

## 1. 概述与核心能力

在大规模推荐系统与稀疏嵌入表训练场景中，嵌入表通常包含数亿到数十亿参数。周期性生成全量快照会带来巨大的 I/O 吞吐开销、引起训练停顿，并在持久化层造成极大的数据冗余。

Sparse Embedding 提供原生**增量检查点**与**流式 Compaction** 能力：

- **一致性切片（Consistent Cut）**：捕获阶段暂时阻止并等待竞争中的 store 调用，使用 Epoch 保护内存。native writer 通道有界，慢 I/O 可以通过背压延长捕获时间。
- **分片原生内存变动日志（Mutation Journaling）**：Shard 本地管理变动条目与单调递增序列号，Forward 前向热路径无任何互斥锁开销与原子写争用。
- **Delta 增量物理模式与墓碑追踪**：底层 Lance 列存保存 `mutation_sequence`、`row_state`（`"active"` 与 `"deleted"`）以及可选的淘汰评分 `score`（LFU/LRU）。
- **原生多版本合流解析**：有界外部排序校验历史 identity 并选出增量链（`FULL` → `INCREMENTAL` → ...）的存活记录。每个 Lance 源的 payload 扫描一次，请求的数据写入临时文件，随后按源顺序回放；只有存活 key 分配目标 slot。
- **离线/在线流式 Compaction**：与 entries restore 共用合流和 payload 解码，再通过有界 writer 通道输出自包含 `FULL`。临时磁盘随历史数据增长，排序与 batch 的 RAM 上限由 `CheckpointStreamPolicy` 控制。

- **Page-Image 恢复边界**：page-image 恢复按有界 page batch 扫描，并在 store 内持有一次空目标导入生命周期。只有 native 证明历史 identity 的 source owner 不变且目标 page geometry 匹配时才可使用。目标 assignment 移动历史 identity 后，entries 恢复的下一次保存必须为 `FULL`；分区数量相同也适用（详见 2.4 节）。
- **生命周期与资源约束**：全部页面和校验成功后才发布 generation；导入尝试失败后，目标在重新开放访问前进入 terminal 状态。训练恢复要求符合目标 optimizer 的实际状态布局；只加载权重是显式 scope。最终目标表、稀疏物理页对应的 optimizer 容量及 Lance 运行时缓存不属于有界临时缓冲。恢复包含控制面和 I/O 同步，不能称为全过程无锁。

---

## 2. 系统架构与数据流

```text
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                     在线训练引擎                                     │
│   Forward / Backward ──► 分片本地变更 ──► CheckpointMutationJournal                  │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │ capture_checkpoint() (无锁快速捕获)
┌──────────────────────────────────────────▼───────────────────────────────────────────┐
│                               一致性切片与分片临时落盘                               │
│   脏数据条目 + 删除墓碑 ──► _staging/<attempt>/<participant_id>/entries.lance        │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │ 2PC 两阶段原子提交 (Manifest 追加)
┌──────────────────────────────────────────▼───────────────────────────────────────────┐
│                            Lance Lakehouse 元数据事务总线                            │
│   snapshot_manifest.lance (记录父版本引用、水位线、检查点模式)                       │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┴────────────────────────┐
                    ▼                                               ▼
┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
│           原生高效合流恢复           │        │         流式 Compaction 合流         │
│  在 C++ 原生流式扫描中遍历           │        │  将增量链条合并为自包含的全量        │
│  Delta 增量链，消除墓碑与冗余        │        │  快照: storage.compact_checkpoint()  │
│  RestorePlan.from_partition()        │        │                                      │
└──────────────────────────────────────┘        └──────────────────────────────────────┘
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

### 2.4 物理 Source 分片存储与跨拓扑检查点不变量

增量检查点在数学上定义为基线快照之上的增量变动：$f(\text{Base}) + \Delta$。当集群拓扑变更（如 Worker 数量扩缩容或分片算法更新）导致 Key 所有权发生转移时，拓扑重构后的首个检查点必须保存为 `FULL` 全量检查点，而非 `INCREMENTAL` 增量检查点。

#### 1. 物理架构：独立的 Participant 分片存储
在分布式训练写路径上，为实现零互斥锁争用与无阻塞捕获，分布式 Worker 不会并发写入同一个物理 Lance 表文件。每个训练 Worker 独立维护本地 Shard 的变动日志，并在切片时将变动条目写出至独占的物理子目录：
```
_staging/<attempt>/<participant_id>/entries.lance
```
全局清单 `snapshot_manifest.lance` 则以轻量级元数据记录这一组离散的物理 Lance 片段（Fragment）。因此，Lance 底层天然是以 Participant（即 source 分片）为独立单元组织的。

#### 2. 跨拓扑不变量与状态物化
增量检查点依赖于确定性的基线状态与各 Participant 日志内单调递增的变动序号。当拓扑重组导致键归属发生迁移时，必须遵循以下状态规约：

```text
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ 拓扑 1 (2 Workers 训练):                                                             │
│  Worker 0 ──► 片段 0 [Key 100 @ seq=10] ─────┐                                       │
│  Worker 1 ──► 片段 1 [Key 200 @ seq=12]      │                                       │
│                                              │ 拓扑变更 (2 -> 4 Workers)             │
│                                              ▼ 恢复路由: Key 100 迁移给 Worker 3     │
│ 拓扑 2 (4 Workers 训练) ── 拓扑重组后的检查点不变量:                                 │
│                                                                                      │
│ [禁止跨拓扑 INCREMENTAL]                     [系统强制规约: FULL 基线物化]           │
│   • Worker 3 仅记录本步新修改 Key              • Worker 3 立即物化自包含全量         │
│   • 未修改的 Key 100 遗失在新增量中            • Key 100 完整收录至片段 3            │
│   • 下次恢复需全量遍历所有历史旧 Worker 碎片   • 彻底切断旧拓扑长链依赖              │
│   • 读放大爆炸 + 墓碑复活 + 旧数据无法 GC      • 允许安全 GC 回收旧拓扑历史文件      │
│   (❌ 破坏一致性与有界性能)                    (✅ 后续检查点恢复极速 INCREMENTAL)   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

拓扑重构后首个检查点强制为 `FULL` 的技术考量：
1. **分片基线连续性（Base Slice Continuity）**：增量片段仅记录当前步变更的条目。若在新节点接管键后直接保存增量，未在该步被修改的历史键将遗失在增量之外，后续恢复将无法仅依赖新节点的父链还原完整状态，退化为跨节点递归回溯。
2. **有界读取开销（Bounded Read Amplification）**：若允许多次跨拓扑增量链级联，节点恢复时需扫描历史上所有拓扑结构下的所有碎片，读取复杂度从 $O(N_{\text{checkpoints}})$ 退化为 $O(N_{\text{participants}} \times N_{\text{checkpoints}})$。
3. **序列号域隔离与墓碑消除（Sequence Domain Isolation & Tombstone Elimination）**：变动序号仅在各 Worker 本地单调递增，无全局物理时钟。跨节点的删除墓碑若直接多源归并，极易因缺乏绝对偏序导致已删除条目被误判或复活。
4. **血缘截断与存储垃圾回收（Lineage Truncation & Safe GC）**：若增量跨拓扑延续，历史拓扑的物理文件将被下游增量依赖永久锚定无法释放。强制执行一次 `FULL` 本质上是**状态物化与血缘截断边界**，使新节点建立完整自包含基线，旧拓扑历史文件得以被安全回收。完成该全量保存后，后续训练步骤可继续执行轻量级 `INCREMENTAL` 保存。

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
    IngestionEngine,
    PartitionSpec,
    RestorePlan,
)

# 场景 A：目标拓扑不变，计划直接描述当前分区。
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

# 场景 B：同一种计划类型也处理目标拓扑变化。
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
    RestorePlan.from_partition(
        PartitionSpec.modulo(),
        partition_count=1,
        partition_id=0,
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
    RestorePlan.from_partition(
        PartitionSpec.modulo(),
        partition_count=1,
        partition_id=0,
        key_dtype="uint64",
    ),
)
```

---

## 4. 可靠性与一致性保证

- **严格 Lineage 隔离与 Head 解析保护**：增量检查点强制校验变动序号的单调递增性与 Lineage 名称一致性，跨血缘父引用直接触发 `CheckpointConflictError`。未被确认的检查点（持有活跃 `seintent` 意图标签或 `sereclaiming` 排除标记）严格排除在 Lineage Head 解析之外（`resolve_lineage_head`）。
- **原子 Abort 与孤儿回收协议**：中止未决保存或回收孤儿 committed 检查点严格遵循故障安全的生命周期时序：
  1. 首先持久化 `sereclaiming.<locator>` 排除标记，确保即便后续流程异常，该版本也不会被误作为 Lineage Head；
  2. 物理删除碎片文件（Fragment Roots）；
  3. 清理父链引用（Parent Chain References）、引用标签及 Page 写入意图；
  4. 从 `snapshot_manifest.lance` 删除元数据记录并使缓存失效；
  5. 删除 `seintent` 意图标签、Pending Outcome 与 Committed 标签；
  6. 最后删除 `sereclaiming` 排除标记。
  该不变量确保在进程崩溃、I/O 中断或清理失败等任何中间故障下，未确认或已孤儿的检查点绝不会被错误暴露为血缘头指针。
- **父版本生命周期锚定（Parent Pinning）**：活跃的父数据集由内部引用标签加锁保护。存储垃圾回收任务绝不会清理仍被下游增量链依赖的数据文件。
- **空变动开销消除（Empty Delta Optimization）**：若训练间隔内无任何权重更新，系统仅向 `snapshot_manifest.lance` 追加紧凑的元数据记录，跳过空数据集的无谓创建。
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
