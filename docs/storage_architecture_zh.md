# 架构与 API 契约

[English](storage_architecture.md) | [性能与可靠性报告](performance_and_reliability_report_zh.md) | [Lance 存储与 Gravitino 集成指南](lance_storage_and_gravitino_zh.md)

本文档面向系统工程师与架构师，系统性阐述 Sparse Embedding 存储引擎的总体架构、并发与内存模型、分层存储拓扑、淘汰调度以及分布式接口契约。

---

## 1. 系统概述与设计哲学

Sparse Embedding 存储引擎专为现代深度推荐系统、搜索排序与大模型稀疏检索场景设计，用于高效托管高达数亿乃至数十亿容量的高维稀疏参数表。

针对推荐场景中高并发离散访存、参数倾斜严重以及超大显存/内存容量诉求，引擎秉承以下核心设计哲学：

- **读路径完全无锁 (Lock-Free Hot Paths)**：前向查询（Forward / Lookup）完全绕过全局锁与分片互斥锁，定位槽采用乐观 SeqLock 读取，实现零加锁、零原子计数写开销。
- **单周期分支消除寻址 (Branchless Addressing)**：物理页表采用平坦紧凑的指针目录，寻址通过位移与掩码位运算单步完成，消除虚函数与多级指针跳转开销。
- **软硬件协同自适应 (Hardware Auto-Tuning)**：自动探测系统物理内存与 NUMA 拓扑，启动时自适应微基准校准最优物理页大小。
- **分层存储与透明溢写 (Tiered Storage & Spill)**：支持纯内存模式（MemoryOnly）与自适应磁盘溢写（AutoStorage），内存超额时通过虚拟内存重映射（`MAP_FIXED`）实现匿名页到文件后备页的平滑换出。
- **防御式设计与 Fail-Fast 契约**：拒绝任何可能导致数据损坏或静默不一致的隐式回退；参数非法、容量耗尽或系统异常时严格抛出异常。

---

## 2. 总体分层架构

Sparse Embedding 存储引擎自顶向下分为五层：

```text
┌─────────────────────────────────────────────────────────────┐
│                       Python API 层                         │
│       EmbeddingStore / EmbeddingConfig / Partitioner        │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    并发哈希路由与定位层                     │
│  ParlayHash Shards ──► MapBundle (RCU) ──► SegmentedLocator │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    物理分页存储管理层                       │
│  StablePageDirectory ──► StoragePage (RAM / Spill SSD)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    缓存淘汰与自适应局部性                   │
│  Eviction Thread ──► Boltzmann Policy ──► Coaccess Sketch   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    分布式分片与持久化层                     │
│  PartitionerRegistry (SIMD)  │  Dual-Lance 2PC Checkpoint   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 并发模型与无锁查询

### 3.1 哈希路由与不可变 MapBundle (RCU 发布)

查询的核心解析路径为：`Key Hash -> Shard ID -> slot_id -> SegmentedLocatorTable -> arena_index`。

- **读写解耦**：每个 Shard 独立管理一个不可变的 `MapBundle`。Reader 仅通过 acquire-load 获取当前的 `MapBundle` 指针，不获取任何共享互斥锁。
- **写时复制与原子发布**：结构性修改（插入、删除、扩容、槽位轮转）在对应 Shard 的独立 Writer Mutex 下克隆当前 Bundle 并完成修改，最后通过单个 atomic exchange 完成发布。各 Shard 之间互不阻塞，单 Shard 扩容不会影响其他 Shard 的并发读写。
- **无二次查找**：冷热分层与物理页不参与路由决策，单次查询只需一次哈希目录查找与一次定位槽读取，杜绝二级哈希二次探测。

### 3.2 定位槽与纯乐观 SeqLock

定位槽管理由 `SegmentedLocatorTable` 负责，由 32 KiB 的固定段组成，每个槽位包含 8 字节对齐的 `{sequence, arena_index}`：

- **乐观读取与原地更新**：Reader 与权重就地更新操作（如 `optimizer_step`、`assign`）通过纯乐观 `try_read` 读取 sequence 与 `arena_index`，经过编译器屏障后直接就地访问物理内存，最后校验 sequence 一致性。热路径上完全无锁且无原子计数修改。
- **搬迁感知与 ABA 防护**：当后台 Compaction 迁移物理页修改 `arena_index` 时，通过 CAS 将 sequence 置为奇数，完成数据拷贝后再原子发布新的偶数 sequence 与新地址。并发 Reader 检测到版本变动会自动重试。当 sequence 接近回绕时，目录会自动将该键轮转至新槽位并退休旧槽位。

### 3.3 两级平坦物理页目录 (`StablePageDirectory`)

物理页目录采用两级紧凑指针目录 `StablePageDirectory<T>`（统一管理数据页、状态页、属主信息与锁），目录常驻 CPU L1 Cache：

$$\text{page\_id} = \text{index} \gg \text{entries\_shift}, \quad \text{slot\_offset} = \text{index} \ \& \ \text{slot\_mask}$$

寻址直接通过单周期无分支位运算完成，彻底消除堆指针多级跳跃。

### 3.4 四级软件流水线预取

稀疏嵌入检索与训练更新属于跳跃式非连续访存，硬件 Stream Prefetcher 无法有效预测。引擎在前向与反向热路径中构建了四级软件流水线预取体系：

1. **第 1 级（输入特征预取）**：预取未来样本的 `keys[i + dist]`、`row_lengths` 与 `offsets`。
2. **第 2 级（输出缓冲区预取）**：对目标输出内存 `out_flat_values + offset` 发出写意图预取。
3. **第 3 级（哈希目录桶预取）**：计算未来 key 的哈希并提前将目录桶缓存行载入 L1/L2 缓存。
4. **第 4 级（定位槽与优化器状态预取）**：预取 8 字节对齐的 `LocatorSlot` 及优化器一阶动量（$m$）、二阶动量（$v$）状态行，有效掩盖 LLC Cache Miss 带来的内存停顿。

---

## 4. 存储分层与自适应硬件调优

### 4.1 硬件资源感知与自适应调优

存储引擎在底层独立维护硬件自适应组件：

- **硬件资源探测 (`SystemResourceProbe`)**：直接从 `/proc/meminfo` 解析高精度的 `MemAvailable`，并通过 `libnuma` 探测 NUMA 节点与物理核心分布。
- **启动自适应校准 (`StorageResourceTuner`)**：在引擎启动时针对 `{4096, 8192, 16384}` 候选页大小执行短微基准测试（结合维度、Worker 数与工作负载类型），选取中位耗时最优的页大小。若显式配置了 `entries_per_page` 则跳过校准。
- **动态最大页数 (`max_pages`)**：根据可用物理内存的 75% 预算自动推导最大页数，并严格对齐至 NUMA 节点倍数。

### 4.2 内存模式与磁盘溢写 (Spill-to-Disk)

- **`MemoryOnly`**：纯内存模式，禁用磁盘溢写。当分配超出内存预算时直接抛出异常。
- **`AutoStorage`**：自适应溢写模式。后台线程监控驻留内存的水位线（High Watermark 与 Low Watermark）：
  - 当内存达到高水位时，后台淘汰线程唤醒，将冷页面写入 Spill 目录；
  - 利用 Linux `mmap(MAP_FIXED)` 系统调用，将虚拟内存地址从匿名物理内存透明重映射到磁盘文件后备页，上层无感；
  - 内存回落至低水位后，后台换页线程自动休眠。

---

## 5. 缓存淘汰与自适应局部性机制

### 5.1 分数阈值淘汰 (`evict_by_score`)

- 淘汰接口通过每 Worker 有界堆收集分数不高于 `max_score` 的存活 Entry，临时内存开销仅与淘汰预算成正比，而非与整表大小线性增长。
- 最终删除在目标 Shard 的 Writer Mutex 下重新校验目录项、定位槽与存活标记，无需全表全局独占锁。

### 5.2 Boltzmann 效用模型与共访局部性感知

为提高缓存命中率，引擎实现了自适应页面效用评价系统：

- **自适应 Boltzmann 策略 (`AdaptiveBoltzmannPolicy`)**：
  $$\text{weight}_i = \frac{\exp(-(\text{utility}_i - \min(\text{utility})) / T)}{\sum_j \exp(-(\text{utility}_j - \min(\text{utility})) / T)}$$
  页面效用结合了热度衰减、共访密度（Co-access Density）与驻留时长。效用越高的页面被淘汰的概率越低。
- **共访草图 (`CoaccessSketch`)**：前向查询异步无阻塞采样共现特征键，后台轻量更新页间共访计数。高频共访的特征通过整页搬迁（Compaction）聚集在相邻的连续内存页中，显著提升 CPU 缓存局部性。

---

## 6. 生命周期管理与 Epoch 反压机制

所有可能被旧并发 Reader 访问的对象（包括 `MapBundle`、`StoreSnapshot`、定位槽与物理页）均统一托管在基于 Epoch 的 `RetirementPressure` 域中：

- **三级反压控制**：
  1. **Normal 阶段**（低于 `backpressure_fraction = 0.15`）：系统全速吞吐。
  2. **Backpressure 阶段**（介于 0.15 与 0.25 之间）：写操作与 Compaction 执行有限 CPU pause，放缓写入。
  3. **Circuit 阶段**（达到 `circuit_fraction = 0.25`）：触发熔断保护，拒绝生成新退休对象的控制面修改并直接抛出异常，防止内存暴涨。
- **安全资源回收**：当确认没有并发 Reader 引用时，旧物理页与元数据方被释放，`retired_bytes` 计数清零。

---

## 7. 分布式分片体系 (Partitioner Registry)

### 7.1 分片器注册中心 (`PartitionerRegistry`)

C++ 核心层通过统一 Facade 管理分片算法工厂：
- `sparse_embedding.modulo`：规范化整数取模（支持负数安全映射）。
- `sparse_embedding.hash`：MurmurHash3 雪崩哈希，支持指定 seed。
- `sparse_embedding.fast_range`：Lemire FastRange 无除法快速范围分片。
- 支持通过 `register_factory` 动态注入自定义分片算法。

### 7.2 向量化切片过滤 (`filter_span`)

在连续 Arrow 键切片上调用 AVX-512 `filter_span` 内核，批量生成字节掩码并提取匹配行索引，供离线恢复或通信调度器极速过滤。

### 7.3 Python 弹性分片策略 (`PartitionStrategy`)

Python 端分片策略与物理集群拓扑完全解耦：

```python
import numpy as np
import sparse_embedding as se

keys = np.array([10, 20, 30, 40], dtype=np.uint64)

# 1. 声明式分片策略 (拓扑晚绑定)
strategy = se.HashPartition(seed=42)

# 2. 传入目标卡数直接计算目标分片
ranks = strategy(keys, partition_count=4)

# 3. 向量化过滤指定分片下的键
filtered_keys = strategy.filter(keys, target_shard=0, partition_count=4)

# 4. 函数式便捷接口
ranks_func = se.partition(keys, partition_count=4, algorithm="sparse_embedding.hash")
```

---

## 8. Checkpoint 子系统接口契约

存储引擎与持久化 Checkpoint 子系统深度协同，遵循以下契约：

- **二元组唯一标识**：Embedding 与优化器状态使用 `(dimension, key)` 二元组作为全局唯一标识，支持不同维度表包含相同 key。
- **训练与推理状态隔离**：
  - 训练实例：分配并管理优化器状态页（Adam 一阶/二阶动量）；
  - 推理实例：配置 `EmbeddingConfig(enable_optimizer=False)` 构建权重专属 Store，杜绝分配优化器状态内存。
- **两阶段提交与 Lance 湖仓持久化**：底层通过 `snapshot_manifest.lance`（控制面元数据账本）与 `entries.lance`（数据面物理切片）实现 2PC 原子发布与恢复。

> 完整的 Lance 检查点存储架构、两阶段提交、增量恢复与 Gravitino 集成规范，请参阅专门文档：[Lance 存储与 Gravitino 集成指南](lance_storage_and_gravitino_zh.md)。

---

## 9. Python 配置指南

```python
import sparse_embedding as se

# 1. 配置分层存储策略 (AutoStorage 支持内存超额自动溢写至 SSD)
storage = se.AutoStorage(
    memory_limit_bytes=4 << 30,          # 驻留内存上限 4 GiB
    spill_directory="/mnt/nvme/spill",    # 磁盘溢写目录
    max_spill_bytes=100 << 30,           # 磁盘最大容量 100 GiB
    high_watermark=0.8,                  # 达到 80% 触发后台换出
    low_watermark=0.6,                   # 降至 60% 换出休眠
)

# 2. 配置 EmbeddingStore 参数
config = se.EmbeddingConfig(
    dims=[128],                          # Embedding 向量维度
    thread_budget=8,                     # 内部工作线程数
    storage=storage,                     # 存储后端
    key_dtype="uint64",                  # 键数据类型 (支持 int32/uint32/int64/uint64)
    value_dtype="float32",               # 物理值类型
    storage_tuning=se.StorageTuningOptions(
        target_capacity_keys=10_000_000, # 预估键容量
        memory_budget_bytes=4 << 30,     # 内存调优预算
        entries_per_page=4096,           # 物理页大小 (4096/8192/16384)
    ),
)

# 3. 创建 Store 实例
store = se.EmbeddingStore(config, score_policy="lfu")
```

---

## 10. 遥测指标与运维排障

调用 `store.storage_stats()` 获取当前存储运行时的累计计数器与内存指标：

| 指标名称 | 类型 | 含义说明与运维指导 |
|---|---|---|
| `memory_bytes` | 字节 | 当前计入存储的驻留物理内存用量 |
| `spill_bytes` | 字节 | 当前存活的文件后备磁盘用量 |
| `memory_limit_bytes` | 字节 | 生效的驻留物理内存上限 |
| `spill_limit_bytes` | 字节 | 生效的磁盘 Spill 容量上限 |
| `spill_count` | 计数 | 磁盘后备页分配次数（含复用 Extent） |
| `swap_out_count` | 计数 | 后台匿名页向文件重映射的换出次数 |
| `eviction_selection_count` | 计数 | 后台淘汰线程已完成的选择轮数 |
| `eviction_evaluated_pages` | 计数 | 所有轮次累计评估打分的页面数 |
| `eviction_selection_ns` | 纳秒 | 累计淘汰选择所耗费的纳秒数 |
| `eviction_candidate_invalidated_count` | 计数 | 最终校验阶段失效的候选数（正常并发变动） |
| `retired_bytes` | 字节 | 已进入 Epoch 退休但尚未回收的字节数 |
| `retirement_pressure_state` | 状态 | 当前 Epoch 压力状态：`normal`、`backpressure` 或 `circuit` |
| `retirement_backpressure_count` | 计数 | 因 Epoch 压力主动执行 CPU pause 的写操作数 |
| `oldest_unreclaimed_age_ns` | 纳秒 | 最早未回收退休记录的存活年龄 |
| `epoch_stall_count` | 计数 | 发生严重延迟并记录 CRITICAL 日志的 Epoch 停顿次数 |
| `spill_directory` | 字符串 | 当前生效的磁盘 Spill 文件目录路径 |
