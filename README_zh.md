# sparse_embedding

[English](README.md)

高性能稀疏存储与计算库，专为 **PB 级（Petabyte-scale）** 超大规模稀疏 Embedding（工业级推荐系统）、稀疏 Engram（条件记忆与知识检索）、稀疏 MoE（专家路由与参数存储）以及稀疏 KV Cache（长上下文长尾缓存）等场景设计。结合 Python 接口与 C++23 底层引擎，采用 Hogwild! 松弛更新、Slot Directory 读路径、x86 SeqLock 发布和分段物理分页存储架构，高效解决海量参数下的内存爆炸、高并发哈希冲突与 I/O 吞吐瓶颈。

## 核心特性

- **硬件资源自适应调优 (StorageResourceTuner)**：自动探测物理核心数、NUMA 拓扑与 `/proc/meminfo` 可用内存，数学推导 2MB HugePage 最佳对齐页容量与 75% 内存预算最大页数
- **通用存储拓扑 (StorageTopology)**：不可变全局拓扑 SSOT 契约，预计算全位移掩码（`>> entries_shift`、`& slot_mask`），实现单周期 branchless 坐标寻址
- **Slot Directory 读路径**：基于 `parlayhash` 实现稳定 Key→slot lookup，再通过 segmented locator table 发布 Slot→arena；warm 后的 lookup/forward 读侧不再经过 shard shared lock 或 lifecycle lock
- **常驻页表目录 (StablePageDirectory)**：物理页表采用两级分段指针目录，零堆跳转且 L1 Cache 友好，分配新页时原子发布
- **SIMD 加速**：AVX-512 向量化哈希计算与梯度 Scatter 操作
- **Hogwild! 梯度累积**：分片 TransientArena 实现无锁梯度聚合，避免跨线程同步
- **批级优化器扩展**：`OptimizerExtensionFacade<ParamT, OptT>` 是唯一的优化器类型擦除 seam。Store 在 shard-local SeqLock 写保护范围内解析 locator row，并按 key shard 调用一次 `update(OptimizerBatchView, global_step)`，不再对每行调用 proxy。
- **C++23 标准并发**：基于 NVIDIA `stdexec` + `exec::static_thread_pool` 实现 Sender/Receiver 任务调度
- **Epoch 内存安全**：通过 `epoch::with_epoch` 批次保护并发哈希表读写，避免 ABA 问题
- **分布式 Lance Checkpoint**：两阶段提交协议，分为控制面 `snapshot_manifest.lance` 与数据面 `entries.lance`。支持 4 种 Arrow 原生整型键（`int32/uint32/int64/uint64`）与 DuckDB 谓词下推；支持可选物理页缓存。
- **优化器参数选择性加载**：`EmbeddingConfig(enable_optimizer=False)` 创建推理 weight-only store。`checkpoint.load_inference_checkpoint`（或存储层 `load_inference_weights`）仅加载权重。
- **分片注册中心与并行调度 (PartitionerRegistry)**：基于 `pro::proxy<PartitionerFacade>` 支持 `modulo`、`hash`、`fast_range` 及自定义扩展。离线恢复支持 AVX-512 `filter_span`；在线调度（`parallel_partitioner.hpp`）对接 Fluxway `FunctionPartition`。
- **跨仓参考向量校验 (Reference Vectors)**：提供 `generate_reference_vectors`，验证分布式路由与离线恢复在各键类型下的分区一致性。
- **驱逐与内存管理**：高效的打分策略（如 LFU），使用线程安全的 `std::atomic_ref` 实现无锁并发状态更新
- **页级冷热数据置换 (Page-Level Hot/Cold Swapping)**：后台线程基于可配置的内存水位线，通过保持虚拟地址稳定的 `MAP_FIXED` 重映射将冷数据页置换至磁盘后备存储。

## 核心文档

- [存储架构设计与 API 契约](docs/storage_architecture_zh.md)：无锁并发模型、SeqLock 槽位寻址、硬件自适应校准与 SSD 溢写机制。
- [Lance 存储与 Gravitino 检查点集成指南](docs/lance_storage_and_gravitino_zh.md)：双 Lance 两阶段提交协议、Arrow 原生算子下推与湖仓元数据治理。
- [增量检查点与流式 Compaction](docs/incremental_checkpoint_zh.md)：分片变动日志、无阻塞一致性切片与多版本合流压缩。
- [动态学习率](docs/dynamic_learning_rate_zh.md)：基于访问频次分数的动态自适应学习率策略。
- [性能与可靠性报告](docs/performance_and_reliability_report_zh.md)：基准评测数据、常驻内存水位线与可靠性校验。

## 系统依赖

| 组件 | 版本 |
|------|------|
| GCC | ≥ 13.4（已验证） |
| stdexec | NVIDIA main branch `931f256c` |
| parlayhash | main branch（含 56-bit 指针 mask 补丁） |
| lance-c | main branch (基于 Rust 的 C 共享库) |
| Rust | 1.70+ (2024 Edition) |
| Bazel | ≥ 7.0 |
| libnuma | numactl 2.0.19（Bazel module `2.0.19.bcr.1`） |

Bazel 配置使用 `-std=c++23` 编译 C++ 源码；支持 GCC 13+ 和 Bazel 7+。

---

## 性能基准

### 测试机器配置

| 项目 | 实测配置 |
|---|---|
| CPU | 2 x Intel Xeon Silver 4116，标称 2.10 GHz |
| 拓扑 | 每 socket 12 核、每核 2 线程，共 24 个物理核、48 个在线逻辑 CPU |
| NUMA | 2 个 node；node 0 CPU 为 `0-11,24-35`，node 1 CPU 为 `12-23,36-47` |
| Cache | lscpu 汇总：L1d 768 KiB、L1i 768 KiB、L2 24 MiB、L3 33 MiB |
| 内存 | OS 可见 125 GiB；禁用 swap |
| 操作系统 | Ubuntu 24.04.4 LTS，x86-64 |
| 工具链 | GCC 13.4、Bazel 7.7、C++23（`-std=c++23`）、`-c opt` |
| 优化参数 | `-O3 -march=native -mtune=native -mavx512bf16 -fno-strict-aliasing -DNDEBUG`；链接使用 `-flto` |
| CPU affinity | Slot Directory forward proof 通过 benchmark affinity helper 绑定 worker；允许使用 48 个 CPU |

### Forward scaling

测试使用 memory-only store、固定 batch、预热和物理 CPU 亲和性。完整 workload 定义和参考表格统一维护在性能报告中。

#### Slot-directory forward scaling

参考 workload 使用 dim=32、4,096-key memory-only table、4,096-key 固定 power-law
batch、400 次计时 `forward_lookup` 重复，并在 48 个 allowed CPU 上使用物理核 affinity。

| 线程数 | Forward 吞吐 |
|---:|---:|
| 1 | 13.40 Mops/s |
| 2 | 20.25 Mops/s |
| 4 | 34.37 Mops/s |
| 8 | **44.46 Mops/s** |
| 16 | 35.69 Mops/s |
| 24 | 34.21 Mops/s |
| 32 | 36.56 Mops/s |
| 64 | 31.38 Mops/s |

完整 workload 定义、高并发表格、compaction 扩展和可靠性校验统一维护在性能报告中。

### NUMA-aware 执行与压力校验

`--numa-aware-sharding` 同时启用执行侧与内存侧 NUMA locality：worker 绑定物理核，arena
page 按 shard 固定到对应 node。公开 pressure 契约和验证命令统一见性能与可靠性报告。

### 无锁页迁移与扩表

当前八 worker `training_path_profile` 的 `assign`、`accumulate_gradients` 和
`optimizer_step` 吞吐分别约为 20.00、5.68 和 11.80 Mops/s。
扩表使用精确 per-shard 预算，不再统一向上取二次幂：目录采用 2.5× 探测预算并向 16 对齐，
运行期 `MapBundle` 容量按 `<64K: 2×`、`64K≤capacity<1M: 1.5×`、`≥1M: 1.25×` 增长。

Mutation benchmark 在调用要求唯一的 mutation API 前会先对 power-law 候选去重，并按实际
处理行数报告吞吐。纯 assign/accumulate/optimizer 测量请使用 `training_path_profile`。
该 profile 会在梯度路由、transient 累积和 optimizer lookup 之间复用同一次 hash；
workload 定义和统计口径见性能报告。
NUMA-aware sharding 通过 `--numa-aware-sharding` opt-in；它按 node 放置 arena page 和
worker，并在复用线程改变目标 CPU 时重新绑定。benchmark profile 以物理核为上限，不把
SMT sibling 当成额外内存带宽。lookup-only forward 直接使用 shard `ReadView` 并复用路由
hash。估算 payload 小于 512 KiB 时跳过 NUMA 二次分桶，因为小 batch 的调度成本高于
locality 收益；更大的 payload 仍使用 node-local routing。API 语义保持不变。

长期存在的 `exec::static_thread_pool` 消除了线程创建/销毁，但每次 `bulk + sync_wait`
仍需付出 task 入队、唤醒、steal、完成计数和 barrier 成本。空 bulk proof 在 1/2/8/24
worker 下约为 3.47/21.7/41.6/87.4 微秒。

如需复现覆盖全部公开操作的磁盘换页压力测试，可运行
`small_memory_all_paths_pressure_test`。64 MiB Auto-storage 配置会真实触发 swap-out，
并校验换页后的 payload：

```bash
bazel test -c opt //sparse_embedding/src/backend/hogwild:small_memory_all_paths_pressure_test \
  --test_output=errors
```

### 自动存储管理

存储拓扑在 store 构造阶段完成一次校准。调优器根据 embedding 维度、worker 数、workload
和 NUMA 模式，对 `4096/8192/16384` 小候选集执行稳健中位数采样并按请求形状缓存结果；显式
`entries_per_page` 仍是可复现实验的覆盖参数，并会跳过校准。

默认无需配置存储参数。`EmbeddingStore` 会探测系统与 cgroup 内存限制、进程资源限制、文件系统容量和本地设备属性。在安全内存预算仍有余量时，64 字节对齐页面保留在 RAM。

后台线程由条件变量和水位事件驱动，跟踪 `high_watermark` 与 `low_watermark`（默认 0.8 与 0.6）；它不会定时轮询，也不会按概率跳过必须执行的淘汰。触及高水位后，启用打分的 Eviction Proxy 会选择冷页，通过 `MAP_FIXED` 置换到未链接的文件后备映射，直到内存降至低水位。

可淘汰页不超过 1,024 页时进行精确选择；更大集合每轮最多评估 256 个不重复采样页，并持续执行到低水位。候选快照和 proxy generation 会阻止过期的候选页生效，打分阶段不持有 storage mutex。

第一次使用磁盘页面时会向 stderr 输出 `SparseEmbeddingSpillWarning`。调用 `storage_stats()` 还会发出一次标准 Python 警告，并返回当前内存与 spill 使用量：

```python
import sparse_embedding

config = sparse_embedding.EmbeddingConfig(
    dims=[128],
    storage_tuning=sparse_embedding.StorageTuningOptions(
        target_capacity_keys=10_000_000,
        memory_budget_bytes=4 << 30,
    ),
)
store = sparse_embedding.EmbeddingStore(config)
print(store.storage_stats())
```

`spill_count` 统计磁盘后备页面的分配次数（包括复用文件区间），并不表示独立的内存压力事件次数。

`storage_stats()` 还会返回 `swap_out_count`、`eviction_selection_count`、`eviction_evaluated_pages`、`eviction_selection_ns` 和 `eviction_candidate_invalidated_count`。这些计数器是累计诊断数据，不表示当前驻留页面数量。其中 `swap_out_count` 与 `spill_count` 语义不同：前者统计后台 RAM→文件映射置换，后者统计磁盘后备页分配。

高级用户可以指定存储设备，不需要改变 embedding 操作：

```python
storage = sparse_embedding.AutoStorage(
    memory_limit_bytes=0,
    spill_directory="/mnt/nvme/sparse-embedding",
    max_spill_bytes=1 << 40,
)
config = sparse_embedding.EmbeddingConfig(dims=[128], storage=storage)
```

`MemoryOnly()` 会禁用磁盘 spill，并在探测出的安全内存预算耗尽时抛出异常。key hash 元数据始终常驻 RAM，按 shard 动态增长；结构性 embedding index 上限和实际 storage 分配各自报告精确的资源错误，磁盘 spill 不会让 hash 元数据驻盘。

对于容量已知的大 workload，应在计时前调用 `store.reserve(keys_count, dim)` 或 `store.reserve_per_shard(counts, dim)`，将 ParlayHash 容量准备移出热路径，同时不对所有 shard 的 Upsert 做全局串行化。

---

## 架构说明

```
HogwildEmbeddingStore
├── StorageTopology (SSOT)       # 不可变全局存储拓扑与位移预计算
├── StorageResourceTuner         # 2MB HugePage 与硬件资源自适应调优
├── parlayhash (KeyMap)          # 分片 Key→uint32_t mapping
├── EmbeddingArena               # Cache-line aligned embedding storage
│   └── StablePageDirectory      # 零堆跳转 L1 Cache 常驻页目录
├── PageStorage                  # 自动探测 RAM 预算与文件映射 spill
├── TransientArena[thread_budget]  # Per-shard gradient accumulation (lock-free)
├── exec::static_thread_pool     # C++23 stdexec thread pool
├── PartitionerRegistry (ops)    # 零虚表 proxy 多态分片注册中心 (modulo/hash/fast_range)
├── ParallelPartitioner          # 1.03B keys/s 多线程分片调度与 AVX-512 掩码过滤
└── pro::proxy<OptimizerExtensionFacade<ParamT, OptT>> # 批级优化器 seam
```

### 关键设计决策

1. **键值解耦架构（Decoupled KV Storage）**：将传统哈希表的 `Key -> Vector` 拆分为 `Key -> Index`（`parlayhash`）和 `Index -> Vector`（`EmbeddingArena`）两级结构。哈希表仅存储轻量级的整数索引，使嵌入向量的存储保持紧凑、连续且对齐，极大提升了 CPU 缓存命中率与 SIMD 向量化效率。

2. **分片局部同步（Shard-Local Synchronization）**：ParlayHash 查询使用共享锁；Upsert、删除、重建和内部扩容仅对目标 shard 使用独占锁。其他 shard 继续并发，既避免无同步扩容的正确性问题，也避免全局 Upsert 锁造成的吞吐崩塌。

3. **分页内存管理（Paged Arena Allocation）**：嵌入表底层的 `EmbeddingArena` 采用类似操作系统的虚拟内存页表（PageTable）结构。通过惰性分配（Lazy Allocation）大块连续内存页，不仅避免了启动时的巨量内存开销，也防止了高频小规模内存分配导致的内存碎片化问题。

4. **无冲突梯度累积（Conflict-Free Gradient Accumulation）**：反向传播时为每个线程分配独立的 `TransientArena` 分片。各线程先在本地分片中累积梯度，最后再进行 SIMD 加速的汇总与优化器更新，避免了多线程同时更新同一 embedding 时的伪共享（False Sharing）和原子竞争。

5. **批级优化器分发（Batch-Scoped Optimizer Dispatch）**：locator 解析与参数修改位于同一个 packed SeqLock guard callback 内。优化器 proxy 每个已解析 shard batch 只进入一次，参数 span 不会被保存用于延迟更新。
6. **深存储模块（Storage as a Deep Module）**：embedding、optimizer state、score 和 transient gradient arena 共享同一个页面存储接口。资源探测、磁盘选择、警告和 extent 复用全部留在分配冷路径，不进入 lookup 与 optimizer 热路径。

### 训练更新路径

训练路径（`accumulate_gradients()` 后接 `optimizer_step()`）将梯度保存在按
shard 划分的 `TransientArena` 中。`optimizer_step()` 解析出 shard 批次后只进入一次
`OptimizerExtensionFacade`，并在 locator 的 `PackedWriteGuardBatch` 释放前消费全部参数
span。

优化器扩展分发明确限定在批级。静态分派是同一契约的实现选项，`std::variant` 只用于冷配置，
不会成为逐行运行时路径。

Mutation API 要求 `(key, dimension)` 行唯一。`lookup_or_insert` 可以归并重复 miss，
但 `assign`、`accumulate_gradients` 和 `optimizer_step` 的 batch 必须来自已发布 key；
release 构建遇到并发 erase/recycle 竞态时由 producer 增加 relaxed counter，并向有界
telemetry ring 发布固定大小诊断事件；单一 consumer 可以写入 `std::clog`。producer 不
执行 I/O，ring 满载时丢弃事件且不阻塞；受影响梯度仍被丢弃，debug 构建立即抛异常。

---

## 构建与安装

本库现已集成标准的 Python packaging 流程，在底层自动调用 Bazel 完成 C++ 扩展的构建：

```bash
# 从源码编译并安装
pip install .

# 推荐在本地开发时使用 editable mode
pip install -e .

# 构建二进制 wheel 分发包
pip wheel . --no-build-isolation
```

运行 Python 测试套件：

```bash
# 安装测试依赖并运行 pytest
pip install -e ".[test]"
pytest
```

默认启用 Lance。设置环境变量 `SE_BUILD_WITH_LANCE=0` 可进行不含 Lance 的独立构建：

```bash
SE_BUILD_WITH_LANCE=0 pip install -e .
```

直接通过 Bazel 构建与测试 C++ 目标：

```bash
bazel build -c opt //sparse_embedding/...
bazel test -c opt //sparse_embedding/...
```

详细的架构规范与构建契约请参阅 [构建与测试指南](docs/building_and_testing_zh.md)。

## 运行基准测试

```bash
# 全量 API 基准（推荐系统场景，power-law 访问分布）
bazel run -c opt //benchmarks:embedding_store_benchmark

# 热路径微基准（隔离 epoch / parlayhash / memcpy 各项开销）
bazel run -c opt //benchmarks:hotpath_microbench

# 淘汰压力下的 Python 同步/异步训练
PYTHONPATH="$PWD" python3 benchmarks/python/benchmark_e2e.py

# 运行 5 次的存储压力遥测 benchmark
PYTHONPATH="$PWD" python3 benchmarks/python/benchmark_storage_pressure.py
```
