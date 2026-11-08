# Sparse Embedding 性能与可靠性

[English](performance_and_reliability_report.md) | [架构与 API 契约](storage_architecture_zh.md) | [Lance 存储与 Gravitino 集成指南](lance_storage_and_gravitino_zh.md) | [增量检查点与流式 Compaction](incremental_checkpoint_zh.md) | [动态学习率](dynamic_learning_rate_zh.md)

本文档给出 Sparse Embedding 的公开性能参考、推荐配置和可靠性保证。

## 能力概览

Sparse Embedding 提供 shard-local 并发存储、无锁目录查询、miss-only 批量归并、
基于 score 的淘汰、自适应 page 选择、整页 Compaction 和可选热 Key 提升。

`lookup_or_insert` 对已存在 key 只执行目录查询；未命中 key 进入轻量 insertion
slow path。同一批次的重复 miss 共享一个 arena slot，输出位置复用同一 embedding。

`assign`、`accumulate_gradients` 与 optimizer update 的 mutation batch 要求
`(key, dimension)` 行唯一且 key 已发布。并发 erase/recycle 竞态只会使受影响梯度批次
失效：debug 构建立即抛异常用于诊断；release 构建使用 relaxed counter 计数，并向有界
telemetry ring 发布固定大小诊断事件。单一后台 consumer 可以把事件写入 `std::clog`；
producer 不执行 I/O，ring 满载时丢弃诊断且不阻塞训练路径。

Lance checkpoint 以规范 `entries` 数据集作为 SSOT，物理 `page_images` 作为可选导入缓存，manifest 固定可见性边界。存储分层为控制面 `snapshot_manifest.lance` 与数据面 `entries.lance`，支持 4 种 Arrow 原生键类型（`int32/uint32/int64/uint64`）及 DuckDB 谓词下推。恢复引擎通过 `PartitionerRegistry` 与 AVX-512 `filter_span` 过滤切片。分片路由通过 `parallel_partitioner.hpp` 对接 Fluxway `FunctionPartition`，8 线程吞吐达 1.03B keys/s。写入与 Compaction 采用有界内存管道流式处理。本地目录使用 POSIX 原子 rename；远端对象存储使用暂存目录与 Pointer 原子发布并级联清理。
page image 保存 slot 布局、`seq_count`、参数、score、optimizer state/step 和 SHA-256 checksum。
增量 checkpoint 只追加变化 entry 与 tombstone，并由外部 coordinator 完成全局提交。
推理使用 `EmbeddingConfig(enable_optimizer=False)` 与 `checkpoint.load_inference_checkpoint`（或存储层 `load_inference_weights`），
避免分配和加载优化器状态。

Compaction、热 Key 提升和 page eviction 都通过唯一权威目录更新 locator。源 page 在
epoch 安全前保持可读，不会被重新用作 destination。

## 测试标准机器与编译配置

| 项目 | 配置 |
|---|---|
| CPU | 2 × Intel Xeon Silver 4116，2.10 GHz |
| 拓扑 | 24 个物理核、48 个逻辑 CPU、2 个 NUMA node |
| NUMA CPU | node 0: `0-11,24-35`；node 1: `12-23,36-47` |
| Cache | L1d 768 KiB、L1i 768 KiB、L2 24 MiB、L3 33 MiB |
| 内存 | OS 可见 125 GiB，禁用 swap |
| 系统 | Ubuntu 24.04.4 LTS，x86-64 |
| 工具链 | GCC 13.4、Bazel 7.7、C++23 |
| 优化 | `-O3 -march=native -mtune=native -mavx512bf16 -DNDEBUG -flto` |

推荐使用物理核数量作为 execution worker 上限。需要 NUMA locality 时，同时启用
worker affinity 和 arena page placement：

```bash
bazel run -c opt //benchmarks:embedding_store_benchmark -- \
  --threads=16 --numa-aware-sharding --skip-pressure
```

`--numa-aware-sharding` 同时启用 execution 和 memory NUMA locality。worker 绑定
物理 CPU，arena page 按 shard 固定到对应 NUMA node。小 payload 会跳过额外 NUMA 分桶，
避免调度成本超过 locality 收益。

默认组合模式的 `embedding_store_benchmark` 在计时前预生成 power-law 批次，只计时
`forward_lookup`。扩展门槛使用检测到的同一 NUMA 节点放置内存和被测 worker，并在
forward 计时期间把协调线程绑定到同节点的独立 CPU。允许运行的 CPU 拓扑不足时直接报错；
`--threads=N` 模式不执行组合扩展门槛。

## 代表性性能结果

以下数据来自上面的标准机器。高并发与 compaction 使用一次 benchmark
调用内五次采样的中位数；混合 lookup/insert 使用五次独立进程运行的中位数。
吞吐单位为 Mops/s 或 M keys/s，仅用于 workload 参考，不构成跨机器容量保证。

### 并发内存查询

配置为 dim=128、预留容量的 memory-only table、10,000,000 个 `int64` key、8,192-key
batch、128 个并发调用方和最多 128 个内部 worker。

| 指标 | 结果 |
|---|---:|
| 查询吞吐 | **22.13 M keys/s** |
| telemetry relaxed atomic update 降幅 | **93.75%** |

### Slot-directory forward scaling

配置为 dim=32、4,096-key memory-only table、4,096-key 固定 power-law batch、400 次
计时 `forward_lookup` 重复，进程使用 `taskset -c 0-47`，并启用默认自动存储调优。

| 线程数 | Forward 吞吐 |
|---:|---:|
| 1 | 14.61 Mops/s |
| 2 | 27.04 Mops/s |
| 4 | 43.11 Mops/s |
| 8 | **50.39 Mops/s** |
| 16 | 45.74 Mops/s |
| 24 | 40.11 Mops/s |
| 32 | 39.57 Mops/s |
| 64 | 37.42 Mops/s |

默认 memory-work 与 row-grain 限制有效 task 数，避免计时路径被 enqueue、steal、completion 和 barrier 成本主导。存储资源调优器 (`StorageResourceTuner`) 会在构造阶段针对目标 workload 形状在候选页大小（4096、8192、16384）间自动评估并缓存最优页几何，同时支持显式指定 `entries_per_page` 实现完全确定性的复现。

调优器采用保守准入机制：只有当候选页大小的中位耗时比基准 4096 profile 至少优化 50% 时才会切换，从而有效抵御系统偶发噪声并保证在大容量或宽维度场景下稳定选择最佳页几何。

Forward、训练和 optimizer 热点表格取 5 次独立进程运行的中位数，使用 `taskset -c 0-47`（48 个允许使用的逻辑 CPU、24 个物理核）。高并发和 compaction 表格取单进程内 5 次样本；混合 lookup/insert 表格取 5 次独立进程运行的中位数。

### 完整存储前向扩展门槛

独立的 `embedding_store_benchmark` workload 使用 dim=128、500,000-key 词表、
65,536-key power-law 批次和 20 次计时重复。5 次完整默认运行均通过，包含存储压力阶段。
下表为 5 次运行的吞吐中位数，以及每次运行相对 1 worker 的最低扩展比。

| 原生 worker 数 | Forward 中位吞吐 | 相对 1 worker 的最低比值 | 门槛 |
|---:|---:|---:|---:|
| 1 | 4.23 Mops/s | — | — |
| 2 | 7.86 Mops/s | 1.85× | 1.6× |
| 4 | 14.51 Mops/s | 3.40× | 2.5× |
| 8 | **20.06 Mops/s** | **4.72×** | 3.5× |

### 训练路径

dim=128、65,536 个已存在且唯一的 row、8 workers、进程使用 `taskset -c 0-47`，每次计时 20 次重复。吞吐为 5 次运行中位数：

| 操作 | 吞吐 | 带宽 |
|---|---:|---:|
| `assign` | **13.43 Mops/s** | 6.40 GiB/s |
| `accumulate_gradients`（含必需 step） | **2.92 Mops/s** | 1.39 GiB/s |
| `optimizer_step` | **14.63 Mops/s** | 6.98 GiB/s |

热点 optimizer-only profile 使用相同 dim 和进程亲和性、200 次计时重复、单个原生
`optimizer_step()` 调用者；各结果为 5 次独立运行的中位数。65,536-key 组是单独的
对照 workload，不替代上表 20 次重复的训练路径结果。

| 唯一 key 数 | 原生 worker 数 | Optimizer 吞吐 |
|---:|---:|---:|
| 4,096 | 1 | 3.99 Mops/s |
| 4,096 | 8 | **17.89 Mops/s** |
| 4,096 | 16 | 16.46 Mops/s |
| 65,536 | 1 | 2.32 Mops/s |
| 65,536 | 8 | **13.43 Mops/s** |
| 65,536 | 16 | 10.13 Mops/s |

批级 optimizer profile 与下表使用不同 workload。比较 assign、梯度累积和 optimizer step
时请使用 `training_path_profile`；比较存储和 compaction 时请使用本文末尾的 benchmark 命令。

### 高并发压力与可靠性

8 producers、1,048,576 个 key、dim=8、Auto storage、8 MiB resident limit、512 MiB
spill limit、0.75/0.50 watermarks 的 pressure profile：

| 指标 | 结果 |
|---|---:|
| `full_lookup_verified` | `1` |
| pressure pages | `0` |
| spill / swap-out | `0 / 0` |
| retirement sink hit | `1,044,480` |
| allocation p99 | `1.55 ms` |

小内存 all-path pressure test 覆盖 `lookup_or_insert`、`forward_lookup`、`assign`、
`contains`、`erase`、重复 gradient accumulation 和 `optimizer_step`，并校验换页后
payload 未改变。

## 运行机制与性能参考

当前路径为 `key → shard MapBundle → locator_slot → arena_index → payload`。逻辑目录
仍使用 COW 快照，物理页目录使用稳定 segment（`StablePageDirectory`）；lookup、optimizer 和普通 compaction
relocation 不获取 mutex。扩表只发生在 writer/control plane：`reserve` 在计时前完成，
运行期超出预算时按以下精确系数增长：

| 层次 | 扩容规则 |
|---|---|
| 目录 bucket 预算 | `ceil(2.5 × key_capacity)`，再向上对齐到 16；若较小的 `ceil_pow2(4 × key_capacity)` 更省则采用它 |
| `MapBundle` 运行期容量 | `<64K`：2×；`64K≤capacity<1M`：1.5×；`≥1M`：1.25× |
| shard 预算 | `reserve(keys_count, dim)` 精确按 shard 分摊；`reserve_per_shard(counts, dim)` 使用调用方给出的独立预算 |

扩容 clone 当前 `MapBundle` 后 atomic exchange，旧 bundle 由 epoch retirement 回收；locator
slot 与 arena index 地址保持稳定。bucket/slot/arena 上限或分配失败直接抛异常，不使用
fallback。

### 最新性能结果

高并发 lookup/insert：`dim=32`、warm keys=`1<<20`、batch=1024、lookup 64 轮、insert 8
轮、memory-only 1 GiB、每配置 5 次 median/p95/p99。

| threads | lookup median/p95/p99 (Mops/s) | insert median/p95/p99 (Mops/s) |
|---:|---:|---:|
| 1 | 15.38 / 15.52 / 15.52 | 0.092 / 0.093 / 0.093 |
| 16 | 60.74 / 68.57 / 68.57 | 0.339 / 0.355 / 0.355 |
| 64 | **80.42 / 86.77 / 86.77** | **0.528 / 0.559 / 0.559** |

混合 lookup/insert：同一 dim、warm keys、batch，64 轮；16/64 总线程各半，Auto storage
memory=1 GiB、spill=2 GiB。

| 总线程 | lookup/insert 线程 | lookup / insert (Mops/s) | lookup/insert p99 (ns) |
|---:|---:|---:|---:|
| 16 | 8 / 8 | 48.44 / 0.338 | 191,635 / 61,571,904 |
| 64 | 32 / 32 | **57.71 / 0.412** | 695,113 / 141,594,266 |

compaction 扩展：`dim=128`、64 shards、每 source worker 4 pages；使用自动校准页几何
（本机为每页 4096 entries）、`store_threads=64`，固定工作量，1/2/4/8/16 workers，
每配置 5 次 median。

| workers | relocated keys/s |
|---:|---:|
| 1 | 1.94 M |
| 2 | 3.83 M |
| 4 | 7.12 M |
| 8 | 10.59 M |
| 16 | **12.04 M** |

未预分配 destination 时，计时阶段还包含目标页分配，因此结果通常更低。对应可执行目标
为 `high_concurrency_lookup_insert_benchmark`、`mixed_lookup_insert_concurrency_benchmark`
和 `page_compaction_benchmark`。

### 并行分片调度器与 SIMD 批掩码过滤 (Partitioner & Filter Benchmarks)

测试配置：8 线程并发、测试键规模 $10^7$、覆盖 4 种 `key_dtype` (`int32`, `uint32`, `int64`, `uint64`)。

| 算法 / 算子 | 单核 C++ 原生内核 (keys/s) | 8 线程并发吞吐 (keys/s) |
|---|---:|---:|
| `sparse_embedding.modulo` | 2.33 B | **789.2 M** |
| `sparse_embedding.hash` (MurmurHash3) | 2.41 B | **893.5 M** |
| `sparse_embedding.fast_range` (Lemire) | 2.41 B | **1.03 B** |
| AVX-512 SIMD 连续切片批掩码过滤 (`filter_span`) | 1.82 B | **1.45 B** |

- **在线路由**：多线程并行调度内核（`parallel_partitioner.hpp`）基于 `std::execution` 分块调度，各算法吞吐稳定维持在 789M ~ 1.03B keys/s，单核心微基准达 2.4B keys/s，为在线 SPMD 通信规划提供微秒级响应。
- **离线恢复过滤**：原生 `filter_span` 内核在 Arrow 连续物理内存切片上利用 AVX-512 向量化掩码并行比对，单批次处理 8 个（64-bit）或 16 个（32-bit）键，单核过滤吞吐达到 1.82B keys/s，彻底消除了每行迭代与中间反序列化开销。

换页测试应验证 `swap_out_count > 0` 与 `spill_count > 0`。安全回归应覆盖 dim=1、Adam、
`optimizer_step()` 与并发 `compact_page()`，并在独立子进程中检查非零终止码。完整 native
测试、TSAN、ASAN/UBSAN 均应作为发布前验证，不依赖异常捕获替代信号检测。

## 公开验证

```bash
bazel test -c opt --test_output=errors //sparse_embedding/...
bazel build -c opt //:package_core_extension
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py
PYTHONPATH="$PWD" pytest -q
```

压力验证：

```bash
bazel test -c opt //sparse_embedding/src/backend/hogwild:small_memory_all_paths_pressure_test \
  --test_output=errors
```

性能测试命令：

```bash
bazel run -c opt //benchmarks:high_concurrency_lookup_insert_benchmark
bazel run -c opt //benchmarks:mixed_lookup_insert_concurrency_benchmark
bazel run -c opt //benchmarks:page_compaction_benchmark -- \
  --strong-scaling --preallocate-destinations
```

## 设计边界

热 Key 提升是控制面操作，不在每次查询中执行。权威目录直接映射 key 到当前 arena
locator，不维护独立 HotMap 和 ColdMap。

Checkpoint 通过 SeqLock reader 物化稳定 native page image，不复制 spill 文件，也不
依赖文件系统 copy-on-write。Compaction 在释放源 page 前搬迁全部 live key，保持单一路径查询。
