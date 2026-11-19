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
避免分配和加载优化器状态。对于纯训练或仅全量快照场景，配置 `EmbeddingConfig(enable_mutation_journal=False)` 可关闭 CDC 变动追踪，完全免除热路径原子序号记账与记录内存开销。此外，Adam 优化器采用对齐隔离架构，一阶与二阶动量保持 64 字节 AVX-512 SIMD 向量对齐，标量 step 计数器采用 4 字节自然对齐（`OptimizerRowLayout::kNaturalAligned`，单页 16 KiB 紧凑打包），兼顾向量计算吞吐与 CPU 缓存密度。

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

### 大表混合查询与插入

FP16，128 维，预填 16,777,216 个 Key。四个并发请求各查 983,040 个 Key（单次累计查询 3,932,160 个 Key），每次整批提交或分为 30 块执行。表中为每种模式两次测量的墙钟中位数（秒）。混合负载每 512 个位置包含一个新 Key（每轮新增 7,680 行）。表中记录测试耗时中位数（秒）及对应等效吞吐：

| 负载类型 | 分块模式耗时 (s) | 整批模式耗时 (s) | 分块模式吞吐 | 整批模式吞吐 |
|---|---:|---:|---:|---:|
| 全命中 (All hits) | 0.975 | 1.132 | 4.03 M keys/s | 3.47 M keys/s |
| 混合负载 (99.8% 命中, 0.2% 插入) | 1.149 | 1.092 | 3.42 M keys/s | 3.60 M keys/s |

### 并发内存查询

配置为 dim=128、预留容量的 memory-only table、10,000,000 个 `int64` key、8,192-key
batch、128 个并发调用方和最多 128 个内部 worker。

| 指标 | 结果 |
|---|---:|
| 查询吞吐 | **22.13 M keys/s** |
| telemetry relaxed atomic update 降幅 | **93.75%** |

### Slot-directory forward scaling

在参考平台上，默认 `forward_scaling_benchmark` 在 4,096 个打乱唯一键（dim=32）下 8 线程前向查表达到 **52.16 Mops/s**；全功能 `embedding_store_benchmark` 在 65,536 个幂律分布采样键（dim=128）下 16 线程达到 **26.26 Mops/s**，并稳定满足 2、4、8 worker 的并行扩展门槛。下表给出多次独立运行测试的吞吐中位数与扩展特性：

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

### 16 线程 NUMA 存储吞吐

双路 Xeon Silver 4116，16 个工作线程绑定在 CPU 0–23，按 NUMA 节点分片。向量维度为 128，词表含 500,000 个键，每批从幂律分布采样 65,536 次。写操作按去重后实际处理的行数计算吞吐。下表取 12 次独立进程运行的中位数。

| 操作 | 吞吐（Mops/s） |
|---|---:|
| `lookup_or_insert` | 1.02 |
| `forward_lookup` | 11.365 |
| `assign` | 32.76 |
| `accumulate_gradients` | 4.25 |
| `optimizer_step` | 20.065 |
| `contains` | 22.295 |
| `erase`（10% 批次） | 5.26 |

### LFU 存储压力下的 Python 训练

每批 4,096 个键，维度为 16，使用 SGD 和 LFU。线程预算为 4，进程绑定 CPU 0–11，内存限额为 256 KiB，spill 限额为 1 GiB。每次运行预热后计时 5 次，表中吞吐与延迟分别取 12 次运行的吞吐中位数和批次 p95 延迟的中位数。

| 操作 | 吞吐（Mops/s） | 批次 p95（ms） |
|---|---:|---:|
| 异步查表 | 14.95 | 0.2825 |
| 异步赋值 | 9.195 | 0.454 |
| 异步 SGD 更新 | 6.426 | 0.6405 |
| 同步训练步 | 1.724 | 2.3885 |

### Arena 槽位元数据吞吐

8 个工作线程绑定 CPU 0–7，对 262,144 个槽位遍历 32 次。每次运行取 5 次采样的中位数，下表取 12 次独立运行的中位数。组合操作对每个槽位发布 owner 后清除存活标记，计为一次操作。

| 模式 | 十亿槽位/秒 |
|---|---:|
| `publish` | 1.999 |
| `clear` | 3.47 |
| `publish_clear` | 1.408 |

在仓库根目录运行以下命令可复现上述负载。Python 环境需安装本项目的扩展。

```bash
bazel build -c opt //benchmarks:embedding_store_benchmark //benchmarks:page_compaction_owner_benchmark
taskset -c 0-23 bazel-bin/benchmarks/embedding_store_benchmark --threads=16 --numa-aware-sharding --bind-affinity --skip-pressure
taskset -c 0-11 python3 benchmarks/python/benchmark_e2e.py
for mode in publish clear publish_clear; do
  taskset -c 0-7 bazel-bin/benchmarks/page_compaction_owner_benchmark "$mode" 8
done
```

### Python 并发表操作吞吐与横向扩展

基于 `benchmarks/python/benchmark_embedding_table.py` 在维度 128 下评测 8、16、32 线程的扩展表现：

| 操作类型 | 批大小 | 8 线程 | 16 线程 | 32 线程 | 扩展趋势 |
|---|---:|---:|---:|---:|---|
| **就地赋值 (Assign)** | 65,536 | 10.00 MOPS | 11.79 MOPS | **14.05 MOPS** | 线性写扩展 |
| **就地赋值 (Assign)** | 131,072 | 9.65 MOPS | 11.29 MOPS | **13.69 MOPS** | 缓存行写入饱和 |
| **前向查表 (Lookup)** | 65,536 | 4.65 MOPS | 7.47 MOPS | **9.64 MOPS** | 强线性加速 (2.07x) |
| **前向查表 (Lookup)** | 131,072 | 5.44 MOPS | 8.35 MOPS | **9.88 MOPS** | 内存带宽打满 |
| **插入 (Insert)** | 65,536 | 1.14 MOPS | **1.17 MOPS** | 1.16 MOPS | 分片互斥锁上限 |
| **端到端训练步** | 65,536 | 1.53 MOPS | 1.77 MOPS | **1.98 MOPS** | 查表+聚合+更新完整流水线 |
| **端到端训练步** | 131,072 | 1.53 MOPS | 1.84 MOPS | **2.02 MOPS** | 查表+聚合+更新完整流水线 |

### 优化器状态对齐架构与 Cacheline 局部性

Adam 优化器状态包含一阶动量 ($m$)、二阶动量 ($v$) 与迭代步数计数器 ($t$)，采用向量与标量分层对齐：

1. **向量与标量分层对齐 (`OptimizerRowLayout`)**：
   - 浮点向量 $m$ 与 $v$（$D \times \text{sizeof(float)}$）：采用 64 字节对齐（`kCacheLineAligned`），确保 512 位 AVX-512 FMA 向量化指令无未对齐惩罚。
   - 步数计数器 $t$：标量 `uint32_t`（4 字节）采用自然紧凑对齐（`kNaturalAligned`），行跨度严格为 4 字节，物理页起始地址保持 64 字节对齐。
2. **物理页内存规格**：
   - 默认每页 4,096 个槽位时，`step_state` 物理页规格为 **16 KiB**（$4096 \times 4\text{ B}$），槽位内存密度达 100%。
3. **缓存工作集密度与并发局部性**：
   - 单条 64 字节缓存行高密度容纳 16 个槽位的 step 计数器。
   - 预留 262,144 个槽位时，step 计数器共占 **1 MiB**。
   - 不同线程更新相邻槽位时仍可能产生伪共享。

### 变动日志 (Mutation Journal) 内存与操作模式

`enable_mutation_journal` 配置项控制底层变更数据捕获（CDC）机制：

| 配置选项 | 内存常驻结构 | 热路径原子操作 | 检查点能力支持 | 适用场景 |
|---|---|---|---|---|
| `enable_mutation_journal = True` (默认) | 分配变动记录与指针管理数组（约 65 B/key RSS） | 写路径执行原子序号预留 | 全量 (FULL) + 增量 (INCREMENTAL) 检查点 | 在线增量快照、流式 CDC |
| `enable_mutation_journal = False` | **零变动记录分配** | **无原子序号预留开销** | 全量 (FULL) 检查点与恢复 | 纯模型训练、推理部署、周期性全量快照 |

当关闭变动日志时，调用 `capture_checkpoint(incremental=True)` 或 `checkpoint_mutations()` 会立即抛出明确的 `RuntimeError`，严格杜绝生成静默损坏或缺失基线的不完整增量；而全量快照、恢复以及确认协议则具备完备的正确性保证。

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

目录的读写语义见[分片发布契约](storage_architecture_zh.md#31-哈希路由与分片发布)。各层容量按下表增长。

| 层次 | 扩容规则 |
|---|---|
| 目录桶数 | 分片预留容量 `C > 0` 时使用 `4 × bit_floor(C)` 个桶，容量为零时使用 2 个桶 |
| `MapBundle` 运行期容量 | `<64K`：2×；`64K≤capacity<1M`：1.5×；`≥1M`：1.25× |
| shard 预算 | `reserve(keys_count, dim)` 精确按 shard 分摊；`reserve_per_shard(counts, dim)` 使用调用方给出的独立预算 |

### 运行时基准评测结果

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

验证覆盖 45 个原生测试目标和 486 项 Python 测试。恢复基准使用 16,384、65,536 和 262,144 行的数据集，覆盖空结果、单线程与四线程条目恢复，以及单线程页镜像恢复。每种配置的恢复行数均与预期一致。

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
