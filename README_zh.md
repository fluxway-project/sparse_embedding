# sparse_embedding

[English](README.md)

高性能稀疏嵌入表库（Sparse Embedding Table），面向大规模推荐系统训练，采用 Hogwild! 松弛更新、Slot Directory 读路径、x86 SeqLock 发布和分页存储架构。

## 核心特性

- **Slot Directory 读路径**：基于 `parlayhash` 实现稳定 Key→slot lookup，再通过 segmented locator table 发布 Slot→arena；warm 后的 lookup/forward 读侧不再经过 shard shared lock 或 lifecycle lock
- **SIMD 加速**：AVX-512 向量化哈希计算与梯度 Scatter 操作
- **Hogwild! 梯度累积**：分片 TransientArena 实现无锁梯度聚合，避免跨线程同步
- **C++23 标准并发**：基于 NVIDIA `stdexec` + `exec::static_thread_pool` 实现 Sender/Receiver 任务调度
- **Epoch 内存安全**：通过 `epoch::with_epoch` 批次保护并发哈希表读写，避免 ABA 问题
- **分布式 Lance Checkpoint**：框架无关的 prepare/write/commit 协议写入 manifest 固定版本的 `entries` 与物理 `page_images`，保存 slot 布局、sequence、payload/state，并进行 SHA-256 校验。精确版本恢复支持 physical participant 或调用方提供的弹性 partition function；单调提交权会拒绝过期 coordinator，不依赖特定分布式框架。
- **动态分片与路由 (Partitioner OP)**：新增底层 C++ Partitioner OPs (`ModuloPartitioner`, `HashPartitioner`, `FastRangePartitioner`)，支持多路特征高效路由，支持在 Checkpoint 加载阶段进行灵活的集群弹性扩缩容（Resharding）。
- **驱逐与内存管理**：高效的打分策略（如 LFU），使用线程安全的 `std::atomic_ref` 实现无锁并发状态更新
- **页级冷热数据置换 (Page-Level Hot/Cold Swapping)**：后台线程基于可配置的内存水位线，通过保持虚拟地址稳定的 `MAP_FIXED` 重映射将冷数据页置换至磁盘后备存储。

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

Bazel 配置使用 `-std=c++23` 编译 C++ 源码；当前验收构建使用 GCC 13.4 和 Bazel 7.7。

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

### 已接受 C++ Proof（2026-08-04）

当前已接受的重构性能数据来自两个专用 proof binary。更完整的 `embedding_store_benchmark --bind-affinity --skip-pressure` 运行违反了新的 caller-dedup 契约，并以 `key shard batch contains duplicate key` 抛异常，因此只记录为失败尝试，不能作为已接受性能结果。

#### Slot Directory Forward Scaling

隔离 forward proof 使用 dim=32 的 memory-only store、128 个已发布 key、4,096-key 固定 power-law batch、seed `0x5EED5CA1E`、400 次计时 `forward_lookup` 重复、48 个 allowed CPU，以及 1/2/4/8/16/64 线程。

| 线程数 | Forward 吞吐 |
|---:|---:|
| 1 | 9.632 Mops/s |
| 2 | 16.436 Mops/s |
| 4 | 27.914 Mops/s |
| 8 | 40.707 Mops/s |
| 16 | 42.049 Mops/s |
| 64 | 12.591 Mops/s |

相对单线程的 scaling ratio 为：2 线程 1.706x、4 线程 2.898x、8 线程 4.226x，超过当前 1.6x/2.5x/3.5x forward-scaling gate。

### 同机重构后复测（2026-08-10）

同一台测试机器上使用相同 workload，当前源码单次结果为 1/2/4/8/16/64 线程分别
14.988/21.852/35.165/45.945/41.928/33.446 Mops/s。16 线程追加三次为
36.557/39.012/39.973 Mops/s，存在明显调度和 NUMA 波动；固定物理核 affinity 与
NUMA placement 后再作优化结论。8 producers 高并发 pressure proof 通过，未发生
pressure page、spill 或 swap-out。

`--numa-aware-sharding` 同时启用执行侧与内存侧 NUMA locality：worker 绑定物理核，arena
page 按 shard 固定到对应 node。复现命令：`bazel run -c opt //benchmarks:embedding_store_benchmark -- --threads=16 --numa-aware-sharding --skip-pressure`。

#### 持续 Retirement Pressure 与 Sink

sustained-pressure proof 使用 8 个 producer、1,048,576 个总 key、dim=8、Auto storage、8 MiB 驻留内存限制、512 MiB spill 限制，以及 0.75/0.50 水位。该测试在压力阶段后校验已保留 payload，报告当前 retirement/Sink 契约，而不是 disk spill/swap 吞吐。

| 指标 | 结果 |
|---|---:|
| Pressure pages (`spill_count + swap_out_count`) | 0 |
| `spill_count` | 0 |
| `swap_out_count` | 0 |
| `memory_bytes` | 294,912 |
| `spill_bytes` | 0 |
| `retired_bytes` | 2,109,456 |
| `retirement_sink_hit_count` | 1,047,040 |
| Allocation p95 | 1.163 ms |
| Allocation p99 | 2.599 ms |
| Backlog drain | 0.003 ms |
| Producer elapsed | 129.397 ms |

该 run 有意在 page spill/swap 发生前打开 retirement-pressure circuit。已接受结论是 Sink admission 能在无 payload corruption 的情况下处理压力。

这些数据用于复现和回归诊断，不是跨机器性能承诺。架构设计与遥测语义见[架构与 API 契约](docs/storage_architecture_zh.md)，workload 定义和验证命令见[性能与可靠性报告](docs/performance_and_reliability_report_zh.md)。

Mutation benchmark 在调用要求唯一的 mutation API 前会先对 power-law 候选去重，并按实际
处理行数报告吞吐。纯 assign/accumulate/optimizer 测量请使用 `training_path_profile`。
该 profile 会在梯度路由、transient 累积和 optimizer lookup 之间复用同一次 hash；
当前开启 affinity 的测量结果见性能报告。
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

默认无需配置存储参数。`EmbeddingStore` 会探测系统与 cgroup 内存限制、进程资源限制、文件系统容量和本地设备属性。在安全内存预算仍有余量时，64 字节对齐页面保留在 RAM。

后台线程由条件变量和水位事件驱动，跟踪 `high_watermark` 与 `low_watermark`（默认 0.8 与 0.6）；它不会定时轮询，也不会按概率跳过必须执行的淘汰。触及高水位后，启用打分的 Eviction Proxy 会选择冷页，通过 `MAP_FIXED` 置换到未链接的文件后备映射，直到内存降至低水位。

可淘汰页不超过 1,024 页时进行精确选择；更大集合每轮最多评估 256 个不重复采样页，并持续执行到低水位。候选快照和 proxy generation 会阻止旧候选生效，打分阶段不持有 storage mutex。

第一次使用磁盘页面时会向 stderr 输出 `SparseEmbeddingSpillWarning`。调用 `storage_stats()` 还会发出一次标准 Python 警告，并返回当前内存与 spill 使用量：

```python
import sparse_embedding

config = sparse_embedding.EmbeddingConfig(dims=[128])
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
├── parlayhash (KeyMap)          # 分片 Key→uint32_t mapping
├── EmbeddingArena               # Cache-line aligned contiguous embedding storage
│   └── PageTable[8192]          # Atomic page pointers, lazy allocated
├── PageStorage                  # 自动探测 RAM 预算与文件映射 spill
├── TransientArena[thread_budget]  # Per-shard gradient accumulation (lock-free)
├── exec::static_thread_pool     # C++23 stdexec thread pool
└── pro::proxy<OptimizerFacade<ParamT, OptT>> # Type-erased optimizer
```

### 关键设计决策

1. **键值解耦架构（Decoupled KV Storage）**：将传统哈希表的 `Key -> Vector` 拆分为 `Key -> Index`（`parlayhash`）和 `Index -> Vector`（`EmbeddingArena`）两级结构。哈希表仅存储轻量级的整数索引，使嵌入向量的存储保持紧凑、连续且对齐，极大提升了 CPU 缓存命中率与 SIMD 向量化效率。

2. **分片局部同步（Shard-Local Synchronization）**：ParlayHash 查询使用共享锁；Upsert、删除、重建和内部扩容仅对目标 shard 使用独占锁。其他 shard 继续并发，既避免无同步扩容的正确性问题，也避免全局 Upsert 锁造成的吞吐崩塌。

3. **分页内存管理（Paged Arena Allocation）**：嵌入表底层的 `EmbeddingArena` 采用类似操作系统的虚拟内存页表（PageTable）结构。通过惰性分配（Lazy Allocation）大块连续内存页，不仅避免了启动时的巨量内存开销，也防止了高频小规模内存分配导致的内存碎片化问题。

4. **无冲突梯度累积（Conflict-Free Gradient Accumulation）**：反向传播时为每个线程分配独立的 `TransientArena` 分片。各线程先在本地分片中累积梯度，最后再进行 SIMD 加速的汇总与优化器更新，避免了多线程同时更新同一 embedding 时的伪共享（False Sharing）和原子竞争。

5. **深存储模块（Storage as a Deep Module）**：embedding、optimizer state、score 和 transient gradient arena 共享同一个页面存储接口。资源探测、磁盘选择、警告和 extent 复用全部留在分配冷路径，不进入 lookup 与 optimizer 热路径。

---

## 构建与安装

本库现已集成标准的 Python packaging 流程，在底层自动调用 Bazel 完成 C++ 扩展的构建：

```bash
# 从源码编译并安装 wheel
pip install .

# 推荐在开发时使用 editable mode
pip install -e .
```

当然，如果您只需运行 C++ 二进制或单独测试：

```bash
bazel build -c opt //sparse_embedding/...
```

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
