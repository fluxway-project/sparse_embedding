# sparse_embedding

[English](README.md)

高性能稀疏嵌入表库（Sparse Embedding Table），面向大规模推荐系统训练，采用 Hogwild! 松弛更新、分片哈希表同步和分页存储架构。

## 核心特性

- **分片并发**：基于 `parlayhash` 实现期望 O(1) 的 Key→Index 访问；读写锁仅作用于单个 shard，插入、删除、重建和扩容时其他 shard 仍可并行
- **SIMD 加速**：AVX-512 向量化哈希计算与梯度 Scatter 操作
- **Hogwild! 梯度累积**：分片 TransientArena 实现无锁梯度聚合，避免跨线程同步
- **C++23 标准并发**：基于 NVIDIA `stdexec` + `exec::static_thread_pool` 实现 Sender/Receiver 任务调度
- **Epoch 内存安全**：通过 `epoch::with_epoch` 批次保护并发哈希表读写，避免 ABA 问题
- **分布式 Lance Checkpoint**：支持分布式两阶段提交 (Two-Phase Commit) 的 Lance 存储后端，通过零拷贝 PyArrow 导出直接写入 Lance Dataset，解耦 embedding 与优化器状态，支持百 GB 级极速存储与加载。
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

Bazel 配置使用 `-std=c++2b` 编译 C++ 源码；当前验收构建使用 GCC 13.4 和 Bazel 7.7。

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
| 工具链 | GCC 13.4、Bazel 7.7、C++23（`-std=c++2b`）、`-c opt` |
| 优化参数 | `-O3 -march=native -mtune=native -mavx512bf16 -fno-strict-aliasing -DNDEBUG`；链接使用 `-flto` |
| CPU affinity | 本次快照未绑定 CPU |

### C++ 生产测试场景

| 参数 | 值 |
|---|---|
| Embedding 维度 | 128 个 `float`（512 bytes） |
| 词表 | 500,000 个 `uint64_t` key，单 slot |
| Batch | 65,536 keys |
| 访问分布 | 普通 operation batch 的 80% 流量落在最热 10%；erase key 均匀采样 |
| 随机种子 | 42 |
| 线程数 | 1、4、8、16、32、64 |
| 容量准备 | 计时前执行 `reserve(500000, 128)` |
| 优化器 | benchmark SGD，学习率 0.01 |
| 计时操作 | lookup/insert、forward lookup、assign、梯度累积、optimizer step、contains、erase |

LFU 压力场景使用 dim=128、16,384 个可能 key、4,096-key batch、2 个 worker、seed=42、相同的 80/20 热冷分布、学习率 0.01 的 benchmark SGD、5 MiB 内存限制、1 GiB spill 限制和 0.8/0.6 水位。它在插入前调用 `reserve(16384, 128)`，之后执行 4 次包含 lookup、assign、梯度累积和 optimizer step 的计时重复，并启用后台淘汰。计时结束后等待 100 ms 再读取的 telemetry 只能证明最终触发了淘汰，不能证明淘汰与计时区间重叠。

最新复现快照采集于 2026-07-21。该 C++ 数据来自一次包含内部重复的 benchmark 执行，不等同于 5 次独立运行、固定 CPU affinity 的 baseline/optimized 协议。

| benchmark 报告区间（Mops/s） | 1T | 4T | 8T | 16T | 32T | 64T |
|---|---:|---:|---:|---:|---:|---:|
| `lookup_or_insert` | 1.84 | 1.26 | 1.88 | 3.18 | 4.35 | 4.20 |
| `forward_lookup` | 2.90 | 5.34 | 7.91 | 10.18 | 12.10 | 11.61 |
| `assign` | 1.98 | 1.55 | 2.46 | 3.91 | 5.46 | 6.02 |
| `accumulate_gradients` | 2.35 | 6.30 | 8.60 | 10.37 | 11.54 | 10.80 |
| `optimizer_step` | 5.53 | 12.99 | 20.66 | 27.77 | 32.54 | 30.30 |
| `contains` | 6.25 | 5.90 | 8.62 | 11.13 | 15.59 | 16.86 |
| `erase`（10% batch） | 8.05 | 4.06 | 5.53 | 8.41 | 13.08 | 16.99 |

这些数据不是隔离 API timing。lookup/insert、forward lookup、assign、梯度累积和 contains 的计时区间包含随机 batch 生成。optimizer 报告区间还包含下一批 key 生成与梯度累积，却使用当前 store size 作为吞吐分子，因此不能解释为独立 optimizer 吞吐。erase 对同一组 key 连续执行 10 次且不重新插入，后续重复主要测量 not-found 路径。

LFU 内存压力场景报告 2.75 Mops/s、最终 2 次 swap-out、评估 4 页，累计选择时间 9,616 ns。Python 压力场景的同步训练 step 为 4.69 Mops/s；异步 lookup、assign 和 optimizer step 分别为 12.38、13.45 和 17.16 Mops/s。

这些数据用于复现和回归诊断，不是跨机器性能承诺。workload 定义、遥测语义和验证命令见[存储、并发与性能](docs/storage_and_performance_zh.md)。

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
storage = sparse_embedding.DiskSpill("/mnt/nvme/sparse-embedding", 1 << 40)
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
├── TransientArena[num_threads]  # Per-shard gradient accumulation (lock-free)
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
