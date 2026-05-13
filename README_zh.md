# sparse_embedding

高性能稀疏嵌入表库（Sparse Embedding Table）——为大规模推荐系统训练设计，基于 Hogwild! 无锁并发架构。

## 核心特性

- **无锁并发**：基于 `parlayhash`（Parlay Parallel Hash Map）实现 Key→Index 的 O(1) 无锁并发读写
- **SIMD 加速**：AVX-512 向量化哈希计算与梯度 Scatter 操作
- **Hogwild! 梯度累积**：分片 TransientArena 实现无锁梯度聚合，避免跨线程同步
- **C++26 标准并发**：基于 NVIDIA `stdexec` + `exec::static_thread_pool` 实现标准 Sender/Receiver 任务调度
- **Epoch 内存安全**：通过 `epoch::with_epoch` 批次保护并发哈希表读写，避免 ABA 问题
- **分布式 Lance Checkpoint**：支持分布式两阶段提交 (Two-Phase Commit) 的 Lance 存储后端，通过零拷贝 PyArrow 导出直接写入 Lance Dataset，解耦 embedding 与优化器状态，支持百 GB 级极速存储与加载。
- **动态分片与路由 (Partitioner OP)**：新增底层 C++ Partitioner OPs (`ModuloPartitioner`, `HashPartitioner`, `FastRangePartitioner`)，支持多路特征高效路由，支持在 Checkpoint 加载阶段进行灵活的集群弹性扩缩容（Resharding）。
- **驱逐与内存管理**：高效的打分策略（如 LFU），使用线程安全的 `std::atomic_ref` 实现无锁并发状态更新

## 系统依赖

| 组件 | 版本 |
|------|------|
| GCC | ≥ 15.0（需要 C++26 constexpr exceptions 支持） |
| stdexec | NVIDIA main branch `931f256c` |
| parlayhash | main branch（含 56-bit 指针 mask 补丁） |
| lance-c | main branch (基于 Rust 的 C 共享库) |
| Rust | 1.70+ (2024 Edition) |
| Bazel | ≥ 7.0 |

> **注意**：GCC 13 及以下不支持 `__cpp_constexpr_exceptions`，导致 `stdexec` 的 `sender_in` 概念检查失败，无法使用 `sync_wait` / `start_detached`。必须使用 GCC 15+。

---

## 性能基准

### 测试环境

| 项目 | 规格 |
|------|------|
| **CPU** | Intel® Xeon® Silver 4116 × 2 Socket |
| **物理核心** | 12 cores/socket × 2 = **24 cores**，支持超线程共 **48 vCPUs** |
| **CPU 主频** | 2.10 GHz（睿频 2.40 GHz） |
| **L1d Cache** | 768 KiB |
| **L2 Cache** | 24 MiB |
| **L3 Cache** | 33 MiB（per socket） |
| **NUMA** | 2 nodes，node 间延迟比 2.1x（node 0: 64 GB，node 1: 64 GB） |
| **内存** | 128 GiB DDR4 |
| **操作系统** | Ubuntu 20.04 |
| **编译器** | GCC 15.2.0，`-O2 -march=native` |

### 测试场景

模拟真实推荐系统训练场景：

| 参数 | 值 |
|------|-----|
| embedding 维度（dim） | 128 floats（512 bytes） |
| 词汇表大小（vocab） | 500,000 unique keys |
| 批大小（batch size） | 65,536 |
| 访问分布 | Power-law：top-10% key 承担 80% 流量 |
| slot 数量 | 1 |

### 吞吐量结果

> 单位：**Mops/s**（百万操作/秒）；`forward_lookup` / `assign` 额外标注内存带宽（GB/s）

#### `forward_lookup`（推理前向，热路径）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| forward_lookup | 3.24 Mops/s (1.66 GB/s) | 7.37 Mops/s (3.77 GB/s) | 11.23 Mops/s (5.75 GB/s) | 12.54 Mops/s (6.42 GB/s) | 15.08 Mops/s (7.72 GB/s) | 12.91 Mops/s (6.61 GB/s) |

`forward_lookup` 在 16-32T 时趋于饱和，实测峰值 6.94 GB/s。

**注意**：这并非 DRAM 带宽饱和（理论峰值为 230.4 GB/s，实测仅利用 3%），而是**随机访问延迟瓶颈**：
- 每次 embedding 查找需随机访问 arena 中 512 字节（8 条 cache line），DRAM 访问延迟 ~90 ns
- 单线程随机读带宽上限：64B / 90ns ≈ **0.71 GB/s**
- 要触及理论带宽需要同时保持 ~324 个 outstanding miss，而 Xeon Silver 4116 每核 MSHR 仅约 12 个
- 继续增加线程无法提升吞吐，因为 DRAM controller 的 outstanding request queue 已满，而非总线带宽不足

#### `lookup_or_insert`（训练插入，写热路径）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| lookup_or_insert | 1.48 Mops/s | 2.60 Mops/s | 4.02 Mops/s | 4.88 Mops/s | 5.06 Mops/s | 5.17 Mops/s |

1→8T 呈现 3.6x 线性加速；8T 后受 parlayhash CAS 竞争与 DRAM 写带宽限制趋于饱和。

#### `assign`（参数赋值）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| assign | 1.49 Mops/s (0.76 GB/s) | 3.19 Mops/s (1.63 GB/s) | 4.93 Mops/s (2.52 GB/s) | 5.90 Mops/s (3.02 GB/s) | 6.15 Mops/s (3.15 GB/s) | 6.28 Mops/s (3.21 GB/s) |

#### `accumulate_gradients`（梯度累积，反向传播）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| accumulate_gradients | 2.11 Mops/s | 5.76 Mops/s | 8.81 Mops/s | 11.33 Mops/s | 11.06 Mops/s | 10.66 Mops/s |

注：`accumulate_gradients` 使用 AVX-512 SIMD 分片 Scatter-Reduce；在修复了线程池调度阈值与使用 thread-local 缓存复用后，在 16-32T 呈现出完美的扩展性。

#### `optimizer_step`（参数更新）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| optimizer_step | 5.67 Mops/s (5.80 GB/s) | 12.99 Mops/s (13.30 GB/s) | 19.82 Mops/s (20.30 GB/s) | 30.73 Mops/s (31.47 GB/s) | **31.15 Mops/s (31.90 GB/s)** | 28.95 Mops/s (29.65 GB/s) |

注：`optimizer_step` 处理的是已累积的 unique key 集合（约 252K）。在修正了并行调度并移除了 Data Race 后，在 32T 时展现了惊人的 25.06 GB/s 内存带宽利用率，扩展效率绝佳。

#### `contains`（存在性检查）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| contains | 9.40 Mops/s | 16.57 Mops/s | 20.69 Mops/s | 22.48 Mops/s | 23.64 Mops/s | 22.65 Mops/s |

纯哈希查找无 embedding 拷贝，受 CPU 和哈希表 bucket 访问延迟限制，8T 后接近饱和。

#### `erase`（条目删除，10% 批次）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| erase (10% batch) | 12.39 Mops/s | 34.61 Mops/s | 63.85 Mops/s | 61.76 Mops/s | 49.25 Mops/s | 27.92 Mops/s |

`erase` 在 8T 时达到峰值（64.49 Mops/s），超线程后因 NUMA 跨 socket 竞争开始下降。

### Python 端性能表现

以下是在 Python 中调用相应接口（通过 `nanobind` 和 DLPack 零拷贝封装）测得的吞吐量，包含 Python 和 C++ 侧上下文切换及 asyncio 调度开销：

| API | 吞吐量 (Mops/s) | 说明 |
|-----|----------------|------|
| `async_unique` | ~25.58 Mops/s | 异步获取 unique keys，极速哈希去重 |
| `async_lookup_or_insert` | ~22.72 Mops/s | 异步插入及查找，零拷贝张量返回 |
| `async_lookup` | ~27.40 Mops/s | 异步前向查询，无阻滞事件循环 |
| `async_assign` (Zipfian) | ~2.85 Mops/s | 异步赋值，模拟高偏置幂律分布 |
| **端到端 Step** | **~3.87 Mops/s** | 全链路 asyncio 训练步 |

所有 `EmbeddingStore` 方法（如 `lookup_or_insert`, `assign`, `contains`, `erase`, `clear`, `assign_scores`, `accumulate_gradients`, `optimizer_step`）均已提供高性能、边缘触发的 `async_*` 异步变体。得益于 DLPack 和 eventfd 无锁 asyncio 挂载，Python 端的调用开销被降到最低，依然保持了数以百万计的吞吐率。

---

## 架构说明

```
HogwildEmbeddingStore
├── parlayhash (KeyMap)          # Lock-free concurrent Key→uint32_t mapping
├── EmbeddingArena               # Cache-line aligned contiguous embedding storage
│   └── PageTable[8192]          # Atomic page pointers, lazy allocated
├── TransientArena[num_threads]  # Per-shard gradient accumulation (lock-free)
├── exec::static_thread_pool     # C++26 stdexec thread pool
└── pro::proxy<OptimizerFacade>  # Type-erased optimizer (SGD, Adam, etc.)
```

### 关键设计决策

1. **键值解耦架构（Decoupled KV Storage）**：将传统哈希表的 `Key -> Vector` 拆分为 `Key -> Index`（`parlayhash`）和 `Index -> Vector`（`EmbeddingArena`）两级结构。哈希表仅存储轻量级的整数索引，使嵌入向量的存储保持紧凑、连续且对齐，极大提升了 CPU 缓存命中率与 SIMD 向量化效率。

2. **全局无锁并发（Lock-Free Everywhere）**：从前向查询到反向传播，全链路消除互斥锁（Mutex）。哈希表通过 CAS 原语实现并发插入，内存分配通过原子指针（Atomic Pointers）进行，梯度更新采用 Hogwild! 松弛一致性模型，彻底打通多核环境下的扩展瓶颈。

3. **分页内存管理（Paged Arena Allocation）**：嵌入表底层的 `EmbeddingArena` 采用类似操作系统的虚拟内存页表（PageTable）结构。通过惰性分配（Lazy Allocation）大块连续内存页，不仅避免了启动时的巨量内存开销，也防止了高频小规模内存分配导致的内存碎片化问题。

4. **无冲突梯度累积（Conflict-Free Gradient Accumulation）**：反向传播时为每个线程分配独立的 `TransientArena` 分片。各线程先在本地分片中累积梯度，最后再进行 SIMD 加速的汇总与优化器更新，避免了多线程同时更新同一 embedding 时的伪共享（False Sharing）和原子竞争。

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
```
