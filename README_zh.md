# sparse_embedding

高性能稀疏嵌入表库（Sparse Embedding Table）——为大规模推荐系统训练设计，基于 Hogwild! 无锁并发架构。

## 核心特性

- **无锁并发**：基于 `parlayhash`（Parlay Parallel Hash Map）实现 Key→Index 的 O(1) 无锁并发读写
- **SIMD 加速**：AVX-512 向量化哈希计算与梯度 Scatter 操作
- **Hogwild! 梯度累积**：分片 TransientArena 实现无锁梯度聚合，避免跨线程同步
- **C++26 标准并发**：基于 NVIDIA `stdexec` + `exec::static_thread_pool` 实现标准 Sender/Receiver 任务调度
- **Epoch 内存安全**：通过 `epoch::with_epoch` 批次保护并发哈希表读写，避免 ABA 问题

## 系统依赖

| 组件 | 版本 |
|------|------|
| GCC | ≥ 15.0（需要 C++26 constexpr exceptions 支持） |
| stdexec | NVIDIA main branch `931f256c` |
| parlayhash | main branch（含 56-bit 指针 mask 补丁） |
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
| forward_lookup | 3.26 Mops/s (1.67 GB/s) | 8.26 Mops/s (4.23 GB/s) | 11.21 Mops/s (5.74 GB/s) | 11.70 Mops/s (5.99 GB/s) | 13.56 Mops/s (6.94 GB/s) | 12.41 Mops/s (6.36 GB/s) |

`forward_lookup` 在 16-32T 时趋于饱和，实测峰值 6.94 GB/s。

**注意**：这并非 DRAM 带宽饱和（理论峰值为 230.4 GB/s，实测仅利用 3%），而是**随机访问延迟瓶颈**：
- 每次 embedding 查找需随机访问 arena 中 512 字节（8 条 cache line），DRAM 访问延迟 ~90 ns
- 单线程随机读带宽上限：64B / 90ns ≈ **0.71 GB/s**
- 要触及理论带宽需要同时保持 ~324 个 outstanding miss，而 Xeon Silver 4116 每核 MSHR 仅约 12 个
- 继续增加线程无法提升吞吐，因为 DRAM controller 的 outstanding request queue 已满，而非总线带宽不足

#### `lookup_or_insert`（训练插入，写热路径）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| lookup_or_insert | 1.64 Mops/s | 3.39 Mops/s | 5.85 Mops/s | 6.03 Mops/s | 6.29 Mops/s | 6.03 Mops/s |

1→8T 呈现 3.6x 线性加速；8T 后受 parlayhash CAS 竞争与 DRAM 写带宽限制趋于饱和。

#### `assign`（参数赋值）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| assign | 1.58 Mops/s (0.81 GB/s) | 3.39 Mops/s (1.73 GB/s) | 6.42 Mops/s (3.29 GB/s) | 7.68 Mops/s (3.93 GB/s) | 7.65 Mops/s (3.92 GB/s) | 6.98 Mops/s (3.57 GB/s) |

#### `accumulate_gradients`（梯度累积，反向传播）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| accumulate_gradients | 3.02 Mops/s | 2.95 Mops/s | 2.96 Mops/s | 2.96 Mops/s | 3.03 Mops/s | **9.15 Mops/s** |

注：`accumulate_gradients` 使用 AVX-512 SIMD 分片 Scatter-Reduce；1-32T 的瓶颈在于 Scatter 阶段的写冲突，64T 时 TransientArena 分片粒度充分分离，呈现突破性提升。

#### `optimizer_step`（参数更新）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| optimizer_step | 1.95 Mops/s (2.00 GB/s) | 1.93 Mops/s (1.97 GB/s) | 1.90 Mops/s (1.95 GB/s) | 1.91 Mops/s (1.95 GB/s) | 1.89 Mops/s (1.94 GB/s) | **4.50 Mops/s (4.61 GB/s)** |

注：`optimizer_step` 处理的是已累积的 unique key 集合（约 252K），受单 slot optimizer `pro::proxy` 调用开销限制，64T 时 Find + SGD update 并行效率最优。

#### `contains`（存在性检查）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| contains | 8.83 Mops/s | 15.95 Mops/s | 20.58 Mops/s | 22.31 Mops/s | 23.52 Mops/s | 22.84 Mops/s |

纯哈希查找无 embedding 拷贝，受 CPU 和哈希表 bucket 访问延迟限制，8T 后接近饱和。

#### `erase`（条目删除，10% 批次）

| 操作 | 1T | 4T | 8T | 16T | 32T | 64T |
|------|----|----|----|-----|-----|-----|
| erase (6553 keys) | 11.58 Mops/s | 36.27 Mops/s | 64.49 Mops/s | 58.06 Mops/s | 39.33 Mops/s | 24.53 Mops/s |

`erase` 在 8T 时达到峰值（64.49 Mops/s），超线程后因 NUMA 跨 socket 竞争开始下降。

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

1. **EmbeddingArena 预清零**：`ensure_capacity` 在分配新 page 时对整页（4096 × dim × sizeof(float)）做一次 `memset(0)`，`allocate()` 不再对单 entry 重复清零。

2. **Epoch 批次化**：`epoch::with_epoch` 提升到每个 dispatch chunk 的外层，而非每个 key 单独进出，消除高并发下全局 epoch 原子计数器的竞争。

3. **parlayhash 预分配**：以 `max_active_keys * 2` 初始化哈希表容量，避免运行时 rehash 导致的并发死锁（`copy_if_needed` 竞态）。

4. **`lookup_or_insert` 零冗余 Find**：新插入路径从 `upsert_fn` 闭包直接捕获 arena_idx，消除 Upsert 后的额外 Find 调用。

---

## 构建

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
