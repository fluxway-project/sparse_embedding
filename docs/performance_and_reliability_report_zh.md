# Sparse Embedding 性能与可靠性

[English](performance_and_reliability_report.md)

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

Lance checkpoint 包含 `entries`、物理 `page_images` 和固定版本的 manifest。page image
保存 slot 布局、`seq_count`、参数、score、optimizer state/step 和 SHA-256 checksum。
增量 checkpoint 只追加变化 entry 与 tombstone，并由外部 coordinator 完成全局提交。

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

可显式控制 dispatch：

```bash
bazel run -c opt //benchmarks:embedding_store_benchmark -- \
  --threads=16 --operation-physical-workers=16 \
  --operation-memory-parallelism=16 --forward-prefetch-distance=16 \
  --numa-aware-sharding --skip-pressure
```

`--numa-aware-sharding` 同时启用 execution 和 memory NUMA locality。worker 绑定
物理 CPU，arena page 按 shard 固定到对应 NUMA node。小 payload 会跳过额外 NUMA 分桶，
避免调度成本超过 locality 收益。

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
| 1 | 13.40 Mops/s |
| 2 | 20.25 Mops/s |
| 4 | 34.37 Mops/s |
| 8 | **44.46 Mops/s** |
| 16 | 35.69 Mops/s |
| 24 | 34.21 Mops/s |
| 32 | 36.56 Mops/s |
| 64 | 31.38 Mops/s |

默认 memory-work 与 row-grain 限制有效 task 数，避免计时路径被 enqueue、steal、completion 和 barrier 成本主导。存储调优器会在构造阶段针对 workload 形状校准 4096、8192、16384 三个候选页大小，并缓存中位数最优值；本机四类 workload 的校准代理均选择 4096，候选页显式对照记录在 `tmp/forward_page_geometry_pinned_20260822.log`；显式 `entries_per_page` 可用于复现实验。

只有候选页大小的中位耗时至少比已验证的 4096 profile 好 50% 时才会切换，否则保留 4096。该保守门槛同时抵御共享机器噪声和校准代理 workload 与真实路径的偏差，同时允许明显更优的页几何被选中。

本文所有公开数据均为最终工作区在 `taskset -c 0-47` 下的 5 次运行中位数，完整日志见
`tmp/docs_matrix_pinned_20260822.log`。`b61895d` 中的旧峰值不是稳定基线：commit 级 5 次
复测无法复现其 8.69 Mops/s 梯度累积峰值和 11.92 M relocated-keys/s compaction 峰值。
完整 commit 对照见 `tmp/historical_performance_audit_20260822.md`。

### 训练路径

dim=128、65,536 个已存在且唯一的 row、8 workers、进程使用 `taskset -c 0-47`，每次计时 20 次重复。吞吐为 5 次运行中位数：

| 操作 | 吞吐 |
|---|---:|---:|
| `assign` | **20.00 Mops/s** |
| `accumulate_gradients`（含必需 step） | **5.68 Mops/s** |
| `optimizer_step` | **11.80 Mops/s** |

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
| 1 | 10.87 / 11.78 / 11.78 | 0.045 / 0.051 / 0.051 |
| 16 | 65.61 / 67.01 / 67.01 | 0.319 / 0.354 / 0.354 |
| 64 | **77.15 / 84.11 / 84.11** | **0.566 / 0.592 / 0.592** |

混合 lookup/insert：同一 dim、warm keys、batch，64 轮；16/64 总线程各半，Auto storage
memory=1 GiB、spill=2 GiB。

| 总线程 | lookup/insert 线程 | lookup / insert (Mops/s) | lookup/insert p99 (ns) |
|---:|---:|---:|---:|
| 16 | 8 / 8 | 47.93 / 0.343 | 210,386 / 51,778,919 |
| 64 | 32 / 32 | **60.76 / 0.382** | 697,452 / 161,216,723 |

compaction 扩展：`dim=128`、64 shards、每 source worker 4 pages；使用自动校准页几何
（本机为每页 4096 entries）、`store_threads=64`，固定工作量，1/2/4/8/16 workers，
每配置 5 次 median。

| workers | relocated keys/s |
|---:|---:|
| 1 | 1.04 M |
| 2 | 1.99 M |
| 4 | 3.54 M |
| 8 | 6.77 M |
| 16 | **10.02 M** |

未预分配 destination 时，计时阶段还包含目标页分配，因此结果通常更低。对应可执行目标
为 `high_concurrency_lookup_insert_benchmark`、`mixed_lookup_insert_concurrency_benchmark`
和 `page_compaction_benchmark`。

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
