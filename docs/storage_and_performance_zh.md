# 存储、并发与性能

[English](storage_and_performance.md)

本文档描述 2026-07-21 已验证的存储与并发契约。实现代码是常量的唯一事实来源；下文保留常量名称，便于从文档直接定位策略实现。

## C++ API 契约

`PageStorage` 与 `PageEvictionProxy` 是 `pro::proxy` facade。新代码通过 `create_page_storage(StorageOptions)` 创建存储，并按值传递 facade。复制或移动 facade 是浅操作，共享底层存储生命周期。`StoragePage` 是单次分配的 move-only RAII owner。

`set_page_eviction_proxy()` 返回 `PageEvictionRegistration`。每个 registration 必须且只能清理一次；空 token、未知 token 和已清理 token 都会抛异常，不会静默 fallback。active proxy 的 identity 或 lifetime 发生切换时 proxy generation 会递增，因此旧 active-proxy generation 评估出的候选不能执行 swap-out。

旧虚基类存储层级与 `PageStorage::create()` 不属于当前契约。

## Checkpoint 标识

Embedding 与 optimizer checkpoint 行使用 `(dimension, key)` 作为标识。
单独的 `key` 不唯一，因为同一个 store 的多个 dimension table 可以包含相同的
`uint64` key。Checkpoint loader 必须把持久化行组中的物理 dimension 显式传给
`assign_opt_state(keys, optimizer_states, dimension, steps)`。

Store 只选择一次对应 dimension，校验 optimizer-state 宽度与 step 数量，然后只在
该 table 内解析每个 key。未知 dimension、缺少 optimizer、state 宽度错误、step 数量
不匹配或 key 不存在都会立即抛异常，不允许搜索其他 dimension 作为 fallback。

embedding 与 optimizer 模板使用以下规范形态：

```cpp
using Store = se::HogwildEmbeddingStore<KeyT, ParamT, OptT, ScorePolicy>;
using Optimizer = pro::proxy<se::OptimizerFacade<ParamT, OptT>>;
using OptimizerFactory = se::Function<Optimizer(size_t)>;

auto optimizer =
    pro::make_proxy<se::OptimizerFacade<ParamT, OptT>, OptimizerImpl>();
auto optimizer_factory = pro::make_proxy<
    se::FunctionFacade<Optimizer(size_t)>>(
    [](size_t) -> Optimizer {
      return pro::make_proxy<
          se::OptimizerFacade<ParamT, OptT>, OptimizerImpl>();
    });
```

Facade 定义集中在接口头文件中；调用方使用这些完整模板参数，不再重复声明旧 facade 或 store alias。

## Python 配置

```python
import sparse_embedding as se

storage = se.AutoStorage(
    memory_limit_bytes=5 << 20,
    spill_directory="/mnt/nvme/sparse-embedding",
    max_spill_bytes=1 << 40,
    high_watermark=0.8,
    low_watermark=0.6,
)
config = se.EmbeddingConfig(
    dims=[128],
    num_threads=8,
    max_active_keys=1 << 20,
    storage=storage,
)
store = se.EmbeddingStore(config, score_policy="lfu")
```

`MemoryOnly()` 禁用磁盘后备页，并在内存预算耗尽时抛异常。`DiskSpill(directory, max_bytes)` 使用指定 spill 设备。非法存储限制、对齐、水位、容量或 registration token 都直接抛异常，不会选择 fallback 策略。

## ParlayHash 并发

每个 embedding 维度包含多个 key shard。单个 shard 使用 `std::shared_mutex`：查询获取共享锁；Upsert、删除、重建和内部扩容仅获取该 shard 的独占锁。不存在全局 Upsert mutex，因此一个 shard 扩容时，其他 shard 仍可并行。

初始容量策略集中定义在 store 内：

- `kParlayHashCapacityMultiplier = 4`
- `kInitialKeysPerShard = 2048`
- `kEagerPreallocationKeyLimit = 131072`

不超过 eager limit 的表会预分配预期的每 shard key 数。更大的表从有界的每 shard 容量开始，控制启动 RSS 与 cache footprint。容量已知的生产 workload 应在计时前调用 `reserve(keys_count, dim)` 或 `reserve_per_shard(counts, dim)`。reserve 会在热路径之外重建容量，超过 `max_active_keys` 的请求直接抛异常。

回归测试通过八个 worker 分别向无打分和有打分 store 插入 65,536 个唯一 key，并要求 key 数精确一致。该测试覆盖了此前导致 key 丢失与 ParlayHash SIGSEGV 的动态扩容竞态。

## 后台淘汰

淘汰由事件驱动。后台线程在条件变量上等待，直到可淘汰驻留内存触及高水位；触发后持续工作，直到内存降至低水位。它不使用定时轮询，也没有概率门控。

storage 增量维护匿名可淘汰页索引，因此选择轮次不需要扫描完整 page registry。

- `kExactEvictionPageThreshold = 1024`：对全部候选打分。
- `kSampledEvictionPageCount = 256`：超过阈值后，每轮最多评估 256 个不重复候选。
- `kMaxEvictionNoProgressAttempts = 8`：连续候选失效时终止，避免无限自旋。

选择过程在持有 storage mutex 时快照 page identity 与 proxy generation，释放 mutex 后执行打分，并在 `MAP_FIXED` 重映射前再次验证两者。data page evaluator 聚合 entry score；score metadata page 使用常量冷分，避免同一 score array 被重复扫描。

## 遥测字段

`EmbeddingStore.storage_stats()` 返回累计计数器与当前字节用量：

| 字段 | 语义 |
|---|---|
| `memory_bytes` | storage 当前计入的驻留字节 |
| `spill_bytes` | 当前存活的文件后备字节 |
| `memory_limit_bytes` | 生效的驻留内存限制 |
| `spill_limit_bytes` | 生效的 spill 限制 |
| `spill_count` | 磁盘后备页分配次数，包括复用 extent |
| `swap_out_count` | 后台匿名页到文件映射的置换次数 |
| `eviction_selection_count` | 已完成的选择轮数 |
| `eviction_evaluated_pages` | 所有轮次累计评估页数 |
| `eviction_selection_ns` | 累计选择耗时，单位纳秒 |
| `eviction_candidate_invalidated_count` | 最终校验阶段失效的候选数 |
| `spill_directory` | 自动选择或显式配置的 spill 目录 |

`spill_count` 与 `swap_out_count` 有意保持不同语义。直接分配文件后备页会增加前者；后台淘汰会增加后者。

## 测试环境与场景

### 测试机器

| 项目 | 实测配置 |
|---|---|
| CPU | 2 x Intel Xeon Silver 4116，标称 2.10 GHz |
| CPU 拓扑 | 每 socket 12 核、每核 2 线程，共 24 个物理核、48 个在线逻辑 CPU |
| NUMA node 0 | CPU `0-11,24-35` |
| NUMA node 1 | CPU `12-23,36-47` |
| `lscpu` 报告的 Cache | L1d 768 KiB、L1i 768 KiB、L2 24 MiB、L3 33 MiB |
| 内存 | OS 可见 125 GiB；无 swap |
| 操作系统 | Ubuntu 24.04.4 LTS，x86-64 |
| 编译与构建 | GCC 13.4、Bazel 7.7、C++23（`-std=c++2b`）、`-c opt` |
| 编译参数 | `-O3 -march=native -mtune=native -mavx512bf16 -fno-strict-aliasing -DNDEBUG` |
| 链接参数 | `-flto` |
| CPU affinity | 记录本次快照时未设置 |

### C++ 生产 workload

生产 workload 使用一个 dim=128 slot、500,000 个可能的 `uint64_t` key、65,536-key batch、`max_active_keys=1,000,000` 和 seed=42。生成的普通 operation batch 中，最热的 10% 词表承担 80% 访问；erase key 则均匀采样。1、4、8、16、32、64 线程的 store 都在 warmup 前调用 `reserve(500000, 128)`。parameter 与 optimizer 类型均为 `float`；benchmark SGD 的学习率为 0.01。

| 报告区间 | 前置条件 / 独立 warmup | 重复次数 | 计时区间内工作 | 吞吐分子 |
|---|---:|---:|---|---|
| `lookup_or_insert` | 1 batch | 10 | 随机 batch 生成 + 操作 | 65,536 keys |
| `forward_lookup` | store 已部分填充；无独立 warmup | 20 | 随机 batch 生成 + 操作 | 65,536 keys |
| `assign` | store 已部分填充；无独立 warmup | 10 | 随机 batch 生成 + 操作 | 65,536 keys |
| `accumulate_gradients` | store 已部分填充；无独立 warmup | 10 | 随机 batch 生成 + 操作 | 65,536 keys |
| `optimizer_step` | 保留此前 10 次梯度累积，再额外累积 1 batch | 10 | step + 下一批生成 + 下一次梯度累积 | 当前 store size |
| `contains` | store 已部分填充；无独立 warmup | 20 | 随机 batch 生成 + 操作 | 65,536 keys |
| `erase` | 无 | 10 | 不重新插入，重复删除同一组 key | 6,553 keys |

该表记录 benchmark 的实际实现，而不是隔离 API 成本。optimizer 的吞吐分子并不对应其 accumulated batch 中的 unique key，因此会高估独立 optimizer 吞吐。第一次之后的 erase 重复主要是 not-found 查询。

C++ LFU pressure workload 使用 dim=128、16,384 个可能 key、4,096-key batch、2 个 worker、`max_active_keys=20,000`、seed=42、相同的 80/20 热冷分布、学习率 0.01 的 benchmark SGD、5 MiB 驻留内存限制、1 GiB spill 限制以及 0.8/0.6 高低水位。插入后执行 4 次计时重复；每次重复都执行 lookup、assign、梯度累积和 optimizer step。读取淘汰 telemetry 前等待 100 ms。最终计数器证明淘汰已启用且最终触发，但不能证明它与计时区间重叠。

### Python workload

Python E2E workload 使用 dim=16、4,096 个连续 `int64` key、4 个 worker、`max_active_keys=8,192`、学习率为 0.01 的 SGD、LFU 打分、256 KiB 驻留内存限制和 1 GiB spill 限制。同步和异步路径都 warmup 2 次、计时 5 次。每个 async lookup、assign、gradient accumulation 和 optimizer Future 都被 await。

独立 Python storage-pressure workload 会运行 5 个相互独立的 store。每个 store 使用 dim=128、16,384 个连续 key、2 个 worker、`max_active_keys=20,000`、LFU 打分、5 MiB 驻留内存限制和 1 GiB spill 限制。输出吞吐与选择耗时的 median/p95，以及淘汰计数器的 min/max。

## 验证 workload

```bash
bazel test -c opt \
  //sparse_embedding/src/storage:page_storage_test \
  //sparse_embedding/src/backend/hogwild:hogwild_storage_test
bazel build -c opt //:copy_core_so
PYTHONPATH="$PWD" pytest -q
bazel run -c opt //benchmarks:embedding_store_benchmark
PYTHONPATH="$PWD" python3 benchmarks/python/benchmark_e2e.py
PYTHONPATH="$PWD" python3 benchmarks/python/benchmark_storage_pressure.py
```

正确性验证通过 69 个 Python 测试，跳过 11 个。C++ pressure workload 在淘汰启用且最终触发的条件下执行 lookup、assign、梯度累积和 optimizer step：2.75 Mops/s、2 次 swap-out、评估 4 页，累计选择耗时 9,616 ns。Python pressure workload 的同步训练 step 为 4.69 Mops/s，异步 lookup、assign 和 optimizer step 分别为 12.38、13.45 和 17.16 Mops/s。

同一机器上的优化扩展构建耗时 543.8 秒。构建时间与运行时吞吐分开记录。这些数据是暂定复现快照：C++ 数据来自一次包含内部重复的 benchmark 执行，Python storage-pressure 数据使用 5 次独立运行。95% 吞吐或 2 倍选择收益的相对门槛，需要使用相同 CPU affinity、随机种子、batch 与工具链分别生成 baseline 和 optimized 数据后才能判断。

### 测量状态

| 项目 | 当前状态 |
|---|---|
| C++ 生产 workload | 执行 1 次；固定 seed=42；包含内部 warmup/repetition；未设置 CPU affinity |
| C++ pressure workload | 执行 1 次；覆盖完整 lookup/assign/gradient/optimizer 路径；记录 telemetry |
| Python E2E workload | warmup 2 次、计时 5 次；输出吞吐中位数与 p95 延迟 |
| Python pressure workload | 独立运行 5 次；输出选择耗时和吞吐的 median/p95 |
| 异步完成 | lookup、assign、gradient 和 optimizer Future 均被 await 并返回 |
| 泄漏计数 | 进程正常退出，但未记录显式 wake queue 或线程泄漏计数器 |
| 相对门槛 | 等待相同 CPU affinity 下的固定 baseline/optimized/delta 数据 |

最终 Python p95 延迟为：同步训练 step 0.921 ms、async lookup 0.355 ms、async assign 0.322 ms、async optimizer step 0.279 ms。async gradient accumulation 已执行并 await，但没有作为独立指标计时。
