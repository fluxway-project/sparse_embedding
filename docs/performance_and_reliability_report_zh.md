# Sparse Embedding 性能与可靠性

[English](performance_and_reliability_report.md)

本文档面向开源用户说明 Sparse Embedding 的性能方法、实现保证与回归覆盖范围。

## 已交付能力

Sparse Embedding 提供按 shard 并发的存储、基于分数的 entry 淘汰、自适应 page
选择、整页 Compaction，以及可选的热 Key 提升。

整页 Compaction 会 drain 被选中的源页。每个 live key 都通过唯一的 sharded 权威目录
完成搬迁；源页清空后，才在 epoch 安全条件满足时释放。draining 或 retired 源页不会被
重新用作 destination。

Lance checkpoint 格式包含 `entries`、物理 `page_images` 以及固定两个 dataset version
并保存 checkpoint schema version 的 manifest。manifest 是唯一 checkpoint metadata 记录；
不会写 sidecar `metadata.json`。page image 保存 slot 布局、逐 slot 的 `seq_count`、参数、可选 score、optimizer
state/step 与 SHA-256 checksum。分布式加载的实质输入是调用方显式提供的 `filter_fn` 与
`target_shard_id`；`distributed=True` 只是临时兼容期断言，不负责路由或同步。后端不创建、
不持久化 routing plan。PageThermalTier、热度、coaccess、residency 等调度元数据不纳入
checkpoint。

增量 checkpoint 只追加发生变化的 entry 行和删除 tombstone；每个 shard 都维护单调递增的
sequence watermark。未变化 page 的 checksum 不会再次追加，manifest 会把该 delta 链接到已提交
的父 snapshot。事务仍由 Fluxway 负责跨所有 sparse 组件全局协调。Worker 只导出 fragment，
不能独立 commit；全局 barrier 之后，每个组件 commit owner 将该组件所有 worker artifact
传入唯一一次 `commit_fragments`，只有所有组件提交成功后全局 checkpoint 才发布。

启用 score 的 store 可以从一个物理页中提升有界数量的 top-K 热 Key。提升会把选中的
payload 与元数据复制到新的 arena slot，更新唯一权威目录，同时把未选中的冷 Key 保留在
原页。源页会被标记为 draining，不能再作为 destination；在冷 Key 被显式淘汰或 Compaction
前，原页仍可读取。该操作属于控制面，不会增加第二套 hot/cold 查询目录。

弹性恢复使用调用方针对当前拓扑提供的路由器读取 `entries`。同拓扑布局恢复把
`page_images` 导入保存时的 native page 与 slot 位置；它会拒绝逻辑路由参数和 worker
拓扑不匹配。

## 性能方法

所有报告数值均与具体 workload 绑定，用于回归追踪和同条件比较，不是跨机器吞吐承诺。

C++ workload 使用一个 dim=128 table、500,000 个可能的 `uint64_t` key、65,536-key
batch 和固定随机种子。80% 访问落在最热的 10% key；store 在 warmup 前完成容量预留。

| 操作 | 重复次数 | 计时区间 |
|---|---:|---|
| `lookup_or_insert` | 10 | batch 生成与操作 |
| `forward_lookup` | 20 | batch 生成与操作 |
| `assign` | 10 | batch 生成与操作 |
| `accumulate_gradients` | 10 | batch 生成与操作 |
| `optimizer_step` | 10 | step 与下一批梯度准备 |
| `contains` | 20 | batch 生成与操作 |
| `erase` | 10 | 不重新插入的重复删除 |

存储压力 workload 使用 dim=128、16,384 个可能 key、4,096-key batch、两个 worker、
80/20 热冷访问分布，以及配置好的驻留内存和 spill 限制。

## 已测量结果

### 并发内存查询回归门槛

2026-08-03 的查询回归门槛使用预留容量的纯内存 dim=128 table、10,000,000 个
打乱后的 `int64` key、8,192-key batch、128 个并发 Python 调用方、最多 128 个内部
worker、一次计时查询迭代和 seed 42。每个样本都在全新的 Python 进程中运行。

| 测量项 | 查询吞吐 |
|---|---:|
| 修复前源码基线 | 20,842,691 keys/s |
| 修复后样本 1 | 21,567,284 keys/s |
| 修复后样本 2 | 21,500,684 keys/s |
| 修复后样本 3 | 21,539,993 keys/s |
| 修复后中位数 | **21,539,993 keys/s** |
| 历史回归门槛 | 21,348,000 keys/s |

修复后中位数比修复前源码基线高 3.3455%，比历史门槛高 0.8993%；三个独立样本均
通过门槛。结构性修改将逻辑内存工作宽度与查询 task 数分离：4 MiB batch 仍保留 16 个
逻辑 telemetry partition，但可以暴露最多 128 个 task 来隐藏随机访存延迟，同时由活跃
并发调用数限制实际执行份额。该结果没有修改 256 KiB 工作粒度配置。

加权 telemetry 采样将该 batch 的 relaxed atomic update 从 15,872 降至 992，减少
93.75%。每次调用采样一个逻辑 partition，并以 partition 数作为权重；完整轮转后 page
heat 与 co-access 总量守恒。

### Slot Directory Forward Scaling

2026-08-04 的隔离 forward-scaling proof 使用 warmup 后的新 slot-directory 读路径。
workload 为 dim=32 的 memory-only store、128 个已发布 key、4,096-key 固定 power-law
batch、seed `0x5EED5CA1E`、400 次计时 `forward_lookup` 重复、48 个 allowed CPU，以及
1/2/4/8/16/64 线程。

| 线程数 | Forward 吞吐 |
|---:|---:|
| 1 | 9.632 Mops/s |
| 2 | 16.436 Mops/s |
| 4 | 27.914 Mops/s |
| 8 | 40.707 Mops/s |
| 16 | 42.049 Mops/s |
| 64 | 12.591 Mops/s |

相对单线程的 scaling ratio 为：2 线程 1.706x、4 线程 2.898x、8 线程 4.226x，超过当前
forward scaling gate 的 1.6x/2.5x/3.5x。同一次审计也保留了失败的完整
`embedding_store_benchmark --bind-affinity --skip-pressure`：它因
`key shard batch contains duplicate key` 抛异常，因此不能作为已接受性能结果。单线程完整
benchmark 尝试只到达 epoch-stall critical log，也不能作为已接受吞吐结果。

### 持续 Retirement Pressure

2026-08-04 的 sustained-pressure proof 使用 8 个 producer、1,048,576 个总 key、dim=8、
Auto storage、8 MiB 驻留内存限制、512 MiB spill 限制，以及 0.75/0.50 水位。该测试在压力
阶段后校验已保留 payload，报告当前 retirement/Sink 契约，而不是 disk spill 吞吐。

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

该 run 有意在 page spill/swap 发生前打开 retirement-pressure circuit。已接受结论是 Sink
admission 能在无 payload corruption 的情况下处理压力；它不是 disk spill/swap 吞吐证据。

阈值淘汰对比使用 dim=128 table、512-key batch、8,192-key resident budget、两个
worker，并独立运行五次。

| 指标 | Baseline | 阈值淘汰 |
|---|---:|---:|
| 吞吐 | 329,593 ops/s | 304,496 ops/s |
| 直接 spill 分配 | 10 | 1 |
| 配对吞吐比 | — | 92.125% |

阈值淘汰能够减少直接 spill 分配，但会增加维护成本。它适合约束增长，不能被视为无代价的
查询吞吐优化。

自适应 Boltzmann 指数与归一化循环已由编译器验证使用 32-byte 向量指令。数值测试保证
近似值有限、权重归一化和为一，并将指数相对误差限制在 `0.001` 以下。
分布形式为 weight_i = exp(-(utility_i - min_utility) / temperature) 后对候选归一化；
utility 越高，淘汰概率越低。候选循环使用有界的 base-2 多项式 approximate_exp，目标
workload benchmark 用于验证局部性收益。

Checkpoint 回归覆盖并发 worker 导出、manifest 固定版本恢复、checksum、optimizer state、
score state、worker 数变化的弹性恢复、精确布局恢复以及 drained page 空洞恢复。

### NUMA-aware sharding 与训练路径

NUMA-aware sharding 是 opt-in 特性，依赖 Bazel 的 `numactl@2.0.19.bcr.1`。Shard 范围
映射到 NUMA node，arena page 使用页对齐匿名 mapping 并通过 `numa_tonode_memory` 显式
放置，worker 分配到 node-local CPU 组。亲和性路径缓存拓扑，并在复用的线程改变逻辑
目标 CPU 时重新绑定；本机 benchmark 的 NUMA profile 将物理 worker 限制在 24 个物理核，
不把 48 个 SMT 上下文当成独立物理执行资源。隔离 forward proof 的 2/4/8/16/32/64
logical worker 吞吐为 14.055/17.992/20.858/21.141/22.713/21.888 Mops/s；32/16 为
1.074x，64/32 为 0.964x。这是 socket 带宽边界，而不是功能限制；64 worker 运行不会
依赖 SMT sibling 传输更多工作。

lookup-only NUMA forward 使用 shard `ReadView` 并复用路由 hash。估算 payload 小于
512 KiB 时跳过 NUMA 二次分桶，因为额外遍历和调度成本高于小 batch 的 locality 收益。
在 dim=32、4,096 行 proof 中，吞吐从强制 routing 的约 23–24 Mops/s 提升到连续分片的
44.66–45.25 Mops/s；更大 payload 仍使用 node-local routing。长期存在的
`exec::static_thread_pool` 每次 dispatch 仍需入队、唤醒、steal、完成计数和 barrier；空
`bulk + sync_wait` 在 1/2/8/24 worker 下约为 3.47/21.7/41.6/87.4 微秒。

新增的 `small_memory_all_paths_pressure_test` 在 Auto storage、64 MiB 驻留预算、
0.75/0.50 水位和临时 spill 文件下，并发覆盖 `lookup_or_insert`、`forward_lookup`、
`assign`、`contains`、`erase`、重复 key 的 `accumulate_gradients` 以及
`optimizer_step`。测试使用 64 个可淘汰的 1 MiB 页面，实际完成 45 次 swap-out、11 次
直接 spill 分配、2 次 eviction selection，并校验换页后页面 payload 未改变：

```bash
bazel test -c opt //sparse_embedding/src/backend/hogwild:small_memory_all_paths_pressure_test \
  --test_output=errors
```

最新 profiling pass 还测得 24 worker 下 assign 26.5–27.0 Mops/s、accumulate
7.89–8.82 Mops/s；这些是受主机负载影响的快照，与下面的 8-worker 回归序列分开。

训练 profile 使用 8 worker、dim=128、65,536 个已存在且唯一的行和 100 次重复。最近三次
开启 affinity 的结果为：纯 assign 21.67–21.82 Mops/s，accumulate 加必需 step
7.24–7.33 Mops/s，optimizer step 单独 13.62–15.67 Mops/s。纯 assign 排除了缺失 key
的 directory rebuild，因此与混合 full benchmark 分开报告。accumulate 每个输入行只计算
一次 hash，并将其带入 transient slot 和 optimizer lookup。另测了关闭 affinity 的对照；
在本机 NUMA 拓扑上它并未稳定更快，accumulate/optimizer 反而更慢。

```bash
bazel run -c opt //benchmarks:embedding_store_benchmark -- \
  --threads=8 --numa-aware-sharding --skip-pressure
bazel run -c opt //benchmarks:training_path_profile -- assign
```

## 公开验证

运行优化 C++ 回归套件：

```bash
bazel test -c opt --test_output=errors \
  //sparse_embedding/...
```

构建 Python 扩展并运行 checkpoint 回归：

```bash
bazel build -c opt //:copy_core_so
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py
```

根据目标机器运行通用 Python 测试和 benchmark：

```bash
PYTHONPATH="$PWD" pytest -q
PYTHONPATH="$PWD" python3 benchmarks/python/benchmark_e2e.py
PYTHONPATH="$PWD" python3 benchmarks/python/benchmark_storage_pressure.py
bazel run -c opt //benchmarks:embedding_store_benchmark
```

## 设计边界

热 Key 提升是显式控制面操作，不会在每次查询时执行。权威目录直接将每个 key 映射到当前
arena locator，因此查询不需要第二次分层查找；系统不会维护彼此独立的 HotMap 和 ColdMap。

当前 Compaction 策略会在释放被选中 page 前搬迁其中的全部 live key。该策略保持单一路径
查询，并使物理页回收具有明确的完成条件。

Lance checkpoint 通过 SeqLock reader 协议物化稳定 native page image。checkpoint 不克隆
spill 文件，也不依赖文件系统 copy-on-write 原语。entries 与 page_images 仍是普通 Lance
dataset，由 manifest version 与 checksum 固定。
