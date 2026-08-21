# 架构与 API 契约

[English](storage_architecture.md)

[性能与可靠性报告](performance_and_reliability_report_zh.md)

[Lance 存储与 Gravitino 验证](lance_storage_and_gravitino.md)

本文档描述存储、并发与核心组件机制契约。实现代码是常量的唯一事实来源；下文保留常量名称，便于从文档直接定位策略实现。

## C++ API 契约

`PageStorage` 与 `PageEvictionProxy` 是仅有的 storage `pro::proxy` facade。新代码通过 `create_page_storage(StorageOptions)` 创建存储，并按值传递 facade。复制或移动 facade 是浅操作，共享底层存储生命周期。`StoragePage` 是单次分配的 move-only RAII owner。

`set_page_eviction_proxy()` 返回 `PageEvictionRegistration`。每个 registration 必须且只能清理一次；空 token、未知 token 和已清理 token 都会抛异常，不会静默 fallback。proxy 接口只暴露 score evaluation 与 evaluator registration，不再暴露生命周期锁；Hogwild 的具体 adapter 会在注销 evaluator 时等待正在运行的 callback 退出。active proxy 的 identity 或 lifetime 发生切换时 proxy generation 会递增，因此旧 active-proxy generation 评估出的候选不能执行 swap-out。

旧虚基类存储层级与 `PageStorage::create()` 不属于当前契约。

运行时替换只保留在 `PageStorage` 与 `PageEvictionProxy` facade。
`SwapScheduler`、`SpillAllocator` 与 `PageRegistry` 是普通私有类，不是 facade：
scheduler 管理 wake、stop、join 与异常传播；spill allocator 管理 fd、目录、容量和
extent free-list；registry 管理 page identity、地址索引、淘汰快照和 evaluator 记录。
三者的锁彼此独立，不会在内存分配或查询热路径增加虚调用或类型擦除调用。

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
using Optimizer = se::OptimizerExtension<ParamT, OptT>;
using OptimizerFactory = se::Function<Optimizer(size_t)>;

auto optimizer =
    pro::make_proxy<se::OptimizerExtensionFacade<ParamT, OptT>,
                    OptimizerImpl>();
auto optimizer_factory = pro::make_proxy<
    se::FunctionFacade<Optimizer(size_t)>>(
    [](size_t) -> Optimizer {
      return pro::make_proxy<
          se::OptimizerExtensionFacade<ParamT, OptT>, OptimizerImpl>();
    });
```

`OptimizerExtensionFacade<ParamT, OptT>` 是唯一的优化器扩展 seam。必需操作是
`update(const OptimizerBatchView<ParamT>&, uint64_t)`；状态、checkpoint 和生命周期
操作仍属于同一个 facade。Store 先解析 key 与 arena index，再按 key shard 调用一次
批级 update。参数 span 都在对应 locator 的 `PackedWriteGuardBatch` 持有期间消费，不会
收集裸 `ParamT*` 延迟更新。

`accumulate_gradients()` 与 `optimizer_step()` 是训练 API。`std::variant` 只用于 optimizer
与 learning-rate 冷配置，不会成为逐行长驻运行时表示。

### 优化器执行不变量

训练路径把梯度累积在 `TransientArena`，再按执行 shard 执行一次批级更新，并遵守以下不变量：

| 不变量 | 契约 |
|---|---|
| Locator 生命周期 | key 解析、参数 span 构造与优化器修改都发生在同一个 `PackedWriteGuardBatch` 内。 |
| Proxy 次数 | 优化器扩展 proxy 每个已解析 shard batch 只进入一次，不按行进入。 |
| 禁止延迟指针 | 裸 `ParamT*` 或参数 span 不得逃逸 locator 写回调。 |
| 失败策略 | 非法 dimension、payload 宽度、重复 locator slot 或缺失 optimizer state 立即抛异常。 |

批级分发是生产 seam。静态分派和类型擦除分派实现同一批级契约，不改变 locator 生命周期，
也不允许参数 span 逃逸写保护范围。

## Python 配置

```python
import sparse_embedding as se

storage = se.AutoStorage(
    memory_limit_bytes=5 << 20,
    spill_directory="/path/to/local-spill-device",
    max_spill_bytes=1 << 40,
    high_watermark=0.8,
    low_watermark=0.6,
)
config = se.EmbeddingConfig(
    dims=[128],
    thread_budget=8,
    storage=storage,
)
store = se.EmbeddingStore(config, score_policy="lfu")
```

`MemoryOnly()` 禁用磁盘后备页，并在内存预算耗尽时抛异常。`AutoStorage(memory_limit_bytes=0, spill_directory=..., max_spill_bytes=...)` 使用指定 spill 设备保存可训练页面。非法存储限制、对齐、水位、容量或 registration token 都直接抛异常，不会选择 fallback 策略。

## 运行时原理与并发设计

### 查询与存储

唯一权威查询路径是 `ParlayHash key -> slot_id -> SegmentedLocatorTable -> arena_index`。shard 只由 key hash 决定；热/冷 tier 与物理 page 不参与路由。因此一次查询只需一次目录查找与一次 locator 读取，不会出现 HotMap 未命中后再查 ColdMap 的第二次 hash 查找。

每个 shard 原子发布不可变 `MapBundle*`。map value 是非零 `slot_id`，不是 arena index。reader acquire-load 当前 bundle，不获取 shard shared mutex。结构性 writer（insert、erase、reserve、locator rotation、checkpoint import）才持有该 shard writer mutex，克隆已发布 bundle，应用目录修改，再通过一次 atomic exchange 发布替换版本。普通 lookup、optimizer 和 compaction relocation 不获取该 mutex；它们只使用 locator packed seqlock/CAS。旧 bundle 通过共享 epoch-retirement pressure 域回收。

`SegmentedLocatorTable` 拥有固定 32 KiB segment，每个 segment 含 4,096 个 8-byte locator slot。slot 保存 `{sequence, arena_index}` 并按 8-byte 对齐。slot ID、locator slot、arena index、page 与 store snapshot 都只能在 epoch 安全回收后复用。locator sequence 接近 wraparound 时，目录会把该 key rotate 到新 slot，并永久退休旧 slot，避免 ABA。

payload 访问使用 Linux x86_64 GCC/Clang 纯乐观 SeqLock 契约。Compaction 物理搬迁等修改 `arena_index` 映射的操作通过 CAS 把偶数 sequence 改为奇数，迁移物理页 payload，再 release-publish 偶数新 sequence 与新地址。reader 与就地权重更新者（`optimizer_step`、`assign_existing`）通过纯乐观 `try_read` acquire-load sequence 与 arena 索引，经过 compiler barrier 后直接在目标内存就地读取或修改，再经过第二个 barrier 并复读 sequence 校验一致性。reader 完全 lock-free 且读路径零原子计数写入，在 writer 搬迁时自动感知版本变动并重试。

store 生命周期由原子 `StoreSnapshot*` 与 invocation gate 保护。forward、lookup、mutation、export 与 metadata 调用在入口登记一次，acquire-load snapshot，退出时离开 gate。`clear`、reserve、page-image import 等重配置在私有 control-plane mutex 下等待已进入调用退出，再发布新 snapshot。析构先停止 telemetry、关闭 invocation gate、注销 page-eviction callback、退休 snapshot，最后销毁 slot state。

查询调度将三个量分开处理。逻辑内存工作宽度用于判断 batch 是否值得并行，同时定义
telemetry partition；task 数提供足够的独立随机访存工作来隐藏延迟；实际执行宽度再根据
活跃查询调用数限流，避免嵌套调度为每个并发调用方分配整个线程池。在 128-worker 拓扑下，
8,192-key、dim=128 batch 对应 16 个逻辑 partition、最多 128 个 task；当 128 个调用方
活跃时，新进入调用的执行宽度为一个 worker。执行份额在每次调用进入时确定，不会追溯
调整更早进入的调用。显式 `memory_parallelism` 仍是 task decomposition 与实际执行宽度
的共同上限。

### 流水线软件预取机制

稀疏 Embedding 表检索与训练更新中，访存模式高度离散（随机键哈希、分段 locator 槽位定位、物理数据页索引以及优化器动量状态寻址）。硬件 Stream Prefetcher 无法有效预测此类跳跃式访存，导致大量执行周期消耗在 LLC Cache Miss 与内存停顿上。系统在前向检索（Forward / Lookup）与反向训练（Accumulate / Optimizer Step）热路径中构建了多级软件流水线预取体系：

1. **SIMD 与预取原语层 (`simd/prefetch.hpp`)**：
   封装跨平台的 L1/L2 预取原语（`prefetch_l1`、`prefetch_l2` 及 GNU `__builtin_prefetch`），支持指定读写意图（Read/Write Intent）与缓存局部性提示（Locality Hints）。

2. **前向查询多级流水线 (`HogwildEmbeddingStore::forward`)**：
   基于可配置的流水线步长 `forward_prefetch_distance`（默认 8～16 步），在处理当前样本时对未来的流水线阶段提前发出内存预取：
   - **第 1 级（输入特征预取）**：预取未来位置的 `keys[i + dist]`、`row_lengths` 与 `offsets`。
   - **第 2 级（输出缓冲区预取）**：对目标输出内存 `out_flat_values + offset` 发出写意图预取。
   - **第 3 级（哈希目录桶预取）**：计算未来 key 的哈希并调用 `prefetch_entry_cacheline` -> `FlatKeyDirectory::prefetch_bucket_cacheline(hash)`，提前将目录哈希桶缓存行装入 L1/L2 缓存。
   - **第 4 级（定位槽位预取）**：调用 `SegmentedLocatorTable::prefetch_slot_cacheline(slot_id)` 预拉取 8 字节对齐的 `LocatorSlot`（含 `sequence` 与 `arena_index`）。

3. **优化器状态页流水线 (`OptimizerExtensionFacade` & `AdamOptimizer`)**：
   - `OptimizerExtensionFacade` 将 `prefetch_state(uint32_t idx)` 作为优化器扩展 seam 上的可选操作。
   - 优化器内核在每行 SIMD 更新前预取该行 state；proxy 分发仍按已解析 batch 执行，而不是逐行进入。
   - `OptimizerState::prefetch_state_cacheline(index)` 定位物理页地址并发出写意图预取 `__builtin_prefetch(ptr, 1, 3)`，明确指示为 CPU 缓存行预热，避免与底层存储换页（I/O Page-in）混淆。
   - `AdamOptimizer` 在 SIMD 更新前预取一阶动量（$m$）、二阶动量（$v$）和每槽 `step_state` 缓存行，降低可避免的状态加载延迟。

4. **暂态内存池累加流水线 (`TransientArena`)**：
   - 在 `TransientArena::for_each_accumulated` 反向遍历梯度时，提前 8 步预取后续槽位的 `feature_ids_` 与 `feature_hashes_`。

### ParlayHash 并发

每个 embedding 维度包含多个 key shard。查询不获取 shard mutex。insert、erase、reserve、locator rotation 与 checkpoint import 只在受影响 shard 的 writer mutex 下重建并原子发布不可变目录状态；compaction publication 不重建 `MapBundle`，而是直接对 locator 执行 packed seqlock/CAS 发布。不存在全局 Upsert mutex，因此一个 shard 扩容时，其他 shard 仍可并行。

扩表策略集中定义在 store 与 `FlatKeyDirectory` 内。每个 shard 只按自己的实际目标 key 数扩容；`reserve(keys_count, dim)` 将总预算精确分摊到 shard，`reserve_per_shard(counts, dim)` 则直接接受调用方已知的每 shard 预算，不再把所有 shard 向上取相同二次幂。目录 bucket 数按 2.5 倍探测预算并向 16 对齐；只有较小容量仍低于旧 4 倍二次幂上限时才使用二次幂，因此大表不会因 per-shard rounding 快速膨胀。扩容倍率为 `<64K` 时 2x、`64K–1M` 时 1.5x、`>=1M` 时 1.25x，并且每次都至少覆盖本次 required capacity。

扩容只发生在 writer/control plane：writer 在受影响 shard 的 mutex 下 clone 当前 `MapBundle`，把 key→locator slot 复制到新目录后 atomic exchange；旧 bundle 通过 epoch retirement 回收，locator slot 与 arena index 地址不因目录扩容改变。已知容量的生产 workload 应在计时前调用 `reserve(keys_count, dim)` 或 `reserve_per_shard(counts, dim)`，将目录复制和 arena/optimizer reserve 移出热路径。运行期若插入超出预算，系统按上述精确增长策略扩容；容量上溢、bucket/slot 上限或 storage 分配失败直接抛出明确异常，不使用 fallback。

## 淘汰机制与后台线程

### 分数阈值淘汰机制

`evict_by_score(max_score, max_evict_count, ...)` 只选择分数不大于
`max_score` 的 live entry。候选通过每 worker 有界堆收集，临时内存与淘汰预算成正比，
而不随整张表线性增长。

最终删除会在目标 shard 的 writer mutex 下重新验证目录项、locator slot、score 与 live bit。
扫描阶段不使用 store 生命周期共享锁，也不使用全局独占扫描锁。

live bitmap 会在 index 退休前清除；page heat 计算忽略非 live slot，避免已删除 entry
继续污染页热度。该能力是 entry 级 GC，用于限制 arena 增长，不会释放部分占用的
storage page。

淘汰输出 buffer 只能表达一个物理维度。如果候选同时来自多个维度，接口会在
复制 value 前直接抛出异常；调用方必须按维度分别执行淘汰。

### 后台淘汰线程

淘汰由事件驱动。后台线程在条件变量上等待，直到可淘汰驻留内存触及高水位；触发后持续工作，直到内存降至低水位。它不使用定时轮询，也没有概率门控。

storage 增量维护匿名可淘汰页索引，因此选择轮次不需要扫描完整 page registry。

- `kExactEvictionPageThreshold = 1024`：对全部候选打分。
- `kSampledEvictionPageCount = 256`：超过阈值后，每轮最多评估 256 个不重复候选。
- `kMaxEvictionNoProgressAttempts = 8`：连续候选失效时终止，避免无限自旋。

选择过程在持有 storage mutex 时快照 page identity 与 proxy generation，释放 mutex 后执行打分，并在 `MAP_FIXED` 重映射前再次验证两者。data page evaluator 聚合 entry score；score metadata page 使用常量冷分，避免同一 score array 被重复扫描。

storage facade 不接收 store 生命周期锁。启用 score 的 store 注销 proxy 时，具体 proxy 会移除 page evaluator，并等待正在运行的 evaluator callback 退出后，store 才释放 slot state。

### Epoch 压力与 Sink

所有可能仍被旧 reader 看到的对象都通过同一个 epoch-backed `RetirementPressure` 域退休。该域统计已退休但未回收的 bundle、snapshot、page、arena index、locator slot 与 page image 字节。默认配置为 `backpressure_fraction = 0.15`、`circuit_fraction = 0.25`、`epoch_stall_timeout = 30s`；构造时拒绝非法比例，Auto storage 还要求 circuit 阈值低于 spill high watermark。

低于 backpressure 时正常运行。处于 backpressure 与 circuit 之间时，每个 writer 或 compactor 调用最多执行一次 CPU pause。进入 circuit 后，凡是会产生新 retirement 的控制面 mutation，除非已在本次调用内获得 admission，否则直接抛异常。已有 key 继续正常读取和更新；缺失 key 的 lookup-or-insert 和 assign 明确失败，不会伪造成功或增长目录。压力降回 backpressure 阈值以下后，新插入恢复。

### 自适应局部性信号

`AdaptiveBoltzmannPolicy` 将衰减热度、共访密度和驻留时间组合为 page utility，
温度随内存压力和 utility 离散度变化。接口使用调用方提供的结构化数组 span，策略本身
不分配内存。

`CoaccessSketch` 是有界共访 telemetry。forward lookup 最多把 16 个 key 采样到栈上，
并为每个采样 table 尝试发布一个有界 ring batch；每个 store 的单一后台 consumer 批量
修改 page heat 与 co-access counter。发布不会阻塞或扩容；ring 满载时只增加
`dropped_batches` 并丢弃该次样本，不影响 embedding 数值。forward lookup 每次调用轮转
采样一个逻辑 partition，并以逻辑 partition 数作为观测权重，因此 partition 不依赖 task
decomposition 或并发限流。store 析构会先停止并 join consumer，再销毁 queued event 引用的
slot。

仅 release 生效的 optimizer locator revalidation 失败也使用同一 ring。producer 只增加
relaxed counter 并发布固定大小诊断事件，不格式化字符串，也不执行 iostream I/O。consumer
调用配置好的 `noexcept` 诊断 sink；默认实现向 `std::clog` 写一行。诊断事件受 ring 容量
限制，满载时允许丢弃；`dropped_diagnostic_events` 单独暴露这类丢弃数量。

自适应策略使用 Boltzmann 分布：`weight_i = exp(-(utility_i - min_utility) / temperature) /
sum_j exp(...)`。utility 越高，淘汰概率越低。min-utility 平移保持归一化分布不变，同时让
指数保持非正。温度随内存压力和 utility 离散度变化，并限制在配置边界内。候选循环使用
approximate_exp，它是有界的 base-2 加多项式 exp 近似；测试将相对误差限制在 0.001 以下，
并验证权重有限且归一化为一。共访密度进入 utility，因此经常一起使用的 key 获得更低淘汰
概率，也可以通过整页搬迁聚集；预期局部性收益通过目标 workload benchmark 验证。

`compact_adaptive_pages()` 将选择出的物理 page 预算交给与显式 compaction 相同的整页搬迁事务：源 page 先进入 `Available → Draining`，再逐 key 重新校验 locator、owner 和 sequence；目的地先完成 payload、score 和 Adam state 拷贝，再由 locator packed seqlock 发布新 index。普通 relocation 不获取全局 compaction mutex、shard writer mutex 或操作系统阻塞原语；冲突只产生有限重试或明确的 retry 失败。只有 live bitmap 为空后，数据页与 score 页才会进入 epoch 延迟回收，随后转为 `Retired → Available`。draining 或 retired 源页的 slot 不能作为 destination 复用。目录始终是唯一查询权威，因此 page 移动不会引入 HotMap/ColdMap 的二次查询。

启用 score 的 store 还提供 `promote_hot_keys(dim, page_id, maximum_keys, minimum_score)`。该操作从一个源页确定性选择 top-K，只通过权威 shard 目录搬迁这些 Key，并发布不可变 Hot Page 快照。源页保持分配状态，未选中的冷 Key 继续从原页读取；源页会标记为 draining，因此其 slot 不会被复用为搬迁 destination。调用方可通过 `promoted_hot_page_size(dim)` 查看已发布快照中的 Key 数量。

`seal_adaptive_pages()` 生成 64-byte 对齐的不可变 page 快照，用于 checkpoint 与布局控制；它不是查询索引，不改变路由。selector 与 planner 还接收可复用 scratch 对象，因此调度器能够跨控制周期复用 vector 容量，避免反复分配临时选择 buffer。

compaction 目的地按 shard 批量执行 ticket-CAS 预留；普通 compaction 先按 shard 形成迁移顺序以保持 cache-local 连续布局，co-access compaction 则先按共访 cluster 稳定排序，再逐 key 预留，以保留热 key 共址语义。预留、payload 拷贝和发布均在控制面调用内完成，不把 destination 分配或 page migration 引入 lookup/optimizer 热路径。

### 生命周期与无阻塞边界

`MapBundle`、`StoreSnapshot`、locator slot、arena index、`StoragePage` 和 sealed page image 共用 epoch retirement。reader 在 invocation gate 与 epoch 中取得快照；clear、import 和析构先关闭 invocation gate，再等待活动调用退出，最后释放 payload。旧物理页只有在 source page 无 live entry 且最后一个 epoch reader 退出后才析构，`retired_bytes` 随回收归零。PageStorage 的 swap/spill mutex 和条件变量只存在于后台换页控制面，不进入 lookup、optimizer 或 relocation 热路径。

## 框架无关的 Lance checkpoint 协议

分布式集成使用 `SparseCheckpointProtocol`，提供一个
`CheckpointSaveIdentity`、每个 writer 对应的 `CheckpointParticipant`，以及一次包含
全部 fragment 的最终 commit。`CheckpointCommitAuthority` 只包含 opaque scope、单调
generation 和 authority identity，不包含任何特定编排框架的类型或标识。

`LocalLanceStorage` 与 `GravitinoLanceStorage` 复用唯一的 Lance checkpoint 事务引擎。
该引擎保存物化的 `(dim, key)` entry 状态、物理 page image 和 snapshot manifest；两个
storage adapter 只提供 dataset 定位、staging prefix、持久小记录和 commit lock。
manifest 是唯一可见性边界，并固定 entries 与 page-image 的精确 Lance version。跨
participant 重复 key、participant 覆盖不完整、schema 不一致和过期提交权都会显式失败。

恢复选择精确的 `SparseCheckpointLocator`。`PhysicalRestore` 要求保存时的 participant
拓扑并重建原生 page 布局。`ElasticRestore` 接收调用方稳定的
`partition_fn(keys, partition_count)` 和目标 partition；实现先只扫描 key 与 row state，
再读取分配给当前 partition 的 payload 列。Sparse Embedding 内部不提供拓扑或路由
fallback。

原生 `ModuloPartitioner` 遵循 Python/NumPy 的语言无关规范化求余约定：即使 key 为负数，
owner 也始终位于 `[0, partition_count)`；用户自定义的 `partition_fn` 负责自身的路由语义。

thermal tier、热度、coaccess 和 residency 等调度态不进入 checkpoint，恢复后重新观测。

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
| `retired_bytes` | 已进入 epoch retirement 但尚未回收的字节 |
| `retirement_pressure_state` | 当前 pressure 状态：normal、backpressure 或 circuit |
| `retirement_backpressure_count` | 因 retirement pressure 执行 yield 的 writer 调用数 |
| `oldest_unreclaimed_age_ns` | 最早未回收 retirement record 的年龄 |
| `epoch_stall_count` | 发出 CRITICAL JSON log 的 epoch stall 次数 |
| `spill_directory` | 自动选择或显式配置的 spill 目录 |

`spill_count` 与 `swap_out_count` 有意保持不同语义。直接分配文件后备页会增加前者；后台淘汰会增加后者。
