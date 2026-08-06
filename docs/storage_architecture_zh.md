# 架构与 API 契约

[English](storage_architecture.md)

[性能与可靠性报告](performance_and_reliability_report_zh.md)

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

每个 shard 原子发布不可变 `MapBundle*`。map value 是非零 `slot_id`，不是 arena index。reader acquire-load 当前 bundle，不获取 shard shared mutex。结构性 writer 持有该 shard writer mutex，克隆已发布 bundle，应用 insert/erase/reserve 修改，再通过一次 atomic exchange 发布替换版本。旧 bundle 通过共享 epoch-retirement pressure 域回收。

`SegmentedLocatorTable` 拥有固定 32 KiB segment，每个 segment 含 4,096 个 8-byte locator slot。slot 保存 `{sequence, arena_index}` 并按 8-byte 对齐。slot ID、locator slot、arena index、page 与 store snapshot 都只能在 epoch 安全回收后复用。locator sequence 接近 wraparound 时，目录会把该 key rotate 到新 slot，并永久退休旧 slot，避免 ABA。

payload 读写使用 Linux x86_64 GCC/Clang SeqLock 契约。writer 通过 CAS 把偶数 sequence 改为奇数，更新 value、score、optimizer state 或 compaction relocation payload，再 release-publish 偶数 sequence。reader acquire-load sequence，经过 compiler barrier 后执行 SIMD copy，再经过第二个 barrier 并复读 sequence 校验。reader lock-free，在 writer 压力下允许重试；reader 不写 counter，也不做同步 telemetry mutation。

store 生命周期由原子 `StoreSnapshot*` 与 invocation gate 保护。forward、lookup、mutation、export 与 metadata 调用在入口登记一次，acquire-load snapshot，退出时离开 gate。`clear`、reserve、page-image import 等重配置在私有 control-plane mutex 下等待已进入调用退出，再发布新 snapshot。析构先停止 telemetry、关闭 invocation gate、注销 page-eviction callback、退休 snapshot，最后销毁 slot state。

查询调度将三个量分开处理。逻辑内存工作宽度用于判断 batch 是否值得并行，同时定义
telemetry partition；task 数提供足够的独立随机访存工作来隐藏延迟；实际执行宽度再根据
活跃查询调用数限流，避免嵌套调度为每个并发调用方分配整个线程池。在 128-worker 拓扑下，
8,192-key、dim=128 batch 对应 16 个逻辑 partition、最多 128 个 task；当 128 个调用方
活跃时，新进入调用的执行宽度为一个 worker。执行份额在每次调用进入时确定，不会追溯
调整更早进入的调用。显式 `memory_parallelism` 仍是 task decomposition 与实际执行宽度
的共同上限。

### ParlayHash 并发

每个 embedding 维度包含多个 key shard。查询不获取 shard mutex。insert、erase、reserve、locator rotation、checkpoint import 与 compaction publication 只在受影响 shard 的 writer mutex 下重建并原子发布不可变目录状态。不存在全局 Upsert mutex，因此一个 shard 扩容时，其他 shard 仍可并行。

初始容量策略集中定义在 store 内：

- `kParlayHashCapacityMultiplier = 4`
- `kInitialKeysPerShard = 1`
每个 shard 以 1 个预期 key 起步并独立增长，在不引入逻辑 key 配额的前提下控制启动 RSS 与 cache footprint。容量已知的生产 workload 应在计时前调用 `reserve(keys_count, dim)` 或 `reserve_per_shard(counts, dim)`。reserve 在热路径之外重建容量；结构性 index 溢出和 storage 分配失败会报告精确错误。

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

低于 backpressure 时正常运行。处于 backpressure 与 circuit 之间时，每个 writer 或 compactor 调用最多 yield 一次。进入 circuit 后，凡是会产生新 retirement 的控制面 mutation，除非已在本次调用内获得 admission，否则直接抛异常。已有 key 继续正常读取和更新。缺失 key 的 lookup-or-insert 返回该 dimension 的 64-byte 对齐零值 `SinkImage`，`inserted_mask = 0`、`contains = false`，不增长容量，也不进入 checkpoint。对 sink key 的写操作是 no-op；压力降回 backpressure 阈值以下后，新插入恢复。

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

自适应策略使用 Boltzmann 分布：`weight_i = exp(-(utility_i - min_utility) / temperature) /
sum_j exp(...)`。utility 越高，淘汰概率越低。min-utility 平移保持归一化分布不变，同时让
指数保持非正。温度随内存压力和 utility 离散度变化，并限制在配置边界内。候选循环使用
approximate_exp，它是有界的 base-2 加多项式 exp 近似；测试将相对误差限制在 0.001 以下，
并验证权重有限且归一化为一。共访密度进入 utility，因此经常一起使用的 key 获得更低淘汰
概率，也可以通过整页搬迁聚集；预期局部性收益通过目标 workload benchmark 验证。

`compact_adaptive_pages()` 将选择出的物理 page 预算交给与显式 compaction 相同的整页搬迁事务：源 page 先进入 draining 状态，再逐 key 在所属 shard 的独占访问范围内重新校验并搬迁；只有 live bitmap 为空后，数据页与 score 页才会进入 epoch 延迟回收。draining 或 retired 源页的 slot 不能作为 destination 复用。目录始终是唯一查询权威，因此 page 移动不会引入 HotMap/ColdMap 的二次查询。

启用 score 的 store 还提供 `promote_hot_keys(dim, page_id, maximum_keys, minimum_score)`。该操作从一个源页确定性选择 top-K，只通过权威 shard 目录搬迁这些 Key，并发布不可变 Hot Page 快照。源页保持分配状态，未选中的冷 Key 继续从原页读取；源页会标记为 draining，因此其 slot 不会被复用为搬迁 destination。调用方可通过 `promoted_hot_page_size(dim)` 查看已发布快照中的 Key 数量。

`seal_adaptive_pages()` 生成 64-byte 对齐的不可变 page 快照，用于 checkpoint 与布局控制；它不是查询索引，不改变路由。selector 与 planner 还接收可复用 scratch 对象，因此调度器能够跨控制周期复用 vector 容量，避免反复分配临时选择 buffer。

## Fluxway 与 Lance checkpoint 协议

Fluxway 是跨所有 SPMD 组件的全局 checkpoint 事务唯一协调者：

1. Rank 0 准备外部 artifact。
2. Worker 保存本地状态并返回 checkpoint metadata。
3. Fluxway coordinator 等待所有组件和 rank 完成后，为每个组件的 commit hook 各调用一次。
4. 失败时由 Fluxway 执行 abort。
5. Worker 从已提交状态恢复。

Sparse Embedding 只提供 worker artifact 和校验规则，不维护第二个全局 coordinator 或可变的
跨组件事务状态。Worker 进程禁止调用 `commit_fragments`。

提交单元是一个 Fluxway checkpoint 事务，而不是一个 worker。对于一个 Lance
组件，所有 worker artifact 必须在全局 barrier 之后由该组件 commit owner
传给唯一一次 `commit_fragments`。多个 sparse 组件可以使用各自的 Lance dataset
root，但它们的提交属于同一个 Fluxway 事务：只有所有组件提交成功后全局 checkpoint
才发布。如果多个组件必须共用一个物理 Lance dataset，schema 中必须显式增加组件
标识列；Worker 不能各自独立 append。

分布式 `entries` row 保存 `snapshot_id`、`worker_id` 与 `shard_id`；`page_images` 保存物理 page id、逐 slot 布局、`seq_counts`、参数、可选 score、optimizer state/step、逐 key shard id 以及 SHA-256 checksum。保存期 `worker_id` 只用于 artifact 完整性校验：每个 worker 必须且只能提交一个 artifact，且同一 `(dim,key)` 不得跨 worker 重复。协调者提交 `entries` 与 `page_images`，最后追加包含两个 Lance version 与 checkpoint schema version 的 snapshot manifest。manifest 是唯一 checkpoint metadata 记录，也是 checkpoint 的可见性边界；不会写 sidecar `metadata.json`。恢复时固定这些 version，校验全部 page-image checksum；弹性恢复对路由到当前 worker 的 `entries` 执行 assign。显式同拓扑模式则使用同一份 pinned `page_images` 恢复精确 native page/slot，必须提供保存时 worker 标识，且拓扑不匹配会直接抛异常。

恢复时先解析目标 `snapshot_id` 的唯一 manifest row，以其中保存的 Lance versions 打开数据集，
扫描完整已保存 snapshot，并应用调用方显式提供的 router：

```text
target_rank = filter_fn(key)
```

后端仅用 `snapshot_id = S` 过滤，不按保存时的 `worker_id` 过滤。因此 checkpoint
可以在不同 worker 数量之间恢复。分布式加载的实质输入是调用方提供的 `filter_fn`
和 `target_shard_id`；单独的 `distributed=True` 只是重复表达调用模式，不应成为长期
接口的必填参数。在兼容期间，传入 `filter_fn` 的调用仍可能需要同时传入
`distributed=True`，但这个 flag 不得改变路由或存储行为。Sparse Embedding 不创建、
不序列化、不解释 routing plan，也不回退到任何固定拓扑路由。

`page_images` 只恢复实际物理 page/slot/value/state 布局，不保存或恢复 `PageThermalTier`、
热度、cluster、coaccess、residency 等调度态；新集群启动后重新观测这些状态。

`LanceCheckpointBackend` 在 native SeqLock 协议下读取每个存活 slot 的稳定镜像，再把已
物化的 payload 写入 Lance `page_images`。checkpoint 从不克隆 spill 文件，也不会改变训练
写路径。增量 entries 仍使用 per-shard sequence watermark、delete tombstone 与按 checksum
寻址的 page reference。

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
| `retirement_sink_hit_count` | 缺失 key 被导向 Sink 的次数 |
| `oldest_unreclaimed_age_ns` | 最早未回收 retirement record 的年龄 |
| `epoch_stall_count` | 发出 CRITICAL JSON log 的 epoch stall 次数 |
| `spill_directory` | 自动选择或显式配置的 spill 目录 |

`spill_count` 与 `swap_out_count` 有意保持不同语义。直接分配文件后备页会增加前者；后台淘汰会增加后者。
