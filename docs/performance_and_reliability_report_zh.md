# Sparse Embedding 性能与可靠性

[English](performance_and_reliability_report.md)

本文档给出 Sparse Embedding 的公开性能参考、推荐配置和可靠性保证。

## 能力概览

Sparse Embedding 提供 shard-local 并发存储、无锁目录查询、miss-only 批量归并、
基于 score 的淘汰、自适应 page 选择、整页 Compaction 和可选热 Key 提升。

`lookup_or_insert` 对已存在 key 只执行目录查询；未命中 key 进入轻量 insertion
slow path。同一批次的重复 miss 共享一个 arena slot，输出位置复用同一 embedding。

Lance checkpoint 包含 `entries`、物理 `page_images` 和固定版本的 manifest。page image
保存 slot 布局、`seq_count`、参数、score、optimizer state/step 和 SHA-256 checksum。
增量 checkpoint 只追加变化 entry 与 tombstone，并由外部 coordinator 完成全局提交。

Compaction、热 Key 提升和 page eviction 都通过唯一权威目录更新 locator。源 page 在
epoch 安全前保持可读，不会被重新用作 destination。

## 标准机器与编译配置

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

### 并发内存查询

配置为 dim=128、预留容量的 memory-only table、10,000,000 个 `int64` key、8,192-key
batch、128 个并发调用方和最多 128 个内部 worker。

| 指标 | 结果 |
|---|---:|
| 查询吞吐 | **21.54 M keys/s** |
| telemetry relaxed atomic update 降幅 | **93.75%** |

### Forward scaling

配置为 dim=32、4,096-key 固定 batch、128 个已发布 key、48 个 allowed CPU，使用
slot-directory read path 和连续分片调度。

| 线程数 | Forward 吞吐 |
|---:|---:|
| 1 | 14.988 Mops/s |
| 2 | 21.852 Mops/s |
| 4 | 35.165 Mops/s |
| 8 | **45.945 Mops/s** |
| 16 | 42.049 Mops/s |
| 64 | 33.446 Mops/s |

默认 256 KiB memory-work 粒度和 64 KiB 最小 row 粒度共同限制有效 task 数，避免
线程数增加后被 enqueue、steal、completion 和 barrier 成本抵消。

### 训练路径

dim=128、65,536 个已存在且唯一的 row、8 workers、affinity enabled：

| 操作 | 吞吐 |
|---|---:|
| `assign` | **21.82 Mops/s** |
| `accumulate_gradients`（含必需 step） | **7.33 Mops/s** |
| `optimizer_step` | **15.67 Mops/s** |

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

## 公开验证

```bash
bazel test -c opt --test_output=errors //sparse_embedding/...
bazel build -c opt //:copy_core_so
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_checkpoint.py
PYTHONPATH="$PWD" pytest -q
```

压力验证：

```bash
bazel test -c opt //sparse_embedding/src/backend/hogwild:small_memory_all_paths_pressure_test \
  --test_output=errors
```

## 设计边界

热 Key 提升是控制面操作，不在每次查询中执行。权威目录直接映射 key 到当前 arena
locator，不维护独立 HotMap 和 ColdMap。

Checkpoint 通过 SeqLock reader 物化稳定 native page image，不复制 spill 文件，也不
依赖文件系统 copy-on-write。Compaction 在释放源 page 前搬迁全部 live key，保持单一路径查询。
