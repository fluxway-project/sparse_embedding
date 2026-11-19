# 动态学习率 (Score-Based LR)

[English](dynamic_learning_rate.md) | [架构与 API 契约](storage_architecture_zh.md) | [增量检查点与流式 Compaction](incremental_checkpoint_zh.md) | [Lance 存储与 Gravitino 集成指南](lance_storage_and_gravitino_zh.md) | [性能与可靠性报告](performance_and_reliability_report_zh.md)

## 使用前提

该 API 随仓库 Python package 提供，并使用项目 Bazel toolchain 构建。选择会读取特征得分的
学习率策略时必须同时配置 score policy；不读取得分的策略可以独立使用。

Sparse Embedding 提供**基于特征得分的动态学习率策略 (Score-Based Dynamic Learning Rate Policies)**，依据 LFU 或 LRU 等评分策略实时维护的访问计数动态调节每个特征的有效学习率。

## 可用策略 (Available Policies)

1. **`ConstantLrPolicy()`**: 
   保持固定学习率，不随得分改变（这是不指定策略时的默认行为）。
2. **`SqrtDecayLrPolicy()`**: 
   学习率随得分的平方根衰减。即 `lr = base_lr / sqrt(score)`。
3. **`LogDecayLrPolicy()`**: 
   学习率随得分基于对数衰减。即 `lr = base_lr / log2(2.0 + score)`。
4. **`InverseDecayLrPolicy()`**: 
   学习率随得分按反比衰减。即 `lr = base_lr / score`。
5. **`CosineFrequencyLrPolicy(min_lr_ratio=0.1, max_score=1024)`**: 
   采用余弦衰减（Cosine Decay），得分从 0 达到 `max_score` 时，学习率从 `base_lr` 平滑衰减至 `base_lr * min_lr_ratio`。
6. **`CustomLutLrPolicy(lut=[...])`**: 
   允许用户传入一个自定义乘数查找表 (LUT, Look-Up Table)，根据得分索引到相应的学习率乘数。

## 如何使用

要启用动态学习率，你必须在 `EmbeddingStore` 初始化时配置两个组件：
1. **`score_policy`**: 评分策略（例如 `"lfu"`, `"lru"` 或 `"manual"`）。动态学习率需要依赖得分表来工作。
2. **`dynamic_lr_policy`**: 将动态策略传给优化器配置（例如 `se.Adam` 或 `se.Sgd`）。

### 示例代码

```python
import numpy as np
import sparse_embedding as se

# 定义 Embedding 配置
embedding_config = se.EmbeddingConfig(
    dims=[128],
    thread_budget=16,
    value_dtype="float32"
)

# 使用 Adam 优化器，配置平方根衰减策略 (lr = 0.01 / sqrt(score))
optimizer = se.Adam(
    lr=0.01,
    dynamic_lr_policy=se.SqrtDecayLrPolicy()
)

# 创建 EmbeddingStore (配置 score_policy)
store = se.EmbeddingStore(
    embedding_config=embedding_config,
    initializer_config=se.Zeros(),
    optimizer_config=optimizer,
    score_policy="lfu"
)

# 查表操作自动更新特征访问得分
keys = np.array([1, 2, 3], dtype=np.int64)
embeddings, lengths = store.lookup_or_insert(keys, dim=128)

# 累积梯度并提交优化器更新
gradients = np.ones((3, 128), dtype=np.float32)
store.accumulate_gradients(keys, gradients)
store.optimizer_step(1)
```

## 使用约束

- **得分策略依赖**：依赖特征得分的策略（如 `SqrtDecayLrPolicy`）要求建表时配置 `score_policy`，未指定时初始化报错。
- **训练模式限定**：动态学习率为优化器行为，仅在 `enable_optimizer=True` 时生效。

## 验证

公开行为由动态学习率测试模块覆盖：

```bash
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_dynamic_lr.py
```
