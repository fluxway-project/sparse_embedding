# 动态学习率 (Score-Based LR)

在大规模稀疏嵌入 (Sparse Embeddings) 中，根据特征（Feature）的访问频率或重要性动态调整学习率是非常有益的。对于高频特征，由于它们会被频繁更新，使用较小的学习率可以避免过度震荡或覆盖信息；而对于长尾（低频）特征，使用相对较大的学习率可以确保它们在有限的样本下充分学习。

Sparse Embedding 提供了一组 **基于特征得分的动态学习率策略 (Score-Based Dynamic Learning Rate Policies)**。这些策略会根据特征当前的得分 (通常由 LFU 或 LRU 评分策略实时维护)，在更新时动态调节每个特征的学习率。

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
    num_threads=16,
    value_dtype="float32"
)

# 使用 Adam 优化器，并配置平方根衰减策略 (SqrtDecay)
# 对于得分为 S 的特征，其实际学习率为：0.01 / sqrt(S)
optimizer = se.Adam(
    lr=0.01,
    dynamic_lr_policy=se.SqrtDecayLrPolicy()
)

# 创建 EmbeddingStore
# 注意：你必须开启 score_policy（例如 "lfu"），否则动态学习率将无法工作。
store = se.EmbeddingStore(
    embedding_config=embedding_config,
    initializer_config=se.Zeros(),
    optimizer_config=optimizer,
    score_policy="lfu"
)

# 在训练循环中，只要对特征进行了查询 (lookup/lookup_or_insert)，
# 它们的 LFU 得分就会在后台自动更新。
keys = np.array([1, 2, 3], dtype=np.int64)
embeddings, valid_mask = store.lookup_or_insert(keys, dim=128)

# 计算梯度...
gradients = np.ones((3, 128), dtype=np.float32)
store.accumulate_gradients(keys, gradients)

# 当调用 optimizer_step() 提交更新时，
# 引擎将自动根据每个特征最新的 LFU 得分，应用衰减后的学习率！
store.optimizer_step()
```

## 重要注意事项

* **依赖得分策略 (Score Policy is Required)：**如果你在优化器中指定了需要依赖得分的策略（例如 `SqrtDecayLrPolicy`），你就**必须**在创建 `EmbeddingStore` 时同时启用得分策略（例如 `score_policy="lfu"`）。如果未启用，系统在初始化时会抛出 `ValueError`。
* **极致性能 (Performance)：**动态学习率引入的额外开销微乎其微。底层的 C++ 实现使用了 `constexpr` 来在编译期决议静态和动态学习率代码路径，确保对于普通的静态学习率没有任何额外开销；而对于动态学习率，其计算逻辑已完全融合到了底层深度优化的 SIMD (如 AVX2/AVX-512) 向量化更新流程中，做到了高效并行计算。
