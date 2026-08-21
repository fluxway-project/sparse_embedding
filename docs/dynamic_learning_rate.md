# Dynamic Learning Rate (Score-Based LR)

## Requirements

This API is available from the Python package built with the repository's
Bazel toolchain. Use a score policy when the selected learning-rate policy
reads feature scores; policies that do not read scores can be used without one.

In large-scale sparse embeddings, it is often beneficial to adjust the learning rate dynamically based on the access frequency or importance of each feature. Features that appear frequently might need a smaller learning rate to avoid oscillating or overwriting information, while rare features might need larger updates to learn effectively from limited samples.

Sparse Embedding provides a set of **Score-Based Dynamic Learning Rate Policies** that can adjust the learning rate for each feature based on its score (which is typically updated by a scoring policy like LFU or LRU).

## Available Policies

1. **`ConstantLrPolicy()`**: 
   Keeps the learning rate constant regardless of the score. (Default behavior when no policy is specified).
2. **`SqrtDecayLrPolicy()`**: 
   The learning rate decays with the square root of the score. `lr = base_lr / sqrt(score)`.
3. **`LogDecayLrPolicy()`**: 
   The learning rate decays logarithmically based on the score. `lr = base_lr / log2(2.0 + score)`.
4. **`InverseDecayLrPolicy()`**: 
   The learning rate decays inversely proportional to the score. `lr = base_lr / score`.
5. **`CosineFrequencyLrPolicy(min_lr_ratio=0.1, max_score=1024)`**: 
   Uses a cosine decay schedule based on the score, decaying from `base_lr` to `base_lr * min_lr_ratio` as the score reaches `max_score`.
6. **`CustomLutLrPolicy(lut=[...])`**: 
   Allows you to define a custom Look-Up Table (LUT) of multipliers based on the score.

## How to Use

To use dynamic learning rates, you must configure two components in your `EmbeddingStore`:
1. **`score_policy`**: A policy that maintains a score for each feature (e.g., `"lfu"`, `"lru"`, or `"manual"`). The dynamic learning rate depends on this score.
2. **`dynamic_lr_policy`**: Passed to the `optimizer_config` (e.g., `se.Adam` or `se.Sgd`).

### Example Code

```python
import numpy as np
import sparse_embedding as se

# Define the Embedding configuration
embedding_config = se.EmbeddingConfig(
    dims=[128],
    thread_budget=16,
    value_dtype="float32"
)

# Use Adam optimizer with a Square Root Decay policy
# For a feature with score S, its effective learning rate will be: 0.01 / sqrt(S)
optimizer = se.Adam(
    lr=0.01,
    dynamic_lr_policy=se.SqrtDecayLrPolicy()
)

# Create the EmbeddingStore
# Note: You MUST provide a score_policy (like "lfu") for dynamic LR to work.
store = se.EmbeddingStore(
    embedding_config=embedding_config,
    initializer_config=se.Zeros(),
    optimizer_config=optimizer,
    score_policy="lfu"
)

# During the training loop, whenever features are queried, 
# their LFU scores will be updated automatically.
keys = np.array([1, 2, 3], dtype=np.int64)
embeddings, lengths = store.lookup_or_insert(keys, dim=128)

# Calculate gradients ...
gradients = np.ones((3, 128), dtype=np.float32)
store.accumulate_gradients(keys, gradients)

# When optimizer_step() is called, 
# the learning rate will automatically scale based on the LFU score of each feature!
store.optimizer_step()
```

## Important Notes

* **Score Policy is Required:** If you specify a dynamic LR policy that depends on scores (like `SqrtDecayLrPolicy`), you **must** also enable a score policy (e.g. `score_policy="lfu"`) in the `EmbeddingStore`. Failing to do so will result in a `ValueError` during initialization.
* **Performance:** The dynamic learning rate calculation introduces minimal overhead. The static/dynamic execution paths are resolved via `constexpr` at compile time to ensure zero overhead for static LR configurations, while dynamic LR computations are fully fused with the highly optimized SIMD vectorization updates (like AVX2/AVX-512).

## Validation

The public behavior is covered by the dynamic-learning-rate test module:

```bash
PYTHONPATH="$PWD" pytest -q sparse_embedding/tests/test_dynamic_lr.py
```
