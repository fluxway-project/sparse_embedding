# coding=utf-8

import jax
import jax.numpy as jnp
from jax import grad, jit
from typing import NamedTuple, Any

# 使用 PyTree 来组织参数和优化器状态
# PyTree 可以是任何嵌套的 dicts, lists, tuples
PyTree = Any


# 1. 定义优化器的状态结构
# 使用 NamedTuple 可以让状态更具可读性
class AdamState(NamedTuple):
    step: int
    m: PyTree  # 一阶矩估计 (moving average of gradient)
    v: PyTree  # 二阶矩估计 (moving average of squared gradient)


# 2. 实现 Adam 优化器类
class Adam:
    def __init__(self, learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def init(self, params: PyTree) -> AdamState:
        """
        根据模型参数的结构，初始化优化器的状态。
        m 和 v 的形状与参数完全相同，初始值为0。
        """
        # jax.tree_util.tree_map 可以将一个函数递归地作用于 PyTree 的每个叶子节点
        m = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        v = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        return AdamState(step=0, m=m, v=v)

    # 使用 JIT 编译 update 函数以获得最高性能
    def update(
        self, grads: PyTree, state: AdamState, params: PyTree
    ) -> tuple[PyTree, AdamState]:
        """
        执行一步 Adam 更新。
        这是一个纯函数，不修改任何输入，只返回新的值。
        """
        # 更新步数
        step = state.step + 1

        # 更新一阶矩 m
        # m_t = b1 * m_{t-1} + (1 - b1) * g_t
        new_m = jax.tree_util.tree_map(
            lambda m, g: self.b1 * m + (1 - self.b1) * g, state.m, grads
        )

        # 更新二阶矩 v
        # v_t = b2 * v_{t-1} + (1 - b2) * g_t^2
        new_v = jax.tree_util.tree_map(
            lambda v, g: self.b2 * v + (1 - self.b2) * jnp.square(g), state.v, grads
        )

        # 计算偏差修正 (Bias Correction)
        m_hat = jax.tree_util.tree_map(lambda m: m / (1 - self.b1**step), new_m)
        v_hat = jax.tree_util.tree_map(lambda v: v / (1 - self.b2**step), new_v)

        # 更新模型参数
        # p_t = p_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
        new_params = jax.tree_util.tree_map(
            lambda p, m, v: p - self.learning_rate * m / (jnp.sqrt(v) + self.eps),
            params,
            m_hat,
            v_hat,
        )

        # 返回更新后的参数和优化器状态
        return new_params, AdamState(step=step, m=new_m, v=new_v)


# 3. 使用示例：拟合一个线性回归问题
if __name__ == "__main__":
    # 创建一个 PRNG Key
    key = jax.random.PRNGKey(0)

    # 生成一些伪数据
    # y = 2x + 1 + noise
    true_w, true_b = 2.0, 1.0
    key, subkey = jax.random.split(key)
    xs = jax.random.normal(subkey, (100, 1))
    noise = jax.random.normal(key, (100, 1)) * 0.5
    ys = xs * true_w + true_b + noise

    # 定义一个简单的线性模型
    def model(params, x):
        return x @ params["w"] + params["b"]

    # 定义损失函数 (MSE)
    def loss_fn(params, x, y):
        preds = model(params, x)
        return jnp.mean((preds - y) ** 2)

    # 初始化模型参数
    key, w_key, b_key = jax.random.split(key, 3)
    params = {
        "w": jax.random.normal(w_key, (1, 1)),
        "b": jax.random.normal(b_key, (1,)),
    }
    print(f"Initial params: w={params['w'].item():.2f}, b={params['b'].item():.2f}")

    # 实例化我们的 Adam 优化器
    optimizer = Adam(learning_rate=0.1)

    # 初始化优化器状态
    opt_state = optimizer.init(params)

    # 定义并 JIT 编译训练步骤
    def train_step(params, opt_state, x, y):
        # 计算损失和梯度
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)

        # 使用优化器更新参数和状态
        new_params, new_opt_state = optimizer.update(grads, opt_state, params)

        return new_params, new_opt_state, loss

    # 开始训练循环
    print("\n--- Starting Training ---")
    for i in range(101):
        params, opt_state, loss = train_step(params, opt_state, xs, ys)
        if i % 10 == 0:
            print(f"Step {i:3d}, Loss: {loss:.4f}")

    print("--- Training Finished ---")
    print(f"Final params:   w={params['w'].item():.2f}, b={params['b'].item():.2f}")
    print(f"True params:    w={true_w:.2f}, b={true_b:.2f}")
