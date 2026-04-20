"""
与 pipeline-rocksdb.py 相同的 Ray + JAX 流水线训练逻辑；
区别：Embedding 表使用 C++ EmbeddingTable（nanobind），高性能内存 KV 存储。

运行: python pipeline-memdict.py（需先构建 emb_table_ext：cd pipeline/build && cmake .. && cmake --build . && cp emb_table_ext*.so ..）
对照: pipeline-rocksdb.py（落盘、可承载超内存表）
"""

import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
import emb_table_ext

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
import jax
import jax.numpy as jnp
from jax import jit
import ray
import threading
import time
import numpy as np

# ============================================================
# 架构说明（参考图示）
#
#   DP 0 Dispatcher ──┐
#   DP 1 Dispatcher ──┼──▶  GPU Embedding Server  ──▶  GPU Worker
#   DP x Dispatcher ──┘   (Stage 1: 稀疏查表)     ◀──  (Stage 2: Dense计算)
#                                                        ↑ AllReduce
#   核心设计：
#   1. Dispatcher 独立于 GPU，CPU 部署，Cache-miss 友好
#   2. Embedding Server 与 Dense Worker 解耦成两个流水线 Stage
#      - "共卡问题" → 变成"是否共用 GPU 资源"的调度问题
#   3. Pipeline 重叠：Dense Worker 处理 Step N 时，
#                     Embedding Server 已在预取 Step N+1
#   灵感来源：大模型 MoE 大 EP 训练 + Attention-FeedForward 分离
# ============================================================


# ============================================================
# 1. Dispatcher / Preprocessor Actor
#    - 不与 GPU Worker 共卡，部署在 CPU 节点
#    - 负责原始特征预处理、ID 分组，Cache-miss 友好
#    - 每个 DP 分支有独立的 Dispatcher（DP 0 / DP 1 / ... / DP x）
# ============================================================
@ray.remote
class DispatcherActor:
    """
    数据分发器 / 预处理器（Pipeline 入口）
    - 不绑定 GPU，可部署在 CPU 节点，节省 GPU 资源
    - 负责稀疏 ID 预处理、批次构建
    - Round-Robin 分发到多个 Embedding Server（负载均衡）
    """

    def __init__(self, dispatcher_id: int, num_emb_servers: int):
        self.dispatcher_id = dispatcher_id
        self.num_emb_servers = num_emb_servers
        self.dispatch_count = 0
        print(f"[Dispatcher {dispatcher_id}] 初始化完成（CPU 节点，不占 GPU 资源）")

    def preprocess(self, batch_id: int, raw_ids: list, x_dense: list, y_true: list):
        """
        CPU 端预处理：
        - ID 格式校验
        - 目标 Embedding Server 路由（Round-Robin 负载均衡）
        返回: 预处理好的 batch，供 Embedding Server 消费
        """
        self.dispatch_count += 1
        target_server = batch_id % self.num_emb_servers  # Round-Robin
        return {
            "batch_id": batch_id,
            "dispatcher_id": self.dispatcher_id,
            "target_server": target_server,
            "ids": jnp.array(raw_ids, dtype=jnp.int32),
            "x_dense": jnp.array(x_dense, dtype=jnp.float32),
            "y_true": jnp.array(y_true, dtype=jnp.float32),
        }

    def get_stats(self):
        return {
            "dispatcher_id": self.dispatcher_id,
            "dispatch_count": self.dispatch_count,
        }


# ============================================================
# 2. GPU Embedding Server Actor - Pipeline Stage 1
#    - 稀疏 Embedding 存 C++ EmbeddingTable（nanobind）；rocksdb 版见 pipeline-rocksdb.py
# ============================================================
class AsyncEmbeddingTable:
    """把 C++ EmbeddingTable 的阻塞调用封装为 async 协程接口。"""

    def __init__(self, embedding_dim: int):
        self._db = emb_table_ext.EmbeddingTable(embedding_dim)
        # 单线程执行器保证同一 actor 内调用顺序与状态一致性
        self._executor = ThreadPoolExecutor(max_workers=1)

    def insert(self, key: int, vec):
        self._db.insert(key, vec)

    async def batch_lookup(self, ids_np):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._db.batch_lookup, ids_np)

    async def batch_sgd_update(self, ids_np, grads_np, lr: float):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._db.batch_sgd_update, ids_np, grads_np, lr
        )

    async def get(self, key: int):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._db.get, key)


@ray.remote
class EmbeddingServerActor:
    """
    GPU Embedding Server（流水线 Stage 1）
    - 参数服务器（PS）模式管理 Sparse Embedding
    - Unique + Gather 减少冗余通信量
    - 保存 forward 中间态，供 backward Scatter-Add 使用
    """

    def __init__(
        self,
        server_id: int,
        num_servers: int,
        num_features: int,
        embedding_dim: int,
        seed: int = 42,
    ):
        self.server_id = server_id
        self.num_servers = num_servers
        self.embedding_dim = embedding_dim

        self.db = AsyncEmbeddingTable(embedding_dim)

        self._num_features = 0
        for fid in range(num_features):
            if fid % num_servers == server_id:
                vec = jax.random.normal(
                    jax.random.PRNGKey(fid + seed), (embedding_dim,)
                )
                # C++ insert 优先 __dlpack__ → PyCapsule → DLManagedTensor*，不经 nb::ndarray 转换层。
                self.db.insert(fid, vec)
                self._num_features += 1

        self._pipeline_buf: dict = {}
        self._lock = threading.Lock()

        self.forward_count = 0
        self.backward_count = 0
        print(
            f"[Embedding Server {server_id}] 初始化（C++ EmbeddingTable）: "
            f"负责 {self._num_features}/{num_features} 个特征"
        )

    async def forward(self, batch_id: int, ids):
        with self._lock:
            self.forward_count += 1

        unique_ids, inverse_indices = jnp.unique(ids, return_inverse=True)
        ids_np = np.array(unique_ids, dtype=np.int32)
        capsules = await self.db.batch_lookup(ids_np)
        unique_weights = jnp.stack([jnp.from_dlpack(c) for c in capsules])
        batch_emb = unique_weights[inverse_indices]

        with self._lock:
            self._pipeline_buf[batch_id] = {
                "unique_ids": unique_ids,
                "inverse_indices": inverse_indices,
            }

        return {
            "batch_id": batch_id,
            "server_id": self.server_id,
            "batch_emb": batch_emb,
            "num_unique": int(len(unique_ids)),
        }

    async def backward(self, batch_id: int, batch_emb_grads, lr: float = 0.01):
        with self._lock:
            self.backward_count += 1
            buf = self._pipeline_buf.pop(batch_id, None)

        if buf is None:
            return {"ok": False, "reason": f"batch_id={batch_id} 不在 buffer 中"}

        unique_ids = buf["unique_ids"]
        inverse_indices = buf["inverse_indices"]

        unique_grads = jnp.zeros((len(unique_ids), self.embedding_dim))
        unique_grads = unique_grads.at[inverse_indices].add(batch_emb_grads)

        ids_np = np.array(unique_ids, dtype=np.int32)
        grads_np = np.array(unique_grads, dtype=np.float32)
        await self.db.batch_sgd_update(ids_np, grads_np, lr)

        return {
            "ok": True,
            "batch_id": batch_id,
            "updated": int(len(unique_ids)),
        }

    async def get_embedding(self, idx: int):
        vec_np = await self.db.get(int(idx))
        return jnp.array(vec_np)

    def get_stats(self):
        return {
            "server_id": self.server_id,
            "forward_count": self.forward_count,
            "backward_count": self.backward_count,
            "num_features": self._num_features,
            "pipeline_buf_size": len(self._pipeline_buf),
        }


# ============================================================
# 3. Dense Worker Actor - Pipeline Stage 2
# ============================================================
@ray.remote
class DenseWorkerActor:
    """GPU Dense Worker（流水线 Stage 2）"""

    def __init__(
        self, worker_id: int, num_workers: int, embedding_dim: int, dense_dim: int
    ):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.embedding_dim = embedding_dim
        self.dense_dim = dense_dim

        self.params = {
            "w": jax.random.normal(
                jax.random.PRNGKey(worker_id * 100), (embedding_dim + dense_dim, 1)
            ),
            "b": jnp.zeros((1,)),
        }

        self.compute_count = 0
        self._lock = threading.Lock()
        self._fwd_bwd_jit = jit(self._fwd_bwd_fn)

        print(
            f"[Dense Worker {worker_id}] 初始化完成"
            f"（数据并行 x{num_workers}，Pipeline Stage 2）"
        )

    def _fwd_bwd_fn(self, params, batch_emb, x_dense, y_true):
        def loss_fn(p, emb, x, y):
            features = jnp.concatenate([emb, x], axis=-1)
            pred = jnp.dot(features, p["w"]) + p["b"]
            return jnp.mean((pred - y) ** 2)

        loss, (grads_p, grads_emb) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
            params, batch_emb, x_dense, y_true
        )
        return loss, grads_p, grads_emb

    def compute(self, emb_result: dict, x_dense, y_true, lr: float = 0.01):
        with self._lock:
            self.compute_count += 1

        batch_id = emb_result["batch_id"]
        server_id = emb_result["server_id"]
        batch_emb = emb_result["batch_emb"]

        loss, dense_grads, batch_emb_grads = self._fwd_bwd_jit(
            self.params, batch_emb, x_dense, y_true
        )

        self.params = jax.tree_util.tree_map(
            lambda p, g: p - lr * g, self.params, dense_grads
        )

        return {
            "batch_id": batch_id,
            "server_id": server_id,
            "worker_id": self.worker_id,
            "loss": float(loss),
            "batch_emb_grads": batch_emb_grads,
        }

    def get_params(self):
        return self.params

    def get_stats(self):
        return {
            "worker_id": self.worker_id,
            "compute_count": self.compute_count,
        }


def main():
    print("=" * 70)
    print("Pipeline（C++ EmbeddingTable）：Embedding Server ↔ Dense Worker")
    print("=" * 70)
    ray.init(ignore_reinit_error=True)

    num_features = 100
    embedding_dim = 4
    dense_dim = 2
    num_dispatchers = 2
    num_emb_servers = 2
    num_dense_workers = 4
    num_steps = 12

    dispatchers = [
        DispatcherActor.remote(
            dispatcher_id=i,
            num_emb_servers=num_emb_servers,
        )
        for i in range(num_dispatchers)
    ]
    ray.get([d.get_stats.remote() for d in dispatchers])

    emb_servers = [
        EmbeddingServerActor.remote(
            server_id=i,
            num_servers=num_emb_servers,
            num_features=num_features,
            embedding_dim=embedding_dim,
        )
        for i in range(num_emb_servers)
    ]
    ray.get([s.get_stats.remote() for s in emb_servers])

    dense_workers = [
        DenseWorkerActor.remote(
            worker_id=i,
            num_workers=num_dense_workers,
            embedding_dim=embedding_dim,
            dense_dim=dense_dim,
        )
        for i in range(num_dense_workers)
    ]
    ray.get([w.get_stats.remote() for w in dense_workers])

    raw_batches = []
    for i in range(num_steps):
        ids = [(i * 7 + j * 3) % num_features for j in range(6)]
        x_dense = [[float((i + j) % 3) * 0.4 + 0.2] * dense_dim for j in range(6)]
        y_true = [[float((i + j) % 2) * 0.8 + 0.1] for j in range(6)]
        raw_batches.append({"ids": ids, "x_dense": x_dense, "y_true": y_true})

    print("\n开始 Pipeline 训练（Embedding 表 = C++ EmbeddingTable）\n")

    t0 = time.time()

    preproc_futures = []
    for step in range(num_steps):
        dp_id = step % num_dispatchers
        b = raw_batches[step]
        future = dispatchers[dp_id].preprocess.remote(
            step, b["ids"], b["x_dense"], b["y_true"]
        )
        preproc_futures.append(future)

    emb_futures = []
    for step, preproc_f in enumerate(preproc_futures):
        batch = ray.get(preproc_f)
        srv_id = batch["target_server"]
        emb_f = emb_servers[srv_id].forward.remote(step, batch["ids"])
        emb_futures.append(
            {
                "step": step,
                "srv_id": srv_id,
                "dp_id": step % num_dispatchers,
                "x_dense": batch["x_dense"],
                "y_true": batch["y_true"],
                "emb_f": emb_f,
            }
        )

    compute_futures = []
    for item in emb_futures:
        emb_result = ray.get(item["emb_f"])
        wk_id = item["step"] % num_dense_workers
        compute_f = dense_workers[wk_id].compute.remote(
            emb_result, item["x_dense"], item["y_true"]
        )
        compute_futures.append({**item, "wk_id": wk_id, "compute_f": compute_f})

    backward_futures = []
    for item in compute_futures:
        result = ray.get(item["compute_f"])
        bwd_f = emb_servers[result["server_id"]].backward.remote(
            result["batch_id"], result["batch_emb_grads"]
        )
        backward_futures.append({**item, "result": result, "bwd_f": bwd_f})

    t1 = time.time()

    total_loss = 0.0
    for item in backward_futures:
        bwd = ray.get(item["bwd_f"])
        r = item["result"]
        total_loss += r["loss"]

    avg_loss = total_loss / num_steps
    print(f"平均 Loss: {avg_loss:.4f} | 耗时: {t1 - t0:.3f}s")

    for fid in [0, 1, 7, 14]:
        sid = fid % num_emb_servers
        emb = ray.get(emb_servers[sid].get_embedding.remote(fid))
        print(f"  ID {fid:3d} → Server {sid}: {emb}")

    ray.shutdown()
    print("Ray 已关闭")


if __name__ == "__main__":
    main()
