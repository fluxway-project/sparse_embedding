# coding=utf-8
"""
SPMD training loop using shard_map (replacing pmap).

Architecture:
- ID feature: Independent SparseEmbedding (CPU KVStore) with all_to_all
- Regular features (1-9): Part of model parameters or independent JAX embeddings
- Both forward and backward support all_to_all via shard_map
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from flax import nnx

try:
    from jax.experimental import mesh_utils
    from jax.experimental.shard_map import shard_map
except ImportError:
    # Fallback for older JAX versions
    try:
        from jax.sharding import Mesh, PartitionSpec
        from jax.experimental.shard_map import shard_map
    except ImportError:
        raise ImportError(
            "shard_map requires JAX >= 0.4.0. "
            "Install with: pip install --upgrade jax jaxlib"
        )

from .sparse_embedding import SparseEmbedding


@dataclass
class ModelParams:
    """Model parameters including regular feature embeddings."""
    regular_embeddings: Dict[int, jnp.ndarray]  # feature_idx -> [vocab_size, emb_dim]
    dense_layers: Dict[str, jnp.ndarray]  # layer_name -> weights/biases


class DenseEmbedding(nnx.Module):
    """
    Regular feature embeddings (features 1-9) as JAX parameters.
    Can be part of model or managed independently.
    """
    
    def __init__(
        self,
        feature_metas: Dict[str, int],  # code -> vocab_size
        embed_dims: Dict[str, int],  # code -> embedding_dim
        rng_key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        self.feature_metas = feature_metas
        self.embed_dims = embed_dims
        
        # Initialize embeddings
        self.embeddings = {}
        for code in feature_metas.keys():
            vocab_size = feature_metas[code]
            emb_dim = embed_dims[code]
            key, rng_key = jax.random.split(rng_key)
            self.embeddings[code] = nnx.Embeddings(vocab_size, emb_dim, rngs=rng_key)
    
    def __call__(
        self,
        regular_ids: Dict[str, jnp.ndarray],  # feature_idx -> [B] int32
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass for regular features using vmap for parallel lookup.
        
        Returns: Dict[feature_idx, [B, emb_dim]]
        """
        codes = sorted(regular_ids.keys())
        if not codes:
            return {}
        
        # Stack embedding tables: [num_features, vocab_size, emb_dim]
        emb_tables = jnp.stack([
            self.embeddings[code] for code in codes
        ])
        # Stack IDs: [num_features, batch_size]
        ids_stack = jnp.stack([regular_ids[code] for code in codes])
        
        # Vectorized lookup: vmap over features dimension
        def lookup_single_feat(emb_table, ids):
            return emb_table[ids]  # [batch_size, emb_dim]
        
        # vmap over features: [num_features, batch_size, emb_dim]
        embs_stack = jax.vmap(lookup_single_feat, in_axes=(0, 0))(emb_tables, ids_stack)
        
        # Convert back to dict
        return {code: embs_stack[i] for i, code in enumerate(codes)}


class DenseModel(nnx.Module):
    """
    Hybrid model with:
    - ID feature: Independent SparseEmbedding (managed separately)
    - Regular features: JAX parameter embeddings (part of model)
    """
    
    def __init__(
        self,
        id_embedding_dim: int,
        regular_vocab_sizes: Dict[str, int],
        regular_embed_dims: Dict[str, int],
        rng_key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        """
        Args:
            id_slot: slot ID for ID feature
            id_dim: embedding dimension for ID feature
            regular_vocab_sizes: feature_idx (1-9) -> vocab_size
            regular_embed_dims: feature_idx (1-9) -> embedding_dim
            hidden_dims: hidden layer dimensions for MLP
        """
        
        # Regular features: Part of model parameters
        self.regular_emb_layer = DenseEmbedding(
            regular_vocab_sizes, regular_embed_dims, rng_key=rng_key
        )
        
        # MLP layers
        key, rng_key = jax.random.split(rng_key)
        total_emb_dim = id_embedding_dim + sum(regular_embed_dims.values())
        
        self.mlp = nnx.Linear(total_emb_dim, 1, rngs=rng_key)
    
    def forward(
        self,
        id_emb: jnp.ndarray,
        regular_ids: Dict[int, jnp.ndarray],  # feature_idx -> [B] int32
    ) -> jnp.ndarray:
        """
        Forward pass with SPMD support.
        
        Returns: [B, 1] predictions
        """
        # Regular feature embeddings (already uses vmap internally)
        regular_embs = self.regular_emb_layer(regular_ids)
        
        # Concatenate all embeddings
        all_embs = [id_emb] + [regular_embs[i] for i in sorted(regular_ids.keys())]
        x = jnp.concatenate(all_embs, axis=-1)  # [B, total_dim]
        
        # MLP forward
        x = self.mlp(x)
        
        return x


class Trainer:

    def __init__(self, model, 
                 learning_rate: float = 1e-3):
        self.optimizer = optax.adam(learning_rate)
        # dense model
        self.model = model

        # id model
        self.id_slot = id_slot
        self.id_dim = id_dim
        # ID feature: Independent SparseEmbedding
        id_feature_metas = {id_slot: id_dim}
        id_default_values = {id_slot: [0.0] * id_dim}
        self.id_embedding = SparseEmbedding(
            id_feature_metas, id_default_values, num_shards=num_shards
        )

    
    def train_step(self,
        id_keys: jnp.ndarray,
        regular_ids: Dict[int, jnp.ndarray],
        labels: jnp.ndarray,
    ):
        """
        Single training step with shard_map.
        
        Args:
            params: Model parameters
            id_keys: [B] ID feature keys
            regular_ids: Dict[feat_idx, [B]] regular feature IDs
            labels: [B, 1] labels
        
        Returns:
            (loss, updated_params)
        """
        
        def loss_fn(id_emb, reg_ids, lab):
            # dense model
            x = self.model.forward(id_emb, reg_ids)
            logtis = optax.softmax_cross_entropy_with_integer_labels(x, lab)
            return logtis
        
        # id model
        id_emb = self.id_embedding.forward(id_keys)

        grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        grads = grad_fn(id_emb, regular_ids, labels)

        self.model.optimizer.update(self.model, grads)
        
        # id model
        self.id_embedding.backward(id_keys, grads)


def create_mesh_and_specs(num_devices: int):
    """
    Create mesh and partition specs for shard_map.
    
    Args:
        num_devices: Number of devices
    
    Returns:
        (mesh, in_specs, out_specs)
    """
    devices = jax.devices()[:num_devices]
    
    # Create 1D mesh
    mesh = mesh_utils.create_device_mesh((num_devices,))
    mesh = jax.sharding.Mesh(mesh, axis_names=["devices"])
    
    # Partition specs
    # Batch dimension is sharded across devices
    in_specs = (
        None,  # params (replicated)
        jax.sharding.PartitionSpec("devices"),  # id_keys
        {k: jax.sharding.PartitionSpec("devices") for k in range(1, 10)},  # regular_ids
        jax.sharding.PartitionSpec("devices"),  # labels
    )
    
    out_specs = (
        jax.sharding.PartitionSpec("devices"),  # loss
        None,  # updated_params (replicated)
    )
    
    return mesh, in_specs, out_specs


def train_with_shard_map(
    model,
    data_loader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    num_devices: Optional[int] = None,
):
    """
    Main training loop using shard_map.
    
    Args:
        model: HybridModel instance
        data_loader: Data loader iterator
        num_epochs: Number of epochs
        learning_rate: Learning rate
        num_devices: Number of devices (auto-detect if None)
    """
    devices = jax.devices()
    if num_devices is None:
        num_devices = len(devices)
    
    print(f"Training on {num_devices} devices: {devices[:num_devices]}")
    
    # Create mesh and specs
    mesh, in_specs, out_specs = create_mesh_and_specs(num_devices)
    
    trainer = Trainer(model, learning_rate=learning_rate)
    # Create training step
    train_step_fn = trainer.train_step
    
    # Wrap with shard_map
    @shard_map(
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )
    def sharded_train_step(params, id_keys, regular_ids, labels):
        return train_step_fn(params, id_keys, regular_ids, labels)
    
    # Initialize parameters
    params = model.get_params()
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for step, batch in enumerate(data_loader):
            # Prepare data with proper sharding
            id_keys = jax.device_put(
                batch["id_keys"],
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices")),
            )
            regular_ids = {
                k: jax.device_put(
                    v,
                    jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices")),
                )
                for k, v in batch["regular_ids"].items()
            }
            labels = jax.device_put(
                batch["labels"],
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices")),
            )
            
            # Replicate params across devices
            params_replicated = jax.device_put(
                params,
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
            )
            
            # Run training step
            loss, updated_params = sharded_train_step(
                params_replicated, id_keys, regular_ids, labels
            )
            
            # Update params (take from device 0 or average)
            params = jax.tree_map(lambda x: x[0], updated_params)
            
            epoch_losses.append(float(loss[0]))
            
            if step % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Step {step}, Loss: {loss[0]:.6f}"
                )
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.6f}")
        
        # Update model
        model.update_params(params)


def create_data_loader(
    batch_size: int,
    num_features: int = 10,
    id_range: int = 1_000_000,
    regular_vocab_sizes: Dict[int, int] = None,
    num_batches: int = 100,
):
    """
    Create a dummy data loader for testing.
    """
    if regular_vocab_sizes is None:
        regular_vocab_sizes = {i: 10_000 for i in range(1, 10)}
    
    def _generate_batch():
        # ID feature: random uint64 keys
        id_slot = 121
        slot_prefix = np.uint64(id_slot) << np.uint64(48)
        local_ids = np.random.randint(0, id_range, size=batch_size, dtype=np.uint64)
        id_keys = (slot_prefix | local_ids).astype(np.uint64)
        
        # Regular features: random int32 IDs
        regular_ids = {
            i: np.random.randint(0, regular_vocab_sizes[i], size=batch_size, dtype=np.int32)
            for i in range(1, 10)
        }
        
        # Labels: random float32
        labels = np.random.randn(batch_size, 1).astype(np.float32)
        
        return {
            "id_keys": id_keys,
            "regular_ids": regular_ids,
            "labels": labels,
        }
    
    for _ in range(num_batches):
        yield _generate_batch()


if __name__ == "__main__":
    # Example usage
    print("Shard Map Trainer for Hybrid Embedding Model")
    print("=" * 50)
    
    # Configuration
    id_slot = 121
    id_dim = 64
    regular_vocab_sizes = {i: 10_000 for i in range(1, 10)}
    regular_embed_dims = {i: 32 for i in range(1, 10)}
    hidden_dims = [128, 64]

    jax.distributed.initialize()
    rank = jax.process_index()      # 当前主机编号
    local_devices = jax.local_devices()
    total_devices = jax.device_count()

    print(f"Rank {rank} initialized. Total devices: {total_devices}")
    
    # Create model
    model = DenseModel(
        regular_vocab_sizes=regular_vocab_sizes,
        regular_embed_dims=regular_embed_dims,
        hidden_dims=hidden_dims,
    )
    
    # Create data loader
    data_loader = create_data_loader(
        batch_size=128,
        regular_vocab_sizes=regular_vocab_sizes,
        num_batches=100,
    )
    
    # Train
    train_with_shard_map(
        model=model,
        data_loader=data_loader,
        num_epochs=5,
        learning_rate=1e-3,
    )
