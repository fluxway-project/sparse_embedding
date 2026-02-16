import jax
import jax.numpy as jnp
from flax import nnx


class CategoricalEmbedding(nnx.Module):
    """A categorical embedding layer.
    
    This layer applies a one-hot encoding to the input categorical features.
    """
    
    def __init__(self, num_categories: int, embedding_size: int, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.num_categories = num_categories
        self.embedding_size = embedding_size
        self.embedding = nnx.Embedding(num_categories, embedding_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the CategoricalEmbedding layer.
        
        Args:
            x: Input tensor of shape (batch_size, num_categories).
            
        Returns:
            Output tensor of shape (batch_size, embedding_size).
        """
        x = self.embedding(x)
        return x


class RankMixerModel(nnx.Module):
    """A neural network model using rank mixing layers.

    This model processes input sequences through multiple rank mixing layers
    and applies mean pooling to produce output representations.
    """

    def __init__(
        self,
        embedding_size: int,
        tokenization_intermediate_size: int,
        num_layers: int,
        num_tokens: int,
        num_heads: int,
        hidden_size: int,
        ffn_intermediate_size: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        """Initialize the RankMixerModel.

        Args:
            embedding_size: Dimension of the input embedding.
            tokenization_intermediate_size: Dimension of the tokenization intermediate layer.
            num_layers: Number of rank mixer layers in the model.
            num_tokens: Number of tokens in the input sequence.
            num_heads: Number of attention heads for token mixing.
            hidden_size: Dimension of the hidden state.
            ffn_intermediate_size: Dimension of the FFN intermediate layer.
            rngs: Random number generators for initialization.
        """
        self.embedding_size = embedding_size
        self.tokenization_intermediate_size = tokenization_intermediate_size
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.ffn_intermediate_size = ffn_intermediate_size

        self.tokenizer = Tokenization(
            embedding_size,
            num_tokens,
            tokenization_intermediate_size,
            hidden_size,
            rngs,
        )
        self.layers = nnx.Sequential(
            *[
                RankMixerLayer(
                    num_tokens, num_heads, hidden_size, ffn_intermediate_size, rngs
                )
                for _ in range(num_layers)
            ]
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the RankMixerModel.

        Args:
            x: Input tensor of shape (batch_size, embedding_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size) after mean pooling.
        """

        # tokenization
        x = self.tokenizer(x)

        # layers
        x = self.layers(x)

        # mean pooling
        x = x.mean(axis=1)

        return x


class Tokenization(nnx.Module):
    """Tokenization layer that transforms embeddings into token representations.

    This layer applies a two-stage MLP transformation: first projects partitions
    of the input embedding to an intermediate dimension, then projects to the
    hidden dimension with GELU activation.
    """

    def __init__(
        self,
        embedding_size: int,
        num_tokens: int,
        intermediate_size: int,
        hidden_size: int,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ) -> None:
        """Initialize the Tokenization layer.

        Args:
            embedding_size: Dimension of the input embedding (must be divisible by num_tokens).
            num_tokens: Number of tokens to partition the embedding into.
            intermediate_size: Dimension of the intermediate layer.
            hidden_size: Dimension of the output hidden state.
            rngs: Random number generators for initialization.

        Raises:
            AssertionError: If embedding_size is not divisible by num_tokens.
        """
        assert (
            embedding_size % num_tokens == 0
        ), f"embedding_size={embedding_size} must be divisible by num_tokens={num_tokens}"
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.partition_size = embedding_size // num_tokens
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # mlp1
        self.linear1 = nnx.Einsum(
            "btd,tdi->bti",
            kernel_shape=(self.num_tokens, self.partition_size, self.intermediate_size),
            bias_shape=(self.num_tokens, self.intermediate_size),
            rngs=rngs,
        )

        # mlp2
        self.linear2 = nnx.Einsum(
            "bti,tih->bth",
            kernel_shape=(self.num_tokens, self.intermediate_size, self.hidden_size),
            bias_shape=(self.num_tokens, self.hidden_size),
            rngs=rngs,
        )

        self.act_fn = nnx.gelu

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply tokenization transformation to the input.

        Args:
            x: Input tensor of shape (batch_size, embedding_size).

        Returns:
            Output tensor of shape (batch_size, num_tokens, hidden_size).
        """
        x = x.reshape(-1, self.num_tokens, self.partition_size)

        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)

        return x


class RankMixerLayer(nnx.Module):
    """A single rank mixing layer combining token mixing and per-token FFN.

    This layer applies multi-head token mixing followed by a per-token feed-forward
    network, with layer normalization and residual connections.
    """

    def __init__(
        self,
        num_tokens: int,
        num_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        """Initialize the RankMixerLayer.

        Args:
            num_tokens: Number of tokens in the input sequence.
            num_heads: Number of attention heads for token mixing.
            hidden_size: Dimension of the hidden state.
            intermediate_size: Dimension of the intermediate FFN layer.
            rngs: Random number generators for initialization.
        """
        assert (
            num_tokens == num_heads
        ), f"num_tokens={num_tokens} must be equal to num_heads={num_heads}"

        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.multi_head_token_mixing = MultiHeadTokenMixing(num_heads, hidden_size)
        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.per_token_ffn = DensePerTokenFFN(
            num_tokens, hidden_size, intermediate_size, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the RankMixerLayer.

        Args:
            x: Input tensor of shape (batch_size, num_tokens, hidden_size).

        Returns:
            Output tensor of shape (batch_size, num_tokens, hidden_size).
        """
        x = self.norm1(x + self.multi_head_token_mixing(x))
        x = self.norm2(x + self.per_token_ffn(x))

        return x


class MultiHeadTokenMixing(nnx.Module):
    """Multi-head token mixing layer for cross-token information exchange.

    This layer reshapes the input to split across multiple heads, transposes
    to mix tokens across heads, and merges the results.
    """

    def __init__(self, num_heads: int, hidden_size: int) -> None:
        """Initialize the MultiHeadTokenMixing layer.

        Args:
            num_heads: Number of attention heads.
            hidden_size: Dimension of the hidden state (must be divisible by num_heads).

        Raises:
            AssertionError: If hidden_size is not divisible by num_heads.
        """
        assert (
            hidden_size % num_heads == 0
        ), f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply multi-head token mixing to the input.

        Args:
            x: Input tensor of shape (batch_size, num_tokens, hidden_size).

        Returns:
            Output tensor of shape (batch_size, num_heads, num_tokens * head_size).
        """
        batch_size, num_tokens, _ = x.shape
        x = (
            x.reshape(batch_size, num_tokens, self.num_heads, self.head_size)  # split
            .transpose(0, 2, 1, 3)  # mix
            .reshape(batch_size, self.num_heads, -1)  # merge
        )
        return x


class DensePerTokenFFN(nnx.Module):
    """A per-token feed-forward network with gated linear units.

    This layer applies separate transformations to each token position using
    a gated architecture with GELU activation.
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        """Initialize the DensePerTokenFFN layer.

        Args:
            num_tokens: Number of token positions.
            hidden_size: Dimension of the input and output hidden state.
            intermediate_size: Dimension of the intermediate layer.
            rngs: Random number generators for initialization.
        """
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_up_proj = nnx.Einsum(
            "btd,tdI->btI",
            kernel_shape=(num_tokens, hidden_size, 2 * intermediate_size),
            bias_shape=(num_tokens, 2 * intermediate_size),
            rngs=rngs,
        )
        self.down_proj = nnx.Einsum(
            "bti,tid->btd",
            kernel_shape=(num_tokens, intermediate_size, hidden_size),
            bias_shape=(num_tokens, hidden_size),
            rngs=rngs,
        )

        self.act_fn = nnx.gelu

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the per-token FFN to the input.

        Args:
            x: Input tensor of shape (batch_size, num_tokens, hidden_size).

        Returns:
            Output tensor of shape (batch_size, num_tokens, hidden_size).
        """

        gate, up = jnp.split(self.gate_up_proj(x), 2, axis=-1)
        x = self.act_fn(gate) * up
        x = self.down_proj(x)

        return x


if __name__ == "__main__":
    from jax import random

    batch_size = 2
    seed = 0

    # Configuration hyperparameters
    config = {
        "embedding_size": 32,
        "tokenization_intermediate_size": 16,
        "num_layers": 8,
        "num_tokens": 4,
        "num_heads": None,  # = num_tokens
        "hidden_size": 8,
        "ffn_intermediate_size": 12,
    }
    config["num_heads"] = config["num_tokens"]

    key = random.key(seed)
    x = random.uniform(key, (batch_size, config["embedding_size"]))

    # x = random.uniform(
    #    key, (config["batch_size"], config["num_tokens"], config["hidden_size"])
    # )
    # x = jnp.arange(config["batch_size"] * config["num_tokens"] * config["hidden_size"]).reshape(
    #     config["batch_size"], config["num_tokens"], config["hidden_size"]
    # )

    model = RankMixerModel(
        config["embedding_size"],
        config["tokenization_intermediate_size"],
        config["num_layers"],
        config["num_tokens"],
        config["num_heads"],
        config["hidden_size"],
        config["ffn_intermediate_size"],
        rngs=nnx.Rngs(seed),
    )
    y = model(x)

    print(x.shape, x)
    print("-" * 10)
    print(y.shape, y)
