import numpy as np

import sparse_embedding


def test_embedding_table_basic():
    # 1 slot, dimension 128
    slot_dims = {0: 128}
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            slot_dims=slot_dims, num_threads=4, dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
    )

    assert store.num_slots() == 1
    assert store.dim_for_slot(0) == 128

    keys = np.array([1, 2, 3], dtype=np.int64)
    # lookup_or_insert
    vals, lens, mask = store.lookup_or_insert(keys)

    assert vals is not None
    assert lens is not None
    assert mask is not None

    # Check if elements are inserted (mask should indicate all new insertions: 1 or 0 depending on implementation)
    # The mask usually indicates if it was newly inserted.
    mask_np = np.from_dlpack(mask)
    assert mask_np.shape[0] == 3

    # lookup
    vals_lookup, lens_lookup = store.lookup(keys)
    assert vals_lookup is not None


def test_embedding_table_assign():
    slot_dims = {0: 128}
    store = sparse_embedding.EmbeddingStore(
        embedding_config=sparse_embedding.EmbeddingConfig(
            slot_dims=slot_dims, num_threads=4, dtype="float32"
        ),
        initializer_config=sparse_embedding.Zeros(),
    )

    keys = np.array([10, 20], dtype=np.int64)
    values = np.ones((2, 128), dtype=np.float32)

    # Pre-insert using assign with lookup_or_insert implicitly or explicitly
    store.assign(keys, values)

    vals, lens = store.lookup(keys)
    vals_np = np.from_dlpack(vals)
    assert np.allclose(vals_np, 1.0)
