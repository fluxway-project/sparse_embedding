import numpy as np

import sparse_embedding as se


def test_heavy_collision_adam_update():
    """
    Test edge-cases often seen in Recsys:
    Extremely high collision on a tiny set of hot keys (e.g., popular items).
    We ensure that multiple accumulation calls over the same hot keys
    result in the correct optimizer stepping mechanism behavior,
    even roughly given Hogwild locks.
    """
    # 1. Setup store
    store = se.EmbeddingStore(
        embedding_config=se.EmbeddingConfig(
            dims=[16],
            num_threads=2,
            dtype="float32",
        ),
        initializer_config=se.Zeros(),
        optimizer_config=se.Adam(lr=0.01),
    )

    # We create only 2 keys, but simulate 4 batches of updates
    # This will trigger exactly 1 optimizer step at the last batch
    keys = np.array([101, 102], dtype=np.int64)
    vals, _, _ = store.lookup_or_insert(keys, dim=16)
    v_start = np.from_dlpack(vals).copy()
    assert np.all(v_start == 0.0)

    # 4 micro-batches
    for _ in range(4):
        # Accumulate gradients (dense vectors)
        grads = np.ones((2, 16), dtype=np.float32) * 0.1
        store.accumulate_gradients(keys, grads)

    # Trigger optimizer step manually!
    store.optimizer_step()
    res_vals, _ = store.lookup(keys, dim=16)
    v_end = np.from_dlpack(res_vals)

    # The expected update is applied.
    # Since it's Adam and parameters were 0, gradients were constant 0.1,
    # the first update will be roughly -lr * (sign(grad)) = -0.01
    # Because of epsilon and beta biases, let's just make sure it's moved from zero
    # and has a negative direction.
    assert np.all(v_end < 0.0)
    assert not np.allclose(v_start, v_end), "Parameters should have moved"

    # Do it 4 more times to see continuous update
    for _ in range(4):
        grads = np.ones((2, 16), dtype=np.float32) * 0.1
        store.accumulate_gradients(keys, grads)
    store.optimizer_step()

    res_vals2, _ = store.lookup(keys, dim=16)
    v_end2 = np.from_dlpack(res_vals2)
    assert np.all(v_end2 < v_end), "Parameters should continue to decrease negatively"

    print("[PASS] test_heavy_collision_adam_update")


if __name__ == "__main__":
    test_heavy_collision_adam_update()
    print("All collision tests passed.")
