from typing import assert_type

from sparse_embedding._core import EmbeddingStore, checkpoint


def check_native_checkpoint_types(store: EmbeddingStore) -> None:
    assert_type(
        checkpoint.validate_native_checkpoint_chain(
            [["source"]], [[1]], [1], "uint64", {}
        ),
        bool,
    )
    assert_type(
        checkpoint.restore_native_page_images(
            store,
            [],
            0,
            0,
            "uint64",
            page_layout_generation=1,
        ),
        None,
    )
    checkpoint.restore_native_page_images(
        store,
        [],
        0,
        0,
        "uint64",
        {},
        True,
        page_layout_generation=1,
    )
