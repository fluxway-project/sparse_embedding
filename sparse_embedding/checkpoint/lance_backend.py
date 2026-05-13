import json
import logging
from typing import List

import lance
import pyarrow as pa
import pyarrow.compute as pc
from lance.fragment import FragmentMetadata, write_fragments

from .base import DistributedCheckpointBackend

logger = logging.getLogger(__name__)

EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field("slot_id", pa.uint32()),
        pa.field("key", pa.uint64()),
        pa.field("embedding", pa.list_(pa.float32())),
    ]
)

OPT_STATE_SCHEMA = pa.schema(
    [
        pa.field("slot_id", pa.uint32()),
        pa.field("key", pa.uint64()),
        pa.field("opt_state", pa.list_(pa.float32())),
    ]
)


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.Table.from_arrays(
        [pa.array([], type=f.type) for f in schema],
        schema=schema,
    )


def _build_embedding_batches(store, batch_size: int = 65536) -> List[pa.RecordBatch]:
    batches: List[pa.RecordBatch] = []

    def visitor(array_addr: int, schema_addr: int):
        batches.append(pa.RecordBatch._import_from_c(array_addr, schema_addr))

    store.export_batches_arrow(
        visitor, batch_size=batch_size, include_optimizer_state=False
    )
    return batches


def _build_opt_state_batches(store, batch_size: int = 65536) -> List[pa.RecordBatch]:
    batches: List[pa.RecordBatch] = []

    def visitor(array_addr: int, schema_addr: int):
        batches.append(pa.RecordBatch._import_from_c(array_addr, schema_addr))

    store.export_batches_arrow(
        visitor, batch_size=batch_size, include_optimizer_state=True
    )
    return batches


class LanceCheckpointBackend(DistributedCheckpointBackend):
    """
    Lance-based unified-dataset checkpoint backend.

    Embedding parameters and optimizer states are stored in two separate
    Lance datasets under ``{uri}/embeddings`` and ``{uri}/opt_states``.
    Both datasets share the same ``slot_id`` / ``key`` primary key, with
    variable-length ``pa.list_(pa.float32())`` columns for the data.

    This design keeps the embedding schema constant regardless of optimizer
    type, and allows opt-state to be omitted entirely for inference-only
    deployments.

    Data flow during distributed save:
      1. Driver: ``init_dataset`` — creates empty Lance datasets.
      2. Workers (parallel): ``export_fragments`` — writes Lance fragment
         files and returns serialised ``FragmentMetadata``.
      3. Driver: ``commit_fragments`` — atomically appends all fragments.

    Thread-safety: the underlying C++ ``export_batches_dlpack`` locks each
    shard during iteration.  PyArrow arrays constructed from the DLPack
    capsule are consumed synchronously inside the visitor, so no additional
    Python-side locking is required by default.

    Args:
        use_lock: Reserved for future use.  Currently a no-op.
    """

    def __init__(self, use_lock: bool = False):
        self.use_lock = use_lock

    # ------------------------------------------------------------------
    # CheckpointBackend interface (single-node)
    # ------------------------------------------------------------------

    def save(self, store, uri: str, **kwargs):
        emb_uri = f"{uri}/embeddings"
        opt_uri = f"{uri}/opt_states"

        emb_batches = _build_embedding_batches(store)
        emb_table = (
            pa.Table.from_batches(emb_batches, schema=EMBEDDING_SCHEMA)
            if emb_batches
            else _empty_table(EMBEDDING_SCHEMA)
        )
        lance.write_dataset(emb_table, emb_uri, mode="create")

        opt_batches = _build_opt_state_batches(store)
        if opt_batches:
            opt_table = pa.Table.from_batches(opt_batches, schema=OPT_STATE_SCHEMA)
            lance.write_dataset(opt_table, opt_uri, mode="create")

    def load(self, store, uri: str, filter_fn=None, target_shard_id: int = 0, **kwargs):
        import os

        emb_uri = f"{uri}/embeddings"
        opt_uri = f"{uri}/opt_states"

        _load_embeddings(store, emb_uri, filter_fn, target_shard_id)

        if os.path.exists(opt_uri):
            _load_opt_states(store, opt_uri, filter_fn, target_shard_id)

    # ------------------------------------------------------------------
    # DistributedCheckpointBackend interface (multi-node two-phase commit)
    # ------------------------------------------------------------------

    def init_dataset(self, store, uri: str, **kwargs):
        emb_uri = f"{uri}/embeddings"
        opt_uri = f"{uri}/opt_states"
        lance.write_dataset(_empty_table(EMBEDDING_SCHEMA), emb_uri, mode="create")
        lance.write_dataset(_empty_table(OPT_STATE_SCHEMA), opt_uri, mode="create")

    def export_fragments(self, store, uri: str, **kwargs) -> List[bytes]:
        emb_uri = f"{uri}/embeddings"
        opt_uri = f"{uri}/opt_states"
        result: List[bytes] = []

        emb_batches = _build_embedding_batches(store)
        if emb_batches:
            emb_table = pa.Table.from_batches(emb_batches, schema=EMBEDDING_SCHEMA)
            for frag in write_fragments(emb_table, emb_uri):
                result.append(f"emb:{json.dumps(frag.to_json())}".encode())

        opt_batches = _build_opt_state_batches(store)
        if opt_batches:
            opt_table = pa.Table.from_batches(opt_batches, schema=OPT_STATE_SCHEMA)
            for frag in write_fragments(opt_table, opt_uri):
                result.append(f"opt:{json.dumps(frag.to_json())}".encode())

        return result

    def commit_fragments(
        self, uri: str, all_fragment_metas: List[List[bytes]], **kwargs
    ):
        emb_uri = f"{uri}/embeddings"
        opt_uri = f"{uri}/opt_states"
        emb_frags: List[FragmentMetadata] = []
        opt_frags: List[FragmentMetadata] = []

        for rank_metas in all_fragment_metas:
            for meta_bytes in rank_metas:
                decoded = meta_bytes.decode()
                if decoded.startswith("emb:"):
                    emb_frags.append(FragmentMetadata.from_json(decoded[4:]))
                else:
                    opt_frags.append(FragmentMetadata.from_json(decoded[4:]))

        if emb_frags:
            ds = lance.dataset(emb_uri)
            op = lance.LanceOperation.Append(emb_frags)
            lance.LanceDataset.commit(emb_uri, op, read_version=ds.version)

        if opt_frags:
            ds = lance.dataset(opt_uri)
            op = lance.LanceOperation.Append(opt_frags)
            lance.LanceDataset.commit(opt_uri, op, read_version=ds.version)


def _apply_filter(keys_arr: pa.Array, filter_fn, target_shard_id: int):
    """
    Apply filter_fn to keys_arr and return a pa.BooleanArray mask.
    filter_fn can be a Partitioner with .filter() or a callable returning shard ids.
    Returns None when no filter is needed.
    """
    if filter_fn is None:
        return None

    import numpy as np

    if hasattr(filter_fn, "filter"):
        mask_np = np.from_dlpack(filter_fn.filter(keys_arr, target_shard_id))
    else:
        shard_ids_np = np.from_dlpack(filter_fn(keys_arr))
        mask_np = shard_ids_np == target_shard_id

    return pa.array(mask_np, type=pa.bool_())


def _load_embeddings(store, uri: str, filter_fn, target_shard_id: int):
    ds = lance.dataset(uri)
    for slot_id in store.slot_ids():
        scanner = ds.scanner(filter=f"slot_id = {slot_id}")
        for batch in scanner.to_batches():
            keys_arr = batch["key"]
            if len(keys_arr) == 0:
                continue

            emb_arr = batch["embedding"]
            mask = _apply_filter(keys_arr, filter_fn, target_shard_id)

            if mask is not None:
                keys_arr = pc.filter(keys_arr, mask)
                emb_arr = pc.filter(emb_arr, mask)

            if len(keys_arr) == 0:
                continue

            flat_vals = pc.list_flatten(emb_arr)
            store.assign(keys_arr, flat_vals)


def _load_opt_states(store, uri: str, filter_fn, target_shard_id: int):
    ds = lance.dataset(uri)
    for slot_id in store.slot_ids():
        scanner = ds.scanner(filter=f"slot_id = {slot_id}")
        for batch in scanner.to_batches():
            keys_arr = batch["key"]
            if len(keys_arr) == 0:
                continue

            opt_arr = batch["opt_state"]
            mask = _apply_filter(keys_arr, filter_fn, target_shard_id)

            if mask is not None:
                keys_arr = pc.filter(keys_arr, mask)
                opt_arr = pc.filter(opt_arr, mask)

            if len(keys_arr) == 0:
                continue

            flat_opt = pc.list_flatten(opt_arr)
            store.assign_opt_state(slot_id, keys_arr, flat_opt)
