import hashlib
import json
import logging
import struct
from typing import List, Union

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from lance.fragment import FragmentMetadata, write_fragments

from .base import DistributedCheckpointBackend

logger = logging.getLogger(__name__)

ENTRY_SCHEMA = pa.schema(
    [
        pa.field("dim", pa.uint32()),
        pa.field("key", pa.int64()),
        pa.field("physical_page_id", pa.uint64()),
        pa.field("slot_ordinal", pa.uint32()),
        pa.field("seq_count", pa.uint64()),
        pa.field("vector_stride", pa.uint32()),
        pa.field("embedding_params", pa.list_(pa.float32())),
        pa.field("score", pa.uint64()),
        pa.field("optimizer_states", pa.list_(pa.float32())),
        pa.field("optimizer_state_stride", pa.uint32()),
        pa.field("step", pa.uint32()),
        pa.field("snapshot_id", pa.uint64()),
        pa.field("worker_id", pa.uint32()),
        pa.field("shard_id", pa.uint32()),
        pa.field("operation", pa.uint8()),
        pa.field("shard_sequence", pa.uint64()),
    ]
)

PAGE_IMAGES_SCHEMA = pa.schema(
    [
        pa.field("dim", pa.uint32()),
        pa.field("physical_page_id", pa.uint64()),
        pa.field("vector_dim", pa.uint32()),
        pa.field("vector_stride", pa.uint32()),
        pa.field("slot_ordinals", pa.list_(pa.uint32())),
        pa.field("keys", pa.list_(pa.uint64())),
        pa.field("seq_counts", pa.list_(pa.uint64())),
        pa.field("values", pa.list_(pa.float32())),
        pa.field("scores", pa.list_(pa.uint64())),
        pa.field("optimizer_states", pa.list_(pa.float32())),
        pa.field("optimizer_state_stride", pa.uint32()),
        pa.field("steps", pa.list_(pa.uint32())),
        pa.field("checksum", pa.binary(32)),
        pa.field("snapshot_id", pa.uint64()),
        pa.field("worker_id", pa.uint32()),
        pa.field("shard_ids", pa.list_(pa.uint32())),
        pa.field("page_operation", pa.uint8()),
        pa.field("page_ref_checksum", pa.binary(32)),
    ]
)

PARTITIONED_MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.uint32()),
        pa.field("snapshot_id", pa.uint64()),
        pa.field("worker_count", pa.uint32()),
        pa.field("shard_count", pa.uint32()),
        pa.field("entries_version", pa.uint64()),
        pa.field("page_images_version", pa.uint64()),
        pa.field("parent_snapshot_id", pa.uint64()),
        pa.field("shard_high_watermarks", pa.list_(pa.uint64())),
        pa.field("checkpoint_mode", pa.string()),
        pa.field("parent_uri", pa.string()),
    ]
)

_FORMAT_VERSION = 5
_MANIFEST_DATASET = "snapshot_manifest"
_ENTRIES_DATASET = "entries"
_PAGE_IMAGES_DATASET = "page_images"


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.Table.from_arrays(
        [pa.array([], type=f.type) for f in schema],
        schema=schema,
    )


def _reject_legacy_kwargs(kwargs: dict) -> None:
    legacy = {"snapshot_epoch", "parent_snapshot_epoch"}
    used = sorted(legacy.intersection(kwargs))
    if used:
        raise TypeError(
            "unsupported legacy checkpoint keyword(s): "
            + ", ".join(used)
            + "; use snapshot_id and parent_snapshot_id"
        )


def _unsigned(value, name: str, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < 0 or parsed >= 1 << bits:
        raise ValueError(f"{name} is outside unsigned {bits}-bit range")
    return parsed


def _page_array(image: dict, name: str, dtype, expected_count: int | None = None):
    if name not in image:
        raise ValueError(f"page image is missing {name}")
    array = np.asarray(image[name], dtype=dtype).reshape(-1)
    if expected_count is not None and array.size != expected_count:
        raise ValueError(
            f"page image {name} has {array.size} values; expected {expected_count}"
        )
    return array


def _page_image_checksum(
    dim: int,
    physical_page_id: int,
    vector_dim: int,
    vector_stride: int,
    optimizer_state_stride: int,
    slot_ordinals: np.ndarray,
    keys: np.ndarray,
    seq_counts: np.ndarray,
    values: np.ndarray,
    scores: np.ndarray,
    optimizer_states: np.ndarray,
    steps: np.ndarray,
) -> bytes:
    digest = hashlib.sha256()
    digest.update(
        struct.pack(
            "<IQIIII",
            dim,
            physical_page_id,
            vector_dim,
            vector_stride,
            optimizer_state_stride,
            slot_ordinals.size,
        )
    )
    for array, dtype in (
        (slot_ordinals, "<u4"),
        (keys, "<u8"),
        (seq_counts, "<u8"),
        (values, "<f4"),
        (scores, "<u8"),
        (optimizer_states, "<f4"),
        (steps, "<u4"),
    ):
        digest.update(np.asarray(array, dtype=dtype).tobytes(order="C"))
    return digest.digest()


def _build_page_image_tables(
    store,
    snapshot_id: int,
    worker_id: int,
    worker_count: int,
    shard_count: int,
    mode: str = "full",
    parent_state: dict | None = None,
) -> tuple[pa.Table, pa.Table]:
    if snapshot_id <= 0:
        raise ValueError(f"snapshot_id must be positive, got {snapshot_id}")
    if worker_count <= 0 or shard_count <= 0:
        raise ValueError("worker_count and shard_count must be positive")
    if worker_id < 0 or worker_id >= worker_count:
        raise ValueError(
            f"worker_id={worker_id} is outside worker_count={worker_count}"
        )
    if mode not in {"full", "incremental"}:
        raise ValueError("checkpoint mode must be full or incremental")
    if mode == "incremental" and parent_state is None:
        raise ValueError("incremental export requires parent checkpoint state")

    entry_rows = []
    page_rows = []
    exported_images = store.export_page_images()
    if not isinstance(exported_images, list):
        raise TypeError("native page-image export must return a list")
    for image in exported_images:
        if not isinstance(image, dict):
            raise TypeError("native page-image export must return dictionaries")
        dim = _unsigned(image.get("dim"), "page image dim", 32)
        physical_page_id = _unsigned(
            image.get("physical_page_id"), "physical_page_id", 64
        )
        vector_dim = _unsigned(image.get("vector_dim"), "vector_dim", 32)
        vector_stride = _unsigned(image.get("vector_stride"), "vector_stride", 32)
        if dim != vector_dim or vector_stride < vector_dim or vector_dim == 0:
            raise ValueError("page image has an invalid vector layout")
        slot_ordinals = _page_array(image, "slot_ordinals", np.uint32)
        count = slot_ordinals.size
        if count == 0:
            raise ValueError("native page-image export returned an empty page")
        if np.unique(slot_ordinals).size != count:
            raise ValueError("page image contains duplicate slot ordinals")
        keys = _page_array(image, "keys", np.uint64, count)
        seq_counts = _page_array(image, "seq_counts", np.uint64, count)
        values = _page_array(image, "values", np.float32, count * vector_dim)
        scores = _page_array(image, "scores", np.uint64)
        if scores.size not in (0, count):
            raise ValueError("page image scores must be absent or per-entry")
        optimizer_states = _page_array(image, "optimizer_states", np.float32)
        if optimizer_states.size % count != 0:
            raise ValueError("page image optimizer state is not entry-aligned")
        optimizer_state_stride = optimizer_states.size // count
        steps = _page_array(image, "steps", np.uint32)
        if steps.size not in (0, count):
            raise ValueError("page image steps must be absent or per-entry")
        if (optimizer_state_stride == 0) != (steps.size == 0):
            raise ValueError("page image optimizer state and steps disagree")
        logical_keys = keys.view(np.int64)
        shard_ids = np.remainder(logical_keys, shard_count).astype(
            np.uint32, copy=False
        )
        checksum = _page_image_checksum(
            dim,
            physical_page_id,
            vector_dim,
            vector_stride,
            optimizer_state_stride,
            slot_ordinals,
            keys,
            seq_counts,
            values,
            scores,
            optimizer_states,
            steps,
        )
        page_rows.append(
            {
                "dim": dim,
                "physical_page_id": physical_page_id,
                "vector_dim": vector_dim,
                "vector_stride": vector_stride,
                "slot_ordinals": slot_ordinals.tolist(),
                "keys": keys.tolist(),
                "seq_counts": seq_counts.tolist(),
                "values": values.tolist(),
                "scores": scores.tolist(),
                "optimizer_states": optimizer_states.tolist(),
                "optimizer_state_stride": optimizer_state_stride,
                "steps": steps.tolist(),
                "checksum": checksum,
                "snapshot_id": snapshot_id,
                "worker_id": worker_id,
                "shard_ids": shard_ids.tolist(),
                "page_operation": 1,
                "page_ref_checksum": None,
            }
        )
        for offset, key in enumerate(keys):
            entry_rows.append(
                {
                    "dim": dim,
                    "key": int(logical_keys[offset]),
                    "physical_page_id": physical_page_id,
                    "slot_ordinal": int(slot_ordinals[offset]),
                    "seq_count": int(seq_counts[offset]),
                    "vector_stride": vector_stride,
                    "embedding_params": values[
                        offset * vector_dim : (offset + 1) * vector_dim
                    ].tolist(),
                    "score": int(scores[offset]) if scores.size else None,
                    "optimizer_states": optimizer_states[
                        offset * optimizer_state_stride : (offset + 1)
                        * optimizer_state_stride
                    ].tolist(),
                    "optimizer_state_stride": optimizer_state_stride,
                    "step": int(steps[offset]) if steps.size else None,
                    "snapshot_id": snapshot_id,
                    "worker_id": worker_id,
                    "shard_id": int(shard_ids[offset]),
                    "operation": 1,
                    "shard_sequence": 0,
                }
            )
    if mode == "full":
        watermarks = [0] * shard_count
        for row in entry_rows:
            shard = row["shard_id"]
            watermarks[shard] += 1
            row["shard_sequence"] = watermarks[shard]
        return (
            pa.Table.from_pylist(entry_rows, schema=ENTRY_SCHEMA)
            if entry_rows
            else _empty_table(ENTRY_SCHEMA),
            pa.Table.from_pylist(page_rows, schema=PAGE_IMAGES_SCHEMA)
            if page_rows
            else _empty_table(PAGE_IMAGES_SCHEMA),
        )

    previous_entries = {
        identity: row
        for identity, row in parent_state["entries"].items()
        if row["worker_id"] == worker_id
    }
    previous_pages = parent_state["pages"]
    watermarks = list(parent_state["watermarks"])
    current_by_key = {(row["dim"], row["key"]): row for row in entry_rows}
    delta_rows = []
    for identity, row in sorted(current_by_key.items()):
        previous = previous_entries.get(identity)
        comparable = previous is not None and all(
            row.get(name) == previous.get(name)
            for name in (
                "dim",
                "key",
                "physical_page_id",
                "slot_ordinal",
                "seq_count",
                "vector_stride",
                "embedding_params",
                "score",
                "optimizer_states",
                "optimizer_state_stride",
                "step",
                "shard_id",
            )
        )
        if comparable:
            continue
        shard = row["shard_id"]
        watermarks[shard] += 1
        row["shard_sequence"] = watermarks[shard]
        delta_rows.append(row)
    for identity, previous in sorted(previous_entries.items()):
        if identity in current_by_key:
            continue
        shard = previous["shard_id"]
        watermarks[shard] += 1
        delta_rows.append(
            {
                "dim": previous["dim"],
                "key": previous["key"],
                "physical_page_id": 0,
                "slot_ordinal": 0,
                "seq_count": previous["seq_count"],
                "vector_stride": previous["vector_stride"],
                "embedding_params": [],
                "score": None,
                "optimizer_states": [],
                "optimizer_state_stride": 0,
                "step": None,
                "snapshot_id": snapshot_id,
                "worker_id": worker_id,
                "shard_id": shard,
                "operation": 2,
                "shard_sequence": watermarks[shard],
            }
        )
    previous_checksums = set(previous_pages)
    delta_pages = []
    for page in page_rows:
        checksum = page["checksum"]
        if checksum not in previous_checksums:
            page["page_operation"] = 1
            delta_pages.append(page)
        else:
            page["page_operation"] = 3
            page["page_ref_checksum"] = checksum
            page["slot_ordinals"] = []
            page["keys"] = []
            page["seq_counts"] = []
            page["values"] = []
            page["scores"] = []
            page["optimizer_states"] = []
            page["optimizer_state_stride"] = 0
            page["steps"] = []
            delta_pages.append(page)
    current_page_ids = {(page["dim"], page["physical_page_id"]) for page in page_rows}
    for (worker, dim, page_id), checksum in parent_state["page_id_checksums"].items():
        if worker != worker_id:
            continue
        if (dim, page_id) not in current_page_ids:
            delta_pages.append(
                {
                    "dim": dim,
                    "physical_page_id": page_id,
                    "vector_dim": 0,
                    "vector_stride": 0,
                    "slot_ordinals": [],
                    "keys": [],
                    "seq_counts": [],
                    "values": [],
                    "scores": [],
                    "optimizer_states": [],
                    "optimizer_state_stride": 0,
                    "steps": [],
                    "checksum": checksum,
                    "snapshot_id": snapshot_id,
                    "worker_id": worker_id,
                    "shard_ids": [],
                    "page_operation": 2,
                    "page_ref_checksum": None,
                }
            )
    return (
        pa.Table.from_pylist(delta_rows, schema=ENTRY_SCHEMA)
        if delta_rows
        else _empty_table(ENTRY_SCHEMA),
        pa.Table.from_pylist(delta_pages, schema=PAGE_IMAGES_SCHEMA)
        if delta_pages
        else _empty_table(PAGE_IMAGES_SCHEMA),
    )


class LanceCheckpointBackend(DistributedCheckpointBackend):
    """
    Lance-based unified-dataset checkpoint backend.

    Physical pages are exported from the native store and written as the
    ``entries`` and ``page_images`` Lance datasets. ``entries`` is the
    restore-oriented row view; ``page_images`` preserves each physical page's
    slot layout, seq_count, payload, score, optimizer state, and SHA-256
    checksum. ``snapshot_manifest`` pins both dataset versions atomically.

    Data flow during distributed save:
      1. Driver: ``init_dataset`` — creates empty Lance datasets.
      2. Workers (parallel): ``export_fragments`` — writes Lance fragment
         files and returns serialised ``FragmentMetadata``.
      3. Driver: ``commit_fragments`` — atomically appends all fragments.

    Thread-safety: native export validates every locator under its SeqLock and
    materializes page images before Python writes Lance. No Python-side lock
    participates in the hot path.

    Args:
        use_lock: Reserved for future use. Currently a no-op.
    """

    def __init__(self, use_lock: bool = False):
        self.use_lock = use_lock

    @staticmethod
    def _read_parent_state(uri: str, snapshot_id: int) -> dict:
        return LanceCheckpointBackend._read_snapshot_state(uri, snapshot_id)

    @staticmethod
    def latest_snapshot_id(uri: str) -> int:
        rows = lance.dataset(f"{uri}/{_MANIFEST_DATASET}").to_table().to_pylist()
        epochs = [int(row["snapshot_id"]) for row in rows]
        if not epochs:
            raise RuntimeError("checkpoint manifest has no committed snapshots")
        return max(epochs)

    @staticmethod
    def _read_snapshot_state(uri: str, snapshot_id: int) -> dict:
        manifest = LanceCheckpointBackend._read_partitioned_manifest(uri, snapshot_id)
        if manifest["checkpoint_mode"] not in {"full", "incremental"}:
            raise RuntimeError("checkpoint manifest has an unsupported mode")
        if manifest["checkpoint_mode"] == "incremental":
            parent_uri = manifest["parent_uri"]
            parent_epoch = manifest["parent_snapshot_id"]
            if not parent_uri or parent_epoch <= 0:
                raise RuntimeError("incremental manifest has no valid parent")
            state = LanceCheckpointBackend._read_snapshot_state(
                parent_uri, parent_epoch
            )
        else:
            state = {
                "entries": {},
                "pages": set(),
                "page_id_checksums": {},
                "page_rows": {},
            }
        snapshot_filter = f"snapshot_id = {snapshot_id}"
        _verify_page_images(
            f"{uri}/{_PAGE_IMAGES_DATASET}",
            manifest["page_images_version"],
            snapshot_filter,
        )
        entries = dict(state["entries"])
        entry_dataset = lance.dataset(
            f"{uri}/{_ENTRIES_DATASET}", version=manifest["entries_version"]
        )
        for row in sorted(
            entry_dataset.to_table(filter=snapshot_filter).to_pylist(),
            key=lambda row: (row["snapshot_id"], row["shard_sequence"]),
        ):
            identity = (row["dim"], row["key"])
            if row["operation"] == 2:
                entries.pop(identity, None)
            else:
                entries[identity] = row
        page_rows = dict(state["page_rows"])
        page_id_checksums = dict(state["page_id_checksums"])
        page_dataset = lance.dataset(
            f"{uri}/{_PAGE_IMAGES_DATASET}", version=manifest["page_images_version"]
        )
        for row in sorted(
            page_dataset.to_table(filter=snapshot_filter).to_pylist(),
            key=lambda row: row["snapshot_id"],
        ):
            identity = (row["worker_id"], row["dim"], row["physical_page_id"])
            if row["page_operation"] == 2:
                page_id_checksums.pop(identity, None)
                page_rows.pop(identity, None)
                continue
            if row["page_operation"] == 3:
                previous = state["page_rows"].get(identity)
                if previous is None or previous["checksum"] != row["page_ref_checksum"]:
                    raise RuntimeError("page-image reference checksum is unresolved")
                page_rows[identity] = previous
                page_id_checksums[identity] = previous["checksum"]
                continue
            page_id_checksums[identity] = row["checksum"]
            page_rows[identity] = row
        pages = set(page_id_checksums.values())
        return {
            "entries": entries,
            "pages": pages,
            "page_id_checksums": page_id_checksums,
            "page_rows": page_rows,
            "watermarks": list(manifest["shard_high_watermarks"]),
        }

    # ------------------------------------------------------------------
    # CheckpointBackend interface (single-node)
    # ------------------------------------------------------------------

    def save(self, store, uri: str, **kwargs):
        entries, page_images = _build_page_image_tables(
            store,
            snapshot_id=1,
            worker_id=0,
            worker_count=1,
            shard_count=1,
        )
        lance.write_dataset(entries, f"{uri}/{_ENTRIES_DATASET}", mode="create")
        lance.write_dataset(page_images, f"{uri}/{_PAGE_IMAGES_DATASET}", mode="create")
        lance.write_dataset(
            pa.Table.from_pylist(
                [
                    {
                        "schema_version": _FORMAT_VERSION,
                        "snapshot_id": 1,
                        "worker_count": 1,
                        "shard_count": 1,
                        "entries_version": lance.dataset(
                            f"{uri}/{_ENTRIES_DATASET}"
                        ).version,
                        "page_images_version": lance.dataset(
                            f"{uri}/{_PAGE_IMAGES_DATASET}"
                        ).version,
                        "parent_snapshot_id": 0,
                        "shard_high_watermarks": self._artifact_watermarks(
                            entries, None, shard_count=1
                        ),
                        "checkpoint_mode": "full",
                        "parent_uri": "",
                    }
                ],
                schema=PARTITIONED_MANIFEST_SCHEMA,
            ),
            f"{uri}/{_MANIFEST_DATASET}",
            mode="create",
        )

    def load(self, store, uri: str, filter_fn=None, target_shard_id: int = 0, **kwargs):
        _reject_legacy_kwargs(kwargs)
        distributed = kwargs.pop("distributed", False)
        if not isinstance(distributed, bool):
            raise TypeError("distributed must be a bool")
        if distributed and filter_fn is None:
            raise ValueError("distributed checkpoint load requires filter_fn")
        if filter_fn is not None and not distributed:
            raise ValueError("filter_fn requires distributed=True")
        if filter_fn is None and target_shard_id != 0:
            raise ValueError("target_shard_id requires filter_fn")
        snapshot_id = kwargs.get("snapshot_id")
        restore_layout = kwargs.get("restore_layout", False)
        saved_worker_id = kwargs.get("saved_worker_id")
        current_worker_count = kwargs.get("worker_count")
        if not isinstance(restore_layout, bool):
            raise TypeError("restore_layout must be a bool")
        if snapshot_id is None:
            snapshot_id = self.latest_snapshot_id(uri)
        manifest = self._read_partitioned_manifest(uri, snapshot_id)
        page_images_version = manifest["page_images_version"]
        worker_filter = f"snapshot_id = {snapshot_id}"

        snapshot_state = self._read_snapshot_state(uri, snapshot_id)

        _verify_page_images(
            f"{uri}/{_PAGE_IMAGES_DATASET}", page_images_version, worker_filter
        )
        if restore_layout:
            if filter_fn is not None or target_shard_id != 0:
                raise ValueError(
                    "layout restore does not accept logical routing parameters"
                )
            if (
                not isinstance(saved_worker_id, int)
                or isinstance(saved_worker_id, bool)
                or saved_worker_id < 0
                or saved_worker_id >= manifest["worker_count"]
            ):
                raise ValueError("layout restore saved_worker_id is invalid")
            if current_worker_count != manifest["worker_count"]:
                raise ValueError(
                    "layout restore requires the checkpoint worker topology"
                )
            if store.total_size() != 0:
                raise RuntimeError("layout restore requires an empty store")
            pages = [
                row
                for row in snapshot_state["page_rows"].values()
                if row["worker_id"] == saved_worker_id
            ]
            _import_page_rows(store, pages)
            return
        if snapshot_state is not None:
            _load_entry_rows(
                store,
                list(snapshot_state["entries"].values()),
                filter_fn,
                target_shard_id,
            )
            return
        _load_entries(
            store,
            f"{uri}/{_ENTRIES_DATASET}",
            filter_fn,
            target_shard_id,
            entries_version,
            worker_filter,
        )

    # ------------------------------------------------------------------
    # DistributedCheckpointBackend interface (multi-node two-phase commit)
    # ------------------------------------------------------------------

    def init_dataset(self, store, uri: str, **kwargs):
        lance.write_dataset(
            _empty_table(ENTRY_SCHEMA), f"{uri}/{_ENTRIES_DATASET}", mode="create"
        )
        lance.write_dataset(
            _empty_table(PAGE_IMAGES_SCHEMA),
            f"{uri}/{_PAGE_IMAGES_DATASET}",
            mode="create",
        )

        lance.write_dataset(
            _empty_table(PARTITIONED_MANIFEST_SCHEMA),
            f"{uri}/{_MANIFEST_DATASET}",
            mode="create",
        )

    def export_fragments(self, store, uri: str, **kwargs) -> List[bytes]:
        _reject_legacy_kwargs(kwargs)
        snapshot_id = kwargs.get("snapshot_id")
        worker_id = kwargs.get("worker_id")
        worker_count = kwargs.get("worker_count")
        shard_count = kwargs.get("shard_count")
        checkpoint_mode = kwargs.get("checkpoint_mode", "full")
        parent_snapshot_id = kwargs.get("parent_snapshot_id", 0)
        parent_uri = kwargs.get("parent_uri", "")
        if (
            snapshot_id is None
            or worker_id is None
            or worker_count is None
            or shard_count is None
        ):
            raise ValueError(
                "distributed Lance export requires snapshot_id, worker_id, "
                "worker_count, and shard_count"
            )
        entries_uri = f"{uri}/{_ENTRIES_DATASET}"
        page_images_uri = f"{uri}/{_PAGE_IMAGES_DATASET}"
        result: List[bytes] = []
        parent_state = None
        if checkpoint_mode == "incremental":
            if not isinstance(parent_snapshot_id, int) or parent_snapshot_id <= 0:
                raise ValueError("incremental export requires parent_snapshot_id")
            if not parent_uri:
                raise ValueError("incremental export requires parent_uri")
            parent_state = self._read_parent_state(parent_uri, parent_snapshot_id)
        entries, page_images = _build_page_image_tables(
            store,
            snapshot_id,
            worker_id,
            worker_count,
            shard_count,
            checkpoint_mode,
            parent_state,
        )
        if entries.num_rows:
            for fragment in write_fragments(entries, entries_uri):
                result.append(f"entries:{json.dumps(fragment.to_json())}".encode())
        if page_images.num_rows:
            for fragment in write_fragments(page_images, page_images_uri):
                result.append(f"pages:{json.dumps(fragment.to_json())}".encode())

        artifact = {
            "snapshot_id": snapshot_id,
            "worker_id": worker_id,
            "worker_count": worker_count,
            "shard_count": shard_count,
            "checkpoint_mode": checkpoint_mode,
            "parent_snapshot_id": parent_snapshot_id,
            "parent_uri": parent_uri,
            "shard_high_watermarks": self._artifact_watermarks(
                entries, parent_state, shard_count
            ),
            "fragments": [fragment.decode() for fragment in result],
        }
        return [json.dumps(artifact, sort_keys=True).encode()]

    @staticmethod
    def _artifact_watermarks(entries, parent_state, shard_count):
        watermarks = (
            list(parent_state["watermarks"]) if parent_state else [0] * shard_count
        )
        for row in entries.to_pylist():
            watermarks[row["shard_id"]] = max(
                watermarks[row["shard_id"]], row["shard_sequence"]
            )
        return watermarks

    def commit_fragments(
        self,
        uri: str,
        all_fragment_metas: Union[List[bytes], List[List[bytes]]],
        **kwargs,
    ):
        artifacts = self._decode_worker_artifacts(all_fragment_metas)
        (
            snapshot_id,
            worker_count,
            shard_count,
            checkpoint_mode,
            parent_epoch,
            watermarks,
        ) = self._validate_artifacts(artifacts)

        entries_uri = f"{uri}/{_ENTRIES_DATASET}"
        page_images_uri = f"{uri}/{_PAGE_IMAGES_DATASET}"
        entry_fragments: List[FragmentMetadata] = []
        page_fragments: List[FragmentMetadata] = []

        for artifact in artifacts:
            for decoded in artifact["fragments"]:
                if decoded.startswith("entries:"):
                    entry_fragments.append(FragmentMetadata.from_json(decoded[8:]))
                elif decoded.startswith("pages:"):
                    page_fragments.append(FragmentMetadata.from_json(decoded[6:]))
                else:
                    raise ValueError("unknown Lance worker fragment kind")

        entries_version = lance.dataset(entries_uri).version
        page_images_version = lance.dataset(page_images_uri).version
        if entry_fragments:
            dataset = lance.dataset(entries_uri)
            operation = lance.LanceOperation.Append(entry_fragments)
            entries_version = lance.LanceDataset.commit(
                entries_uri, operation, read_version=dataset.version
            ).version
            self._validate_snapshot_entry_uniqueness(
                entries_uri, entries_version, snapshot_id
            )
        if page_fragments:
            dataset = lance.dataset(page_images_uri)
            operation = lance.LanceOperation.Append(page_fragments)
            page_images_version = lance.LanceDataset.commit(
                page_images_uri, operation, read_version=dataset.version
            ).version

        manifest = pa.Table.from_pylist(
            [
                {
                    "schema_version": _FORMAT_VERSION,
                    "snapshot_id": snapshot_id,
                    "worker_count": worker_count,
                    "shard_count": shard_count,
                    "entries_version": entries_version,
                    "page_images_version": page_images_version,
                    "parent_snapshot_id": parent_epoch,
                    "shard_high_watermarks": watermarks,
                    "checkpoint_mode": checkpoint_mode,
                    "parent_uri": artifacts[0]["parent_uri"],
                }
            ],
            schema=PARTITIONED_MANIFEST_SCHEMA,
        )
        lance.write_dataset(manifest, f"{uri}/{_MANIFEST_DATASET}", mode="append")

    @staticmethod
    def _validate_snapshot_entry_uniqueness(
        entries_uri: str,
        entries_version: int,
        snapshot_id: int,
    ) -> None:
        rows = (
            lance.dataset(entries_uri, version=entries_version)
            .to_table(filter=f"snapshot_id = {snapshot_id}")
            .to_pylist()
        )
        workers_by_identity: dict[tuple[int, int], int] = {}
        for row in rows:
            identity = (row["dim"], row["key"])
            worker_id = row["worker_id"]
            previous_worker = workers_by_identity.get(identity)
            if previous_worker is not None and previous_worker != worker_id:
                raise ValueError(
                    "checkpoint snapshot contains duplicate key across workers"
                )
            workers_by_identity[identity] = worker_id

    @staticmethod
    def _decode_worker_artifacts(
        all_fragment_metas: Union[List[bytes], List[List[bytes]]],
    ) -> List[dict]:
        artifacts: List[dict] = []
        for item in all_fragment_metas:
            values = [item] if isinstance(item, bytes) else item
            if len(values) != 1:
                raise ValueError(
                    "each worker must publish exactly one checkpoint artifact"
                )
            artifact = json.loads(values[0].decode())
            if not isinstance(artifact, dict):
                raise ValueError("checkpoint artifact must be an object")
            artifacts.append(artifact)
        return artifacts

    @staticmethod
    def _validate_artifacts(artifacts: List[dict]) -> tuple[int, int, int]:
        if not artifacts:
            raise ValueError("checkpoint commit requires at least one worker artifact")
        required = {
            "snapshot_id",
            "worker_id",
            "worker_count",
            "shard_count",
            "checkpoint_mode",
            "parent_snapshot_id",
            "shard_high_watermarks",
            "parent_uri",
            "fragments",
        }
        if any(set(artifact) != required for artifact in artifacts):
            raise ValueError("checkpoint artifact fields do not match the protocol")
        epochs = {artifact["snapshot_id"] for artifact in artifacts}
        worker_counts = {artifact["worker_count"] for artifact in artifacts}
        shard_counts = {artifact["shard_count"] for artifact in artifacts}
        if len(epochs) != 1 or len(worker_counts) != 1 or len(shard_counts) != 1:
            raise ValueError(
                "checkpoint worker artifacts disagree on topology or epoch"
            )
        snapshot_id = next(iter(epochs))
        worker_count = next(iter(worker_counts))
        shard_count = next(iter(shard_counts))
        if (
            not isinstance(snapshot_id, int)
            or snapshot_id <= 0
            or not isinstance(worker_count, int)
            or worker_count <= 0
            or not isinstance(shard_count, int)
            or shard_count <= 0
        ):
            raise ValueError("checkpoint topology or epoch is invalid")
        worker_ids = [artifact["worker_id"] for artifact in artifacts]
        if sorted(worker_ids) != list(range(worker_count)):
            raise ValueError(
                "checkpoint artifacts do not cover every worker exactly once"
            )
        modes = {artifact["checkpoint_mode"] for artifact in artifacts}
        parents = {artifact["parent_snapshot_id"] for artifact in artifacts}
        parent_uris = {artifact["parent_uri"] for artifact in artifacts}
        artifact_watermarks = [
            artifact["shard_high_watermarks"] for artifact in artifacts
        ]
        if modes not in [{"full"}, {"incremental"}]:
            raise ValueError("checkpoint artifacts disagree on checkpoint mode")
        mode = next(iter(modes))
        parent_epoch = next(iter(parents))
        if len(parent_uris) != 1:
            raise ValueError("checkpoint artifacts disagree on parent URI")
        parent_uri = next(iter(parent_uris))
        if mode == "full" and parent_epoch != 0:
            raise ValueError("full checkpoint cannot have a parent snapshot")
        if mode == "incremental" and parent_epoch <= 0:
            raise ValueError("incremental checkpoint requires a parent snapshot")
        if any(len(values) != shard_count for values in artifact_watermarks):
            raise ValueError("checkpoint shard watermark count mismatch")
        watermarks = [
            max(values[shard] for values in artifact_watermarks)
            for shard in range(shard_count)
        ]
        return snapshot_id, worker_count, shard_count, mode, parent_epoch, watermarks

    @staticmethod
    def _read_partitioned_manifest(uri: str, snapshot_id: int) -> dict:
        if not isinstance(snapshot_id, int) or snapshot_id <= 0:
            raise ValueError("snapshot_id must be positive")
        dataset = lance.dataset(f"{uri}/{_MANIFEST_DATASET}")
        rows = dataset.to_table(filter=f"snapshot_id = {snapshot_id}").to_pylist()
        if len(rows) != 1:
            raise RuntimeError("checkpoint snapshot manifest is missing or duplicated")
        manifest = rows[0]
        if manifest["schema_version"] != _FORMAT_VERSION:
            raise RuntimeError("checkpoint snapshot manifest schema is unsupported")
        return manifest


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


def _verify_page_images(
    uri: str,
    version: int | None = None,
    worker_filter: str | None = None,
):
    dataset = lance.dataset(uri, version=version)
    for batch in dataset.scanner(filter=worker_filter).to_batches():
        for row in pa.Table.from_batches([batch]).to_pylist():
            if row["page_operation"] == 2:
                if row["slot_ordinals"] or row["keys"] or row["values"]:
                    raise RuntimeError("page-image tombstone contains payload")
                continue
            if row["page_operation"] == 3:
                if row["page_ref_checksum"] is None or row["slot_ordinals"]:
                    raise RuntimeError("page-image reference contains payload")
                continue
            checksum = _page_image_checksum(
                _unsigned(row["dim"], "page image dim", 32),
                _unsigned(row["physical_page_id"], "physical_page_id", 64),
                _unsigned(row["vector_dim"], "vector_dim", 32),
                _unsigned(row["vector_stride"], "vector_stride", 32),
                _unsigned(row["optimizer_state_stride"], "optimizer_state_stride", 32),
                np.asarray(row["slot_ordinals"], dtype=np.uint32),
                np.asarray(row["keys"], dtype=np.uint64),
                np.asarray(row["seq_counts"], dtype=np.uint64),
                np.asarray(row["values"], dtype=np.float32),
                np.asarray(row["scores"], dtype=np.uint64),
                np.asarray(row["optimizer_states"], dtype=np.float32),
                np.asarray(row["steps"], dtype=np.uint32),
            )
            if checksum != row["checksum"]:
                raise RuntimeError("page-image checksum mismatch")


def _load_page_images(
    store,
    uri: str,
    version: int,
    worker_filter: str,
):
    dataset = lance.dataset(uri, version=version)
    latest = {}
    for batch in dataset.scanner(filter=worker_filter).to_batches():
        for row in pa.Table.from_batches([batch]).to_pylist():
            identity = (row["dim"], row["physical_page_id"])
            old = latest.get(identity)
            order = (row["snapshot_id"], row["page_operation"])
            if old is None or (old["snapshot_id"], old["page_operation"]) < order:
                latest[identity] = row
    pages = []
    for row in latest.values():
        if row["page_operation"] == 2:
            continue
        if row["page_operation"] == 3:
            raise RuntimeError("layout restore requires resolved page references")
        checksum = _page_image_checksum(
            _unsigned(row["dim"], "page image dim", 32),
            _unsigned(row["physical_page_id"], "physical_page_id", 64),
            _unsigned(row["vector_dim"], "vector_dim", 32),
            _unsigned(row["vector_stride"], "vector_stride", 32),
            _unsigned(row["optimizer_state_stride"], "optimizer_state_stride", 32),
            np.asarray(row["slot_ordinals"], dtype=np.uint32),
            np.asarray(row["keys"], dtype=np.uint64),
            np.asarray(row["seq_counts"], dtype=np.uint64),
            np.asarray(row["values"], dtype=np.float32),
            np.asarray(row["scores"], dtype=np.uint64),
            np.asarray(row["optimizer_states"], dtype=np.float32),
            np.asarray(row["steps"], dtype=np.uint32),
        )
        if checksum != row["checksum"]:
            raise RuntimeError("page-image checksum mismatch")
        pages.append(
            {
                "dim": row["dim"],
                "physical_page_id": row["physical_page_id"],
                "vector_dim": row["vector_dim"],
                "vector_stride": row["vector_stride"],
                "slot_ordinals": row["slot_ordinals"],
                "keys": row["keys"],
                "seq_counts": row["seq_counts"],
                "values": row["values"],
                "scores": row["scores"],
                "optimizer_states": row["optimizer_states"],
                "steps": row["steps"],
            }
        )
    if not pages:
        raise RuntimeError("layout restore has no page images for worker")
    store.import_page_images(pages)


def _import_page_rows(store, rows):
    if not rows:
        raise RuntimeError("layout restore has no page images for worker")
    pages = []
    for row in rows:
        checksum = _page_image_checksum(
            _unsigned(row["dim"], "page image dim", 32),
            _unsigned(row["physical_page_id"], "physical_page_id", 64),
            _unsigned(row["vector_dim"], "vector_dim", 32),
            _unsigned(row["vector_stride"], "vector_stride", 32),
            _unsigned(row["optimizer_state_stride"], "optimizer_state_stride", 32),
            np.asarray(row["slot_ordinals"], dtype=np.uint32),
            np.asarray(row["keys"], dtype=np.uint64),
            np.asarray(row["seq_counts"], dtype=np.uint64),
            np.asarray(row["values"], dtype=np.float32),
            np.asarray(row["scores"], dtype=np.uint64),
            np.asarray(row["optimizer_states"], dtype=np.float32),
            np.asarray(row["steps"], dtype=np.uint32),
        )
        if checksum != row["checksum"]:
            raise RuntimeError("page-image checksum mismatch")
        pages.append(
            {
                name: row[name]
                for name in (
                    "dim",
                    "physical_page_id",
                    "vector_dim",
                    "vector_stride",
                    "slot_ordinals",
                    "keys",
                    "seq_counts",
                    "values",
                    "scores",
                    "optimizer_states",
                    "steps",
                )
            }
        )
    store.import_page_images(pages)


def _load_entry_rows(store, rows, filter_fn, target_shard_id):
    for dim in store.table_dims():
        dim_rows = [row for row in rows if row["dim"] == dim]
        if not dim_rows:
            continue
        keys = pa.array([row["key"] for row in dim_rows], type=pa.int64())
        mask = _apply_filter(keys, filter_fn, target_shard_id)
        if mask is not None:
            dim_rows = [
                row for row, include in zip(dim_rows, mask.to_pylist()) if include
            ]
        if not dim_rows:
            continue
        keys = pa.array([row["key"] for row in dim_rows], type=pa.int64())
        store.assign(
            keys,
            pc.list_flatten(
                pa.array(
                    [row["embedding_params"] for row in dim_rows],
                    type=pa.list_(pa.float32()),
                )
            ),
        )
        if any(row["score"] is not None for row in dim_rows):
            if any(row["score"] is None for row in dim_rows):
                raise RuntimeError("entries score nullability is inconsistent")
            store.assign_scores(
                keys, pa.array([row["score"] for row in dim_rows], type=pa.uint64())
            )
        strides = [row["optimizer_state_stride"] for row in dim_rows]
        if any(strides):
            if any(stride == 0 for stride in strides) or len(set(strides)) != 1:
                raise RuntimeError("entries optimizer-state stride is inconsistent")
            if any(row["step"] is None for row in dim_rows):
                raise RuntimeError("entries optimizer-state step is missing")
            store.assign_opt_state(
                keys,
                pc.list_flatten(
                    pa.array(
                        [row["optimizer_states"] for row in dim_rows],
                        type=pa.list_(pa.float32()),
                    )
                ),
                dim,
                pa.array([row["step"] for row in dim_rows], type=pa.uint32()),
            )
