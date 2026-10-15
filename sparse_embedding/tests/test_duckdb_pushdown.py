import json
import os
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

import sparse_embedding as se
from sparse_embedding.checkpoint import (
    LocalLanceStorage,
    SparseCheckpoint,
)
from sparse_embedding.checkpoint.schema import (
    ENTRIES_DATASET,
    SNAPSHOT_MANIFEST_DATASET,
)

duckdb = pytest.importorskip("duckdb", reason="DuckDB is not installed")

_KEY_CASES = (
    ("int32", np.asarray([-(1 << 31), -1, 0, (1 << 31) - 1], dtype=np.int32)),
    ("uint32", np.asarray([0, 1, (1 << 32) - 1], dtype=np.uint32)),
    ("int64", np.asarray([-(1 << 63), -1, 0, (1 << 63) - 1], dtype=np.int64)),
    ("uint64", np.asarray([0, 1, 1 << 63, (1 << 64) - 1], dtype=np.uint64)),
)


def _store(key_dtype: str) -> se.EmbeddingStore:
    config = se.EmbeddingConfig(
        dims=[2],
        thread_budget=1,
        value_dtype="float32",
        key_dtype=key_dtype,
        enable_optimizer=False,
    )
    return se.EmbeddingStore(config)


def _assign(store: se.EmbeddingStore, keys: np.ndarray) -> None:
    values = np.repeat((np.arange(keys.size, dtype=np.float32))[:, None], 2, axis=1)
    store.assign(keys, values)


@pytest.fixture(scope="session")
def duckdb_lance_extensions(tmp_path_factory: pytest.TempPathFactory) -> Path:
    configured = os.environ.get("DUCKDB_EXTENSION_DIRECTORY")
    directory = (
        Path(configured)
        if configured
        else tmp_path_factory.mktemp("duckdb-lance-extensions")
    )
    try:
        with duckdb.connect(
            config={"extension_directory": str(directory)}
        ) as connection:
            connection.execute("INSTALL lance FROM 'https://extensions.duckdb.org'")
            connection.execute("LOAD lance")
            row = connection.execute(
                "SELECT installed, loaded FROM duckdb_extensions() "
                "WHERE extension_name = 'lance'"
            ).fetchone()
            if not (row and row[0] and row[1]):
                pytest.skip("DuckDB lance extension is not installed or loaded")
    except Exception as exc:
        pytest.skip(f"DuckDB lance extension is unavailable: {exc}")
    return directory


def _lance_scans(plan: dict) -> list[dict]:
    scans = []
    if "extra_info" in plan and "Lance Path" in plan["extra_info"]:
        scans.append(plan["extra_info"])
    for child in plan.get("children", []):
        scans.extend(_lance_scans(child))
    return scans


@pytest.mark.parametrize(("key_dtype", "keys"), _KEY_CASES)
def test_duckdb_direct_scan_preserves_native_keys_and_pushes_filters(
    tmp_path: Path, key_dtype: str, keys: np.ndarray, duckdb_lance_extensions: Path
) -> None:
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    source = _store(key_dtype)
    _assign(source, keys)
    SparseCheckpoint(storage).save_training_checkpoint(source, "duckdb-key-columns")

    manifest = (
        lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET)).to_table().to_pylist()[0]
    )
    fragment_root = root / storage._manifest_fragment_root_for_participant(manifest, 0)
    entries_path = fragment_root / ENTRIES_DATASET
    table_reference = "'" + str(entries_path).replace("'", "''") + "'"
    predicates = [(f"key = {int(key)}", [(int(key),)]) for key in keys]
    if np.issubdtype(keys.dtype, np.signedinteger):
        predicates.append(("key < 0", [(int(key),) for key in keys if key < 0]))
    else:
        predicates.append(("key >= 1", [(int(key),) for key in keys if key >= 1]))

    with duckdb.connect(
        config={"extension_directory": str(duckdb_lance_extensions)}
    ) as connection:
        connection.execute("LOAD lance")
        expected_type = connection.from_arrow(
            pa.table({"key": pa.array([], type=getattr(pa, key_dtype)())})
        ).types[0]
        assert (
            connection.execute(f"SELECT key FROM {table_reference}").description[0][1]
            == expected_type
        )
        diagnostics = []
        for predicate, expected_rows in predicates:
            query = f"SELECT key FROM {table_reference} WHERE {predicate}"
            assert sorted(connection.execute(query).fetchall()) == sorted(expected_rows)
            plan = json.loads(
                connection.execute(
                    "EXPLAIN (ANALYZE, FORMAT JSON) " + query
                ).fetchone()[1]
            )
            diagnostics.append({"query": query, "plan": plan})
            (tmp_path / "duckdb_lance_scan_diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2)
            )
            scans = _lance_scans(plan)
            assert len(scans) == 1
            scan = scans[0]
            assert scan["Lance Path"] == str(entries_path)
            assert scan["Lance Projection"] == "key"
            assert int(scan["Lance Filter IR Bytes"]) > 0
            assert int(scan["Lance Filter Pushdown Fallbacks"]) == 0
            assert int(scan["Lance Rows Out"]) == len(expected_rows)
