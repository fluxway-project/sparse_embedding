"""Canonical opaque restore-options codec owned by sparse_embedding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import msgpack

from .base import (
    SUPPORTED_KEY_DTYPES,
    IngestionEngine,
    PartitionSpec,
    PartitionTarget,
    RestorePlan,
)

_MAX_PAYLOAD_BYTES = 4096
_MAX_MAP_ITEMS = 16
_MAX_ARRAY_ITEMS = 8
_MAX_STRING_BYTES = 128
_MAX_BINARY_BYTES = 0


def _reject_extension(code: int, data: bytes) -> Any:
    del code, data
    raise ValueError("restore options do not support extension values")


def _reject_duplicate_map_keys(
    pairs: list[tuple[object, object]],
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ValueError("restore options map keys must be strings")
        if key in result:
            raise ValueError("restore options contain duplicate fields")
        result[key] = value
    return result


def _canonicalize(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("restore option map keys must be strings")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def _pack(payload: Mapping[str, object]) -> bytes:
    return msgpack.packb(
        _canonicalize(payload),
        use_bin_type=True,
        strict_types=True,
    )


def _unpack(payload: bytes) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise TypeError("restore options must be bytes")
    if not payload or len(payload) > _MAX_PAYLOAD_BYTES:
        raise ValueError("restore options have an invalid size")
    try:
        decoded = msgpack.unpackb(
            payload,
            raw=False,
            strict_map_key=True,
            object_pairs_hook=_reject_duplicate_map_keys,
            ext_hook=_reject_extension,
            max_map_len=_MAX_MAP_ITEMS,
            max_array_len=_MAX_ARRAY_ITEMS,
            max_str_len=_MAX_STRING_BYTES,
            max_bin_len=_MAX_BINARY_BYTES,
            max_ext_len=0,
        )
    except (TypeError, ValueError, msgpack.exceptions.UnpackException) as error:
        raise ValueError("restore options are malformed") from error
    if not isinstance(decoded, dict):
        raise ValueError("restore options must contain a map")
    if _pack(decoded) != payload:
        raise ValueError("restore options are not canonical")
    return decoded


def _require_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def encode_restore_options(plan: RestorePlan) -> bytes:
    """Encode one typed restore plan as canonical opaque bytes."""

    if not isinstance(plan, RestorePlan):
        raise TypeError("plan must be a RestorePlan")
    assignment = plan.target.spec
    payload: dict[str, object] = {
        "engine": plan.engine.value,
        "key_dtype": plan.key_dtype,
        "partition_count": plan.target.partition_count,
        "partition_id": plan.target.partition_id,
        "restore_kind": "partition",
        "sparse_assignment_name": assignment.name,
        "sparse_assignment_parameters": assignment.parameters,
    }
    encoded = _pack(payload)
    if len(encoded) > _MAX_PAYLOAD_BYTES:
        raise ValueError("restore options exceed the maximum payload size")
    return encoded


def _decode_restore_options(payload: bytes) -> RestorePlan:
    """Decode owner options; this boundary is private to sparse_embedding."""

    decoded = _unpack(payload)
    required = {
        "engine",
        "key_dtype",
        "partition_count",
        "partition_id",
        "restore_kind",
        "sparse_assignment_name",
        "sparse_assignment_parameters",
    }
    if set(decoded) != required:
        raise ValueError("restore options contain unknown fields")
    restore_kind = decoded["restore_kind"]
    if restore_kind != "partition":
        raise ValueError(f"unsupported restore_kind: {restore_kind!r}")
    assignment_name = decoded["sparse_assignment_name"]
    parameters = decoded["sparse_assignment_parameters"]
    key_dtype = decoded["key_dtype"]
    engine_name = decoded["engine"]
    if not isinstance(assignment_name, str):
        raise TypeError("sparse_assignment_name must be a string")
    if not isinstance(parameters, dict):
        raise TypeError("sparse_assignment_parameters must be a map")
    if not isinstance(key_dtype, str):
        raise TypeError("key_dtype must be a string")
    if key_dtype not in SUPPORTED_KEY_DTYPES:
        raise ValueError(f"unsupported key_dtype: {key_dtype}")
    if not isinstance(engine_name, str):
        raise TypeError("restore ingestion engine must be a string")
    try:
        engine = IngestionEngine(engine_name)
    except ValueError as error:
        raise ValueError("unsupported restore ingestion engine") from error
    assignment = PartitionSpec(assignment_name, parameters)
    return RestorePlan(
        PartitionTarget(
            assignment,
            _require_integer(decoded["partition_count"], "partition_count"),
            _require_integer(decoded["partition_id"], "partition_id"),
        ),
        engine=engine,
        key_dtype=key_dtype,
    )


__all__ = ["encode_restore_options"]
