"""Canonical opaque restore-options codec owned by sparse_embedding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import msgpack

from .base import (
    SUPPORTED_KEY_DTYPES,
    CheckpointParticipant,
    ElasticRestore,
    PartitionSpec,
    PhysicalRestore,
    PhysicalRestoreStrategy,
    RestoreMode,
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
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
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
        raise TypeError("owner restore options must be bytes")
    if not payload or len(payload) > _MAX_PAYLOAD_BYTES:
        raise ValueError("owner restore options have an invalid size")
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
        raise ValueError("owner restore options are malformed") from error
    if not isinstance(decoded, dict):
        raise ValueError("owner restore options must contain a map")
    if _pack(decoded) != payload:
        raise ValueError("owner restore options are not canonical")
    return decoded


def _require_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def encode_restore_options(mode: RestorePlan | RestoreMode) -> bytes:
    """Encode one typed restore plan as canonical owner-owned opaque bytes."""

    if isinstance(mode, PhysicalRestore):
        payload: dict[str, object] = {
            "key_dtype": mode.key_dtype,
            "participant_count": mode.participant.participant_count,
            "participant_index": mode.participant.participant_index,
            "restore_kind": "physical",
            "strategy": mode.strategy.value,
        }
    elif isinstance(mode, ElasticRestore):
        assignment = mode.assignment_spec
        payload = {
            "key_dtype": mode.key_dtype,
            "partition_count": mode.partition_count,
            "partition_id": mode.partition_id,
            "restore_kind": "elastic",
            "sparse_assignment_name": assignment.name,
            "sparse_assignment_parameters": assignment.parameters,
        }
    else:
        raise TypeError("restore mode must be PhysicalRestore or ElasticRestore")
    encoded = _pack(payload)
    if len(encoded) > _MAX_PAYLOAD_BYTES:
        raise ValueError("restore options exceed the maximum payload size")
    return encoded


def _decode_restore_options(payload: bytes) -> RestorePlan:
    """Decode owner options; this boundary is private to sparse_embedding."""

    decoded = _unpack(payload)
    restore_kind = decoded.get("restore_kind")
    if not isinstance(restore_kind, str):
        raise TypeError("restore_kind must be a string")

    if restore_kind == "physical":
        required = {
            "key_dtype",
            "participant_count",
            "participant_index",
            "restore_kind",
            "strategy",
        }
        if set(decoded) != required:
            raise ValueError("physical restore options contain unknown fields")
        key_dtype = decoded["key_dtype"]
        strategy_name = decoded["strategy"]
        if not isinstance(key_dtype, str):
            raise TypeError("physical key_dtype must be a string")
        if not isinstance(strategy_name, str):
            raise TypeError("physical restore strategy must be a string")
        if key_dtype not in SUPPORTED_KEY_DTYPES:
            raise ValueError(f"unsupported physical key_dtype: {key_dtype}")
        try:
            strategy = PhysicalRestoreStrategy(strategy_name)
        except ValueError as error:
            raise ValueError("unsupported physical restore strategy") from error
        return PhysicalRestore(
            CheckpointParticipant(
                _require_integer(decoded["participant_index"], "participant_index"),
                _require_integer(decoded["participant_count"], "participant_count"),
            ),
            strategy,
            key_dtype=key_dtype,
        )

    if restore_kind == "elastic":
        required = {
            "key_dtype",
            "partition_count",
            "partition_id",
            "restore_kind",
            "sparse_assignment_name",
            "sparse_assignment_parameters",
        }
        if set(decoded) != required:
            raise ValueError("elastic restore options contain unknown fields")
        assignment_name = decoded["sparse_assignment_name"]
        parameters = decoded["sparse_assignment_parameters"]
        key_dtype = decoded["key_dtype"]
        if not isinstance(assignment_name, str):
            raise TypeError("sparse_assignment_name must be a string")
        if not isinstance(parameters, dict):
            raise TypeError("sparse_assignment_parameters must be a map")
        if not isinstance(key_dtype, str):
            raise TypeError("key_dtype must be a string")
        if key_dtype not in SUPPORTED_KEY_DTYPES:
            raise ValueError(f"unsupported elastic key_dtype: {key_dtype}")
        assignment = PartitionSpec(
            assignment_name,
            parameters,
        )
        return ElasticRestore(
            assignment,
            _require_integer(decoded["partition_count"], "partition_count"),
            _require_integer(decoded["partition_id"], "partition_id"),
            key_dtype=key_dtype,
        )

    raise ValueError(f"unsupported restore_kind: {restore_kind!r}")


__all__ = ["encode_restore_options"]
