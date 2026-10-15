"""Canonical opaque wire codec for sparse checkpoint fragments."""

from __future__ import annotations

import msgpack

from .base import (
    CheckpointFragment,
    CheckpointParticipant,
    CheckpointSaveContext,
    CheckpointSaveMode,
    SparseCheckpointLocator,
    TableSpecification,
)

_FRAGMENT_FIELDS = frozenset(
    {
        "fragment_id",
        "checkpoint_name",
        "save_attempt",
        "participant_index",
        "participant_count",
        "entry_count",
        "page_count",
        "schema_digest",
        "content_digest",
        "dimensions",
        "embedding_dtype",
        "optimizer_dtype",
        "score_policy",
        "key_dtype",
        "mode",
        "parent_checkpoint_name",
        "parent_checkpoint_version",
        "parent_reference_id",
        "mutation_watermark",
        "include_page_images",
        "entries_per_page",
        "capture_id",
        "journal_identity",
        "page_dataset",
        "page_version",
        "page_layout_generation",
    }
)


def _reject_duplicate_fields(
    pairs: list[tuple[object, object]],
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ValueError("sparse fragment fields must be strings")
        if key in result:
            raise ValueError("sparse fragment contains duplicate fields")
        result[key] = value
    return result


def encode_checkpoint_fragment(fragment: CheckpointFragment) -> bytes:
    """Encode a validated fragment for opaque Fluxway transport."""

    if not isinstance(fragment, CheckpointFragment):
        raise TypeError("fragment must be a CheckpointFragment")
    parent = fragment.save_context.parent
    payload = {
        "fragment_id": fragment.fragment_id,
        "checkpoint_name": fragment.checkpoint_name,
        "save_attempt": fragment.save_attempt,
        "participant_index": fragment.participant.participant_index,
        "participant_count": fragment.participant.participant_count,
        "entry_count": fragment.entry_count,
        "page_count": fragment.page_count,
        "schema_digest": fragment.schema_digest,
        "content_digest": fragment.content_digest,
        "dimensions": list(fragment.specification.dimensions),
        "embedding_dtype": fragment.specification.embedding_dtype,
        "optimizer_dtype": fragment.specification.optimizer_dtype or "",
        "score_policy": fragment.specification.score_policy,
        "key_dtype": fragment.specification.key_dtype,
        "mode": fragment.save_context.mode.value,
        "parent_checkpoint_name": parent.checkpoint_name if parent else "",
        "parent_checkpoint_version": parent.checkpoint_version if parent else 0,
        "parent_reference_id": fragment.save_context.parent_reference_id,
        "mutation_watermark": fragment.save_context.mutation_watermark,
        "include_page_images": fragment.save_context.include_page_images,
        "entries_per_page": fragment.save_context.entries_per_page,
        "capture_id": fragment.save_context.capture_id,
        "journal_identity": fragment.save_context.journal_identity,
        "page_dataset": fragment.page_dataset,
        "page_version": fragment.page_version,
        "page_layout_generation": fragment.page_layout_generation,
    }
    return msgpack.packb(payload, use_bin_type=True, strict_types=True)


def decode_checkpoint_fragment(payload: bytes) -> CheckpointFragment:
    """Decode and validate one opaque sparse fragment payload."""

    if not isinstance(payload, bytes) or not payload:
        raise TypeError("sparse fragment payload must be non-empty bytes")
    try:
        decoded = msgpack.unpackb(
            payload,
            raw=False,
            strict_map_key=True,
            object_pairs_hook=_reject_duplicate_fields,
        )
    except (TypeError, ValueError, msgpack.exceptions.UnpackException) as error:
        raise ValueError("sparse fragment wire payload is malformed") from error
    if not isinstance(decoded, dict) or frozenset(decoded) != _FRAGMENT_FIELDS:
        raise ValueError("sparse fragment wire payload fields are invalid")

    parent_name = decoded["parent_checkpoint_name"]
    parent_version = decoded["parent_checkpoint_version"]
    if not isinstance(parent_name, str) or not isinstance(parent_version, int):
        raise TypeError("sparse fragment parent identity is invalid")
    if bool(parent_name) != (parent_version > 0):
        raise ValueError("sparse fragment parent identity is incomplete")
    parent = (
        SparseCheckpointLocator(parent_name, parent_version) if parent_name else None
    )
    specification = TableSpecification(
        dimensions=tuple(decoded["dimensions"]),
        embedding_dtype=decoded["embedding_dtype"],
        optimizer_dtype=decoded["optimizer_dtype"] or None,
        score_policy=decoded["score_policy"],
        key_dtype=decoded["key_dtype"],
    )
    context = CheckpointSaveContext(
        mode=CheckpointSaveMode(decoded["mode"]),
        parent=parent,
        parent_reference_id=decoded["parent_reference_id"],
        mutation_watermark=decoded["mutation_watermark"],
        include_page_images=decoded["include_page_images"],
        entries_per_page=decoded["entries_per_page"],
        capture_id=decoded["capture_id"],
        journal_identity=decoded["journal_identity"],
    )
    return CheckpointFragment(
        fragment_id=decoded["fragment_id"],
        checkpoint_name=decoded["checkpoint_name"],
        save_attempt=decoded["save_attempt"],
        participant=CheckpointParticipant(
            decoded["participant_index"], decoded["participant_count"]
        ),
        entry_count=decoded["entry_count"],
        page_count=decoded["page_count"],
        schema_digest=decoded["schema_digest"],
        content_digest=decoded["content_digest"],
        specification=specification,
        save_context=context,
        page_dataset=decoded["page_dataset"],
        page_version=decoded["page_version"],
        page_layout_generation=decoded["page_layout_generation"],
    )
