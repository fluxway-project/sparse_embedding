#pragma once

#include <optional>
#include <tuple>

#include "sparse_embedding/src/checkpoint/external_sort.hpp"

namespace se::checkpoint {

struct ActiveKeyIdentity {
  uint64_t dimension;
  uint64_t key;

  bool operator==(const ActiveKeyIdentity &) const = default;
  bool operator<(const ActiveKeyIdentity &other) const noexcept {
    return std::tie(dimension, key) < std::tie(other.dimension, other.key);
  }
};

struct MutationSequenceIdentity {
  uint64_t checkpoint;
  uint64_t participant;
  uint64_t sequence;

  bool operator==(const MutationSequenceIdentity &) const = default;
  bool operator<(const MutationSequenceIdentity &other) const noexcept {
    return std::tie(checkpoint, participant, sequence) <
           std::tie(other.checkpoint, other.participant, other.sequence);
  }
};

// Full cache lines hold named fields only: no implicit padding reaches scratch.
// Physical page sequence belongs to payload, never to logical winner ordering.
struct alignas(64) CheckpointEntryRecord {
  static constexpr uint64_t kNoPayload = std::numeric_limits<uint64_t>::max();
  uint64_t key;
  uint64_t checkpoint;
  uint64_t source;
  uint64_t row;
  uint64_t mutation_sequence;
  uint64_t payload_offset;
  uint64_t deleted;
  uint32_t dimension;
  uint32_t participant;

  bool same_identity(const CheckpointEntryRecord &other) const noexcept {
    return dimension == other.dimension && key == other.key;
  }

  bool operator<(const CheckpointEntryRecord &other) const noexcept {
    return std::tie(dimension, key, checkpoint, source, row) <
           std::tie(other.dimension, other.key, other.checkpoint, other.source,
                    other.row);
  }
};

static_assert(sizeof(CheckpointEntryRecord) == 64);
static_assert(std::has_unique_object_representations_v<CheckpointEntryRecord>);

template <typename SortedRecords>
void validate_unique_records(SortedRecords &records, const char *message) {
  using Record = typename SortedRecords::value_type;
  std::optional<Record> previous;
  records.consume_sorted([&](const Record &record) {
    if (previous && *previous == record)
      throw std::invalid_argument(message);
    previous = record;
    return true;
  });
}

// The consumer stages selection metadata only. Payload publication/import must
// start after this function succeeds for the entire chain, including
// tombstones.
template <typename SortedRecords, typename Consumer>
void reduce_checkpoint_entries(SortedRecords &records, Consumer &&consume) {
  std::optional<CheckpointEntryRecord> previous;
  records.consume_sorted([&](const CheckpointEntryRecord &record) {
    if (record.deleted > 1)
      throw std::runtime_error("checkpoint row state is invalid");
    if (previous && previous->same_identity(record)) {
      if (previous->checkpoint == record.checkpoint)
        throw std::invalid_argument(
            "duplicate active checkpoint key or tombstone identity");
      if (previous->participant != record.participant)
        throw std::runtime_error(
            "checkpoint key ownership changed across incremental chain");
      if (record.mutation_sequence <= previous->mutation_sequence)
        throw std::runtime_error(
            "checkpoint mutation sequence is duplicate or retrograde");
    } else if (previous) {
      consume(*previous);
    }
    previous = record;
    return true;
  });
  if (previous)
    consume(*previous);
}

} // namespace se::checkpoint
