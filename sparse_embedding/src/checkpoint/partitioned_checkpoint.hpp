#pragma once

#ifndef SPARSE_EMBEDDING_PARTITIONED_CHECKPOINT_HPP_
#define SPARSE_EMBEDDING_PARTITIONED_CHECKPOINT_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace se {

enum class CheckpointRowOperation : uint8_t { kUpsert, kDelete };
enum class CheckpointPageTier : uint8_t { kHot, kCold };

struct CheckpointPartitionConfig {
  uint32_t worker_count;
  uint32_t shard_count;
};

struct CheckpointPartition {
  uint32_t worker_id;
  uint32_t shard_id;
};

struct CheckpointLayoutRow {
  uint64_t snapshot_id;
  uint64_t page_layout_generation;
  uint64_t shard_sequence;
  uint64_t key;
  uint64_t seq_count;
  uint64_t page_ordinal;
  uint32_t slot_ordinal;
  uint32_t cluster_id;
  uint32_t worker_id;
  uint32_t shard_id;
  CheckpointRowOperation operation;
  CheckpointPageTier tier;
};

struct CheckpointPageImage {
  uint64_t snapshot_id;
  uint64_t page_layout_generation;
  uint64_t page_ordinal;
  uint64_t checksum;
  uint32_t worker_id;
  uint32_t shard_id;
  uint32_t entry_count;
  uint32_t vector_stride;
  CheckpointPageTier tier;
  std::vector<std::byte> payload;
};

struct WorkerCheckpointArtifact {
  uint64_t snapshot_id;
  uint64_t page_layout_generation;
  uint32_t worker_id;
  std::vector<CheckpointLayoutRow> rows;
  std::vector<CheckpointPageImage> pages;
};

struct PartitionedCheckpointArtifacts {
  uint64_t snapshot_id;
  uint64_t page_layout_generation;
  std::vector<CheckpointLayoutRow> rows;
  std::vector<CheckpointPageImage> pages;
  std::vector<uint64_t> shard_high_watermarks;
};

struct PartitionedSnapshotManifest {
  uint32_t schema_version;
  uint32_t worker_count;
  uint32_t shard_count;
  uint64_t snapshot_id;
  uint64_t page_layout_generation;
  uint64_t entries_lance_version;
  uint64_t page_images_lance_version;
  std::vector<uint64_t> shard_high_watermarks;
};

inline void validate_partitioned_snapshot_manifest(
    const PartitionedSnapshotManifest &manifest) {
  if (manifest.schema_version == 0 || manifest.worker_count == 0 ||
      manifest.shard_count == 0 || manifest.snapshot_id == 0 ||
      manifest.entries_lance_version == 0)
    throw std::invalid_argument("partitioned snapshot manifest is incomplete");
  if (manifest.shard_high_watermarks.size() != manifest.shard_count)
    throw std::invalid_argument(
        "partitioned snapshot high-watermark count mismatch");
}

class CheckpointPartitioner {
public:
  explicit CheckpointPartitioner(CheckpointPartitionConfig config)
      : config_(config) {
    if (config.worker_count == 0 || config.shard_count == 0)
      throw std::invalid_argument(
          "checkpoint worker and shard counts must be non-zero");
  }

  CheckpointPartition partition_for_shard(uint32_t shard_id) const {
    if (shard_id >= config_.shard_count)
      throw std::out_of_range("checkpoint shard id is out of range");
    return {shard_id % config_.worker_count, shard_id};
  }

  bool owns(uint32_t worker_id, uint32_t shard_id) const {
    if (worker_id >= config_.worker_count)
      throw std::out_of_range("checkpoint worker id is out of range");
    return partition_for_shard(shard_id).worker_id == worker_id;
  }

  std::vector<uint32_t> shards_for_worker(uint32_t worker_id) const {
    if (worker_id >= config_.worker_count)
      throw std::out_of_range("checkpoint worker id is out of range");
    std::vector<uint32_t> result;
    result.reserve((config_.shard_count + config_.worker_count - 1) /
                   config_.worker_count);
    for (uint32_t shard_id = worker_id; shard_id < config_.shard_count;
         shard_id += config_.worker_count)
      result.push_back(shard_id);
    return result;
  }

  std::string lance_filter(uint32_t worker_id, uint64_t snapshot_id) const {
    if (worker_id >= config_.worker_count)
      throw std::out_of_range("checkpoint worker id is out of range");
    return "snapshot_id = " + std::to_string(snapshot_id) +
           " AND worker_id = " + std::to_string(worker_id);
  }

  CheckpointPartitionConfig config() const noexcept { return config_; }

private:
  CheckpointPartitionConfig config_;
};

inline void
validate_checkpoint_layout_rows(const CheckpointPartitioner &partitioner,
                                std::span<const CheckpointLayoutRow> rows) {
  for (const CheckpointLayoutRow &row : rows) {
    const CheckpointPartition expected =
        partitioner.partition_for_shard(row.shard_id);
    if (row.worker_id != expected.worker_id)
      throw std::invalid_argument(
          "checkpoint row worker does not own its shard");
    if (row.operation == CheckpointRowOperation::kUpsert && row.seq_count == 0)
      throw std::invalid_argument(
          "checkpoint upsert seq_count must be non-zero");
    if (row.operation == CheckpointRowOperation::kDelete &&
        (row.page_ordinal != std::numeric_limits<uint64_t>::max() ||
         row.slot_ordinal != std::numeric_limits<uint32_t>::max()))
      throw std::invalid_argument(
          "checkpoint delete rows must not reference a page slot");
  }
}

inline void validate_worker_checkpoint_artifact(
    const CheckpointPartitioner &partitioner,
    const WorkerCheckpointArtifact &artifact,
    std::span<const uint64_t> committed_sequences) {
  if (artifact.snapshot_id == 0 || artifact.page_layout_generation == 0)
    throw std::invalid_argument("checkpoint artifact epoch is incomplete");
  if (artifact.worker_id >= partitioner.config().worker_count)
    throw std::out_of_range("checkpoint worker id is out of range");
  if (committed_sequences.size() != partitioner.config().shard_count)
    throw std::invalid_argument(
        "checkpoint committed high-watermark count mismatch");

  validate_checkpoint_layout_rows(partitioner, artifact.rows);
  std::vector<uint64_t> next_sequence(committed_sequences.begin(),
                                      committed_sequences.end());
  for (const CheckpointLayoutRow &row : artifact.rows) {
    if (row.snapshot_id != artifact.snapshot_id ||
        row.page_layout_generation != artifact.page_layout_generation ||
        row.worker_id != artifact.worker_id)
      throw std::invalid_argument(
          "checkpoint row does not belong to its worker artifact");
    if (row.shard_sequence != next_sequence[row.shard_id] + 1)
      throw std::invalid_argument(
          "checkpoint shard sequence contains a gap or duplicate");
    next_sequence[row.shard_id] = row.shard_sequence;
  }
  for (const CheckpointPageImage &page : artifact.pages) {
    if (page.snapshot_id != artifact.snapshot_id ||
        page.page_layout_generation != artifact.page_layout_generation ||
        page.worker_id != artifact.worker_id ||
        !partitioner.owns(artifact.worker_id, page.shard_id))
      throw std::invalid_argument(
          "checkpoint page does not belong to its worker artifact");
    if (page.payload.empty() || page.entry_count == 0 ||
        page.vector_stride == 0)
      throw std::invalid_argument("checkpoint page image is empty");
  }
}

inline PartitionedCheckpointArtifacts aggregate_checkpoint_artifacts(
    const CheckpointPartitioner &partitioner, uint64_t snapshot_id,
    uint64_t page_layout_generation,
    std::span<const WorkerCheckpointArtifact> worker_artifacts,
    std::span<const uint64_t> committed_sequences) {
  if (snapshot_id == 0 || page_layout_generation == 0)
    throw std::invalid_argument("checkpoint aggregation epoch is incomplete");
  if (worker_artifacts.size() != partitioner.config().worker_count)
    throw std::invalid_argument(
        "checkpoint artifacts do not cover all workers");
  if (committed_sequences.size() != partitioner.config().shard_count)
    throw std::invalid_argument(
        "checkpoint committed high-watermark count mismatch");

  std::vector<bool> seen_worker(partitioner.config().worker_count, false);
  size_t row_count = 0;
  size_t page_count = 0;
  for (const WorkerCheckpointArtifact &artifact : worker_artifacts) {
    if (artifact.snapshot_id != snapshot_id ||
        artifact.page_layout_generation != page_layout_generation)
      throw std::invalid_argument(
          "checkpoint artifact belongs to another epoch");
    if (seen_worker[artifact.worker_id])
      throw std::invalid_argument("checkpoint worker artifact is duplicated");
    seen_worker[artifact.worker_id] = true;
    validate_worker_checkpoint_artifact(partitioner, artifact,
                                        committed_sequences);
    row_count += artifact.rows.size();
    page_count += artifact.pages.size();
  }

  PartitionedCheckpointArtifacts result{
      snapshot_id,
      page_layout_generation,
      {},
      {},
      {committed_sequences.begin(), committed_sequences.end()}};
  result.rows.reserve(row_count);
  result.pages.reserve(page_count);
  for (const WorkerCheckpointArtifact &artifact : worker_artifacts) {
    result.rows.insert(result.rows.end(), artifact.rows.begin(),
                       artifact.rows.end());
    result.pages.insert(result.pages.end(), artifact.pages.begin(),
                        artifact.pages.end());
  }
  for (const CheckpointLayoutRow &row : result.rows)
    result.shard_high_watermarks[row.shard_id] = row.shard_sequence;
  return result;
}

inline PartitionedSnapshotManifest make_partitioned_snapshot_manifest(
    const CheckpointPartitioner &partitioner,
    const PartitionedCheckpointArtifacts &artifacts,
    uint64_t entries_lance_version, uint64_t page_images_lance_version) {
  PartitionedSnapshotManifest manifest{
      1,
      partitioner.config().worker_count,
      partitioner.config().shard_count,
      artifacts.snapshot_id,
      artifacts.page_layout_generation,
      entries_lance_version,
      page_images_lance_version,
      artifacts.shard_high_watermarks,
  };
  validate_partitioned_snapshot_manifest(manifest);
  return manifest;
}

inline std::vector<CheckpointLayoutRow>
reduce_worker_checkpoint_rows(const CheckpointPartitioner &partitioner,
                              uint32_t worker_id, uint64_t snapshot_id,
                              std::span<const CheckpointLayoutRow> rows) {
  if (snapshot_id == 0)
    throw std::invalid_argument("snapshot epoch zero is reserved");
  if (worker_id >= partitioner.config().worker_count)
    throw std::out_of_range("checkpoint worker id is out of range");

  struct KeyIdentity {
    uint32_t shard_id;
    uint64_t key;

    bool operator==(const KeyIdentity &) const = default;
  };
  struct KeyIdentityHash {
    size_t operator()(const KeyIdentity &identity) const noexcept {
      uint64_t value =
          identity.key ^ (static_cast<uint64_t>(identity.shard_id) << 32);
      value ^= value >> 30;
      value *= 0xbf58476d1ce4e5b9ULL;
      value ^= value >> 27;
      value *= 0x94d049bb133111ebULL;
      value ^= value >> 31;
      return static_cast<size_t>(value);
    }
  };

  std::unordered_map<KeyIdentity, CheckpointLayoutRow, KeyIdentityHash> latest;
  latest.reserve(rows.size());
  for (const CheckpointLayoutRow &row : rows) {
    if (row.snapshot_id > snapshot_id)
      continue;
    if (row.worker_id != worker_id ||
        !partitioner.owns(worker_id, row.shard_id))
      throw std::invalid_argument(
          "checkpoint reader received a foreign worker row");
    KeyIdentity identity{row.shard_id, row.key};
    auto found = latest.find(identity);
    const auto order = [](const CheckpointLayoutRow &value) {
      return std::tuple(value.snapshot_id, value.shard_sequence,
                        value.seq_count, static_cast<uint8_t>(value.operation));
    };
    if (found == latest.end() || order(found->second) < order(row))
      latest.insert_or_assign(identity, row);
  }

  std::vector<CheckpointLayoutRow> result;
  result.reserve(latest.size());
  for (auto &[_, row] : latest) {
    if (row.operation == CheckpointRowOperation::kUpsert)
      result.push_back(std::move(row));
  }
  std::ranges::sort(result, [](const CheckpointLayoutRow &left,
                               const CheckpointLayoutRow &right) {
    return std::tie(left.shard_id, left.page_ordinal, left.slot_ordinal,
                    left.key) < std::tie(right.shard_id, right.page_ordinal,
                                         right.slot_ordinal, right.key);
  });
  return result;
}

} // namespace se

#endif
