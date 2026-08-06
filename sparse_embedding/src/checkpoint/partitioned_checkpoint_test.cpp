#include "sparse_embedding/src/checkpoint/partitioned_checkpoint.hpp"

#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function>
void require_throws(Function &&function, const char *message) {
  try {
    std::forward<Function>(function)();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(message);
}

} // namespace

int main() {
  constexpr uint32_t kWorkers = 8;
  constexpr uint32_t kShards = 64;
  constexpr uint64_t kEpoch = 7;
  se::CheckpointPartitioner partitioner({kWorkers, kShards});
  require(partitioner.lance_filter(3, kEpoch) ==
              "snapshot_id = 7 AND worker_id = 3",
          "checkpoint Lance worker filter mismatch");

  std::vector<se::CheckpointLayoutRow> rows;
  for (uint32_t worker = 0; worker < kWorkers; ++worker) {
    for (uint32_t shard : partitioner.shards_for_worker(worker)) {
      rows.push_back({kEpoch, 11, 1, shard, 1, shard, 0, shard, worker, shard,
                      se::CheckpointRowOperation::kUpsert,
                      se::CheckpointPageTier::kHot});
    }
  }
  se::validate_checkpoint_layout_rows(partitioner, rows);

  std::vector<se::WorkerCheckpointArtifact> artifacts(kWorkers);
  std::vector<std::thread> workers;
  workers.reserve(kWorkers);
  for (uint32_t worker = 0; worker < kWorkers; ++worker) {
    workers.emplace_back([&, worker] {
      se::WorkerCheckpointArtifact artifact{kEpoch, 11, worker, {}, {}};
      for (uint32_t shard : partitioner.shards_for_worker(worker)) {
        artifact.rows.push_back({kEpoch, 11, 1, shard, 1, shard, 0, shard,
                                 worker, shard,
                                 se::CheckpointRowOperation::kUpsert,
                                 se::CheckpointPageTier::kHot});
        artifact.pages.push_back({kEpoch, 11, shard, shard + 1, worker, shard,
                                  1, 64, se::CheckpointPageTier::kHot,
                                  std::vector<std::byte>(64)});
      }
      artifacts[worker] = std::move(artifact);
    });
  }
  for (std::thread &worker : workers)
    worker.join();
  const std::vector<uint64_t> initial_high_watermarks(kShards, 0);
  se::PartitionedCheckpointArtifacts batch = se::aggregate_checkpoint_artifacts(
      partitioner, kEpoch, 11, artifacts, initial_high_watermarks);
  require(batch.snapshot_id == kEpoch, "checkpoint commit epoch mismatch");
  require(batch.rows.size() == kShards,
          "checkpoint commit row coverage mismatch");
  require(batch.pages.size() == kShards,
          "checkpoint commit page coverage mismatch");
  require(batch.shard_high_watermarks.size() == kShards,
          "checkpoint manifest high-watermark size mismatch");
  for (uint64_t sequence : batch.shard_high_watermarks)
    require(sequence == 1, "checkpoint manifest high-watermark mismatch");

  std::vector<se::WorkerCheckpointArtifact> second_artifacts(kWorkers);
  for (uint32_t worker = 0; worker < kWorkers; ++worker) {
    se::WorkerCheckpointArtifact artifact{kEpoch + 1, 12, worker, {}, {}};
    for (uint32_t shard : partitioner.shards_for_worker(worker)) {
      artifact.rows.push_back(
          {kEpoch + 1, 12, 2, shard, 2, shard, 1, shard, worker, shard,
           se::CheckpointRowOperation::kUpsert, se::CheckpointPageTier::kHot});
    }
    second_artifacts[worker] = std::move(artifact);
  }
  auto second_batch = se::aggregate_checkpoint_artifacts(
      partitioner, kEpoch + 1, 12, second_artifacts,
      batch.shard_high_watermarks);
  for (uint64_t sequence : second_batch.shard_high_watermarks)
    require(sequence == 2, "incremental high-watermark did not advance");
  se::PartitionedSnapshotManifest manifest =
      se::make_partitioned_snapshot_manifest(partitioner, second_batch, 100,
                                             101);
  se::validate_partitioned_snapshot_manifest(manifest);

  auto duplicate_worker_artifacts = artifacts;
  duplicate_worker_artifacts[1].worker_id = 0;
  require_throws(
      [&] {
        (void)se::aggregate_checkpoint_artifacts(partitioner, kEpoch, 11,
                                                 duplicate_worker_artifacts,
                                                 initial_high_watermarks);
      },
      "duplicate Fluxway worker metadata was accepted");

  auto sequence_gap_artifacts = artifacts;
  sequence_gap_artifacts[0].rows[0].shard_sequence = 2;
  require_throws(
      [&] {
        (void)se::aggregate_checkpoint_artifacts(partitioner, kEpoch, 11,
                                                 sequence_gap_artifacts,
                                                 initial_high_watermarks);
      },
      "checkpoint artifact sequence gap was accepted");

  auto foreign_owner_artifacts = artifacts;
  foreign_owner_artifacts[0].rows[0].worker_id = 1;
  require_throws(
      [&] {
        (void)se::aggregate_checkpoint_artifacts(partitioner, kEpoch, 11,
                                                 foreign_owner_artifacts,
                                                 initial_high_watermarks);
      },
      "foreign worker artifact row was accepted");

  std::vector<se::CheckpointLayoutRow> incremental_rows{
      {5, 9, 1, 100, 1, 1, 0, 3, 0, 0, se::CheckpointRowOperation::kUpsert,
       se::CheckpointPageTier::kCold},
      {6, 10, 2, 100, 2, 2, 0, 4, 0, 0, se::CheckpointRowOperation::kUpsert,
       se::CheckpointPageTier::kHot},
      {7, 11, 3, 100, 3, std::numeric_limits<uint64_t>::max(),
       std::numeric_limits<uint32_t>::max(), 4, 0, 0,
       se::CheckpointRowOperation::kDelete, se::CheckpointPageTier::kCold},
      {6, 10, 1, 101, 1, 3, 1, 5, 0, 0, se::CheckpointRowOperation::kUpsert,
       se::CheckpointPageTier::kHot},
  };
  auto epoch_six =
      se::reduce_worker_checkpoint_rows(partitioner, 0, 6, incremental_rows);
  require(epoch_six.size() == 2, "incremental epoch-six reduction mismatch");
  auto epoch_seven =
      se::reduce_worker_checkpoint_rows(partitioner, 0, 7, incremental_rows);
  require(epoch_seven.size() == 1 && epoch_seven[0].key == 101,
          "incremental delete tombstone did not win");
}
