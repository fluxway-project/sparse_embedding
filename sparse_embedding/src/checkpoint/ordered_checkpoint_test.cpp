#include "sparse_embedding/src/checkpoint/checkpoint_test_support.hpp"
#include "sparse_embedding/src/checkpoint/ordered_checkpoint.hpp"

#include <array>
#include <vector>

namespace {

using se::checkpoint::CheckpointEntryRecord;
using Records = se::checkpoint::ExternalSortedRecords<
    CheckpointEntryRecord, std::less<CheckpointEntryRecord>, 2, 3>;

using namespace se::checkpoint::test;

CheckpointEntryRecord record(uint64_t key, uint32_t dimension,
                             uint64_t checkpoint, uint32_t participant,
                             uint64_t sequence, uint64_t deleted,
                             uint64_t payload) {
  return {key,     checkpoint, checkpoint * 2 + participant,
          0,       sequence,   payload,
          deleted, dimension,  participant};
}

void check_updates_deletes_and_dimension_identity() {
  Records records;
  // Deliberately interleaved input: only validated checkpoint/source order
  // wins.
  for (auto entry : {
           record(7, 2, 2, 0, 4, 0, 400),
           record(8, 2, 0, 1, 0, 0, 50),
           record(7, 4, 0, 0, 0, 0, 200),
           record(7, 2, 0, 0, 0, 0, 100),
           record(9, 2, 2, 1, 6, 1, CheckpointEntryRecord::kNoPayload),
           record(8, 2, 1, 1, 1, 0, 500),
           record(7, 2, 1, 0, 2, 1, CheckpointEntryRecord::kNoPayload),
           record(9, 2, 0, 1, 0, 0, 60),
       })
    records.append(entry);
  std::vector<CheckpointEntryRecord> winners;
  se::checkpoint::reduce_checkpoint_entries(
      records, [&](const auto &entry) { winners.push_back(entry); });
  require(winners.size() == 4, "reconciliation lost a logical identity");
  require(winners[0].key == 7 && winners[0].payload_offset == 400 &&
              winners[1].key == 8 && winners[1].payload_offset == 500 &&
              winners[2].key == 9 && winners[2].deleted == 1 &&
              winners[3].dimension == 4 && winners[3].payload_offset == 200,
          "reconciliation chose an old, deleted, or wrong-dimension payload");
}

void check_history_rejection_across_runs() {
  for (const auto &conflict : {
           record(7, 2, 0, 1, 0, 0, 500), // same checkpoint, other participant
           record(7, 2, 0, 0, 1, 1, 500), // same checkpoint, tombstone
       }) {
    Records records;
    records.append(record(7, 2, 0, 0, 0, 0, 100));
    for (uint64_t key = 100; key < 221; ++key)
      records.append(record(key, 2, 0, 0, 0, 0, 0));
    records.append(conflict);
    require_throws<std::invalid_argument>(
        [&] {
          se::checkpoint::reduce_checkpoint_entries(records,
                                                    [](const auto &) {});
        },
        "cross-run duplicate checkpoint identity was accepted");
  }
  for (const auto &conflict : {
           record(7, 2, 2, 1, 3, 0, 500), // owner changes after deletion
           record(7, 2, 2, 0, 2, 0, 500), // mutation sequence repeats
       }) {
    Records records;
    records.append(record(7, 2, 0, 0, 0, 0, 100));
    records.append(record(7, 2, 1, 0, 2, 1, CheckpointEntryRecord::kNoPayload));
    for (uint64_t key = 100; key < 221; ++key)
      records.append(record(key, 2, 0, 0, 0, 0, 0));
    records.append(conflict);
    require_throws<std::runtime_error>(
        [&] {
          se::checkpoint::reduce_checkpoint_entries(records,
                                                    [](const auto &) {});
        },
        "cross-level historical owner or sequence regression was accepted");
  }
}

void check_participant_sequence_namespace() {
  using Identity = se::checkpoint::MutationSequenceIdentity;
  using Sequences =
      se::checkpoint::ExternalSortedRecords<Identity, std::less<Identity>, 2,
                                            3>;
  Sequences independent;
  for (Identity sequence :
       {Identity{1, 0, 7}, Identity{1, 1, 7}, Identity{2, 0, 7}})
    independent.append(sequence);
  se::checkpoint::validate_unique_records(independent, "duplicate mutation");

  Sequences duplicates;
  duplicates.append({1, 0, 7});
  for (uint64_t sequence = 100; sequence < 221; ++sequence)
    duplicates.append({1, 0, sequence});
  duplicates.append({1, 0, 7});
  require_throws<std::invalid_argument>(
      [&] {
        se::checkpoint::validate_unique_records(duplicates,
                                                "duplicate mutation");
      },
      "same participant sequence reused across runs was accepted");
}

} // namespace

int main() {
  check_whole_record_truncation(record(1, 2, 0, 0, 0, 0, 0),
                                record(2, 2, 0, 0, 0, 0, 64));
  check_updates_deletes_and_dimension_identity();
  check_history_rejection_across_runs();
  check_participant_sequence_namespace();
}
