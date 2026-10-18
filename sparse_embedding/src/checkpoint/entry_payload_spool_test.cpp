#include "sparse_embedding/src/checkpoint/checkpoint_test_support.hpp"
#include "sparse_embedding/src/checkpoint/entry_payload_spool.hpp"

#include <cstring>

namespace {

using namespace se::checkpoint::test;

se::checkpoint::EntryPayloadHeader header(uint64_t key, uint32_t dimension,
                                          uint32_t optimizer_stride = 0) {
  return {key,
          key + 10,
          key + 20,
          dimension,
          optimizer_stride,
          7,
          static_cast<uint32_t>(dimension * sizeof(float)),
          static_cast<uint32_t>(optimizer_stride * sizeof(float)),
          se::checkpoint::payload_dtype_code("float32"),
          se::checkpoint::payload_dtype_code("float32")};
}

void check_surviving_payload_and_physical_state() {
  using namespace se::checkpoint;
  EntryPayloadSpool spool(true);
  const std::array<float, 2> old_values{1, 2};
  const std::array<float, 2> new_values{3, 4};
  const std::array<float, 4> optimizer{5, 6, 7, 8};
  (void)spool.append(header(1, 2, 4), std::as_bytes(std::span(old_values)),
                     std::as_bytes(std::span(optimizer)));
  const auto winner =
      spool.append(header(1, 2, 4), std::as_bytes(std::span(new_values)),
                   std::as_bytes(std::span(optimizer)));
  (void)spool.append(header(2, 2, 4), std::as_bytes(std::span(old_values)),
                     std::as_bytes(std::span(optimizer)));
  spool.seal();
  ExternalSortedRecords<uint64_t, std::less<uint64_t>, 2, 3> offsets;
  offsets.append(winner);
  size_t calls = 0;
  spool.consume_selected(offsets, [&](const se::NativeEntryBatch &batch) {
    ++calls;
    require(batch.keys == std::vector<uint64_t>{1} &&
                batch.dimensions == std::vector<uint32_t>{2},
            "obsolete staged key reached materialization");
    require(batch.sequences == std::vector<uint64_t>{21} &&
                batch.scores == std::vector<uint64_t>{11} &&
                batch.steps == std::vector<uint32_t>{7},
            "physical sequence, score, or step was lost");
    require(batch.embedding_values.size() == sizeof(new_values) &&
                std::memcmp(batch.embedding_values.data(), new_values.data(),
                            sizeof(new_values)) == 0 &&
                batch.optimizer_values.size() == sizeof(optimizer) &&
                std::memcmp(batch.optimizer_values.data(), optimizer.data(),
                            sizeof(optimizer)) == 0,
            "staged payload bytes changed");
    return true;
  });
  require(calls == 1 && spool.statistics().rows_staged == 3 &&
              spool.statistics().rows_selected == 1 &&
              spool.statistics().logical_bytes_read * 3 ==
                  spool.statistics().logical_bytes_written,
          "payload replay reread obsolete versions");
}

void check_byte_bounded_batches_and_forward_replay() {
  using namespace se::checkpoint;
  EntryPayloadSpool spool(false);
  constexpr uint32_t dimension = 2048;
  std::array<float, dimension> values{};
  const auto value_bytes = std::as_bytes(std::span(values));
  constexpr size_t rows =
      se::CheckpointStreamPolicy::kEntryImportBatchBytes / sizeof(values) + 3;
  std::vector<uint64_t> locations;
  for (size_t index = 0; index < rows; ++index) {
    auto entry = header(index, dimension);
    entry.optimizer_dtype = kNoPayloadDtype;
    locations.push_back(spool.append(entry, value_bytes, {}));
  }
  spool.seal();
  ExternalSortedRecords<uint64_t, std::less<uint64_t>, 4, 3> offsets;
  for (auto position = locations.rbegin(); position != locations.rend();
       ++position)
    offsets.append(*position);
  uint64_t expected_key = 0;
  size_t calls = 0;
  spool.consume_selected(offsets, [&](const se::NativeEntryBatch &batch) {
    ++calls;
    require(batch.embedding_values.size() <=
                se::CheckpointStreamPolicy::kEntryImportBatchBytes,
            "staged materialization ignored the payload byte budget");
    for (uint64_t key : batch.keys)
      require(key == expected_key++,
              "selected payload was not replayed in source order");
    require(batch.optimizer_values.empty(),
            "weight-only replay loaded optimizer data");
    return true;
  });
  require(calls == 2 && expected_key == rows &&
              spool.statistics().seek_operations == 0 &&
              spool.statistics().peak_batch_rows <
                  se::CheckpointStreamPolicy::kScanBatchRows,
          "byte budget did not bound mixed materialization batches");
}

void check_cancellation_and_errors() {
  using namespace se::checkpoint;
  ScratchRoot root;
  const std::array<float, 1> values{1};
  for (bool fail : {false, true}) {
    EntryPayloadSpool spool(true);
    ExternalSortedRecords<uint64_t> offsets;
    for (size_t row = 0; row <= se::CheckpointStreamPolicy::kScanBatchRows;
         ++row)
      offsets.append(
          spool.append(header(row, 1), std::as_bytes(std::span(values)), {}));
    spool.seal();
    auto consume = [&] {
      return spool.consume_selected(
          offsets, [&](const se::NativeEntryBatch &) -> bool {
            if (fail)
              throw std::domain_error("injected payload visitor failure");
            return false;
          });
    };
    if (fail)
      require_throws<std::domain_error>(consume,
                                        "payload visitor error was hidden");
    else {
      require(!consume(), "payload visitor cancellation was ignored");
      require(spool.statistics().logical_bytes_read ==
                  se::CheckpointStreamPolicy::kScanBatchRows *
                          (sizeof(EntryPayloadHeader) + sizeof(values)) +
                      sizeof(EntryPayloadHeader),
              "cancelled replay did not account for its buffered header");
    }
    require(std::filesystem::is_empty(root.path()),
            "cancelled or failed payload replay leaked scratch");
  }
  for (uint64_t invalid : {uint64_t(0), uint64_t(8), uint64_t(4096)}) {
    EntryPayloadSpool spool(true);
    ExternalSortedRecords<uint64_t> offsets;
    offsets.append(
        spool.append(header(0, 1), std::as_bytes(std::span(values)), {}));
    offsets.append(invalid);
    spool.seal();
    require_throws<std::runtime_error>(
        [&] {
          spool.consume_selected(offsets, [](const auto &) { return true; });
        },
        "overlapping, duplicate or out-of-range payload offset was accepted");
    require(std::filesystem::is_empty(root.path()),
            "invalid payload offset leaked scratch");
  }
}

void check_large_singleton_row() {
  using namespace se::checkpoint;
  const uint32_t dimension =
      se::CheckpointStreamPolicy::kEntryImportBatchBytes / sizeof(float) + 1;
  std::vector<float> values(dimension, 1);
  EntryPayloadSpool spool(true);
  ExternalSortedRecords<uint64_t> offsets;
  offsets.append(
      spool.append(header(0, dimension), std::as_bytes(std::span(values)), {}));
  offsets.append(spool.append(header(1, 1),
                              std::as_bytes(std::span(values).first(1)), {}));
  spool.seal();
  size_t calls = 0;
  spool.consume_selected(offsets, [&](const se::NativeEntryBatch &batch) {
    require(batch.keys.size() == 1,
            "oversized singleton was combined with another row");
    ++calls;
    return true;
  });
  require(calls == 2,
          "valid singleton larger than the byte budget was rejected");
}

} // namespace

int main() {
  check_surviving_payload_and_physical_state();
  check_byte_bounded_batches_and_forward_replay();
  check_cancellation_and_errors();
  check_large_singleton_row();
}
