#include "sparse_embedding/src/backend/hogwild/core/segmented_locator_table.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <latch>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

} // namespace

int main() {
  se::hogwild_internal::SegmentedLocatorTable table;
  const uint32_t zero_index_slot = table.allocate(101, 0, {});
  require(table.key(zero_index_slot) == 101,
          "locator did not preserve its immutable key");
  require(table.load(zero_index_slot).value() == 0,
          "locator did not preserve arena index zero");
  const uint32_t first = table.allocate(102, 7, {});
  require(first == 2, "second locator slot must be two");
  require(table.load(first).value() == 7, "locator load mismatch");

  uint32_t boundary = 0;
  for (uint32_t i = 2;
       i < se::hogwild_internal::SegmentedLocatorTable::kEntriesPerSegment;
       ++i) {
    boundary = table.allocate(i + 100, i + 10, {});
  }
  require(boundary ==
              se::hogwild_internal::SegmentedLocatorTable::kEntriesPerSegment,
          "first segment boundary mismatch");
  const uint32_t next = table.allocate(9001, 9000, {});
  require(next == boundary + 1, "second segment publication mismatch");

  std::array<uint32_t, 8> payload{};
  table.write(first, [&](uint32_t index, std::atomic<uint64_t> &) {
    require(index == 7, "writer did not receive current index");
    payload.fill(11);
  });
  table.read(first, [&](uint32_t index) {
    require(index == 7, "reader did not receive current index");
    for (uint32_t value : payload)
      require(value == 11, "seqlock read observed a torn payload");
  });
  require(table.publish(first, 7, 31), "locator publication failed");
  require(table.load(first).value() == 31, "published locator mismatch");
  require(!table.publish(first, 7, 47), "stale locator publication succeeded");
  require(table.release_after_epoch(first),
          "epoch-safe locator release did not make slot reusable");
  require(!table.load(first).has_value(),
          "released locator remained observable as live");
  const uint32_t reused = table.allocate(171, 71, {});
  require(reused == first, "released locator slot was not reused");
  require(table.load(reused).value() == 71,
          "reused locator did not publish its new arena index");
  require(table.key(reused) == 171,
          "reused locator did not publish its new immutable key");
  require(!table.try_read_key(reused, 102, [](uint32_t) {}),
          "stale key read succeeded after locator reuse");
  require(table.try_read_key(
              reused, 171,
              [](uint32_t index) {
                require(index == 71,
                        "key-aware locator read returned wrong index");
              }),
          "current key read failed after locator reuse");

  const uint32_t batch_a = table.allocate(501, 81, {});
  const uint32_t batch_b = table.allocate(502, 82, {});
  std::array<uint32_t, 2> batch_slots{batch_b, batch_a};
  std::array<uint32_t, 2> batch_indices{};
  require(table.try_write_batch(batch_slots,
                                [&](std::span<const uint32_t> indices) {
                                  batch_indices = {indices[0], indices[1]};
                                }),
          "batch locator write failed");
  require(batch_indices == std::array<uint32_t, 2>{82, 81},
          "batch locator write reordered caller payload");
  bool duplicate_rejected = false;
  try {
    std::array<uint32_t, 2> duplicate{batch_a, batch_a};
    table.try_write_batch(duplicate, [](std::span<const uint32_t>) {});
  } catch (const std::invalid_argument &) {
    duplicate_rejected = true;
  }
  require(duplicate_rejected, "duplicate locator batch was accepted");

  std::array<se::hogwild_internal::SegmentedLocatorTable::KeyedWriteSlot, 2>
      keyed_batch{{{batch_b, 502}, {batch_a, 501}}};
  std::array<uint32_t, 2> keyed_indices{};
  require(table.try_write_key_batch(
              keyed_batch,
              [&](std::span<const uint32_t> indices,
                  std::span<const se::hogwild_internal::CheckpointRecordRef>) {
                keyed_indices = {indices[0], indices[1]};
              }),
          "keyed batch locator write failed");
  require(keyed_indices == std::array<uint32_t, 2>{82, 81},
          "keyed batch locator write reordered caller payload");
  keyed_batch[1].expected_key = 999;
  require(
      !table.try_write_key_batch(
          keyed_batch,
          [](std::span<const uint32_t>,
             std::span<const se::hogwild_internal::CheckpointRecordRef>) {}),
      "stale keyed batch write succeeded");

  uint32_t retired_index = 0;
  require(table.retire_current(
              reused,
              [&](uint32_t index,
                  se::hogwild_internal::CheckpointRecordRef) noexcept {
                retired_index = index;
              }),
          "live locator was not atomically retired");
  require(retired_index == 71, "locator retirement returned the wrong index");
  require(!table.load(reused).has_value(),
          "retired locator remained observable as live");
  require(!table.try_read(reused, [](uint32_t) {}),
          "retired locator remained readable");
  require(!table.try_write(reused, [](uint32_t, std::atomic<uint64_t> &) {}),
          "retired locator remained writable");
  require(table.recycle_retired_after_epoch(reused),
          "retired locator was not recycled after epoch");
  const uint32_t reused_after_retirement = table.allocate(179, 79, {});
  require(reused_after_retirement == reused,
          "atomically retired locator slot was not reused");

  const uint32_t permanent = table.allocate(183, 83, {});
  table.set_sequence_for_test(permanent,
                              std::numeric_limits<uint32_t>::max() - 1U);
  require(table.requires_rotation(permanent),
          "near-wrap locator was not marked for rotation");
  bool write_rejected = false;
  try {
    table.write(permanent, [](uint32_t, std::atomic<uint64_t> &) {});
  } catch (const se::hogwild_internal::LocatorSequenceExhausted &) {
    write_rejected = true;
  }
  require(write_rejected, "near-wrap locator accepted a wrapping write");
  require(!table.release_after_epoch(permanent),
          "near-wrap locator was made reusable");

  int old_record = 1, new_record = 2;
  const uint32_t retiring = table.allocate(
      211, 91, se::hogwild_internal::CheckpointRecordRef::from(&old_record));
  std::latch reading(1), resume_read(1), retire_started(1);
  auto reader = std::async(std::launch::async, [&] {
    return table.try_read_key_with_record(
        retiring, 211,
        [&](uint32_t index, se::hogwild_internal::CheckpointRecordRef record) {
          require(index == 91 && record.opaque_address() == &old_record,
                  "optimistic callback did not obtain original record");
          reading.count_down();
          resume_read.wait();
          require(table.checkpoint_record(retiring).opaque_address() ==
                      record.opaque_address(),
                  "retirement changed record before epoch ended");
        });
  });
  reading.wait();
  auto retirement = std::async(std::launch::async, [&] {
    retire_started.count_down();
    return table.retire_current(
        retiring,
        [&](uint32_t,
            se::hogwild_internal::CheckpointRecordRef record) noexcept {
          if (record.opaque_address() != &old_record)
            std::terminate();
        });
  });
  retire_started.wait();
  require(retirement.wait_for(std::chrono::milliseconds(50)) ==
              std::future_status::timeout,
          "locator retirement did not wait for the optimistic reader");
  const uint32_t replacement = table.allocate(
      211, 92, se::hogwild_internal::CheckpointRecordRef::from(&new_record));
  (void)replacement;
  resume_read.count_down();
  require(!reader.get(), "retired optimistic read reported success");
  require(retirement.get(), "record-bearing locator retirement failed");
  const uint32_t replacement_after_retirement = table.allocate(
      211, 92, se::hogwild_internal::CheckpointRecordRef::from(&new_record));
  require(
      table.checkpoint_record(replacement_after_retirement).opaque_address() ==
          &new_record,
      "reinsert reused an old logical record");
  require(table.recycle_retired_after_epoch(retiring),
          "record-bearing locator did not recycle after epoch");
  require(!table.checkpoint_record(retiring).has_value(),
          "epoch-safe recycle retained old record pointer");

  return 0;
}
