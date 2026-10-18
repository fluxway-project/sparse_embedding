#include "sparse_embedding/src/checkpoint/checkpoint_test_support.hpp"
#include "sparse_embedding/src/checkpoint/external_sort.hpp"

#include <array>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace {

using namespace se::checkpoint::test;

void check_balanced_external_ordering() {
  ScratchRoot root;
  se::checkpoint::ExternalSortedRecords<uint64_t, std::less<uint64_t>, 4, 3>
      records;
  // Multiplication permutes 0..256; the sorted result is known independently
  // of run construction, and a partial final run exercises residual levels.
  for (uint64_t index = 0; index < 257; ++index)
    records.append((index * 37) % 257);
  uint64_t expected = 0;
  const bool complete = records.consume_sorted([&](uint64_t value) {
    require(value == expected++, "external merge did not preserve total order");
    return true;
  });
  require(complete && expected == 257, "external merge lost records");
  const auto &stats = records.statistics();
  require(stats.peak_buffer_records <= 4, "sort buffer exceeded its bound");
  require(stats.peak_merge_fan_in <= 3, "final merge exceeded its fan-in");
  require(stats.peak_run_files <= records.maximum_run_files(),
          "sort run bookkeeping grew beyond its fixed capacity");
  require(stats.bytes_written <= 257 * sizeof(uint64_t) * 7,
          "external ordering repeatedly rewrote an accumulated large run");
  require(stats.active_run_files == 0, "consumed runs were not cleaned up");
  require(std::filesystem::is_empty(root.path()),
          "external sort left scratch files");
}

void check_duplicates_and_empty_stream() {
  se::checkpoint::ExternalSortedRecords<uint64_t, std::less<uint64_t>, 2, 2>
      records;
  for (uint64_t value : {2, 1, 2, 0, 2, 1})
    records.append(value);
  constexpr std::array<uint64_t, 6> expected{0, 1, 1, 2, 2, 2};
  size_t index = 0;
  records.consume_sorted([&](uint64_t value) {
    require(index < expected.size() && value == expected[index++],
            "external infrastructure discarded or reordered a duplicate");
    return true;
  });
  require(index == expected.size(),
          "external sort discarded duplicate records");
  require_throws<std::logic_error>([&] { records.append(7); },
                                   "consumed stream accepted another record");
  require_throws<std::logic_error>(
      [&] { records.consume_sorted([](uint64_t) { return true; }); },
      "sorted stream allowed repeated consumption");

  se::checkpoint::ExternalSortedRecords<uint64_t> empty;
  require(empty.consume_sorted([](uint64_t) -> bool {
    throw std::runtime_error("empty sort emitted a record");
  }),
          "empty stream did not finish");
  require(empty.statistics().initial_runs == 0,
          "empty stream performed temporary I/O");
}

void check_cancel_and_visitor_failure() {
  ScratchRoot root;
  for (bool fail : {false, true}) {
    se::checkpoint::ExternalSortedRecords<uint64_t, std::less<uint64_t>, 2, 3>
        records;
    for (uint64_t value = 0; value < 121; ++value)
      records.append(120 - value);
    if (fail) {
      require_throws<std::domain_error>(
          [&] {
            records.consume_sorted([](uint64_t) -> bool {
              throw std::domain_error("injected visitor failure");
            });
          },
          "external sort replaced the visitor's primary exception");
    } else {
      size_t visits = 0;
      require(!records.consume_sorted([&](uint64_t) { return ++visits < 3; }) &&
                  visits == 3,
              "external sort ignored early cancellation");
    }
    require(std::filesystem::is_empty(root.path()),
            "failed or cancelled consumption left scratch files");
    require(records.statistics().active_run_files == 0,
            "failed or cancelled consumption retained runs");
  }
}

void check_io_and_arithmetic_contracts() {
  using namespace se::checkpoint;
  std::istringstream truncated(std::string(9, '\0'));
  std::array<uint64_t, 2> values{};
  require_throws<std::runtime_error>(
      [&] { (void)read_records(truncated, std::span<uint64_t>(values)); },
      "partial temporary record was accepted");
  require_throws<std::overflow_error>(
      [] { (void)checked_sum(std::numeric_limits<uint64_t>::max(), 1); },
      "temporary byte offset silently overflowed");
  require_throws<std::overflow_error>(
      [] { (void)checked_byte_count(std::numeric_limits<uint64_t>::max(), 2); },
      "temporary byte length silently overflowed");
  std::ofstream full("/dev/full", std::ios::binary);
  require(bool(full), "cannot open deterministic full-device test sink");
  require_throws<std::runtime_error>(
      [&] {
        write_records(full, std::span<const uint64_t>(values));
        finish_output(full);
      },
      "temporary output silently ignored an I/O failure");

  TemporaryRunDirectory directory;
  const auto file = directory.file("private.bin");
  const auto public_permissions =
      std::filesystem::perms::group_all | std::filesystem::perms::others_all;
  require((std::filesystem::status(directory.path()).permissions() &
           public_permissions) == std::filesystem::perms::none,
          "temporary directory was not created privately");
  require_throws<std::invalid_argument>(
      [&] { (void)directory.file("../escape"); },
      "temporary filename escaped its owner");
  const auto path = directory.path();
  directory.cleanup();
  require(!std::filesystem::exists(path), "explicit temporary cleanup failed");
}

} // namespace

int main() {
  check_whole_record_truncation<uint64_t>(1, 2);
  check_balanced_external_ordering();
  check_duplicates_and_empty_stream();
  check_cancel_and_visitor_failure();
  check_io_and_arithmetic_contracts();
}
