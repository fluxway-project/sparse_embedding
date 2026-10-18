#pragma once

// ============================================================================
// Checkpoint Bounded External Sort Engine (ExternalSortedRecords)
// ============================================================================
//
// Pipeline Architecture & Data Flow:
// ----------------------------------
//
//   [Unsorted Multi-Version Stream] (from Lance fragment delta chains)
//                 │
//                 ▼ append(record)
//   ┌────────────────────────────────────────────────────────┐
//   │ In-Memory Buffer (Capacity: RunRecords, e.g., 65,536)  │
//   └─────────────────────────┬──────────────────────────────┘
//                             │ (buffer full: std::sort + spill)
//                             ▼
//   ┌────────────────────────────────────────────────────────┐
//   │ Level 0 Scratch Runs: [Run 0] [Run 1] ... [Run FanIn-1]│
//   └─────────────────────────┬──────────────────────────────┘
//                             │ (accumulate FanIn runs: k-way heap merge)
//                             ▼
//   ┌────────────────────────────────────────────────────────┐
//   │ Level 1 Cascaded Runs: [Run 0] ...                     │
//   └─────────────────────────┬──────────────────────────────┘
//                             │ finish_runs() + heap merge
//                             ▼
//   ┌────────────────────────────────────────────────────────┐
//   │ consume_sorted(visitor) -> Globally Sorted Sequence    │
//   │ Order: (dimension, key, checkpoint_version, seq...)    │
//   └─────────────────────────┬──────────────────────────────┘
//                             │
//                             ▼ Downstream (OrderedCheckpointRecords)
//   ┌────────────────────────────────────────────────────────┐
//   │ Single-Pass Resolution: Discard Overwritten/Tombstones │
//   │ Emit Bit-Identical Winner State in O(1) Memory         │
//   └────────────────────────────────────────────────────────┘
//
// Core Operating Mechanisms:
// --------------------------
// 1. Bounded In-Memory Buffering & Spill:
//    Incoming fixed-width records (e.g. CheckpointEntryRecord, uint64_t
//    offsets, or MutationSequenceIdentity) are buffered in memory up to
//    RunRecords (default: 65,536 records via CheckpointStreamPolicy). When
//    full, the buffer is sorted in place and flushed to a temporary binary run
//    file on disk.
//
// 2. Hierarchical Tree-Structured Fan-In Merging:
//    To prevent file descriptor exhaustion and bound disk I/O, runs are
//    organized into logarithmic tree levels. When FanIn runs (default: 64)
//    accumulate at any level, a min-heap multi-way merge collapses them into a
//    single sorted run at the next level, ensuring $O(N \log N)$ complexity.
//
// 3. Single-Pass Streaming Resolution:
//    When consume_sorted() is called, runs across all levels are merged on the
//    fly via a min-heap reader (or iterated directly in memory if no spill
//    occurred). Downstream consumers (such as OrderedCheckpointRecords in
//    ordered_checkpoint.hpp) resolve the latest active state per (dim, key)
//    in $O(1)$ auxiliary memory and discard superseded versions and tombstones.
//
// Safety and Isolation Guarantees:
// --------------------------------
// - Process-Isolated Scratch Directory:
//   Created via mkdtemp with owner-only (0700) atomic permissions. Scratch
//   files are strictly transient and deleted immediately upon consumption or
//   during exception stack unwinding.
// - Zero Padding Leakage:
//   Enforces static assertions on std::has_unique_object_representations_v to
//   prevent padding or uninitialized bytes from touching temporary disk files.
// ============================================================================

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "sparse_embedding/src/checkpoint/checkpoint_stream_policy.hpp"
#include "sparse_embedding/src/checkpoint/checkpoint_stream_statistics.hpp"

namespace se::checkpoint {

inline uint64_t checked_sum(uint64_t left, uint64_t right) {
  if (right > std::numeric_limits<uint64_t>::max() - left)
    throw std::overflow_error("checkpoint temporary record offset overflows");
  return left + right;
}

inline uint64_t checked_byte_count(uint64_t count, uint64_t item_size) {
  if (item_size != 0 &&
      count > std::numeric_limits<uint64_t>::max() / item_size)
    throw std::overflow_error(
        "checkpoint temporary record byte count overflows");
  return count * item_size;
}

// Scratch belongs to this process and is never a persisted checkpoint format.
// Unique object representations prevent padding bytes from entering scratch.
template <typename Record>
void write_records(std::ostream &output, std::span<const Record> records) {
  static_assert(std::is_trivially_copyable_v<Record> &&
                std::has_unique_object_representations_v<Record>);
  const uint64_t bytes = checked_byte_count(records.size(), sizeof(Record));
  if (!std::in_range<std::streamsize>(bytes))
    throw std::overflow_error("checkpoint temporary write size overflows");
  if (!records.empty())
    output.write(reinterpret_cast<const char *>(records.data()), bytes);
  if (!output)
    throw std::runtime_error("checkpoint temporary record write failed");
}

template <typename Record>
size_t read_records(std::istream &input, std::span<Record> records) {
  static_assert(std::is_trivially_copyable_v<Record> &&
                std::has_unique_object_representations_v<Record>);
  const uint64_t bytes = checked_byte_count(records.size(), sizeof(Record));
  if (!std::in_range<std::streamsize>(bytes))
    throw std::overflow_error("checkpoint temporary read size overflows");
  if (records.empty())
    return 0;
  input.read(reinterpret_cast<char *>(records.data()), bytes);
  const auto received = input.gcount();
  if (received % sizeof(Record) != 0)
    throw std::runtime_error("checkpoint temporary record is truncated");
  if (input.bad() || (!input && !input.eof()))
    throw std::runtime_error("checkpoint temporary record read failed");
  return static_cast<size_t>(received) / sizeof(Record);
}

inline void finish_output(std::ofstream &output) {
  output.flush();
  if (!output)
    throw std::runtime_error("checkpoint temporary output flush failed");
  output.close();
  if (!output)
    throw std::runtime_error("checkpoint temporary output close failed");
}

class TemporaryRunDirectory {
  std::filesystem::path path_;

public:
  TemporaryRunDirectory() = default;
  TemporaryRunDirectory(const TemporaryRunDirectory &) = delete;
  TemporaryRunDirectory &operator=(const TemporaryRunDirectory &) = delete;
  TemporaryRunDirectory(TemporaryRunDirectory &&other) noexcept
      : path_(std::move(other.path_)) {
    other.path_.clear();
  }

  ~TemporaryRunDirectory() { cleanup_after_error(); }

  const std::filesystem::path &path() const noexcept { return path_; }

  std::filesystem::path file(std::string_view name) {
    const std::filesystem::path filename(name);
    if (name.empty() || name == "." || name == ".." ||
        filename.has_parent_path())
      throw std::invalid_argument("checkpoint temporary filename is invalid");
    if (path_.empty()) {
      auto pattern = (std::filesystem::temp_directory_path() /
                      "sparse_embedding_checkpoint-XXXXXX")
                         .string();
      // mkdtemp creates owner-only permissions atomically, without a chmod gap.
      if (!::mkdtemp(pattern.data()))
        throw std::system_error(
            errno, std::generic_category(),
            "checkpoint temporary directory creation failed");
      path_ = std::move(pattern);
    }
    return path_ / filename;
  }

  void cleanup() {
    if (path_.empty())
      return;
    std::error_code error;
    std::filesystem::remove_all(path_, error);
    if (error)
      throw std::filesystem::filesystem_error(
          "checkpoint temporary directory cleanup failed", path_, error);
    path_.clear();
  }

  // Normal completion/cancellation uses checked cleanup. During unwinding the
  // primary error must survive; report an actual cleanup failure independently.
  void cleanup_after_error() noexcept {
    try {
      cleanup();
    } catch (const std::exception &error) {
      std::fprintf(stderr, "%s\n", error.what());
    }
  }
};

template <typename Record, typename Compare = std::less<Record>,
          size_t RunRecords = CheckpointStreamPolicy::kExternalSortRunRecords,
          size_t FanIn = CheckpointStreamPolicy::kExternalSortMergeFanIn>
class ExternalSortedRecords {
  static_assert(RunRecords > 0 &&
                RunRecords <= CheckpointStreamPolicy::kExternalSortRunRecords);
  static_assert(FanIn > 1 &&
                FanIn <= CheckpointStreamPolicy::kExternalSortMergeFanIn);
  static_assert(std::is_trivially_copyable_v<Record> &&
                std::has_unique_object_representations_v<Record>);
  static constexpr size_t kIoRecords = 64;
  static constexpr size_t kLevels = [] {
    uint64_t runs = std::numeric_limits<uint64_t>::max() / RunRecords;
    size_t levels = 1;
    while (runs >= FanIn) {
      runs /= FanIn;
      ++levels;
    }
    return levels + 1;
  }();

  struct Run {
    std::filesystem::path path;
    uint64_t records = 0;
  };

  struct Level {
    std::array<Run, FanIn> runs;
    size_t count = 0;
  };

  class Reader {
    std::ifstream input_;
    std::array<Record, kIoRecords> buffer_;
    size_t position_ = 0;
    size_t size_ = 0;
    uint64_t remaining_ = 0;

  public:
    void open(const Run &run) {
      input_.open(run.path, std::ios::binary);
      if (!input_)
        throw std::runtime_error("checkpoint temporary run open failed");
      if (std::filesystem::file_size(run.path) !=
          checked_byte_count(run.records, sizeof(Record)))
        throw std::runtime_error("checkpoint temporary run length differs");
      remaining_ = run.records;
    }

    bool next(Record &record, ExternalSortStatistics &statistics) {
      if (position_ == size_) {
        if (remaining_ == 0)
          return false;
        const size_t expected = std::min<uint64_t>(remaining_, buffer_.size());
        size_ =
            read_records(input_, std::span<Record>(buffer_.data(), expected));
        if (size_ != expected)
          throw std::runtime_error("checkpoint temporary run is truncated");
        remaining_ -= size_;
        statistics.bytes_read = checked_sum(
            statistics.bytes_read, checked_byte_count(size_, sizeof(Record)));
        position_ = 0;
      }
      record = buffer_[position_++];
      return true;
    }

    void close() {
      input_.clear();
      input_.close();
      if (!input_)
        throw std::runtime_error("checkpoint temporary run close failed");
    }
  };

  TemporaryRunDirectory directory_;
  std::vector<Record> buffer_;
  std::array<Level, kLevels> levels_;
  ExternalSortStatistics statistics_;
  uint64_t next_run_id_ = 0;
  bool consumed_ = false;
  [[no_unique_address]] Compare compare_;

  std::filesystem::path next_run_path() {
    if (next_run_id_ == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("checkpoint temporary run id overflows");
    return directory_.file("run-" + std::to_string(next_run_id_++) + ".bin");
  }

  void register_run() {
    ++statistics_.active_run_files;
    statistics_.peak_run_files =
        std::max(statistics_.peak_run_files, statistics_.active_run_files);
  }

  void write(std::ofstream &output, std::span<const Record> records) {
    write_records(output, records);
    statistics_.bytes_written =
        checked_sum(statistics_.bytes_written,
                    checked_byte_count(records.size(), sizeof(Record)));
  }

  Run merge(std::span<const Run> runs) {
    if (runs.size() < 2 || runs.size() > FanIn)
      throw std::logic_error("checkpoint temporary merge fan-in is invalid");
    statistics_.peak_merge_fan_in =
        std::max(statistics_.peak_merge_fan_in, runs.size());
    std::array<Reader, FanIn> inputs;
    std::array<Record, FanIn> heads;
    std::array<size_t, FanIn> heap;
    size_t heap_size = 0;
    auto greater = [&](size_t left, size_t right) {
      return compare_(heads[right], heads[left]);
    };
    uint64_t expected_records = 0;
    for (size_t index = 0; index < runs.size(); ++index) {
      expected_records = checked_sum(expected_records, runs[index].records);
      inputs[index].open(runs[index]);
      if (inputs[index].next(heads[index], statistics_)) {
        heap[heap_size++] = index;
        std::push_heap(heap.begin(), heap.begin() + heap_size, greater);
      }
    }

    const auto result = next_run_path();
    std::ofstream output(result, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error("checkpoint temporary merge output open failed");
    register_run();
    std::array<Record, kIoRecords> output_records;
    size_t output_size = 0;
    uint64_t output_count = 0;
    while (heap_size != 0) {
      std::pop_heap(heap.begin(), heap.begin() + heap_size, greater);
      const size_t index = heap[--heap_size];
      output_records[output_size++] = heads[index];
      output_count = checked_sum(output_count, 1);
      if (output_size == output_records.size()) {
        write(output, output_records);
        output_size = 0;
      }
      if (inputs[index].next(heads[index], statistics_)) {
        heap[heap_size++] = index;
        std::push_heap(heap.begin(), heap.begin() + heap_size, greater);
      }
    }
    write(output, std::span<const Record>(output_records.data(), output_size));
    if (output_count != expected_records)
      throw std::runtime_error("checkpoint temporary merge lost records");
    finish_output(output);
    for (size_t index = 0; index < runs.size(); ++index)
      inputs[index].close();
    // A failed output never destroys an input that could explain the failure.
    for (const auto &run : runs) {
      std::error_code error;
      if (!std::filesystem::remove(run.path, error) || error)
        throw std::filesystem::filesystem_error(
            "checkpoint temporary input removal failed", run.path, error);
      --statistics_.active_run_files;
    }
    statistics_.merges = checked_sum(statistics_.merges, 1);
    return {result, output_count};
  }

  void add_run(Run run) {
    for (auto &level : levels_) {
      level.runs[level.count++] = std::move(run);
      if (level.count != FanIn)
        return;
      run = merge(level.runs);
      for (auto &input : level.runs)
        input = {};
      level.count = 0;
    }
    throw std::overflow_error("checkpoint temporary run levels exhausted");
  }

  void spill() {
    if (buffer_.empty())
      return;
    std::sort(buffer_.begin(), buffer_.end(), compare_);
    const auto path = next_run_path();
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error("checkpoint temporary run output open failed");
    register_run();
    write(output, buffer_);
    finish_output(output);
    statistics_.initial_runs = checked_sum(statistics_.initial_runs, 1);
    const uint64_t count = buffer_.size();
    buffer_.clear();
    add_run({path, count});
  }

  Run finish_runs() {
    Run result;
    for (auto &level : levels_) {
      if (level.count == 0)
        continue;
      if (result.path.empty()) {
        result = std::move(level.runs[--level.count]);
      }
      if (level.count != 0) {
        level.runs[level.count++] = std::move(result);
        result = merge(std::span<const Run>(level.runs.data(), level.count));
        for (auto &run : level.runs)
          run = {};
        level.count = 0;
      }
    }
    return result;
  }

  void cleanup_after_error() noexcept {
    directory_.cleanup_after_error();
    if (directory_.path().empty())
      statistics_.active_run_files = 0;
  }

public:
  using value_type = Record;

  ExternalSortedRecords() = default;
  ExternalSortedRecords(const ExternalSortedRecords &) = delete;
  ExternalSortedRecords &operator=(const ExternalSortedRecords &) = delete;

  static constexpr size_t maximum_run_files() {
    return kLevels * (FanIn - 1) + 2;
  }

  const ExternalSortStatistics &statistics() const noexcept {
    return statistics_;
  }

  void append(Record record) {
    if (consumed_)
      throw std::logic_error("checkpoint sorted stream is already consumed");
    try {
      statistics_.records = checked_sum(statistics_.records, 1);
      if (buffer_.capacity() == 0)
        buffer_.reserve(RunRecords);
      buffer_.push_back(record);
      statistics_.peak_buffer_records =
          std::max(statistics_.peak_buffer_records, buffer_.size());
      if (buffer_.size() == RunRecords)
        spill();
    } catch (...) {
      consumed_ = true;
      cleanup_after_error();
      throw;
    }
  }

  template <typename Visitor> bool consume_sorted(Visitor &&visitor) {
    if (consumed_)
      throw std::logic_error("checkpoint sorted stream is already consumed");
    consumed_ = true;
    try {
      bool complete = true;
      if (statistics_.initial_runs == 0) {
        std::sort(buffer_.begin(), buffer_.end(), compare_);
        for (const auto &record : buffer_) {
          if (!visitor(record)) {
            complete = false;
            break;
          }
        }
      } else {
        spill();
        const auto final_run = finish_runs();
        if (final_run.records != statistics_.records)
          throw std::runtime_error("checkpoint temporary sort lost records");
        Reader input;
        input.open(final_run);
        Record record;
        while (input.next(record, statistics_)) {
          if (!visitor(record)) {
            complete = false;
            break;
          }
        }
        input.close();
      }
      buffer_.clear();
      directory_.cleanup();
      statistics_.active_run_files = 0;
      return complete;
    } catch (...) {
      cleanup_after_error();
      throw;
    }
  }
};

} // namespace se::checkpoint
