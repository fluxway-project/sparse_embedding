#pragma once

#ifndef SPARSE_EMBEDDING_CHECKPOINT_STREAM_POLICY_HPP_
#define SPARSE_EMBEDDING_CHECKPOINT_STREAM_POLICY_HPP_

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace se {

struct CheckpointStreamPolicy {
  static constexpr std::size_t kChannelCapacity = 8;
  static constexpr std::int64_t kScanBatchRows = 4096;
  static constexpr std::size_t kPageImportBatchPages = 16;
  static constexpr std::size_t kEntryImportBatchBytes = 8U << 20;
  static constexpr std::size_t kExternalSortRunRecords = 1U << 16;
  static constexpr std::size_t kExternalSortMergeFanIn = 64;

  static void validate_scan_batch_rows(std::int64_t rows,
                                       std::int64_t limit = kScanBatchRows) {
    if (rows < 0 || rows > limit)
      throw std::runtime_error(
          "native Lance restore scanner exceeded the configured batch bound");
  }
};

} // namespace se

#endif // SPARSE_EMBEDDING_CHECKPOINT_STREAM_POLICY_HPP_
