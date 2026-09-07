#pragma once

#ifndef SPARSE_EMBEDDING_CHECKPOINT_STREAM_POLICY_HPP_
#define SPARSE_EMBEDDING_CHECKPOINT_STREAM_POLICY_HPP_

#include <cstddef>
#include <cstdint>

namespace se {

struct CheckpointStreamPolicy {
  static constexpr std::size_t kChannelCapacity = 8;
  static constexpr std::int64_t kScanBatchRows = 4096;
  static constexpr std::size_t kPageImportBatchPages = 16;
};

} // namespace se

#endif // SPARSE_EMBEDDING_CHECKPOINT_STREAM_POLICY_HPP_
