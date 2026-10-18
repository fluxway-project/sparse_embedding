#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include "sparse_embedding/src/checkpoint/checkpoint_stream_policy.hpp"
#include "sparse_embedding/src/checkpoint/lance/native_entry_batch.hpp"
#include "sparse_embedding/src/checkpoint/lance/native_payload.hpp"

namespace se {

// Consume the scanner's batch before requesting another one. Scratch capacity
// is reused, and is bounded by the largest scan batch, never the whole chain.
// Only optimizer assignment needs dimension grouping; assign accepts mixed
// row lengths and scores use the same (dimension, key) identity as entries.
template <typename KeyT, typename ParamT, typename OptT, typename Store,
          typename Scanner>
void restore_native_entry_batches(Store &store, Scanner &&scan,
                                  bool load_optimizer_state) {
  std::vector<KeyT> keys, optimizer_keys;
  std::vector<ParamT> values;
  std::vector<int64_t> lengths;
  std::vector<OptT> optimizer;
  std::vector<uint32_t> steps;
  std::array<std::pair<uint32_t, uint32_t>,
             CheckpointStreamPolicy::kScanBatchRows>
      optimizer_rows;
  const bool load_scores = store.has_score_policy();

  std::forward<Scanner>(scan)([&](const NativeEntryBatch &batch) {
    const size_t count = batch.keys.size();
    CheckpointStreamPolicy::validate_scan_batch_rows(count);
    if (batch.dimensions.size() != count ||
        batch.embedding_offsets.size() != count + 1 ||
        batch.optimizer_offsets.size() != count + 1 ||
        batch.optimizer_strides.size() != count ||
        (!batch.steps.empty() && batch.steps.size() != count) ||
        (!batch.scores.empty() && batch.scores.size() != count) ||
        batch.embedding_offsets.front() != 0 ||
        batch.optimizer_offsets.front() != 0 ||
        batch.embedding_offsets.back() != batch.embedding_values.size() ||
        batch.optimizer_offsets.back() != batch.optimizer_values.size())
      throw std::runtime_error("native restore entry batch shape is invalid");

    size_t optimizer_count = 0;
    // Validate every row before copying or mutating the target. In particular,
    // unsigned offset subtraction must never conceal a decreasing offset.
    for (size_t row = 0; row < count; ++row) {
      const size_t begin = batch.embedding_offsets[row];
      const size_t end = batch.embedding_offsets[row + 1];
      if (batch.dimensions[row] == 0 || end < begin ||
          end > batch.embedding_values.size() ||
          end - begin != size_t(batch.dimensions[row]) * sizeof(ParamT))
        throw std::runtime_error(
            "native restore embedding payload shape is invalid");
      const size_t optimizer_begin = batch.optimizer_offsets[row];
      const size_t optimizer_end = batch.optimizer_offsets[row + 1];
      if (optimizer_end < optimizer_begin ||
          optimizer_end > batch.optimizer_values.size() ||
          (load_optimizer_state &&
           optimizer_end - optimizer_begin !=
               size_t(batch.optimizer_strides[row]) * sizeof(OptT)))
        throw std::runtime_error(
            "native restore optimizer payload shape is invalid");
      if (load_optimizer_state && batch.optimizer_strides[row] != 0)
        optimizer_rows[optimizer_count++] = {batch.dimensions[row], row};
    }
    if (count == 0)
      return true;

    auto optimizer_end = optimizer_rows.begin() + optimizer_count;
    std::sort(optimizer_rows.begin(), optimizer_end);
    for (size_t row = 1; row < optimizer_count; ++row) {
      if (optimizer_rows[row - 1].first == optimizer_rows[row].first &&
          batch.optimizer_strides[optimizer_rows[row - 1].second] !=
              batch.optimizer_strides[optimizer_rows[row].second])
        throw std::runtime_error("native restore optimizer strides differ");
    }

    keys.resize(count);
    lengths.resize(count);
    for (size_t row = 0; row < count; ++row) {
      keys[row] = static_cast<KeyT>(batch.keys[row]);
      lengths[row] = batch.dimensions[row];
    }
    copy_native_values(std::span<const std::byte>(batch.embedding_values),
                       values);
    const std::span<const KeyT> key_span(keys);
    const std::span<const int64_t> length_span(lengths);
    if (auto error = store.assign(key_span, std::span<const ParamT>(values),
                                  length_span))
      throw std::system_error(error, "native restore entry assignment failed");
    if (load_scores && !batch.scores.empty()) {
      if (auto error = store.assign_scores(
              key_span, std::span<const uint64_t>(batch.scores), length_span))
        throw std::system_error(error,
                                "native restore score assignment failed");
    }

    for (size_t begin = 0; begin < optimizer_count;) {
      size_t end = begin + 1;
      const auto [dimension, first_row] = optimizer_rows[begin];
      while (end < optimizer_count && optimizer_rows[end].first == dimension)
        ++end;
      const size_t group_count = end - begin;
      const size_t stride = batch.optimizer_strides[first_row];
      optimizer_keys.resize(group_count);
      optimizer.resize(group_count * stride);
      steps.resize(batch.steps.empty() ? 0 : group_count);
      for (size_t index = 0; index < group_count; ++index) {
        const size_t row = optimizer_rows[begin + index].second;
        optimizer_keys[index] = keys[row];
        std::memcpy(optimizer.data() + index * stride,
                    batch.optimizer_values.data() +
                        batch.optimizer_offsets[row],
                    stride * sizeof(OptT));
        if (!steps.empty())
          steps[index] = batch.steps[row];
      }
      store.assign_opt_states(dimension, std::span<const KeyT>(optimizer_keys),
                              std::span<const OptT>(optimizer),
                              std::span<const uint32_t>(steps));
      begin = end;
    }
    return true;
  });
}

} // namespace se
