#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

#include "sparse_embedding/src/interface/embedding_table_interface.hpp"

namespace se::hogwild::io {

template <typename ParamT, typename OptT>
size_t validate_page_image_columns(const PageImageExport<ParamT, OptT> &image,
                                   size_t expected_dim, size_t expected_stride,
                                   bool score_enabled,
                                   size_t optimizer_state_dim,
                                   size_t entries_per_page) {
  if (image.embedding_dim != expected_dim ||
      image.page_slot_stride_bytes != expected_stride)
    throw std::invalid_argument("page-image layout does not match slot");
  const size_t count = image.keys.size();
  if (count == 0 || image.slot_ordinals.size() != count ||
      image.seq_counts.size() != count)
    throw std::invalid_argument("page-image columns are inconsistent");
  if (expected_dim != 0 &&
      count > std::numeric_limits<size_t>::max() / expected_dim)
    throw std::overflow_error("page-image value column size overflows");
  if (image.values.size() != count * expected_dim)
    throw std::invalid_argument("page-image value column is inconsistent");
  if (score_enabled) {
    if (!image.scores.empty() && image.scores.size() != count)
      throw std::invalid_argument("page-image scores are inconsistent");
  } else if (!image.scores.empty()) {
    throw std::invalid_argument("page-image has scores for an unscored slot");
  }
  if (optimizer_state_dim == 0) {
    if (!image.optimizer_states.empty() || !image.steps.empty())
      throw std::invalid_argument(
          "page-image has state for a stateless optimizer");
  } else {
    if (count > std::numeric_limits<size_t>::max() / optimizer_state_dim ||
        image.optimizer_states.size() != count * optimizer_state_dim ||
        image.steps.size() != count)
      throw std::invalid_argument("page-image optimizer state is inconsistent");
  }
  for (size_t ordinal : image.slot_ordinals) {
    if (ordinal >= entries_per_page)
      throw std::out_of_range("page-image slot ordinal is out of range");
  }
  for (uint64_t sequence : image.seq_counts) {
    if (sequence == 0 || (sequence & 1U) != 0)
      throw std::invalid_argument("page-image owner seq_count is invalid");
  }
  return count;
}

template <typename ParamT, typename OptT>
size_t validate_page_image(const PageImageImport<ParamT, OptT> &import,
                           size_t expected_dim, size_t expected_stride,
                           bool score_enabled, size_t optimizer_state_dim,
                           size_t entries_per_page) {
  if (import.embedding_dim != expected_dim)
    throw std::invalid_argument(
        "page-image import dimension does not match slot");
  return validate_page_image_columns(import.image, expected_dim,
                                     expected_stride, score_enabled,
                                     optimizer_state_dim, entries_per_page);
}

} // namespace se::hogwild::io
