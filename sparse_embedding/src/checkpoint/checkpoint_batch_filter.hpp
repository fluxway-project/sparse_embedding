// Copyright 2026 The Fluxway Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#pragma once

#ifndef SPARSE_EMBEDDING_CHECKPOINT_BATCH_FILTER_HPP_
#define SPARSE_EMBEDDING_CHECKPOINT_BATCH_FILTER_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace se {

template <typename KeyT, typename ParamT, typename OptT>
struct FilteredCheckpointBatch {
  std::vector<KeyT> keys;
  std::vector<ParamT> values;
  std::vector<uint64_t> scores;
  std::vector<OptT> optimizer_states;
  std::vector<uint32_t> steps;

  size_t assign(std::span<const KeyT> source_keys,
                std::span<const ParamT> source_values,
                std::span<const uint64_t> source_scores,
                std::span<const OptT> source_optimizer_states,
                std::span<const uint32_t> source_steps, size_t dim,
                uint64_t min_score) {
    if (source_values.size() != source_keys.size() * dim)
      throw std::invalid_argument("checkpoint value batch shape mismatch");
    if (source_scores.size() != source_keys.size())
      throw std::invalid_argument("checkpoint score batch shape mismatch");
    if (!source_steps.empty() && source_steps.size() != source_keys.size())
      throw std::invalid_argument("checkpoint step batch shape mismatch");
    if (source_keys.empty()) {
      keys.clear();
      values.clear();
      scores.clear();
      optimizer_states.clear();
      steps.clear();
      return 0;
    }
    if (source_optimizer_states.size() % source_keys.size() != 0)
      throw std::invalid_argument(
          "checkpoint optimizer-state batch shape mismatch");

    const size_t optimizer_values_per_key =
        source_optimizer_states.size() / source_keys.size();
    const size_t selected = static_cast<size_t>(std::count_if(
        source_scores.begin(), source_scores.end(),
        [min_score](uint64_t score) { return score >= min_score; }));

    keys.resize(selected);
    values.resize(selected * dim);
    scores.resize(selected);
    optimizer_states.resize(selected * optimizer_values_per_key);
    if (source_steps.empty())
      steps.clear();
    else
      steps.resize(selected);

    size_t destination = 0;
    for (size_t source = 0; source < source_keys.size(); ++source) {
      if (source_scores[source] < min_score)
        continue;
      keys[destination] = source_keys[source];
      std::memcpy(values.data() + destination * dim,
                  source_values.data() + source * dim, dim * sizeof(ParamT));
      scores[destination] = source_scores[source];
      if (optimizer_values_per_key != 0)
        std::memcpy(
            optimizer_states.data() + destination * optimizer_values_per_key,
            source_optimizer_states.data() + source * optimizer_values_per_key,
            optimizer_values_per_key * sizeof(OptT));
      if (!source_steps.empty())
        steps[destination] = source_steps[source];
      ++destination;
    }
    return selected;
  }
};

} // namespace se

#endif // SPARSE_EMBEDDING_CHECKPOINT_BATCH_FILTER_HPP_
