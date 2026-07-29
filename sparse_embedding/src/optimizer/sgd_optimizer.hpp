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

#ifndef SPARSE_EMBEDDING_SGD_OPTIMIZER_HPP_
#define SPARSE_EMBEDDING_SGD_OPTIMIZER_HPP_

#include <cstdint>
#include <limits>
#include <span>

#include "sparse_embedding/src/optimizer/learning_rate.hpp"
#include "sparse_embedding/src/optimizer/optimizer_config.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

namespace se {

template <typename ParamT, typename OptT, typename LearningRateT>
class SgdOptimizer {
  SgdConfig config;
  LearningRateT learning_rate;

public:
  explicit SgdOptimizer(size_t /*dim*/, const SgdConfig &cfg,
                        PageStorage /*storage*/, LearningRateT lr)
      : config(cfg), learning_rate(std::move(lr)) {}

  void allocate(uint32_t /*idx*/) {}
  void set_page_eviction_enabled(bool /*enabled*/) {}
  constexpr size_t capacity() const {
    return std::numeric_limits<size_t>::max();
  }
  void reserve(size_t /*capacity*/) {}
  void deallocate(uint32_t /*idx*/) {}
  void clear() {}

  void state_data(uint32_t /*idx*/, std::span<OptT> /*out*/) const {}

  constexpr size_t state_dim() const { return 0; }

  void set_state(uint32_t /*idx*/, std::span<const OptT> /*state*/) {}

  constexpr uint32_t step_data(uint32_t /*idx*/) const { return 0; }

  void set_step(uint32_t /*idx*/, uint32_t /*s*/) {}

  constexpr bool requires_score() const {
    return learning_rate.requires_score();
  }

  void step(uint32_t /*idx*/, std::span<ParamT> param_span,
            std::span<const ParamT> grad_span, uint64_t /*global_step*/,
            uint64_t score) {
    float current_lr = learning_rate.compute(score);
    simd::simd_sgd_update(param_span.data(), grad_span.data(), current_lr,
                          config.weight_decay, param_span.size());
  }
};

} // namespace se

#endif // SPARSE_EMBEDDING_SGD_OPTIMIZER_HPP_
