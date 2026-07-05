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
#include <span>
#include <type_traits>

#include "sparse_embedding/src/optimizer/optimizer_config.hpp"

namespace se {

template <typename ParamT, typename OptT> class SgdOptimizer {
  SgdConfig config;

public:
  explicit SgdOptimizer(size_t dim, const SgdConfig &cfg) : config(cfg) {
    (void)dim;
  }

  void allocate() {}
  void reserve(size_t capacity) { (void)capacity; }
  void deallocate(uint32_t idx) {}
  void clear() {}

  void state_data(uint32_t idx, std::span<OptT> out) const {
    (void)idx;
    (void)out;
  }

  size_t state_dim() const { return 0; }

  void set_state(uint32_t idx, std::span<const OptT> state) {
    (void)idx;
    (void)state;
  }

  uint32_t step_data(uint32_t idx) const {
    (void)idx;
    return 0;
  }

  void set_step(uint32_t idx, uint32_t s) {
    (void)idx;
    (void)s;
  }

  void step(uint32_t idx, std::span<ParamT> param_span,
            std::span<const ParamT> grad_span,
            [[maybe_unused]] uint64_t global_step) {
    size_t dim = param_span.size();
    ParamT *param = param_span.data();
    const ParamT *accumulated_grad = grad_span.data();

    // In SGD, we can just use simd_adam_update? No, we should use a simpler
    // simd operation. For now, simple fallback.
    for (size_t i = 0; i < dim; ++i) {
      float g = static_cast<float>(accumulated_grad[i]);
      float p_val = static_cast<float>(param[i]);
      if (config.weight_decay > 0.0f) {
        g += config.weight_decay * p_val;
      }
      if constexpr (std::is_same_v<ParamT, float>) {
        param[i] = p_val - config.lr * g;
      } else {
        param[i] = static_cast<ParamT>(p_val - config.lr * g);
      }
    }
  }
};

} // namespace se

#endif // SPARSE_EMBEDDING_SGD_OPTIMIZER_HPP_
