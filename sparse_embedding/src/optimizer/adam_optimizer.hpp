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

#ifndef SPARSE_EMBEDDING_ADAM_OPTIMIZER_HPP_
#define SPARSE_EMBEDDING_ADAM_OPTIMIZER_HPP_

#include <cmath>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "sparse_embedding/src/optimizer/optimizer_config.hpp"
#include "sparse_embedding/src/optimizer/optimizer_state.hpp"
#include "sparse_embedding/src/simd/simd_ops.hpp"

namespace se {

template <typename ValT> class AdamOptimizer {
  OptimizerStateArena m_state;
  OptimizerStateArena v_state;
  AdamConfig config;

public:
  explicit AdamOptimizer(size_t dim, const AdamConfig &cfg)
      : m_state(dim), v_state(dim), config(cfg) {}

  void allocate() {
    m_state.allocate();
    v_state.allocate();
  }

  void deallocate(uint32_t idx) {
    m_state.deallocate(idx);
    v_state.deallocate(idx);
  }

  void clear() {
    m_state.clear();
    v_state.clear();
  }

  void step(uint32_t idx, std::span<ValT> param_span,
            std::span<const ValT> grad_span, uint64_t global_step) {
    size_t dim = param_span.size();
    ValT *param = param_span.data();
    const ValT *accumulated_grad = grad_span.data();

    float *m = m_state.data(idx);
    float *v = v_state.data(idx);

    float bc1 =
        1.0f / (1.0f - std::pow(config.beta1, static_cast<float>(global_step)));
    float bc2 =
        1.0f / (1.0f - std::pow(config.beta2, static_cast<float>(global_step)));

    if constexpr (simd::is_simd_supported_v<ValT>) {
      thread_local std::vector<float> avg_grad;
      avg_grad.resize(dim);
      for (size_t d = 0; d < dim; ++d) {
        avg_grad[d] = accumulated_grad[d];
      }
      if (config.weight_decay > 0.0f) {
        for (size_t d = 0; d < dim; ++d) {
          avg_grad[d] += config.weight_decay * param[d];
        }
      }
      simd::simd_adam_update(param, avg_grad.data(), m, v, config.lr,
                             config.beta1, config.beta2, config.eps, bc1, bc2,
                             dim);
    } else {
      for (size_t d = 0; d < dim; ++d) {
        float g = static_cast<float>(accumulated_grad[d]);
        if (config.weight_decay > 0.0f) {
          g += config.weight_decay * static_cast<float>(param[d]);
        }
        m[d] = config.beta1 * m[d] + (1.0f - config.beta1) * g;
        v[d] = config.beta2 * v[d] + (1.0f - config.beta2) * g * g;
        float m_hat = m[d] * bc1;
        float v_hat = v[d] * bc2;
        param[d] = static_cast<ValT>(static_cast<float>(param[d]) -
                                     config.lr * m_hat /
                                         (std::sqrt(v_hat) + config.eps));
      }
    }
  }
};

} // namespace se

#endif // SPARSE_EMBEDDING_ADAM_OPTIMIZER_HPP_
