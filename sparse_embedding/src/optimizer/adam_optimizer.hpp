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

  // Returns a pointer to a flat buffer [m[0..dim-1], v[0..dim-1]] for idx.
  // Caller must copy the data before releasing any lock; the buffer is
  // backed by two separate arenas so we materialise it into a thread-local.
  const float *state_data(uint32_t idx) const {
    size_t d = m_state.dim();
    thread_local std::vector<float> buf;
    buf.resize(d * 2);
    std::copy_n(m_state.data(idx), d, buf.data());
    std::copy_n(v_state.data(idx), d, buf.data() + d);
    return buf.data();
  }

  size_t state_dim() const { return m_state.dim() * 2; }

  void set_state(uint32_t idx, std::span<const float> state) {
    size_t d = m_state.dim();
    std::copy_n(state.data(), d, m_state.data(idx));
    std::copy_n(state.data() + d, d, v_state.data(idx));
  }

  void step(uint32_t idx, std::span<ValT> param_span,
            std::span<const ValT> grad_span, uint64_t global_step) {
    const float step_f = static_cast<float>(global_step);
    const float bc1 = 1.0f / (1.0f - std::pow(config.beta1, step_f));
    const float bc2 = 1.0f / (1.0f - std::pow(config.beta2, step_f));

    size_t dim = param_span.size();
    ValT *param = param_span.data();
    const ValT *accumulated_grad = grad_span.data();

    float *m = m_state.data(idx);
    float *v = v_state.data(idx);

    if constexpr (simd::is_simd_supported_v<ValT>) {
      const float *grad_ptr;
      thread_local std::vector<float> grad_buf;

      if constexpr (std::is_same_v<ValT, float>) {
        if (config.weight_decay > 0.0f) {
          grad_buf.resize(dim);
          for (size_t d = 0; d < dim; ++d) {
            grad_buf[d] = accumulated_grad[d] + config.weight_decay * param[d];
          }
          grad_ptr = grad_buf.data();
        } else {
          grad_ptr = accumulated_grad;
        }
      } else {
        grad_buf.resize(dim);
        if (config.weight_decay > 0.0f) {
          for (size_t d = 0; d < dim; ++d) {
            grad_buf[d] = static_cast<float>(accumulated_grad[d]) +
                          config.weight_decay * static_cast<float>(param[d]);
          }
        } else {
          for (size_t d = 0; d < dim; ++d) {
            grad_buf[d] = static_cast<float>(accumulated_grad[d]);
          }
        }
        grad_ptr = grad_buf.data();
      }

      simd::simd_adam_update(param, grad_ptr, m, v, config.lr, config.beta1,
                             config.beta2, config.eps, bc1, bc2, dim);
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
