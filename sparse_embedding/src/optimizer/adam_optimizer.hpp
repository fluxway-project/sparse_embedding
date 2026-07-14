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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>

#include "sparse_embedding/src/optimizer/optimizer_config.hpp"
#include "sparse_embedding/src/optimizer/optimizer_state.hpp"
#include "sparse_embedding/src/simd/simd_ops.hpp"

namespace se {

template <typename ParamT, typename OptT> class AdamOptimizer {
  OptimizerStateArena<OptT> m_state;
  OptimizerStateArena<OptT> v_state;
  OptimizerStateArena<uint32_t>
      step_state; // Per-ID step counter (uint32); initialized to 0 by
                  // OptimizerStateArena's value-initialized allocation so the
                  // first update sees id_step==1 after the increment, which is
                  // required for correct Adam bias correction.
  AdamConfig config;

public:
  explicit AdamOptimizer(size_t dim, const AdamConfig &cfg,
                         std::shared_ptr<PageStorage> storage)
      : m_state(dim, storage), v_state(dim, storage),
        step_state(1, std::move(storage)), config(cfg) {}

  void allocate(uint32_t idx) {
    m_state.allocate(idx);
    v_state.allocate(idx);
    step_state.allocate(idx);
  }

  size_t capacity() const {
    return std::min(
        {m_state.capacity(), v_state.capacity(), step_state.capacity()});
  }

  void reserve(size_t capacity) {
    m_state.reserve(capacity);
    v_state.reserve(capacity);
    step_state.reserve(capacity);
  }

  void deallocate(uint32_t idx) {
    m_state.deallocate(idx);
    v_state.deallocate(idx);
    step_state.deallocate(idx);
  }

  void clear() {
    m_state.clear();
    v_state.clear();
    step_state.clear();
  }

  void state_data(uint32_t idx, std::span<OptT> out) const {
    size_t d = m_state.dim();
    if (out.size() < d * 2) {
      throw std::invalid_argument("Output buffer too small");
    }
    std::copy_n(m_state.data(idx), d, out.data());
    std::copy_n(v_state.data(idx), d, out.data() + d);
  }

  size_t state_dim() const { return m_state.dim() * 2; }

  void set_state(uint32_t idx, std::span<const OptT> state) {
    size_t d = m_state.dim();
    if (state.size() < d * 2) {
      throw std::invalid_argument("Input state span is too small.");
    }
    std::copy_n(state.data(), d, m_state.data(idx));
    std::copy_n(state.data() + d, d, v_state.data(idx));
  }

  uint32_t step_data(uint32_t idx) const { return *step_state.data(idx); }

  void set_step(uint32_t idx, uint32_t s) { *step_state.data(idx) = s; }

  void step(uint32_t idx, std::span<ParamT> param_span,
            std::span<const ParamT> grad_span,
            [[maybe_unused]] uint64_t global_step) {
    uint32_t &id_step = *step_state.data(idx);
    id_step++;
    const float step_f = static_cast<float>(id_step);
    const float bc1 = 1.0f / (1.0f - std::pow(config.beta1, step_f));
    const float bc2 = 1.0f / (1.0f - std::pow(config.beta2, step_f));

    size_t dim = param_span.size();
    ParamT *param = param_span.data();
    const ParamT *accumulated_grad = grad_span.data();

    OptT *m = m_state.data(idx);
    OptT *v = v_state.data(idx);

    simd::simd_adam_update(param, accumulated_grad, m, v, config.lr,
                           config.beta1, config.beta2, config.eps, bc1, bc2,
                           config.weight_decay, dim);
  }
};

} // namespace se

#endif // SPARSE_EMBEDDING_ADAM_OPTIMIZER_HPP_
