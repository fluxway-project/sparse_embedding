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

#include "optimizer/optimizer_config.hpp"

namespace se {

template <typename ValT> class SgdOptimizer {
  SgdConfig config;

public:
  explicit SgdOptimizer(size_t dim, const SgdConfig &cfg) : config(cfg) {}

  void allocate() {}
  void deallocate(uint32_t idx) {}
  void clear() {}

  void step(uint32_t idx, std::span<ValT> param_span,
            std::span<const ValT> grad_span, uint32_t k_batches,
            uint64_t global_step) {
    size_t dim = param_span.size();
    ValT *param = param_span.data();
    const ValT *accumulated_grad = grad_span.data();

    float inv_k = 1.0f / static_cast<float>(k_batches);

    for (size_t d = 0; d < dim; ++d) {
      float g = static_cast<float>(accumulated_grad[d]) * inv_k;
      if (config.weight_decay > 0.0f) {
        g += config.weight_decay * static_cast<float>(param[d]);
      }
      param[d] =
          static_cast<ValT>(static_cast<float>(param[d]) - config.lr * g);
    }
  }
};

} // namespace se

#endif // SPARSE_EMBEDDING_SGD_OPTIMIZER_HPP_
