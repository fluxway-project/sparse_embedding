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

#ifndef SPARSE_EMBEDDING_SIMD_SGD_OPS_HPP_
#define SPARSE_EMBEDDING_SIMD_SGD_OPS_HPP_

#include <cstddef>
#include <type_traits>

#include "sparse_embedding/src/simd/intrinsics.hpp"
#include "sparse_embedding/src/simd/traits.hpp"

namespace se::simd {

template <typename ValT, typename GradT>
inline void simd_sgd_update(ValT *__restrict__ param,
                            const GradT *__restrict__ grad, float lr,
                            float weight_decay, size_t count) {

  size_t i = 0;
  [[assume(count > 0)]];
#if defined(__AVX512F__)
  if constexpr (is_simd_supported_v<ValT> && is_simd_supported_v<GradT>) {
    {
      const __m512 learning_rate = _mm512_set1_ps(lr);
      const __m512 decay = _mm512_set1_ps(weight_decay);
      for (; i + 16 <= count; i += 16) {
        __m512 parameters = load_to_ps(param + i);
        __m512 gradients = load_to_ps(grad + i);
        gradients = _mm512_fmadd_ps(decay, parameters, gradients);
        parameters = _mm512_fnmadd_ps(learning_rate, gradients, parameters);
        store_from_ps(param + i, parameters);
      }
    }
  }
#elif defined(__AVX2__)
  if constexpr ((std::is_same_v<ValT, float>
#if defined(__F16C__)
                 || std::is_same_v<ValT, float16_t>
#endif
                 )&&(std::is_same_v<GradT, float>
#if defined(__F16C__)
                     || std::is_same_v<GradT, float16_t>
#endif
                     )) {
    {
      const __m256 learning_rate = _mm256_set1_ps(lr);
      const __m256 decay = _mm256_set1_ps(weight_decay);
      for (; i + 8 <= count; i += 8) {
        __m256 parameters = load256_to_ps(param + i);
        __m256 gradients = load256_to_ps(grad + i);
        gradients = _mm256_fmadd_ps(decay, parameters, gradients);
        parameters = _mm256_fnmadd_ps(learning_rate, gradients, parameters);
        store256_from_ps(param + i, parameters);
      }
    }
  }
#endif
  for (; i < count; ++i) {
    float parameter = static_cast<float>(param[i]);
    float gradient = static_cast<float>(grad[i]) + weight_decay * parameter;
    param[i] = static_cast<ValT>(parameter - lr * gradient);
  }
}

} // namespace se::simd

#endif // SPARSE_EMBEDDING_SIMD_SGD_OPS_HPP_
