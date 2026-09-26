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

#ifndef SPARSE_EMBEDDING_SIMD_ADAM_OPS_HPP_
#define SPARSE_EMBEDDING_SIMD_ADAM_OPS_HPP_

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "sparse_embedding/src/simd/intrinsics.hpp"
#include "sparse_embedding/src/simd/traits.hpp"

namespace se::simd {

template <typename ValT, typename GradT, typename OptStateT>
inline void
simd_adam_update(ValT *__restrict__ param, const GradT *__restrict__ grad,
                 OptStateT *__restrict__ m, OptStateT *__restrict__ v, float lr,
                 float beta1, float beta2, float eps, float bc1, float bc2,
                 float weight_decay, size_t count) {
  size_t i = 0;
  [[assume(count > 0)]];
#if defined(__AVX512F__)
  if constexpr (is_simd_supported_v<ValT> && is_simd_supported_v<GradT> &&
                is_simd_supported_v<OptStateT>) {
    {
      __m512 vb1 = _mm512_set1_ps(beta1);
      __m512 vb2 = _mm512_set1_ps(beta2);
      __m512 v1mb1 = _mm512_set1_ps(1.0f - beta1);
      __m512 v1mb2 = _mm512_set1_ps(1.0f - beta2);
      __m512 vlr = _mm512_set1_ps(-lr);
      __m512 veps = _mm512_set1_ps(eps);
      __m512 vbc1 = _mm512_set1_ps(bc1);
      __m512 vbc2 = _mm512_set1_ps(bc2);
      __m512 vwd = _mm512_set1_ps(weight_decay);

      for (; i + 32 <= count; i += 32) {
        __m512 p0 = load_to_ps(param + i);
        __m512 p1 = load_to_ps(param + i + 16);
        __m512 g0 = load_to_ps(grad + i);
        __m512 g1 = load_to_ps(grad + i + 16);
        if (weight_decay > 0.0f) {
          g0 = _mm512_fmadd_ps(vwd, p0, g0);
          g1 = _mm512_fmadd_ps(vwd, p1, g1);
        }
        __m512 mi0 = load_to_ps(m + i);
        __m512 mi1 = load_to_ps(m + i + 16);
        __m512 vi0 = load_to_ps(v + i);
        __m512 vi1 = load_to_ps(v + i + 16);

        mi0 = _mm512_fmadd_ps(vb1, mi0, _mm512_mul_ps(v1mb1, g0));
        mi1 = _mm512_fmadd_ps(vb1, mi1, _mm512_mul_ps(v1mb1, g1));
        vi0 = _mm512_fmadd_ps(vb2, vi0,
                              _mm512_mul_ps(v1mb2, _mm512_mul_ps(g0, g0)));
        vi1 = _mm512_fmadd_ps(vb2, vi1,
                              _mm512_mul_ps(v1mb2, _mm512_mul_ps(g1, g1)));

        store_from_ps(m + i, mi0);
        store_from_ps(m + i + 16, mi1);
        store_from_ps(v + i, vi0);
        store_from_ps(v + i + 16, vi1);

        __m512 m_hat0 = _mm512_mul_ps(mi0, vbc1);
        __m512 m_hat1 = _mm512_mul_ps(mi1, vbc1);
        __m512 v_hat0 = _mm512_mul_ps(vi0, vbc2);
        __m512 v_hat1 = _mm512_mul_ps(vi1, vbc2);
        __m512 denom0 = _mm512_add_ps(_mm512_sqrt_ps(v_hat0), veps);
        __m512 denom1 = _mm512_add_ps(_mm512_sqrt_ps(v_hat1), veps);
        p0 = _mm512_fmadd_ps(vlr, _mm512_div_ps(m_hat0, denom0), p0);
        p1 = _mm512_fmadd_ps(vlr, _mm512_div_ps(m_hat1, denom1), p1);
        store_from_ps(param + i, p0);
        store_from_ps(param + i + 16, p1);
      }
      for (; i + 16 <= count; i += 16) {
        __m512 p = load_to_ps(param + i);
        __m512 g = load_to_ps(grad + i);
        if (weight_decay > 0.0f) {
          g = _mm512_fmadd_ps(vwd, p, g);
        }
        __m512 mi = load_to_ps(m + i);
        __m512 vi = load_to_ps(v + i);

        mi = _mm512_fmadd_ps(vb1, mi, _mm512_mul_ps(v1mb1, g));
        vi =
            _mm512_fmadd_ps(vb2, vi, _mm512_mul_ps(v1mb2, _mm512_mul_ps(g, g)));

        store_from_ps(m + i, mi);
        store_from_ps(v + i, vi);

        __m512 m_hat = _mm512_mul_ps(mi, vbc1);
        __m512 v_hat = _mm512_mul_ps(vi, vbc2);
        __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), veps);
        p = _mm512_fmadd_ps(vlr, _mm512_div_ps(m_hat, denom), p);
        store_from_ps(param + i, p);
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
                     )&&(std::is_same_v<OptStateT, float>
#if defined(__F16C__)
                         || std::is_same_v<OptStateT, float16_t>
#endif
                         )) {
    {
      __m256 vb1 = _mm256_set1_ps(beta1);
      __m256 vb2 = _mm256_set1_ps(beta2);
      __m256 v1mb1 = _mm256_set1_ps(1.0f - beta1);
      __m256 v1mb2 = _mm256_set1_ps(1.0f - beta2);
      __m256 vlr = _mm256_set1_ps(-lr);
      __m256 veps = _mm256_set1_ps(eps);
      __m256 vbc1 = _mm256_set1_ps(bc1);
      __m256 vbc2 = _mm256_set1_ps(bc2);
      __m256 vwd = _mm256_set1_ps(weight_decay);

      for (; i + 16 <= count; i += 16) {
        __m256 p0 = load256_to_ps(param + i);
        __m256 p1 = load256_to_ps(param + i + 8);
        __m256 g0 = load256_to_ps(grad + i);
        __m256 g1 = load256_to_ps(grad + i + 8);
        if (weight_decay > 0.0f) {
          g0 = _mm256_fmadd_ps(vwd, p0, g0);
          g1 = _mm256_fmadd_ps(vwd, p1, g1);
        }
        __m256 mi0 = load256_to_ps(m + i);
        __m256 mi1 = load256_to_ps(m + i + 8);
        __m256 vi0 = load256_to_ps(v + i);
        __m256 vi1 = load256_to_ps(v + i + 8);

        mi0 = _mm256_fmadd_ps(vb1, mi0, _mm256_mul_ps(v1mb1, g0));
        mi1 = _mm256_fmadd_ps(vb1, mi1, _mm256_mul_ps(v1mb1, g1));
        vi0 = _mm256_fmadd_ps(vb2, vi0,
                              _mm256_mul_ps(v1mb2, _mm256_mul_ps(g0, g0)));
        vi1 = _mm256_fmadd_ps(vb2, vi1,
                              _mm256_mul_ps(v1mb2, _mm256_mul_ps(g1, g1)));

        store256_from_ps(m + i, mi0);
        store256_from_ps(m + i + 8, mi1);
        store256_from_ps(v + i, vi0);
        store256_from_ps(v + i + 8, vi1);

        __m256 m_hat0 = _mm256_mul_ps(mi0, vbc1);
        __m256 m_hat1 = _mm256_mul_ps(mi1, vbc1);
        __m256 v_hat0 = _mm256_mul_ps(vi0, vbc2);
        __m256 v_hat1 = _mm256_mul_ps(vi1, vbc2);
        __m256 denom0 = _mm256_add_ps(_mm256_sqrt_ps(v_hat0), veps);
        __m256 denom1 = _mm256_add_ps(_mm256_sqrt_ps(v_hat1), veps);
        p0 = _mm256_fmadd_ps(vlr, _mm256_div_ps(m_hat0, denom0), p0);
        p1 = _mm256_fmadd_ps(vlr, _mm256_div_ps(m_hat1, denom1), p1);
        store256_from_ps(param + i, p0);
        store256_from_ps(param + i + 8, p1);
      }
      for (; i + 8 <= count; i += 8) {
        __m256 p = load256_to_ps(param + i);
        __m256 g = load256_to_ps(grad + i);
        if (weight_decay > 0.0f) {
          g = _mm256_fmadd_ps(vwd, p, g);
        }
        __m256 mi = load256_to_ps(m + i);
        __m256 vi = load256_to_ps(v + i);

        mi = _mm256_fmadd_ps(vb1, mi, _mm256_mul_ps(v1mb1, g));
        vi =
            _mm256_fmadd_ps(vb2, vi, _mm256_mul_ps(v1mb2, _mm256_mul_ps(g, g)));

        store256_from_ps(m + i, mi);
        store256_from_ps(v + i, vi);

        __m256 m_hat = _mm256_mul_ps(mi, vbc1);
        __m256 v_hat = _mm256_mul_ps(vi, vbc2);
        __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), veps);
        p = _mm256_fmadd_ps(vlr, _mm256_div_ps(m_hat, denom), p);
        store256_from_ps(param + i, p);
      }
    }
  }
#endif
  for (; i < count; ++i) {
    float g = static_cast<float>(grad[i]);
    float p_val = static_cast<float>(param[i]);
    if (weight_decay > 0.0f) {
      g += weight_decay * p_val;
    }
    float mi = static_cast<float>(m[i]);
    float vi = static_cast<float>(v[i]);

    mi = beta1 * mi + (1.0f - beta1) * g;
    vi = beta2 * vi + (1.0f - beta2) * g * g;

    if constexpr (std::is_same_v<OptStateT, float>) {
      m[i] = mi;
    } else {
      m[i] = static_cast<OptStateT>(mi);
    }
    if constexpr (std::is_same_v<OptStateT, float>) {
      v[i] = vi;
    } else {
      v[i] = static_cast<OptStateT>(vi);
    }

    float m_hat = mi * bc1;
    float v_hat = vi * bc2;
    float p_new = p_val - lr * m_hat / (std::sqrt(v_hat) + eps);
    if constexpr (std::is_same_v<ValT, float>) {
      param[i] = p_new;
    } else {
      param[i] = static_cast<ValT>(p_new);
    }
  }
}

} // namespace se::simd

#endif // SPARSE_EMBEDDING_SIMD_ADAM_OPS_HPP_
