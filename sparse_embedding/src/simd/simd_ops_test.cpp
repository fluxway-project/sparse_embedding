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

#include "sparse_embedding/src/simd/simd_ops.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "sparse_embedding/src/common/numeric_types.hpp"

int main() {
  static_assert(se::simd::kCacheLineSize == 64);
  static_assert(se::simd::is_simd_supported_v<float>);

  // 1. Test SGD
  {
    alignas(64) std::array<float, 37> parameters{};
    alignas(64) std::array<float, 37> gradients{};
    for (size_t i = 0; i < parameters.size(); ++i) {
      parameters[i] = static_cast<float>(i + 1);
      gradients[i] = 0.5f;
    }

    se::simd::simd_sgd_update(parameters.data(), gradients.data(), 0.1f, -0.01f,
                              parameters.size());

    for (size_t i = 0; i < parameters.size(); ++i) {
      float original = static_cast<float>(i + 1);
      float expected = original - 0.1f * (0.5f - 0.01f * original);
      if (std::abs(parameters[i] - expected) > 1e-6f) {
        throw std::runtime_error("SIMD SGD result mismatch");
      }
    }
  }

  // 2. Test Prefetch & Memory Ops
  {
    std::vector<float> src(64, 2.5f);
    std::vector<float> dst(64, 1.5f);
    se::simd::prefetch_l1(src.data());
    se::simd::prefetch_l2(dst.data());

    se::simd::simd_accumulate(src.data(), dst.data(), 64);
    for (size_t i = 0; i < 64; ++i) {
      if (std::abs(dst[i] - 4.0f) > 1e-6f) {
        throw std::runtime_error("SIMD accumulate mismatch");
      }
    }

    std::vector<float> copied(64, 0.0f);
    se::simd::simd_copy(dst.data(), copied.data(), 64);
    for (size_t i = 0; i < 64; ++i) {
      if (std::abs(copied[i] - 4.0f) > 1e-6f) {
        throw std::runtime_error("SIMD copy mismatch");
      }
    }
  }

  // 3. Test Adam
  {
    const size_t count = 48;
    std::vector<float> param(count, 1.0f);
    std::vector<float> grad(count, 0.2f);
    std::vector<float> m(count, 0.0f);
    std::vector<float> v(count, 0.0f);

    se::simd::simd_adam_update(param.data(), grad.data(), m.data(), v.data(),
                               0.001f, 0.9f, 0.999f, 1e-8f, 1.0f, 1.0f, 0.01f,
                               count);

    for (size_t i = 0; i < count; ++i) {
      float expected_g = 0.2f + 0.01f * 1.0f;
      float expected_m = 0.9f * 0.0f + 0.1f * expected_g;
      float expected_v = 0.999f * 0.0f + 0.001f * expected_g * expected_g;
      float expected_p =
          1.0f - 0.001f * expected_m / (std::sqrt(expected_v) + 1e-8f);
      if (std::abs(param[i] - expected_p) > 1e-5f) {
        throw std::runtime_error("SIMD Adam result mismatch");
      }
    }
  }

  // 4. Test Uniform Init
  {
    std::vector<float> data(128, 0.0f);
    se::simd::simd_uniform_init(data.data(), data.size(), -0.5f, 0.5f, 42ULL);
    for (float val : data) {
      if (val < -0.5f || val > 0.5f) {
        throw std::runtime_error("SIMD uniform init out of range");
      }
    }
  }

  return 0;
}
