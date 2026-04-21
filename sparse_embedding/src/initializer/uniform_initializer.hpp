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

#ifndef SPARSE_EMBEDDING_UNIFORM_INITIALIZER_HPP_
#define SPARSE_EMBEDDING_UNIFORM_INITIALIZER_HPP_

#include <cstdint>
#include <span>

#include "simd/simd_ops.hpp"

namespace se {

template <typename ValT> struct UniformInitializer {
  float min_val;
  float max_val;

  UniformInitializer(float min = -0.05f, float max = 0.05f)
      : min_val(min), max_val(max) {}

  void initialize(std::span<ValT> param, uint64_t key) const {
    if constexpr (simd::is_simd_supported_v<ValT>) {
      se::simd::simd_uniform_init(param.data(), param.size(), min_val, max_val,
                                  key);
    } else {
      constexpr size_t kChunkSize = 256;
      float buf[kChunkSize];
      size_t count = param.size();
      for (size_t i = 0; i < count; i += kChunkSize) {
        size_t chunk = std::min(kChunkSize, count - i);
        se::simd::simd_uniform_init(buf, chunk, min_val, max_val, key + i);
        for (size_t j = 0; j < chunk; ++j) {
          param[i + j] = static_cast<ValT>(buf[j]);
        }
      }
    }
  }
};

} // namespace se

#endif // SPARSE_EMBEDDING_UNIFORM_INITIALIZER_HPP_
