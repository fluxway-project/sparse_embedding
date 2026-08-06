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

#ifndef SPARSE_EMBEDDING_BINDINGS_DTYPE_DISPATCH_HPP_
#define SPARSE_EMBEDDING_BINDINGS_DTYPE_DISPATCH_HPP_

#include <dlpack/dlpack.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include "sparse_embedding/src/common/numeric_types.hpp"

namespace se::py::detail {

enum class KeyDType : uint8_t { kInt32, kUInt32, kInt64, kUInt64 };
enum class ValueDType : uint8_t { kFloat32, kFloat16, kBFloat16 };

inline KeyDType parse_key_dtype(std::string_view dtype) {
  if (dtype == "int32" || dtype == "i32")
    return KeyDType::kInt32;
  if (dtype == "uint32" || dtype == "u32")
    return KeyDType::kUInt32;
  if (dtype == "int64" || dtype == "i64")
    return KeyDType::kInt64;
  if (dtype == "uint64" || dtype == "u64" || dtype.empty())
    return KeyDType::kUInt64;
  throw std::invalid_argument("unsupported key_dtype: " + std::string(dtype));
}

inline ValueDType parse_value_dtype(std::string_view dtype,
                                    std::string_view label) {
  if (dtype == "float32" || dtype == "f32")
    return ValueDType::kFloat32;
  if (dtype == "float16" || dtype == "f16")
    return ValueDType::kFloat16;
  if (dtype == "bfloat16" || dtype == "bf16")
    return ValueDType::kBFloat16;
  throw std::invalid_argument("unsupported " + std::string(label) + ": " +
                              std::string(dtype));
}

inline KeyDType detect_key_dtype(const DLManagedTensor *dlm) {
  const auto code = dlm->dl_tensor.dtype.code;
  const auto bits = dlm->dl_tensor.dtype.bits;
  if (code == static_cast<uint8_t>(kDLInt) && bits == 32)
    return KeyDType::kInt32;
  if (code == static_cast<uint8_t>(kDLUInt) && bits == 32)
    return KeyDType::kUInt32;
  if (code == static_cast<uint8_t>(kDLInt) && bits == 64)
    return KeyDType::kInt64;
  if (code == static_cast<uint8_t>(kDLUInt) && bits == 64)
    return KeyDType::kUInt64;
  throw std::invalid_argument("keys must be int32, uint32, int64 or uint64");
}

template <typename Fn>
decltype(auto) dispatch_value_dtype(ValueDType dtype, Fn &&fn) {
  switch (dtype) {
  case ValueDType::kFloat32:
    return fn.template operator()<float>();
  case ValueDType::kFloat16:
    return fn.template operator()<se::float16_t>();
  case ValueDType::kBFloat16:
    return fn.template operator()<se::bfloat16_t>();
  }
  throw std::invalid_argument("unsupported value dtype code");
}

template <typename Fn>
decltype(auto) dispatch_dtype_pair(ValueDType param_dtype,
                                   ValueDType optimizer_dtype, Fn &&fn) {
  switch (param_dtype) {
  case ValueDType::kFloat32:
    switch (optimizer_dtype) {
    case ValueDType::kFloat32:
      return fn.template operator()<float, float>();
    case ValueDType::kFloat16:
      return fn.template operator()<float, se::float16_t>();
    case ValueDType::kBFloat16:
      return fn.template operator()<float, se::bfloat16_t>();
    }
    break;
  case ValueDType::kFloat16:
    switch (optimizer_dtype) {
    case ValueDType::kFloat32:
      return fn.template operator()<se::float16_t, float>();
    case ValueDType::kFloat16:
      return fn.template operator()<se::float16_t, se::float16_t>();
    case ValueDType::kBFloat16:
      return fn.template operator()<se::float16_t, se::bfloat16_t>();
    }
    break;
  case ValueDType::kBFloat16:
    switch (optimizer_dtype) {
    case ValueDType::kFloat32:
      return fn.template operator()<se::bfloat16_t, float>();
    case ValueDType::kFloat16:
      return fn.template operator()<se::bfloat16_t, se::float16_t>();
    case ValueDType::kBFloat16:
      return fn.template operator()<se::bfloat16_t, se::bfloat16_t>();
    }
    break;
  }
  throw std::invalid_argument("unsupported dtype pair");
}

} // namespace se::py::detail

#endif // SPARSE_EMBEDDING_BINDINGS_DTYPE_DISPATCH_HPP_
