// Copyright 2026 The Fluxway Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "sparse_embedding/src/common/numeric_types.hpp"

namespace se {

// =============================================================================
// Key Data Type System (SSOT)
// =============================================================================

enum class KeyDType : uint8_t {
  kInt32,
  kUInt32,
  kInt64,
  kUInt64,
};

template <KeyDType DType> struct KeyTypeTraits;
template <> struct KeyTypeTraits<KeyDType::kInt32> {
  using type = int32_t;
};
template <> struct KeyTypeTraits<KeyDType::kUInt32> {
  using type = uint32_t;
};
template <> struct KeyTypeTraits<KeyDType::kInt64> {
  using type = int64_t;
};
template <> struct KeyTypeTraits<KeyDType::kUInt64> {
  using type = uint64_t;
};

template <typename T> struct KeyDTypeOf;
template <> struct KeyDTypeOf<int32_t> {
  static constexpr KeyDType value = KeyDType::kInt32;
};
template <> struct KeyDTypeOf<uint32_t> {
  static constexpr KeyDType value = KeyDType::kUInt32;
};
template <> struct KeyDTypeOf<int64_t> {
  static constexpr KeyDType value = KeyDType::kInt64;
};
template <> struct KeyDTypeOf<uint64_t> {
  static constexpr KeyDType value = KeyDType::kUInt64;
};

inline std::string_view key_dtype_name(KeyDType dtype) {
  switch (dtype) {
  case KeyDType::kInt32:
    return "int32";
  case KeyDType::kUInt32:
    return "uint32";
  case KeyDType::kInt64:
    return "int64";
  case KeyDType::kUInt64:
    return "uint64";
  }
  throw std::invalid_argument("unsupported key dtype code");
}

inline KeyDType parse_key_dtype(std::string_view dtype) {
  if (dtype == "int32")
    return KeyDType::kInt32;
  if (dtype == "uint32")
    return KeyDType::kUInt32;
  if (dtype == "int64")
    return KeyDType::kInt64;
  if (dtype == "uint64")
    return KeyDType::kUInt64;
  throw std::invalid_argument("unsupported key dtype: " + std::string(dtype));
}

template <typename Fn>
decltype(auto) dispatch_key_type(KeyDType dtype, Fn &&fn) {
  switch (dtype) {
  case KeyDType::kInt32:
    return fn.template operator()<int32_t>();
  case KeyDType::kUInt32:
    return fn.template operator()<uint32_t>();
  case KeyDType::kInt64:
    return fn.template operator()<int64_t>();
  case KeyDType::kUInt64:
    return fn.template operator()<uint64_t>();
  }
  throw std::invalid_argument("unsupported key dtype code");
}

template <typename KeyT>
uint64_t modulo_partition(KeyT key, uint64_t partition_count) {
  static_assert(std::is_integral_v<KeyT>);
  if (partition_count == 0)
    throw std::invalid_argument("partition_count must be positive");
  if constexpr (std::is_signed_v<KeyT>) {
    const __int128 divisor = static_cast<__int128>(partition_count);
    __int128 remainder = static_cast<__int128>(key) % divisor;
    if (remainder < 0)
      remainder += divisor;
    return static_cast<uint64_t>(remainder);
  } else {
    return static_cast<uint64_t>(key) % partition_count;
  }
}

// =============================================================================
// Value Data Type System (SSOT)
// =============================================================================

enum class ValueDType : uint8_t {
  kFloat32,
  kFloat16,
  kBFloat16,
};

template <ValueDType DType> struct ValueTypeTraits;
template <> struct ValueTypeTraits<ValueDType::kFloat32> {
  using type = float;
};
template <> struct ValueTypeTraits<ValueDType::kFloat16> {
  using type = se::float16_t;
};
template <> struct ValueTypeTraits<ValueDType::kBFloat16> {
  using type = se::bfloat16_t;
};

template <typename T> struct ValueDTypeOf;
template <> struct ValueDTypeOf<float> {
  static constexpr ValueDType value = ValueDType::kFloat32;
};
template <> struct ValueDTypeOf<se::float16_t> {
  static constexpr ValueDType value = ValueDType::kFloat16;
};
template <> struct ValueDTypeOf<se::bfloat16_t> {
  static constexpr ValueDType value = ValueDType::kBFloat16;
};

inline std::string_view value_dtype_name(ValueDType dtype) {
  switch (dtype) {
  case ValueDType::kFloat32:
    return "float32";
  case ValueDType::kFloat16:
    return "float16";
  case ValueDType::kBFloat16:
    return "bfloat16";
  }
  throw std::invalid_argument("unsupported value dtype code");
}

inline ValueDType parse_value_dtype(std::string_view dtype,
                                    std::string_view label = "value dtype") {
  if (dtype == "float32" || dtype == "f32")
    return ValueDType::kFloat32;
  if (dtype == "float16" || dtype == "f16")
    return ValueDType::kFloat16;
  if (dtype == "bfloat16" || dtype == "bf16")
    return ValueDType::kBFloat16;
  throw std::invalid_argument("unsupported " + std::string(label) + ": " +
                              std::string(dtype));
}

template <typename Fn>
decltype(auto) dispatch_value_type(ValueDType dtype, Fn &&fn) {
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
  return dispatch_value_type(param_dtype, [&]<typename ParamT>() {
    return dispatch_value_type(optimizer_dtype, [&]<typename OptT>() {
      return fn.template operator()<ParamT, OptT>();
    });
  });
}

inline size_t value_dtype_itemsize(ValueDType dtype) {
  return dispatch_value_type(dtype,
                             []<typename ValueT>() { return sizeof(ValueT); });
}

} // namespace se
