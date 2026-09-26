#pragma once

#ifndef SPARSE_EMBEDDING_CHECKED_NUMERIC_HPP_
#define SPARSE_EMBEDDING_CHECKED_NUMERIC_HPP_

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace se {

template <typename Dest>
inline Dest checked_narrow(float value, std::string_view operation) {
  if (!std::isfinite(value))
    throw std::overflow_error(std::string(operation) +
                              " produced a non-finite value");
  const float max_value = static_cast<float>(std::numeric_limits<Dest>::max());
  if (value > max_value || value < -max_value)
    throw std::overflow_error(std::string(operation) +
                              " exceeds the destination dtype range");
  return static_cast<Dest>(value);
}

inline void validate_finite(float value, std::string_view name) {
  if (!std::isfinite(value))
    throw std::invalid_argument(std::string(name) + " must be finite");
}

} // namespace se

#endif // SPARSE_EMBEDDING_CHECKED_NUMERIC_HPP_
