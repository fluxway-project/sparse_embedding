#pragma once

#include <cstring>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace se {

template <typename T>
void copy_native_values(std::span<const std::byte> bytes,
                        std::vector<T> &values) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (bytes.size() % sizeof(T) != 0)
    throw std::runtime_error("native checkpoint binary value size is invalid");
  values.resize(bytes.size() / sizeof(T));
  if (!bytes.empty())
    std::memcpy(values.data(), bytes.data(), bytes.size());
}

} // namespace se
