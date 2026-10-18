#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace se {

struct NativeEntryBatch {
  std::vector<uint32_t> dimensions;
  std::vector<uint64_t> keys;
  std::vector<std::byte> embedding_values;
  std::vector<uint32_t> embedding_offsets;
  std::string embedding_dtype;
  std::vector<std::byte> optimizer_values;
  std::vector<uint32_t> optimizer_offsets;
  std::string optimizer_dtype;
  std::vector<uint32_t> optimizer_strides;
  std::vector<uint32_t> steps;
  std::vector<uint64_t> scores;
  std::vector<uint64_t> sequences;
};

} // namespace se
