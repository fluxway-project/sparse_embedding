#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace se {

enum class OptimizerStateResidency : uint8_t { kHost, kDevice, kManaged };

enum class OptimizerStateGranularity : uint8_t {
  kEntry,
  kDimension,
  kTile,
  kTable,
};

struct OptimizerStateBlock {
  std::string_view name;
  size_t element_size = 0;
  size_t elements_per_entry = 0;
  size_t alignment = alignof(std::max_align_t);
  OptimizerStateResidency residency = OptimizerStateResidency::kHost;
  OptimizerStateGranularity granularity = OptimizerStateGranularity::kEntry;
};

class OptimizerStateLayout {
public:
  OptimizerStateLayout() = default;
  OptimizerStateLayout(std::vector<OptimizerStateBlock> blocks, size_t version)
      : blocks_(std::move(blocks)), version_(version) {
    validate();
  }

  const std::vector<OptimizerStateBlock> &blocks() const noexcept {
    return blocks_;
  }

  size_t version() const noexcept { return version_; }

  size_t block_offset(size_t index) const {
    validate();
    if (index >= blocks_.size())
      throw std::out_of_range("optimizer state block index is out of range");
    size_t offset = 0;
    for (size_t i = 0; i < index; ++i) {
      offset = align_up(offset, blocks_[i].alignment);
      offset = checked_add(offset, block_bytes(blocks_[i]));
    }
    return align_up(offset, blocks_[index].alignment);
  }

  size_t block_bytes(size_t index) const {
    if (index >= blocks_.size())
      throw std::out_of_range("optimizer state block index is out of range");
    return block_bytes(blocks_[index]);
  }

  size_t bytes_per_entry() const {
    validate();
    size_t total = 0;
    for (const OptimizerStateBlock &block : blocks_) {
      total = align_up(total, block.alignment);
      total = checked_add(total, block_bytes(block));
    }
    return total;
  }

  void validate() const {
    if (version_ == 0)
      throw std::invalid_argument("optimizer state layout version is zero");
    for (const OptimizerStateBlock &block : blocks_) {
      if (block.name.empty())
        throw std::invalid_argument("optimizer state block name is empty");
      if (block.element_size == 0 || block.elements_per_entry == 0)
        throw std::invalid_argument("optimizer state block has zero extent");
      if (block.alignment == 0 ||
          (block.alignment & (block.alignment - 1)) != 0)
        throw std::invalid_argument("optimizer state alignment is invalid");
      block_bytes(block);
    }
  }

private:
  static size_t checked_add(size_t left, size_t right) {
    if (left > std::numeric_limits<size_t>::max() - right)
      throw std::overflow_error("optimizer state layout overflows size_t");
    return left + right;
  }

  static size_t block_bytes(const OptimizerStateBlock &block) {
    if (block.element_size >
        std::numeric_limits<size_t>::max() / block.elements_per_entry)
      throw std::overflow_error("optimizer state block size overflows size_t");
    return block.element_size * block.elements_per_entry;
  }

  static size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
      throw std::invalid_argument("optimizer state alignment is invalid");
    const size_t remainder = value & (alignment - 1);
    if (remainder == 0)
      return value;
    return checked_add(value, alignment - remainder);
  }

  std::vector<OptimizerStateBlock> blocks_;
  size_t version_ = 0;
};

} // namespace se
