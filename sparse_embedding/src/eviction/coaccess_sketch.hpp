#pragma once

#ifndef SPARSE_EMBEDDING_COACCESS_SKETCH_HPP_
#define SPARSE_EMBEDDING_COACCESS_SKETCH_HPP_

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

namespace se {

class CoaccessSketch {
public:
  static constexpr size_t kRows = 4;
  static constexpr size_t kMaximumBatchSample = 16;

  explicit CoaccessSketch(size_t columns_power_of_two)
      : columns_(columns_power_of_two), mask_(columns_power_of_two - 1),
        counters_(std::make_unique<std::atomic<uint32_t>[]>(
            checked_counter_count(columns_power_of_two))) {
    for (size_t index = 0; index < kRows * columns_; ++index)
      counters_[index].store(0, std::memory_order_relaxed);
  }

  template <typename Key>
  void observe_batch(std::span<const Key> keys, uint32_t weight = 1) noexcept {
    if (weight == 0)
      std::terminate();
    const size_t sample_count = std::min(keys.size(), kMaximumBatchSample);
    std::array<uint64_t, kMaximumBatchSample> sampled{};
    if (sample_count == 0)
      return;
    const size_t stride = std::max(size_t{1}, keys.size() / sample_count);
    for (size_t sample = 0; sample < sample_count; ++sample)
      sampled[sample] = static_cast<uint64_t>(
          keys[std::min(sample * stride, keys.size() - 1)]);
    for (size_t left = 0; left < sample_count; ++left) {
      for (size_t right = left + 1; right < sample_count; ++right)
        observe_pair(sampled[left], sampled[right], weight);
    }
  }

  void observe_pair(uint64_t left, uint64_t right,
                    uint32_t weight = 1) noexcept {
    if (weight == 0)
      std::terminate();
    if (left == right)
      return;
    if (right < left)
      std::swap(left, right);
    for (size_t row = 0; row < kRows; ++row) {
      std::atomic<uint32_t> &counter =
          counters_[row * columns_ + pair_hash(left, right, row)];
      uint32_t observed = counter.load(std::memory_order_relaxed);
      while (observed != std::numeric_limits<uint32_t>::max() &&
             !counter.compare_exchange_weak(
                 observed,
                 static_cast<uint32_t>(std::min<uint64_t>(
                     std::numeric_limits<uint32_t>::max(),
                     static_cast<uint64_t>(observed) + weight)),
                 std::memory_order_relaxed, std::memory_order_relaxed)) {
      }
    }
  }

  uint32_t estimate(uint64_t left, uint64_t right) const noexcept {
    if (right < left)
      std::swap(left, right);
    uint32_t result = std::numeric_limits<uint32_t>::max();
    for (size_t row = 0; row < kRows; ++row) {
      result = std::min(
          result, counters_[row * columns_ + pair_hash(left, right, row)].load(
                      std::memory_order_relaxed));
    }
    return result;
  }

  void decay_range(size_t begin_column, size_t end_column, uint8_t shift) {
    if (begin_column > end_column || end_column > columns_)
      throw std::out_of_range("co-access decay range is invalid");
    if (shift == 0 || shift >= 32)
      throw std::invalid_argument("co-access decay shift must be in [1, 31]");
    for (size_t row = 0; row < kRows; ++row) {
      for (size_t column = begin_column; column < end_column; ++column) {
        std::atomic<uint32_t> &counter = counters_[row * columns_ + column];
        uint32_t observed = counter.load(std::memory_order_relaxed);
        while (!counter.compare_exchange_weak(observed, observed >> shift,
                                              std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
        }
      }
    }
  }

  void clear() noexcept {
    for (size_t index = 0; index < kRows * columns_; ++index)
      counters_[index].store(0, std::memory_order_relaxed);
  }

  size_t columns() const noexcept { return columns_; }

private:
  static size_t checked_counter_count(size_t columns_power_of_two) {
    if (!std::has_single_bit(columns_power_of_two) || columns_power_of_two < 64)
      throw std::invalid_argument(
          "co-access sketch columns must be a power of two >= 64");
    if (columns_power_of_two > std::numeric_limits<size_t>::max() / kRows)
      throw std::overflow_error("co-access sketch size overflows size_t");
    return kRows * columns_power_of_two;
  }

  size_t pair_hash(uint64_t left, uint64_t right, size_t row) const noexcept {
    uint64_t value = left + 0x9e3779b97f4a7c15ULL * (row + 1);
    value ^= std::rotl(right, static_cast<int>(13 + row * 7));
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<size_t>(value) & mask_;
  }

  size_t columns_;
  size_t mask_;
  std::unique_ptr<std::atomic<uint32_t>[]> counters_;
};

} // namespace se

#endif
