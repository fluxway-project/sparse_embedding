#pragma once

#ifndef SPARSE_EMBEDDING_PAGE_TELEMETRY_HPP_
#define SPARSE_EMBEDDING_PAGE_TELEMETRY_HPP_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "sparse_embedding/src/eviction/coaccess_sketch.hpp"

namespace se {

class PageAccessTelemetry {
public:
  PageAccessTelemetry(size_t page_count, size_t coaccess_columns)
      : page_count_(checked_page_count(page_count)),
        hits_(std::make_unique<std::atomic<uint32_t>[]>(page_count_)),
        coaccess_(coaccess_columns) {
    for (size_t page = 0; page < page_count_; ++page)
      hits_[page].store(0, std::memory_order_relaxed);
  }

  void observe(uint32_t page_id, uint32_t weight = 1) noexcept {
    if (page_id >= page_count_)
      std::terminate();
    if (weight == 0)
      std::terminate();
    std::atomic<uint32_t> &hit = hits_[page_id];
    uint32_t current = hit.load(std::memory_order_relaxed);
    while (current != std::numeric_limits<uint32_t>::max() &&
           !hit.compare_exchange_weak(
               current,
               static_cast<uint32_t>(
                   std::min<uint64_t>(std::numeric_limits<uint32_t>::max(),
                                      static_cast<uint64_t>(current) + weight)),
               std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
  }

  void observe_batch(std::span<const uint64_t> keys,
                     std::span<const uint32_t> page_ids) noexcept {
    if (keys.size() != page_ids.size())
      std::terminate();
    for (uint32_t page_id : page_ids)
      observe(page_id);
    coaccess_.observe_batch(keys);
  }

  void observe_coaccess_batch(std::span<const uint64_t> keys,
                              uint32_t weight = 1) noexcept {
    coaccess_.observe_batch(keys, weight);
  }

  uint32_t hits(uint32_t page_id) const {
    if (page_id >= page_count_)
      throw std::out_of_range("page telemetry page id is out of range");
    return hits_[page_id].load(std::memory_order_relaxed);
  }

  float coaccess_density(std::span<const uint64_t> keys) const {
    if (keys.empty())
      throw std::invalid_argument("page telemetry requires a key sample");
    const size_t count =
        std::min(keys.size(), CoaccessSketch::kMaximumBatchSample);
    if (count == 1)
      return 0.0F;
    uint64_t total = 0;
    for (size_t index = 1; index < count; ++index)
      total += coaccess_.estimate(keys[0], keys[index]);
    return static_cast<float>(total) / static_cast<float>(count - 1);
  }

  uint32_t coaccess_estimate(uint64_t left, uint64_t right) const noexcept {
    return coaccess_.estimate(left, right);
  }

  void decay(uint8_t shift) {
    if (shift == 0 || shift >= 32)
      throw std::invalid_argument(
          "page telemetry decay shift must be in [1, 31]");
    for (size_t page = 0; page < page_count_; ++page) {
      std::atomic<uint32_t> &hit = hits_[page];
      uint32_t current = hit.load(std::memory_order_relaxed);
      while (!hit.compare_exchange_weak(current, current >> shift,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
      }
    }
    coaccess_.decay_range(0, coaccess_.columns(), shift);
  }

  void reset() noexcept {
    for (size_t page = 0; page < page_count_; ++page)
      hits_[page].store(0, std::memory_order_relaxed);
    coaccess_.clear();
  }

private:
  static size_t checked_page_count(size_t page_count) {
    if (page_count == 0)
      throw std::invalid_argument("page telemetry requires at least one page");
    return page_count;
  }

  size_t page_count_;
  std::unique_ptr<std::atomic<uint32_t>[]> hits_;
  CoaccessSketch coaccess_;
};

class StripedPageAccessTelemetry {
public:
  StripedPageAccessTelemetry(size_t page_count, size_t coaccess_columns,
                             size_t stripe_count) {
    if (stripe_count == 0)
      throw std::invalid_argument("page telemetry stripe count is zero");
    stripes_.reserve(stripe_count);
    for (size_t stripe = 0; stripe < stripe_count; ++stripe) {
      stripes_.push_back(
          std::make_unique<PageAccessTelemetry>(page_count, coaccess_columns));
    }
  }

  void observe(size_t stripe, uint32_t page_id, uint32_t weight = 1) noexcept {
    stripe_at(stripe).observe(page_id, weight);
  }

  void observe_coaccess_batch(size_t stripe, std::span<const uint64_t> keys,
                              uint32_t weight = 1) noexcept {
    stripe_at(stripe).observe_coaccess_batch(keys, weight);
  }

  uint64_t hits(uint32_t page_id) const {
    uint64_t total = 0;
    for (const auto &stripe : stripes_)
      total += stripe->hits(page_id);
    return total;
  }

  float coaccess_density(std::span<const uint64_t> keys) const {
    float total = 0.0F;
    for (const auto &stripe : stripes_)
      total += stripe->coaccess_density(keys);
    return total;
  }

  uint64_t coaccess_estimate(uint64_t left, uint64_t right) const noexcept {
    uint64_t total = 0;
    for (const auto &stripe : stripes_)
      total += stripe->coaccess_estimate(left, right);
    return total;
  }

  void decay(uint8_t shift) {
    for (const auto &stripe : stripes_)
      stripe->decay(shift);
  }

  void reset() noexcept {
    for (const auto &stripe : stripes_)
      stripe->reset();
  }

private:
  PageAccessTelemetry &stripe_at(size_t stripe) noexcept {
    if (stripe >= stripes_.size())
      std::terminate();
    return *stripes_[stripe];
  }

  std::vector<std::unique_ptr<PageAccessTelemetry>> stripes_;
};

class PageResidencyTelemetry {
public:
  explicit PageResidencyTelemetry(size_t page_count)
      : page_count_(page_count),
        residency_started_at_(std::make_unique<std::atomic<uint64_t>[]>(
            checked_page_count(page_count))) {
    for (size_t page = 0; page < page_count_; ++page) {
      residency_started_at_[page].store(0, std::memory_order_relaxed);
    }
  }

  void register_resident(uint32_t page_id) {
    if (page_id >= page_count_)
      throw std::out_of_range("page telemetry page id is out of range");
    const uint64_t epoch = residency_epoch_.load(std::memory_order_acquire);
    if (epoch == 0)
      throw std::logic_error("page telemetry residency epoch is invalid");
    uint64_t expected = 0;
    if (!residency_started_at_[page_id].compare_exchange_strong(
            expected, epoch, std::memory_order_release,
            std::memory_order_acquire)) {
      throw std::logic_error("page telemetry page was already registered");
    }
  }

  void advance_residency_epoch() {
    uint64_t epoch = residency_epoch_.load(std::memory_order_acquire);
    while (true) {
      if (epoch == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("page telemetry residency epoch overflow");
      if (residency_epoch_.compare_exchange_weak(epoch, epoch + 1,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
        return;
      }
    }
  }

  float residency_age(uint32_t page_id) const {
    if (page_id >= page_count_)
      throw std::out_of_range("page telemetry page id is out of range");
    const uint64_t started =
        residency_started_at_[page_id].load(std::memory_order_acquire);
    if (started == 0)
      throw std::logic_error("page telemetry page was not registered");
    const uint64_t epoch = residency_epoch_.load(std::memory_order_acquire);
    if (epoch < started)
      throw std::logic_error("page telemetry residency epoch moved backward");
    return static_cast<float>(epoch - started);
  }

  void reset() {
    for (size_t page = 0; page < page_count_; ++page) {
      residency_started_at_[page].store(0, std::memory_order_relaxed);
    }
    residency_epoch_.store(1, std::memory_order_release);
  }

private:
  static size_t checked_page_count(size_t page_count) {
    if (page_count == 0)
      throw std::invalid_argument("page telemetry requires at least one page");
    return page_count;
  }

  size_t page_count_;
  std::unique_ptr<std::atomic<uint64_t>[]> residency_started_at_;
  std::atomic<uint64_t> residency_epoch_{1};
};

} // namespace se

#endif
