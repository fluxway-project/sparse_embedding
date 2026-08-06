#pragma once

#ifndef SPARSE_EMBEDDING_ADAPTIVE_PAGE_SELECTOR_HPP_
#define SPARSE_EMBEDDING_ADAPTIVE_PAGE_SELECTOR_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sparse_embedding/src/eviction/adaptive_boltzmann_policy.hpp"

namespace se {

struct AdaptivePageSelection {
  std::vector<uint64_t> page_ids;
  size_t bytes = 0;
};

struct AdaptivePageSelectionScratch {
  struct Candidate {
    uint64_t page_id;
    uint64_t bytes;
    double priority;
  };

  std::vector<uint64_t> sorted_ids;
  std::vector<float> weights;
  std::vector<Candidate> candidates;

  void reserve(size_t count) {
    sorted_ids.reserve(count);
    weights.reserve(count);
    candidates.reserve(count);
  }
};

struct AdaptivePageSelectorConfig {
  float minimum_residency_age = 0.0F;
  float hysteresis_utility_margin = 0.0F;
  float high_pressure_threshold = 0.75F;
  size_t max_pages = 1;
  size_t migration_budget_bytes = 0;
};

class AdaptivePageSelector {
public:
  AdaptivePageSelector() : policy_(AdaptiveBoltzmannConfig{}) {}

  explicit AdaptivePageSelector(AdaptiveBoltzmannPolicy policy)
      : policy_(std::move(policy)) {}

  AdaptivePageSelection select(std::span<const uint64_t> page_ids,
                               std::span<const PageThermalSignal> signals,
                               std::span<const uint64_t> page_bytes,
                               float memory_pressure,
                               const AdaptivePageSelectorConfig &config) const {
    if (page_ids.size() != signals.size() ||
        page_ids.size() != page_bytes.size())
      throw std::invalid_argument(
          "page selection columns have different sizes");
    if (page_ids.empty())
      throw std::invalid_argument("page selection requires at least one page");
    AdaptivePageSelectionScratch scratch;
    scratch.reserve(page_ids.size());
    return select(page_ids, signals, page_bytes, memory_pressure, config,
                  scratch);
  }

  AdaptivePageSelection select(std::span<const uint64_t> page_ids,
                               std::span<const PageThermalSignal> signals,
                               std::span<const uint64_t> page_bytes,
                               float memory_pressure,
                               const AdaptivePageSelectorConfig &config,
                               AdaptivePageSelectionScratch &scratch) const {
    if (page_ids.size() != signals.size() ||
        page_ids.size() != page_bytes.size())
      throw std::invalid_argument(
          "page selection columns have different sizes");
    if (page_ids.empty())
      throw std::invalid_argument("page selection requires at least one page");
    scratch.sorted_ids.assign(page_ids.begin(), page_ids.end());
    std::ranges::sort(scratch.sorted_ids);
    if (std::ranges::adjacent_find(scratch.sorted_ids) !=
        scratch.sorted_ids.end())
      throw std::invalid_argument("page selection contains duplicate ids");
    if (config.max_pages == 0)
      throw std::invalid_argument("page selection maximum must be non-zero");
    if (config.minimum_residency_age < 0.0F ||
        config.hysteresis_utility_margin < 0.0F ||
        !std::isfinite(config.minimum_residency_age) ||
        !std::isfinite(config.hysteresis_utility_margin) ||
        !std::isfinite(config.high_pressure_threshold) ||
        config.high_pressure_threshold < 0.0F ||
        config.high_pressure_threshold > 1.0F)
      throw std::invalid_argument("page selection thresholds are invalid");

    scratch.weights.resize(page_ids.size());
    const auto summary = policy_.compute_eviction_weights(
        signals, memory_pressure, std::span<float>(scratch.weights));
    scratch.candidates.clear();
    for (size_t i = 0; i < page_ids.size(); ++i) {
      if (page_bytes[i] == 0 ||
          signals[i].residency_age < config.minimum_residency_age)
        continue;
      const float utility = policy_.page_utility(signals[i]);
      if (memory_pressure < config.high_pressure_threshold &&
          utility > summary.minimum_utility + config.hysteresis_utility_margin)
        continue;
      const double priority = static_cast<double>(scratch.weights[i]) /
                              static_cast<double>(page_bytes[i]);
      scratch.candidates.push_back({page_ids[i], page_bytes[i], priority});
    }
    std::ranges::sort(scratch.candidates,
                      [](const auto &left, const auto &right) {
                        if (left.priority != right.priority)
                          return left.priority > right.priority;
                        return left.page_id < right.page_id;
                      });

    AdaptivePageSelection selection;
    selection.page_ids.reserve(
        std::min(config.max_pages, scratch.candidates.size()));
    for (const auto &candidate : scratch.candidates) {
      if (selection.page_ids.size() == config.max_pages)
        break;
      if (candidate.bytes > config.migration_budget_bytes - selection.bytes)
        continue;
      selection.page_ids.push_back(candidate.page_id);
      selection.bytes += candidate.bytes;
    }
    return selection;
  }

private:
  AdaptiveBoltzmannPolicy policy_;
};

} // namespace se

#endif
