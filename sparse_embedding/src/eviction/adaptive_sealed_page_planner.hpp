#pragma once

#ifndef SPARSE_EMBEDDING_ADAPTIVE_SEALED_PAGE_PLANNER_HPP_
#define SPARSE_EMBEDDING_ADAPTIVE_SEALED_PAGE_PLANNER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sparse_embedding/src/eviction/adaptive_page_selector.hpp"
#include "sparse_embedding/src/eviction/sealed_hot_page.hpp"

namespace se {

template <typename Value> struct PageLayoutEntryView {
  uint64_t key;
  uint64_t seq_count;
  std::span<const Value> value;
};

template <typename Value> struct PageLayoutInput {
  uint64_t physical_page_id;
  uint64_t bytes;
  PageThermalSignal signal;
  std::span<const PageLayoutEntryView<Value>> entries;
};

template <typename Value> struct AdaptiveSealedPagePlan {
  AdaptivePageSelection selection;
  std::vector<std::shared_ptr<const SealedHotPage<Value>>> pages;
};

template <typename Value> struct AdaptiveSealedPagePlannerScratch {
  AdaptivePageSelectionScratch selection;
  std::vector<uint64_t> ids;
  std::vector<uint64_t> bytes;
  std::vector<PageThermalSignal> signals;

  void reserve(size_t count) {
    selection.reserve(count);
    ids.reserve(count);
    bytes.reserve(count);
    signals.reserve(count);
  }
};

template <typename Value> class AdaptiveSealedPagePlanner {
public:
  AdaptiveSealedPagePlanner(AdaptivePageSelector selector,
                            SealedHotPagePool<Value> pool)
      : selector_(std::move(selector)), pool_(std::move(pool)) {}

  AdaptiveSealedPagePlan<Value>
  plan(std::span<const PageLayoutInput<Value>> inputs, float memory_pressure,
       const AdaptivePageSelectorConfig &config, PageThermalTier tier,
       uint32_t cluster_id, uint64_t page_layout_generation) {
    if (page_layout_generation == 0)
      throw std::invalid_argument("sealed page plan layout epoch is zero");
    if (inputs.empty())
      throw std::invalid_argument("sealed page plan requires input pages");

    AdaptiveSealedPagePlannerScratch<Value> scratch;
    scratch.reserve(inputs.size());
    return plan(inputs, memory_pressure, config, tier, cluster_id,
                page_layout_generation, scratch);
  }

  AdaptiveSealedPagePlan<Value>
  plan(std::span<const PageLayoutInput<Value>> inputs, float memory_pressure,
       const AdaptivePageSelectorConfig &config, PageThermalTier tier,
       uint32_t cluster_id, uint64_t page_layout_generation,
       AdaptiveSealedPagePlannerScratch<Value> &scratch) {
    if (page_layout_generation == 0)
      throw std::invalid_argument("sealed page plan layout epoch is zero");
    if (inputs.empty())
      throw std::invalid_argument("sealed page plan requires input pages");
    scratch.ids.clear();
    scratch.bytes.clear();
    scratch.signals.clear();
    for (const PageLayoutInput<Value> &input : inputs) {
      if (input.bytes == 0 || input.entries.empty())
        throw std::invalid_argument("sealed page plan input is incomplete");
      scratch.ids.push_back(input.physical_page_id);
      scratch.bytes.push_back(input.bytes);
      scratch.signals.push_back(input.signal);
    }

    AdaptiveSealedPagePlan<Value> result;
    result.selection =
        selector_.select(scratch.ids, scratch.signals, scratch.bytes,
                         memory_pressure, config, scratch.selection);
    result.pages.reserve(result.selection.page_ids.size());
    for (uint64_t page_id : result.selection.page_ids) {
      const PageLayoutInput<Value> *input = nullptr;
      for (const PageLayoutInput<Value> &candidate : inputs) {
        if (candidate.physical_page_id == page_id) {
          input = &candidate;
          break;
        }
      }
      if (input == nullptr)
        throw std::logic_error("selector returned an unknown physical page");
      auto reservation =
          pool_.reserve(tier, cluster_id, page_layout_generation);
      for (const PageLayoutEntryView<Value> &entry : input->entries)
        reservation.builder.append(entry.key, entry.seq_count, entry.value);
      result.pages.push_back(pool_.publish(std::move(reservation)));
    }
    return result;
  }

private:
  AdaptivePageSelector selector_;
  SealedHotPagePool<Value> pool_;
};

} // namespace se

#endif
