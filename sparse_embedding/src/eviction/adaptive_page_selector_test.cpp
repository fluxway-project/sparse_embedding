#include "sparse_embedding/src/eviction/adaptive_page_selector.hpp"

#include <array>
#include <cstdint>
#include <ranges>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Function> void require_throws(Function &&function) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected operation to throw");
}

} // namespace

int main() {
  se::AdaptivePageSelector selector;
  constexpr std::array<uint64_t, 4> ids{10, 11, 12, 13};
  constexpr std::array<se::PageThermalSignal, 4> signals{
      se::PageThermalSignal{0.1F, 0.0F, 2.0F},
      se::PageThermalSignal{0.8F, 0.5F, 2.0F},
      se::PageThermalSignal{0.2F, 0.1F, 1.0F},
      se::PageThermalSignal{0.3F, 0.0F, 2.0F}};
  constexpr std::array<uint64_t, 4> bytes{100, 100, 100, 100};
  constexpr se::AdaptivePageSelectorConfig config{2.0F, 0.3F, 0.75F, 2, 200};
  auto selection = selector.select(ids, signals, bytes, 0.25F, config);
  require(selection.bytes == 200, "migration budget was not enforced");
  require(selection.page_ids.size() == 2,
          "maximum selected page count was not enforced");
  require(std::ranges::find(selection.page_ids, 12) == selection.page_ids.end(),
          "minimum residency age was not enforced");
  require(selection.page_ids[0] == 10 && selection.page_ids[1] == 13,
          "cold-page ordering was not deterministic");

  constexpr std::array<uint64_t, 2> pressure_ids{20, 21};
  constexpr std::array<se::PageThermalSignal, 2> pressure_signals{
      se::PageThermalSignal{0.0F, 0.0F, 0.0F},
      se::PageThermalSignal{1.0F, 0.0F, 0.0F}};
  constexpr std::array<uint64_t, 2> pressure_bytes{4, 1};
  constexpr se::AdaptivePageSelectorConfig pressure_config{0.0F, 0.0F, 0.0F, 1,
                                                           4};
  auto low_pressure = selector.select(pressure_ids, pressure_signals,
                                      pressure_bytes, 0.0F, pressure_config);
  auto high_pressure = selector.select(pressure_ids, pressure_signals,
                                       pressure_bytes, 1.0F, pressure_config);
  require(low_pressure.page_ids[0] == 20 && high_pressure.page_ids[0] == 21,
          "adaptive temperature did not affect cost-aware page priority");

  se::AdaptivePageSelectionScratch scratch;
  scratch.reserve(ids.size());
  auto first_reused =
      selector.select(ids, signals, bytes, 1.0F, config, scratch);
  const size_t ids_capacity = scratch.sorted_ids.capacity();
  const size_t weights_capacity = scratch.weights.capacity();
  const size_t candidates_capacity = scratch.candidates.capacity();
  auto second_reused =
      selector.select(ids, signals, bytes, 1.0F, config, scratch);
  require(first_reused.page_ids == second_reused.page_ids &&
              scratch.sorted_ids.capacity() == ids_capacity &&
              scratch.weights.capacity() == weights_capacity &&
              scratch.candidates.capacity() == candidates_capacity,
          "selector scratch storage was not reused");

  auto invalid_config = pressure_config;
  invalid_config.high_pressure_threshold = 2.0F;
  require_throws([&] {
    selector.select(pressure_ids, pressure_signals, pressure_bytes, 0.0F,
                    invalid_config);
  });
  constexpr std::array<uint64_t, 2> duplicate_ids{20, 20};
  require_throws([&] {
    selector.select(duplicate_ids, pressure_signals, pressure_bytes, 0.0F,
                    pressure_config);
  });
}
