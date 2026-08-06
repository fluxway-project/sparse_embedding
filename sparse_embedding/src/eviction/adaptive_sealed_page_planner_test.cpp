#include "sparse_embedding/src/eviction/adaptive_sealed_page_planner.hpp"

#include <array>
#include <ranges>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main() {
  constexpr std::array<float, 2> first_value{1.0F, 2.0F};
  constexpr std::array<float, 2> second_value{3.0F, 4.0F};
  constexpr std::array<float, 2> third_value{5.0F, 6.0F};
  std::array<se::PageLayoutEntryView<float>, 1> first_entries{
      se::PageLayoutEntryView<float>{10, 2, first_value}};
  std::array<se::PageLayoutEntryView<float>, 1> second_entries{
      se::PageLayoutEntryView<float>{20, 4, second_value}};
  std::array<se::PageLayoutEntryView<float>, 1> third_entries{
      se::PageLayoutEntryView<float>{30, 6, third_value}};
  std::array<se::PageLayoutInput<float>, 3> inputs{
      se::PageLayoutInput<float>{100, 128, {0.1F, 0.0F, 2.0F}, first_entries},
      se::PageLayoutInput<float>{101, 128, {0.9F, 1.0F, 2.0F}, second_entries},
      se::PageLayoutInput<float>{102, 128, {0.2F, 0.0F, 2.0F}, third_entries},
  };
  se::AdaptiveSealedPagePlanner<float> planner(
      se::AdaptivePageSelector{}, se::SealedHotPagePool<float>(2, 1, 2));
  constexpr se::AdaptivePageSelectorConfig config{0.0F, 0.0F, 0.75F, 2, 256};
  auto plan =
      planner.plan(inputs, 1.0F, config, se::PageThermalTier::kCold, 9, 3);
  require(plan.selection.bytes == 256 && plan.pages.size() == 2,
          "planner did not enforce the migration budget");
  require(std::ranges::equal(plan.selection.page_ids,
                             std::array<uint64_t, 2>{100, 102}),
          "planner did not select the cold physical pages");
  require(plan.pages[0]->layout().tier == se::PageThermalTier::kCold &&
              plan.pages[0]->layout().cluster_id == 9 &&
              plan.pages[0]->layout().layout_generation == 3,
          "planner did not preserve layout metadata");
  require(
      std::ranges::equal(plan.pages[0]->keys(), std::array<uint64_t, 1>{10}) &&
          std::ranges::equal(plan.pages[1]->value(0), third_value),
      "planner did not preserve selected page entries");

  se::AdaptiveSealedPagePlanner<float> scratch_planner(
      se::AdaptivePageSelector{}, se::SealedHotPagePool<float>(4, 1, 2));
  se::AdaptiveSealedPagePlannerScratch<float> scratch;
  scratch.reserve(inputs.size());
  auto scratch_plan = scratch_planner.plan(
      inputs, 1.0F, config, se::PageThermalTier::kCold, 9, 4, scratch);
  require(scratch_plan.pages.size() == 2 &&
              scratch.ids.capacity() >= inputs.size() &&
              scratch.selection.candidates.capacity() >= inputs.size(),
          "planner scratch storage was not accepted");
}
