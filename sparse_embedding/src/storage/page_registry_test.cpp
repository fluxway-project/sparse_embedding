#include "sparse_embedding/src/storage/page_registry.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main() {
  se::storage_detail::PageRegistry registry;
  std::array<std::byte, 4096> first{};
  std::array<std::byte, 4096> second{};
  const size_t first_index = registry.allocate_page_index();
  const size_t second_index = registry.allocate_page_index();
  registry.insert({first.data(), first.size(), SIZE_MAX, first_index,
                   se::storage_detail::PageBacking::kAnonymousMapping, true});
  registry.insert({second.data(), second.size(), 0, second_index,
                   se::storage_detail::PageBacking::kSpillMapping, false});
  require(registry.find(first.data())->page_idx == first_index,
          "page registry lookup returned the wrong record");

  bool duplicate_rejected = false;
  try {
    registry.insert({first.data(), first.size(), SIZE_MAX, first_index,
                     se::storage_detail::PageBacking::kAnonymousMapping,
                     false});
  } catch (const std::logic_error &) {
    duplicate_rejected = true;
  }
  require(duplicate_rejected, "page registry accepted a duplicate address");

  registry.register_evaluator(first_index, [] { return uint64_t{7}; });
  require(registry.evaluators().at(first_index)() == 7,
          "page evaluator registry lost its callable");
  registry.unregister_evaluator(first_index);
  require(registry.evaluators().empty(),
          "page evaluator registry did not erase its callable");

  std::vector<std::array<std::byte, 1>> pages(1100);
  for (size_t index = 0; index < pages.size(); ++index) {
    registry.register_evictable(
        {pages[index].data(), index + 100, size_t{4096}});
  }
  const auto snapshot = registry.snapshot_evictable_pages();
  require(!snapshot.empty() && snapshot.size() <= 256,
          "page registry sampling exceeded its fixed budget");
  for (size_t index = 0; index < pages.size(); ++index)
    registry.unregister_evictable(index + 100);

  const auto removed = registry.erase(first.data());
  require(removed.page_idx == first_index,
          "page registry erased the wrong record");
  require(registry.find(first.data()) == nullptr,
          "page registry retained an erased address");
  require(registry.find(second.data())->page_idx == second_index,
          "page registry failed to repair a moved address index");
  registry.erase(second.data());
}
