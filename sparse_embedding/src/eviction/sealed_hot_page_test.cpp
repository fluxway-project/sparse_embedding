#include "sparse_embedding/src/eviction/sealed_hot_page.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <ranges>
#include <stdexcept>

namespace {

template <typename Function> void require_throws(Function &&function) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected operation to throw");
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main() {
  se::SealedHotPageBuilder<float> builder(se::PageThermalTier::kHot, 41, 9, 2,
                                          5);
  constexpr std::array<float, 5> first{1, 2, 3, 4, 5};
  constexpr std::array<float, 5> second{6, 7, 8, 9, 10};
  require(builder.append(100, 2, first) == 0, "first slot mismatch");
  require(builder.append(101, 4, second) == 1, "second slot mismatch");
  require_throws([&] { builder.append(102, 6, second); });
  auto page = builder.seal();
  require(page.layout().tier == se::PageThermalTier::kHot,
          "sealed page tier mismatch");
  require(page.layout().cluster_id == 41 &&
              page.layout().layout_generation == 9,
          "sealed page layout metadata mismatch");
  require(page.layout().page_slot_stride_bytes == 64,
          "sealed page payload is not cache-line aligned");
  require(reinterpret_cast<uintptr_t>(page.payload_data()) % 64 == 0,
          "sealed page payload base is not cache-line aligned");
  require(std::ranges::equal(page.keys(), std::array<uint64_t, 2>{100, 101}),
          "sealed page keys mismatch");
  require(std::ranges::equal(page.seq_counts(), std::array<uint64_t, 2>{2, 4}),
          "sealed page seq_counts mismatch");
  require(std::ranges::equal(page.value(1), second),
          "sealed page payload mismatch");
  require_throws([&] { builder.seal(); });
  require_throws([&] { page.value(2); });

  se::SealedHotPagePool<float> pool(1, 2, 5);
  auto reservation = pool.reserve(se::PageThermalTier::kHot, 7, 10);
  require_throws([&] { pool.snapshot(reservation.page_id); });
  reservation.builder.append(200, 6, first);
  auto published = pool.publish(std::move(reservation));
  require(pool.published_count() == 1,
          "sealed page publication count mismatch");
  require(pool.snapshot(0) == published, "sealed page snapshot mismatch");
  require_throws([&] { pool.reserve(se::PageThermalTier::kCold, 8, 11); });

  se::SealedHotPagePool<float> other_pool(1, 2, 5);
  auto foreign_reservation =
      other_pool.reserve(se::PageThermalTier::kCold, 8, 11);
  require_throws([&] { pool.publish(std::move(foreign_reservation)); });
  foreign_reservation.builder.append(300, 8, first);
  other_pool.publish(std::move(foreign_reservation));

  se::SealedHotPagePool<float> recycling_pool(1, 2, 5);
  {
    auto abandoned = recycling_pool.reserve(se::PageThermalTier::kCold, 9, 12);
    abandoned.builder.append(400, 10, first);
  }
  auto replacement = recycling_pool.reserve(se::PageThermalTier::kHot, 10, 13);
  replacement.builder.append(401, 12, second);
  recycling_pool.publish(std::move(replacement));
}
