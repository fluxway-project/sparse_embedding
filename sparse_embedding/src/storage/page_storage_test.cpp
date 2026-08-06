#include "sparse_embedding/src/storage/page_storage.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr size_t kExpectedSampledEvaluationLimit = 256;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class EvictionGuard {
public:
  explicit EvictionGuard(bool score_enabled = true)
      : score_enabled_(score_enabled) {}

  bool evaluate_page(size_t, uint64_t &score) {
    score = 0;
    return true;
  }
  void register_page(size_t, std::function<uint64_t()>) {}
  void unregister_page(size_t) noexcept {}
  bool score_enabled() const noexcept { return score_enabled_; }

private:
  bool score_enabled_;
};

class BlockingEvictionProxy {
public:
  bool evaluate_page(size_t, uint64_t &score) {
    std::unique_lock lock(evaluation_mutex_);
    evaluation_started_ = true;
    evaluation_cv_.notify_all();
    evaluation_cv_.wait(lock, [&] { return evaluation_released_; });
    score = 0;
    evaluation_finished_ = true;
    evaluation_cv_.notify_all();
    return true;
  }

  void register_page(size_t, std::function<uint64_t()>) {}
  void unregister_page(size_t) noexcept { wait_for_evaluation_completion(); }
  bool score_enabled() const noexcept { return true; }

  void wait_for_evaluation() {
    std::unique_lock lock(evaluation_mutex_);
    evaluation_cv_.wait(lock, [&] { return evaluation_started_; });
  }

  void release_evaluation() {
    std::lock_guard lock(evaluation_mutex_);
    evaluation_released_ = true;
    evaluation_cv_.notify_all();
  }

  void wait_for_evaluation_completion() {
    std::unique_lock lock(evaluation_mutex_);
    evaluation_cv_.wait(lock, [&] { return evaluation_finished_; });
  }

private:
  std::mutex evaluation_mutex_;
  std::condition_variable evaluation_cv_;
  bool evaluation_started_ = false;
  bool evaluation_released_ = false;
  bool evaluation_finished_ = false;
};

} // namespace

int main() {
  se::StorageOptions options;
  options.memory_limit_bytes = 0;
  options.spill_directory = std::filesystem::temp_directory_path();
  options.max_spill_bytes = 1 << 20;

  auto storage = se::create_page_storage(std::move(options));
  for (size_t alignment : {size_t{0}, size_t{3}, size_t{3000}}) {
    bool rejected = false;
    try {
      auto invalid_alignment = storage->allocate(4096, alignment);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "invalid allocation alignment was accepted");
  }

  se::StorageOptions late_proxy_options;
  late_proxy_options.memory_limit_bytes = 4096;
  auto late_proxy_storage =
      se::create_page_storage(std::move(late_proxy_options));
  auto heap_page = late_proxy_storage->allocate(4096, 64);
  late_proxy_storage->set_page_eviction_proxy(
      std::make_shared<EvictionGuard>());
  require(late_proxy_storage->stats().swap_out_count == 0,
          "non-evictable heap page entered the swap candidate set");

  se::StorageOptions compact_heap_options;
  compact_heap_options.mode = se::StorageMode::kMemoryOnly;
  compact_heap_options.memory_limit_bytes = 4096;
  auto compact_heap_storage =
      se::create_page_storage(std::move(compact_heap_options));
  auto compact_heap_page = compact_heap_storage->allocate(65, 64);
  require(compact_heap_storage->stats().memory_bytes == 128,
          "heap allocation was rounded to the operating-system page size");
  auto memory_only_scored_proxy = std::make_shared<EvictionGuard>(true);
  auto memory_only_unscored_proxy = std::make_shared<EvictionGuard>(false);
  auto memory_only_registration =
      compact_heap_storage->set_page_eviction_proxy(memory_only_scored_proxy);
  auto second_memory_only_registration =
      compact_heap_storage->set_page_eviction_proxy(memory_only_scored_proxy);
  bool unscored_rejected = false;
  try {
    compact_heap_storage->set_page_eviction_proxy(memory_only_unscored_proxy);
  } catch (const std::invalid_argument &) {
    unscored_rejected = true;
  }
  require(unscored_rejected, "unscored eviction proxy was accepted");
  compact_heap_storage->clear_page_eviction_proxy(memory_only_registration);
  bool repeated_registration_rejected = false;
  try {
    compact_heap_storage->clear_page_eviction_proxy(memory_only_registration);
  } catch (const std::logic_error &) {
    repeated_registration_rejected = true;
  }
  require(repeated_registration_rejected,
          "repeated page eviction registration cleanup was accepted");
  compact_heap_storage->clear_page_eviction_proxy(
      second_memory_only_registration);
  bool empty_registration_rejected = false;
  try {
    compact_heap_storage->clear_page_eviction_proxy({});
  } catch (const std::invalid_argument &) {
    empty_registration_rejected = true;
  }
  require(empty_registration_rejected,
          "empty page eviction registration was accepted");
  bool unknown_registration_rejected = false;
  try {
    compact_heap_storage->clear_page_eviction_proxy({999});
  } catch (const std::logic_error &) {
    unknown_registration_rejected = true;
  }
  require(unknown_registration_rejected,
          "unknown page eviction registration was accepted");

  se::StorageOptions concurrent_memory_options;
  concurrent_memory_options.mode = se::StorageMode::kMemoryOnly;
  concurrent_memory_options.memory_limit_bytes = 4096;
  auto concurrent_memory_storage =
      se::create_page_storage(std::move(concurrent_memory_options));
  std::array<se::StoragePage, 64> concurrent_pages;
  std::atomic<size_t> concurrent_allocation_count{0};
  std::atomic<size_t> concurrent_rejection_count{0};
  std::vector<std::thread> concurrent_allocations;
  concurrent_allocations.reserve(concurrent_pages.size());
  for (size_t index = 0; index < concurrent_pages.size(); ++index) {
    concurrent_allocations.emplace_back([&, index] {
      try {
        concurrent_pages[index] = concurrent_memory_storage->allocate(
            128, 64, {}, se::PageEvictionMode::kDisabled);
        concurrent_allocation_count.fetch_add(1, std::memory_order_relaxed);
      } catch (const std::bad_alloc &) {
        concurrent_rejection_count.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto &allocation : concurrent_allocations)
    allocation.join();
  require(concurrent_allocation_count.load(std::memory_order_relaxed) == 32,
          "concurrent heap allocation exceeded the memory budget");
  require(concurrent_rejection_count.load(std::memory_order_relaxed) == 32,
          "concurrent heap allocation did not reject excess capacity");
  require(concurrent_memory_storage->stats().memory_bytes == 4096,
          "concurrent heap reservation accounting was inconsistent");

  se::StorageOptions allocation_failure_options;
  allocation_failure_options.memory_limit_bytes =
      std::numeric_limits<size_t>::max();
  allocation_failure_options.spill_directory =
      std::filesystem::temp_directory_path();
  allocation_failure_options.max_spill_bytes = 1 << 20;
  auto allocation_failure_storage =
      se::create_page_storage(std::move(allocation_failure_options));
  bool allocation_failure_propagated = false;
  try {
    auto impossible_page = allocation_failure_storage->allocate(
        std::numeric_limits<size_t>::max() - 63, 64, {},
        se::PageEvictionMode::kDisabled);
  } catch (const std::bad_alloc &) {
    allocation_failure_propagated = true;
  }
  require(allocation_failure_propagated,
          "heap allocation failure did not propagate as bad_alloc");
  require(allocation_failure_storage->stats().memory_bytes == 0,
          "failed heap allocation retained a memory reservation");
  require(allocation_failure_storage->stats().spill_count == 0,
          "heap allocation failure fell back to spill storage");

  std::uintptr_t first_address = 0;
  {
    auto page = storage->allocate(4096, 64);
    first_address = reinterpret_cast<std::uintptr_t>(page.data());
    require(first_address % 64 == 0, "page is not cache-line aligned");
    page.data()[0] = std::byte{0x2a};
    page.data()[4095] = std::byte{0x7f};
    require(page.data()[0] == std::byte{0x2a}, "page start is not writable");
    require(page.data()[4095] == std::byte{0x7f}, "page end is not writable");

    auto stats = storage->stats();
    require(stats.memory_bytes == 0, "forced spill allocated heap memory");
    require(stats.spill_bytes == 4096, "spill bytes were not accounted");
    require(stats.spill_count == 1, "spill event was not accounted");
  }

  require(storage->stats().spill_bytes == 0,
          "released spill page remained live");
  auto reused = storage->allocate(4096, 64);
  require(reinterpret_cast<std::uintptr_t>(reused.data()) == first_address,
          "released spill extent was not reused");

  std::vector<se::StoragePage> pages;
  pages.reserve(128);
  for (size_t i = 0; i < 128; ++i) {
    pages.push_back(storage->allocate(4096, 64));
  }
  pages.clear();
  require(storage->stats().spill_bytes == reused.size(),
          "bulk spill release accounting mismatch");

  if (std::filesystem::is_directory("/dev/shm")) {
    se::StorageOptions memory_backed_options;
    memory_backed_options.memory_limit_bytes = 0;
    memory_backed_options.spill_directory = "/dev/shm";
    memory_backed_options.max_spill_bytes = 1 << 20;
    auto memory_backed =
        se::create_page_storage(std::move(memory_backed_options));
    bool rejected = false;
    try {
      memory_backed->allocate(4096, 64);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "memory-backed filesystem was accepted for spill");
  }

  const std::array<std::pair<double, double>, 2> invalid_watermarks = {
      std::pair{std::numeric_limits<double>::quiet_NaN(), 0.6},
      std::pair{0.8, std::numeric_limits<double>::quiet_NaN()}};
  for (const auto &[high, low] : invalid_watermarks) {
    se::StorageOptions invalid_watermark_options;
    invalid_watermark_options.watermark = se::WatermarkConfig{high, low};
    bool rejected = false;
    try {
      auto invalid_storage =
          se::create_page_storage(std::move(invalid_watermark_options));
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "NaN watermark was accepted");
  }

  se::StorageOptions watermark_options;
  watermark_options.memory_limit_bytes = 4096;
  watermark_options.spill_directory = std::filesystem::temp_directory_path();
  watermark_options.max_spill_bytes = 1 << 20;
  auto watermark_storage =
      se::create_page_storage(std::move(watermark_options));
  auto watermark_registration = watermark_storage->set_page_eviction_proxy(
      std::make_shared<EvictionGuard>());
  auto watermark_page =
      watermark_storage->allocate(4096, 64, {}, se::PageEvictionMode::kEnabled);
  const auto watermark_address = watermark_page.data();
  constexpr std::byte watermark_sentinel{0x5a};
  watermark_address[0] = watermark_sentinel;
  for (size_t attempt = 0;
       attempt < 100 && watermark_storage->stats().swap_out_count == 0;
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  auto watermark_stats = watermark_storage->stats();
  require(watermark_stats.swap_out_count == 1,
          "default watermarks did not trigger background swap-out");
  require(watermark_stats.memory_bytes == 0,
          "background swap-out did not release memory accounting");
  require(watermark_page.data() == watermark_address,
          "background swap changed the virtual address");
  require(watermark_page.data()[0] == watermark_sentinel,
          "background swap lost mapped page contents");
  require(watermark_stats.eviction_selection_count > 0,
          "background eviction did not record a selection pass");
  require(watermark_stats.eviction_evaluated_pages > 0,
          "background eviction did not record evaluated pages");
  require(watermark_stats.eviction_selection_ns > 0,
          "background eviction did not record selection time");
  require(watermark_stats.eviction_candidate_invalidated_count == 0,
          "stable eviction candidate was unexpectedly invalidated");
  require(watermark_storage->consume_spill_warning(),
          "background swap-out did not publish spill warning");

  se::StorageOptions default_eviction_options;
  default_eviction_options.memory_limit_bytes = 4096;
  default_eviction_options.spill_directory =
      std::filesystem::temp_directory_path();
  default_eviction_options.max_spill_bytes = 1 << 20;
  auto default_eviction_storage =
      se::create_page_storage(std::move(default_eviction_options));
  auto default_eviction_registration =
      default_eviction_storage->set_page_eviction_proxy(
          std::make_shared<EvictionGuard>());
  auto default_eviction_page = default_eviction_storage->allocate(4096, 64);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  require(default_eviction_storage->stats().swap_out_count == 0,
          "default allocation unexpectedly enabled background eviction");

  se::StorageOptions mixed_watermark_options;
  mixed_watermark_options.memory_limit_bytes = 8192;
  mixed_watermark_options.spill_directory =
      std::filesystem::temp_directory_path();
  mixed_watermark_options.max_spill_bytes = 1 << 20;
  mixed_watermark_options.watermark = se::WatermarkConfig{0.75, 0.5};
  auto mixed_watermark_storage =
      se::create_page_storage(std::move(mixed_watermark_options));
  auto mixed_watermark_registration =
      mixed_watermark_storage->set_page_eviction_proxy(
          std::make_shared<EvictionGuard>());
  auto mixed_heap_page = mixed_watermark_storage->allocate(
      4096, 64, {}, se::PageEvictionMode::kDisabled);
  auto mixed_evictable_page = mixed_watermark_storage->allocate(
      4096, 64, [] { return uint64_t{0}; }, se::PageEvictionMode::kEnabled);
  for (size_t attempt = 0;
       attempt < 100 && mixed_watermark_storage->stats().swap_out_count == 0;
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  require(mixed_watermark_storage->stats().swap_out_count == 1,
          "mixed resident memory did not trigger background eviction");
  require(mixed_watermark_storage->stats().memory_bytes == 4096,
          "background eviction removed swap-disabled memory");

  se::StorageOptions gap_watermark_options;
  gap_watermark_options.memory_limit_bytes = 12288;
  gap_watermark_options.spill_directory =
      std::filesystem::temp_directory_path();
  gap_watermark_options.max_spill_bytes = 1 << 20;
  gap_watermark_options.watermark = se::WatermarkConfig{0.75, 0.5};
  auto gap_watermark_storage =
      se::create_page_storage(std::move(gap_watermark_options));
  auto gap_watermark_registration =
      gap_watermark_storage->set_page_eviction_proxy(
          std::make_shared<EvictionGuard>());
  (void)gap_watermark_registration;
  auto gap_page_a = gap_watermark_storage->allocate(
      4096, 64, [] { return uint64_t{0}; }, se::PageEvictionMode::kEnabled);
  auto gap_page_b = gap_watermark_storage->allocate(
      4096, 64, [] { return uint64_t{0}; }, se::PageEvictionMode::kEnabled);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const auto gap_passive_stats = gap_watermark_storage->stats();
  require(gap_passive_stats.memory_bytes == 8192,
          "low/high gap setup did not keep two resident pages");
  require(gap_passive_stats.evictable_memory_bytes == 8192,
          "low/high gap setup did not expose evictable resident bytes");
  require(gap_passive_stats.swap_out_count == 0,
          "low/high gap unexpectedly triggered passive background eviction");
  gap_watermark_storage->drain_page_eviction_backlog(std::chrono::seconds(1));
  const auto gap_drained_stats = gap_watermark_storage->stats();
  require(gap_drained_stats.memory_bytes <= 6144,
          "explicit eviction drain did not reach low watermark");
  require(gap_drained_stats.swap_out_count == 1,
          "explicit eviction drain did not swap exactly one gap page");

  se::StorageOptions replacement_options;
  replacement_options.memory_limit_bytes = 8192;
  replacement_options.spill_directory = std::filesystem::temp_directory_path();
  replacement_options.max_spill_bytes = 1 << 20;
  replacement_options.watermark = se::WatermarkConfig{0.5, 0.25};
  auto replacement_storage =
      se::create_page_storage(std::move(replacement_options));
  auto first_proxy = std::make_shared<BlockingEvictionProxy>();
  auto first_registration =
      replacement_storage->set_page_eviction_proxy(first_proxy);
  auto replacement_page = replacement_storage->allocate(
      4096, 64, [] { return uint64_t{0}; }, se::PageEvictionMode::kEnabled);
  first_proxy->wait_for_evaluation();
  std::atomic<bool> first_proxy_clear_finished{false};
  std::thread first_proxy_clear([&] {
    replacement_storage->clear_page_eviction_proxy(first_registration);
    first_proxy_clear_finished.store(true, std::memory_order_release);
  });
  for (size_t attempt = 0; attempt < 100 && !first_proxy_clear_finished.load(
                                                std::memory_order_acquire);
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  require(!first_proxy_clear_finished.load(std::memory_order_acquire),
          "page eviction proxy clear did not wait for active evaluator");
  first_proxy->release_evaluation();
  first_proxy_clear.join();
  first_proxy->wait_for_evaluation_completion();
  require(first_proxy_clear_finished.load(std::memory_order_acquire),
          "page eviction proxy clear did not complete after evaluator exit");
  require(replacement_storage->stats().swap_out_count == 0,
          "stale proxy generation swapped a page after proxy clear");
  auto second_registration = replacement_storage->set_page_eviction_proxy(
      std::make_shared<EvictionGuard>());

  se::StorageOptions isolation_options;
  isolation_options.memory_limit_bytes = 8192;
  isolation_options.spill_directory = std::filesystem::temp_directory_path();
  isolation_options.max_spill_bytes = 1 << 20;
  isolation_options.watermark = se::WatermarkConfig{0.5, 0.25};
  auto isolation_storage =
      se::create_page_storage(std::move(isolation_options));
  auto blocking_proxy = std::make_shared<BlockingEvictionProxy>();
  auto blocking_registration =
      isolation_storage->set_page_eviction_proxy(blocking_proxy);
  auto tiered_page = isolation_storage->allocate(
      4096, 64, [] { return uint64_t{0}; }, se::PageEvictionMode::kEnabled);
  blocking_proxy->wait_for_evaluation();
  se::StoragePage memory_page;
  std::atomic<bool> memory_allocation_completed{false};
  std::thread memory_allocation([&] {
    memory_page = isolation_storage->allocate(64, 64, {},
                                              se::PageEvictionMode::kDisabled);
    memory_allocation_completed.store(true, std::memory_order_release);
  });
  for (size_t attempt = 0; attempt < 100 && !memory_allocation_completed.load(
                                                std::memory_order_acquire);
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const bool completed_before_swap_release =
      memory_allocation_completed.load(std::memory_order_acquire);
  blocking_proxy->release_evaluation();
  memory_allocation.join();
  require(completed_before_swap_release,
          "memory-only allocation was blocked by background swap");
  require(isolation_storage->stats().spill_count == 0,
          "memory-only allocation entered direct spill storage");

  se::StorageOptions sampled_options;
  sampled_options.memory_limit_bytes = 8 << 20;
  sampled_options.spill_directory = std::filesystem::temp_directory_path();
  sampled_options.max_spill_bytes = 1 << 30;
  sampled_options.watermark = se::WatermarkConfig{0.8, 0.75};
  auto sampled_storage = se::create_page_storage(std::move(sampled_options));
  auto sampled_registration = sampled_storage->set_page_eviction_proxy(
      std::make_shared<EvictionGuard>());
  std::vector<se::StoragePage> sampled_pages;
  sampled_pages.reserve(1800);
  for (size_t index = 0; index < 1800; ++index) {
    sampled_pages.push_back(sampled_storage->allocate(
        4096, 64, [] { return uint64_t{0}; }, se::PageEvictionMode::kEnabled));
  }
  for (size_t attempt = 0;
       attempt < 100 && sampled_storage->stats().swap_out_count == 0;
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  const auto sampled_stats = sampled_storage->stats();
  require(sampled_stats.swap_out_count > 0,
          "large candidate set did not trigger background swap-out");
  require(sampled_stats.eviction_selection_count > 0,
          "large candidate set did not record selection passes");
  require(sampled_stats.eviction_evaluated_pages > 0,
          "large candidate set did not evaluate eviction candidates");
  if (sampled_stats.eviction_evaluated_pages >
      sampled_stats.eviction_selection_count *
          kExpectedSampledEvaluationLimit) {
    throw std::runtime_error(
        "large candidate set exceeded the sampled evaluation budget: " +
        std::to_string(sampled_stats.eviction_evaluated_pages) + " / " +
        std::to_string(sampled_stats.eviction_selection_count));
  }

  se::StorageOptions copy_options;
  copy_options.mode = se::StorageMode::kMemoryOnly;
  copy_options.memory_limit_bytes = 4096;
  auto original_storage = se::create_page_storage(std::move(copy_options));
  auto copied_storage = original_storage;
  auto moved_storage = std::move(copied_storage);
  auto copied_page = moved_storage->allocate(64, 64);
  require(moved_storage->stats().memory_bytes == 64,
          "copied and moved PageStorage lost shared ownership");

  isolation_storage->clear_page_eviction_proxy(blocking_registration);
  replacement_storage->clear_page_eviction_proxy(second_registration);
  mixed_watermark_storage->clear_page_eviction_proxy(
      mixed_watermark_registration);
  default_eviction_storage->clear_page_eviction_proxy(
      default_eviction_registration);
  watermark_storage->clear_page_eviction_proxy(watermark_registration);
  sampled_storage->clear_page_eviction_proxy(sampled_registration);
}
