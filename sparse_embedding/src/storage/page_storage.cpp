#include "sparse_embedding/src/storage/page_storage.hpp"

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>

#include "sparse_embedding/src/storage/page_eviction_selection.hpp"
#include "sparse_embedding/src/storage/page_registry.hpp"
#include "sparse_embedding/src/storage/spill_allocator.hpp"
#include "sparse_embedding/src/storage/swap_scheduler.hpp"

namespace se {
namespace {

constexpr size_t kAutoMemoryNumerator = 4;
constexpr size_t kAutoMemoryDenominator = 5;
constexpr size_t kCacheLineSize = 64;
constexpr size_t kMaxEvictionNoProgressAttempts = 8;

size_t checked_round_up(size_t value, size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("storage alignment must be a power of two");
  }
  if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    throw std::overflow_error("storage allocation size overflow");
  }
  return (value + alignment - 1) & ~(alignment - 1);
}

size_t watermark_bytes(size_t limit, double fraction) noexcept {
  if (limit == 0 || fraction <= 0.0)
    return 0;
  if (fraction >= 1.0)
    return limit;
  const size_t scaled = static_cast<size_t>(static_cast<long double>(limit) *
                                            static_cast<long double>(fraction));
  return std::max(size_t{1}, scaled);
}

std::optional<size_t> read_keyed_bytes(const std::filesystem::path &path,
                                       std::string_view key) {
  std::ifstream input(path);
  std::string name;
  size_t value;
  std::string unit;
  while (input >> name >> value >> unit) {
    if (name == key) {
      return value * 1024;
    }
  }
  return std::nullopt;
}

std::optional<size_t> read_size(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::string value;
  input >> value;
  if (!input || value == "max") {
    return std::nullopt;
  }
  return static_cast<size_t>(std::stoull(value));
}

std::filesystem::path current_cgroup_path() {
  std::ifstream input("/proc/self/cgroup");
  std::string line;
  while (std::getline(input, line)) {
    auto separator = line.find("::");
    if (separator != std::string::npos && line.substr(0, separator) == "0") {
      return std::filesystem::path(std::string_view("/sys/fs/cgroup")) /
             line.substr(separator + 2).substr(1);
    }
  }
  return {};
}

std::filesystem::path current_cgroup_v1_memory_path() {
  std::ifstream input("/proc/self/cgroup");
  std::string line;
  while (std::getline(input, line)) {
    auto first = line.find(':');
    auto second = line.find(':', first + 1);
    if (first == std::string::npos || second == std::string::npos)
      continue;
    auto controllers = line.substr(first + 1, second - first - 1);
    if (controllers.find("memory") != std::string::npos) {
      auto relative = line.substr(second + 1);
      if (!relative.empty() && relative.front() == '/')
        relative.erase(relative.begin());
      return std::filesystem::path(std::string_view("/sys/fs/cgroup/memory")) /
             relative;
    }
  }
  return {};
}

size_t detect_memory_limit() {
  auto available =
      read_keyed_bytes("/proc/meminfo", "MemAvailable:").value_or(size_t{0});
  auto cgroup = current_cgroup_path();
  if (!cgroup.empty()) {
    auto current = read_size(cgroup / "memory.current");
    auto high = read_size(cgroup / "memory.high");
    auto maximum = read_size(cgroup / "memory.max");
    if (current) {
      for (auto limit : {high, maximum}) {
        if (limit) {
          size_t headroom = *limit > *current ? *limit - *current : 0;
          available = available == 0 ? headroom : std::min(available, headroom);
        }
      }
    }
  } else {
    auto cgroup_v1 = current_cgroup_v1_memory_path();
    if (!cgroup_v1.empty()) {
      auto current = read_size(cgroup_v1 / "memory.usage_in_bytes");
      auto hard_limit = read_size(cgroup_v1 / "memory.limit_in_bytes");
      auto soft_limit = read_size(cgroup_v1 / "memory.soft_limit_in_bytes");
      if (current) {
        for (auto limit : {soft_limit, hard_limit}) {
          if (limit) {
            size_t headroom = *limit > *current ? *limit - *current : 0;
            available =
                available == 0 ? headroom : std::min(available, headroom);
          }
        }
      }
    }
  }
  rlimit address_space{};
  if (getrlimit(RLIMIT_AS, &address_space) == 0 &&
      address_space.rlim_cur != RLIM_INFINITY) {
    auto vm_size =
        read_keyed_bytes("/proc/self/status", "VmSize:").value_or(size_t{0});
    size_t headroom =
        address_space.rlim_cur > vm_size ? address_space.rlim_cur - vm_size : 0;
    available = available == 0 ? headroom : std::min(available, headroom);
  }
  rlimit data_limit{};
  if (getrlimit(RLIMIT_DATA, &data_limit) == 0 &&
      data_limit.rlim_cur != RLIM_INFINITY) {
    auto data_size =
        read_keyed_bytes("/proc/self/status", "VmData:").value_or(size_t{0});
    size_t headroom =
        data_limit.rlim_cur > data_size ? data_limit.rlim_cur - data_size : 0;
    available = available == 0 ? headroom : std::min(available, headroom);
  }
  std::ifstream overcommit_input("/proc/sys/vm/overcommit_memory");
  int overcommit_mode = 0;
  overcommit_input >> overcommit_mode;
  if (overcommit_mode == 2) {
    auto commit_limit = read_keyed_bytes("/proc/meminfo", "CommitLimit:");
    auto committed = read_keyed_bytes("/proc/meminfo", "Committed_AS:");
    if (commit_limit && committed) {
      size_t headroom =
          *commit_limit > *committed ? *commit_limit - *committed : 0;
      available = available == 0 ? headroom : std::min(available, headroom);
    }
  }
  return available / kAutoMemoryDenominator * kAutoMemoryNumerator;
}

} // namespace

using PageBacking = storage_detail::PageBacking;
using PageRecord = storage_detail::PageRecord;

class TieredPageStorage final
    : public std::enable_shared_from_this<TieredPageStorage> {
public:
  explicit TieredPageStorage(StorageOptions options);
  ~TieredPageStorage();

  StoragePage
  allocate(size_t bytes, size_t alignment = 64, PageEvaluator evaluator = {},
           PageEvictionMode eviction_mode = PageEvictionMode::kDisabled,
           PagePlacementMode placement_mode = PagePlacementMode::kDefault);
  StorageStats stats() const noexcept;
  std::string spill_directory() const;
  bool consume_spill_warning() noexcept;
  void drain_page_eviction_backlog(std::chrono::nanoseconds timeout);
  bool supports_background_page_eviction() const noexcept;
  PageEvictionRegistration set_page_eviction_proxy(PageEvictionProxy proxy);
  void clear_page_eviction_proxy(PageEvictionRegistration registration);
  void release(std::byte *data, size_t allocated_size,
               bool heap_backed) noexcept;
  StoragePage make_page(std::byte *data, size_t size, size_t allocated_size,
                        size_t page_idx, bool heap_backed);

private:
  StoragePage allocate_memory(size_t bytes, size_t allocated_size,
                              size_t alignment, PageBacking backing,
                              PageEvaluator evaluator);
  StoragePage allocate_spill(size_t bytes, size_t allocated_size);
  void emit_spill_warning(size_t requested_bytes) noexcept;
  void unregister_page_evaluator_unlocked(size_t page_idx) noexcept;
  bool try_reserve_memory(size_t bytes) noexcept;
  PageBacking
  select_memory_backing(PageEvictionMode eviction_mode) const noexcept;

  static void run_swap_cycle(void *context);
  void swap_cycle();
  enum class EvictionDrainStatus { kDrained, kProxyChanged, kTimedOut };
  EvictionDrainStatus
  drain_eviction_backlog_until(size_t low_threshold,
                               std::chrono::steady_clock::time_point deadline);

  struct ClaimedSwap {
    std::byte *addr;
    size_t allocated_size;
    size_t offset;
    size_t page_idx;
    uint64_t proxy_generation;
  };

  ClaimedSwap claim_swap_out_page(PageRecord &record,
                                  uint64_t proxy_generation);
  void write_claimed_page(const ClaimedSwap &claim);
  bool commit_claimed_page(const ClaimedSwap &claim);
  void rollback_claimed_page(const ClaimedSwap &claim) noexcept;

  // Builds one ordered eviction batch from a mutex-protected page snapshot.
  using EvictionPageSnapshot = storage_detail::EvictionPageSnapshot;
  using EvictionCandidate = storage_detail::EvictionCandidate;

  std::vector<EvictionCandidate>
  select_coldest_pages(const std::vector<EvictionPageSnapshot> &pages,
                       size_t bytes_to_evict, PageEvictionProxy &proxy);

  StorageOptions options_;
  size_t page_size_;
  size_t memory_limit_bytes_;
  std::shared_ptr<RetirementPressure> retirement_pressure_;
  storage_detail::SpillAllocator spill_allocator_;
  mutable std::mutex mutex_;
  std::condition_variable page_transition_cv_;
  storage_detail::SwapScheduler swap_scheduler_;
  std::atomic<size_t> memory_bytes_{0};
  std::atomic<size_t> evictable_memory_bytes_{0};
  std::atomic<size_t> spill_bytes_{0};
  std::atomic<uint64_t> spill_count_{0};
  std::atomic<bool> spill_warning_pending_{false};
  std::atomic<bool> spill_warning_emitted_{false};
  std::atomic<uint64_t> swap_out_count_{0};
  std::atomic<uint64_t> eviction_selection_count_{0};
  std::atomic<uint64_t> eviction_evaluated_pages_{0};
  std::atomic<uint64_t> eviction_selection_ns_{0};
  std::atomic<uint64_t> eviction_candidate_invalidated_count_{0};
  storage_detail::PageRegistry page_registry_;

  PageEvictionProxy proxy_;
  std::unordered_set<uint64_t> proxy_clients_;
  uint64_t next_proxy_registration_ = 1;
  uint64_t proxy_generation_ = 0;
  std::timed_mutex eviction_drain_mutex_;
};

StoragePage::StoragePage(PageStorage owner, std::byte *data, size_t size,
                         size_t allocated_size, size_t page_index,
                         bool heap_backed) noexcept
    : owner_(std::move(owner)), data_(data), size_(size),
      allocated_size_(allocated_size), page_index_(page_index),
      heap_backed_(heap_backed) {}

StoragePage::StoragePage(StoragePage &&other) noexcept {
  *this = std::move(other);
}

StoragePage &StoragePage::operator=(StoragePage &&other) noexcept {
  if (this != &other) {
    reset();
    owner_ = std::move(other.owner_);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    allocated_size_ = std::exchange(other.allocated_size_, 0);
    page_index_ = std::exchange(other.page_index_, 0);
    heap_backed_ = std::exchange(other.heap_backed_, false);
  }
  return *this;
}

StoragePage::~StoragePage() { reset(); }

void StoragePage::reset() noexcept {
  if (owner_ && data_) {
    owner_->release(data_, allocated_size_, heap_backed_);
  }
  owner_.reset();
  data_ = nullptr;
  size_ = 0;
  allocated_size_ = 0;
  page_index_ = 0;
  heap_backed_ = false;
}

PageStorage create_page_storage(StorageOptions options) {
  auto storage = std::make_shared<TieredPageStorage>(std::move(options));
  return PageStorage{std::move(storage)};
}

StoragePage TieredPageStorage::make_page(std::byte *data, size_t size,
                                         size_t allocated_size,
                                         size_t page_index, bool heap_backed) {
  PageStorage owner{shared_from_this()};
  return StoragePage(std::move(owner), data, size, allocated_size, page_index,
                     heap_backed);
}

TieredPageStorage::TieredPageStorage(StorageOptions options)
    : options_(std::move(options)),
      page_size_(static_cast<size_t>(sysconf(_SC_PAGESIZE))),
      memory_limit_bytes_(
          options_.memory_limit_bytes.value_or(detect_memory_limit())),
      retirement_pressure_(std::make_shared<RetirementPressure>(
          memory_limit_bytes_, options_.retirement_pressure)),
      spill_allocator_(options_.spill_directory, options_.max_spill_bytes) {
  if (options_.mode == StorageMode::kAuto && !options_.watermark)
    options_.watermark = WatermarkConfig{};
  if (page_size_ == 0 || (page_size_ & (page_size_ - 1)) != 0) {
    throw std::runtime_error("invalid operating system page size");
  }
  if (options_.watermark && options_.mode == StorageMode::kMemoryOnly) {
    throw std::invalid_argument(
        "watermark swap is incompatible with MemoryOnly storage mode");
  }
  if (options_.watermark) {
    const auto &wm = *options_.watermark;
    if (!std::isfinite(wm.high) || !std::isfinite(wm.low) || wm.high <= 0.0 ||
        wm.high > 1.0 || wm.low < 0.0 || wm.low >= wm.high) {
      throw std::invalid_argument(
          "watermark.high must be in (0,1] and watermark.low must be in "
          "[0, watermark.high)");
    }
  }
  if (options_.mode == StorageMode::kAuto && options_.watermark &&
      options_.retirement_pressure.circuit_fraction >=
          options_.watermark->high) {
    throw std::invalid_argument(
        "retirement pressure circuit fraction must be below the Auto spill "
        "high watermark");
  }
}

TieredPageStorage::~TieredPageStorage() { swap_scheduler_.stop_and_join(); }

PageEvictionRegistration
TieredPageStorage::set_page_eviction_proxy(PageEvictionProxy proxy) {
  if (!proxy)
    throw std::invalid_argument("page eviction proxy must not be empty");
  if (!proxy->score_enabled())
    throw std::invalid_argument("page eviction proxy must enable scoring");

  if (supports_background_page_eviction())
    swap_scheduler_.ensure_started(this, &TieredPageStorage::run_swap_cycle);
  std::unique_lock lock(mutex_);

  if (next_proxy_registration_ == 0) {
    throw std::overflow_error("page eviction registration id exhausted");
  }
  const PageEvictionRegistration registration{next_proxy_registration_++};
  const bool install_proxy = !proxy_ || !proxy_->score_enabled();
  if (install_proxy) {
    if (proxy_generation_ == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("page eviction proxy generation exhausted");
    proxy_ = std::move(proxy);
    ++proxy_generation_;
  }
  proxy_clients_.insert(registration.value);
  if (install_proxy) {
    for (const auto &[page_idx, evaluator] : page_registry_.evaluators())
      proxy_->register_page(page_idx, evaluator);
  }
  const size_t high_threshold =
      options_.watermark
          ? watermark_bytes(memory_limit_bytes_, options_.watermark->high)
          : 0;
  if (supports_background_page_eviction() && high_threshold > 0 &&
      proxy_->score_enabled())
    swap_scheduler_.wake();
  return registration;
}

bool TieredPageStorage::supports_background_page_eviction() const noexcept {
  return options_.mode == StorageMode::kAuto && options_.watermark &&
         watermark_bytes(memory_limit_bytes_, options_.watermark->high) > 0;
}

void TieredPageStorage::drain_page_eviction_backlog(
    std::chrono::nanoseconds timeout) {
  if (timeout <= std::chrono::nanoseconds::zero())
    throw std::invalid_argument("page eviction drain timeout must be positive");
  if (!supports_background_page_eviction())
    throw std::logic_error(
        "page eviction drain requires watermark-backed auto storage");

  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::unique_lock drain_lock(eviction_drain_mutex_, std::defer_lock);
  if (!drain_lock.try_lock_until(deadline))
    throw std::runtime_error(
        "page eviction drain timed out waiting for an active drain");

  const size_t low_threshold =
      watermark_bytes(memory_limit_bytes_, options_.watermark->low);
  while (true) {
    {
      std::unique_lock lock(mutex_);
      swap_scheduler_.rethrow_if_failed();
      if (!proxy_ || !proxy_->score_enabled())
        throw std::logic_error(
            "page eviction drain requires an active score proxy");
    }

    const EvictionDrainStatus status =
        drain_eviction_backlog_until(low_threshold, deadline);
    std::unique_lock lock(mutex_);
    swap_scheduler_.rethrow_if_failed();
    if (status == EvictionDrainStatus::kTimedOut)
      throw std::runtime_error("page eviction drain timed out");
    if (status == EvictionDrainStatus::kProxyChanged) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("page eviction drain timed out");
      continue;
    }
    if (memory_bytes_.load(std::memory_order_relaxed) > low_threshold)
      throw std::runtime_error(
          "page eviction drain stopped with no evictable resident pages");
    return;
  }
}

void TieredPageStorage::clear_page_eviction_proxy(
    PageEvictionRegistration registration) {
  if (registration.value == 0)
    throw std::invalid_argument("page eviction registration must be valid");

  std::unique_lock lock(mutex_);
  if (proxy_clients_.erase(registration.value) == 0)
    throw std::logic_error("page eviction registration was not active");
  if (!proxy_clients_.empty())
    return;
  if (proxy_generation_ == std::numeric_limits<uint64_t>::max())
    throw std::overflow_error("page eviction proxy generation exhausted");
  PageEvictionProxy retired_proxy = proxy_;
  std::vector<size_t> retired_page_indices;
  retired_page_indices.reserve(page_registry_.evaluators().size());
  for (const auto &[page_index, _] : page_registry_.evaluators())
    retired_page_indices.push_back(page_index);
  proxy_.reset();
  ++proxy_generation_;
  lock.unlock();
  for (size_t page_index : retired_page_indices)
    retired_proxy->unregister_page(page_index);
  swap_scheduler_.request_stop();
}

void TieredPageStorage::unregister_page_evaluator_unlocked(
    size_t page_idx) noexcept {
  page_registry_.unregister_evaluator(page_idx);
  if (proxy_)
    proxy_->unregister_page(page_idx);
}

// =============================================================================
// Allocation path
// =============================================================================

PageBacking TieredPageStorage::select_memory_backing(
    PageEvictionMode eviction_mode) const noexcept {
  return options_.mode == StorageMode::kMemoryOnly ||
                 eviction_mode == PageEvictionMode::kDisabled || !proxy_ ||
                 !proxy_->score_enabled()
             ? PageBacking::kHeap
             : PageBacking::kAnonymousMapping;
}

bool TieredPageStorage::try_reserve_memory(size_t bytes) noexcept {
  size_t current = memory_bytes_.load(std::memory_order_relaxed);
  while (current <= memory_limit_bytes_ &&
         bytes <= memory_limit_bytes_ - current) {
    if (memory_bytes_.compare_exchange_weak(current, current + bytes,
                                            std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

StoragePage TieredPageStorage::allocate(size_t bytes, size_t alignment,
                                        PageEvaluator evaluator,
                                        PageEvictionMode eviction_mode,
                                        PagePlacementMode placement_mode) {
  if (bytes == 0) {
    throw std::invalid_argument("storage allocation must not be empty");
  }
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("storage alignment must be a power of two");
  }
  if (alignment > page_size_) {
    throw std::invalid_argument("storage alignment exceeds system page size");
  }
  if (placement_mode == PagePlacementMode::kPageAlignedMapping) {
    const size_t mapped_size = checked_round_up(bytes, page_size_);
    std::unique_lock lock(mutex_);
    if (eviction_mode == PageEvictionMode::kEnabled)
      swap_scheduler_.rethrow_if_failed();
    if (!try_reserve_memory(mapped_size))
      throw std::bad_alloc();
    return allocate_memory(bytes, mapped_size, page_size_,
                           PageBacking::kAnonymousMapping,
                           std::move(evaluator));
  }
  if (eviction_mode == PageEvictionMode::kDisabled ||
      options_.mode == StorageMode::kMemoryOnly) {
    const size_t fast_alignment = std::max(alignment, kCacheLineSize);
    const size_t fast_size = checked_round_up(bytes, fast_alignment);
    if (try_reserve_memory(fast_size)) {
      return allocate_memory(bytes, fast_size, fast_alignment,
                             PageBacking::kHeap, {});
    }
  }
  std::unique_lock lock(mutex_);
  if (eviction_mode == PageEvictionMode::kEnabled)
    swap_scheduler_.rethrow_if_failed();
  const PageBacking memory_backing = select_memory_backing(eviction_mode);
  const size_t memory_alignment = memory_backing == PageBacking::kHeap
                                      ? std::max(alignment, kCacheLineSize)
                                      : page_size_;
  const size_t memory_size = checked_round_up(bytes, memory_alignment);
  if (try_reserve_memory(memory_size)) {
    return allocate_memory(bytes, memory_size, memory_alignment, memory_backing,
                           std::move(evaluator));
  }
  if (options_.mode == StorageMode::kMemoryOnly) {
    throw std::bad_alloc();
  }
  return allocate_spill(bytes, checked_round_up(bytes, page_size_));
}

StoragePage TieredPageStorage::allocate_memory(size_t bytes,
                                               size_t allocated_size,
                                               size_t alignment,
                                               PageBacking backing,
                                               PageEvaluator evaluator) {
  const bool register_evaluator =
      evaluator && backing == PageBacking::kAnonymousMapping;
  void *ptr = nullptr;
  if (backing == PageBacking::kHeap) {
    ptr = std::aligned_alloc(alignment, allocated_size);
    if (!ptr) {
      memory_bytes_.fetch_sub(allocated_size, std::memory_order_relaxed);
      throw std::bad_alloc();
    }
  } else {
    ptr = mmap(nullptr, allocated_size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
      memory_bytes_.fetch_sub(allocated_size, std::memory_order_relaxed);
      throw std::bad_alloc();
    }
  }
  auto *data = static_cast<std::byte *>(ptr);
  if (backing == PageBacking::kAnonymousMapping)
    evictable_memory_bytes_.fetch_add(allocated_size,
                                      std::memory_order_relaxed);

  if (backing == PageBacking::kHeap) {
    return make_page(data, bytes, allocated_size,
                     page_registry_.allocate_page_index(), true);
  }

  // Register this page so the swap thread can find and potentially swap it out.
  // offset is pre-assigned via acquire_extent so that swap-out can write to
  // disk without re-allocating.  We defer the actual file extent allocation
  // to the first actual swap-out of this page to avoid touching the disk on
  // every in-memory allocation.  We use SIZE_MAX as a sentinel for "not yet
  // assigned".
  const size_t page_idx = page_registry_.allocate_page_index();
  bool registry_inserted = false;
  bool evictable_page_registered = false;
  bool evaluator_registered = false;
  try {
    page_registry_.insert({data, allocated_size, SIZE_MAX, page_idx, backing,
                           register_evaluator});
    registry_inserted = true;
    page_registry_.register_evictable({data, page_idx, allocated_size});
    evictable_page_registered = true;
    if (register_evaluator) {
      page_registry_.register_evaluator(page_idx, evaluator);
      evaluator_registered = true;
      if (proxy_)
        proxy_->register_page(page_idx, std::move(evaluator));
    }
  } catch (...) {
    if (evaluator_registered)
      unregister_page_evaluator_unlocked(page_idx);
    if (evictable_page_registered)
      page_registry_.unregister_evictable(page_idx);
    if (registry_inserted)
      page_registry_.erase(data);
    if (backing == PageBacking::kHeap) {
      std::free(data);
    } else {
      munmap(data, allocated_size);
      evictable_memory_bytes_.fetch_sub(allocated_size,
                                        std::memory_order_relaxed);
    }
    memory_bytes_.fetch_sub(allocated_size, std::memory_order_relaxed);
    throw;
  }

  if (supports_background_page_eviction() &&
      evictable_memory_bytes_.load(std::memory_order_relaxed) > 0 &&
      memory_bytes_.load(std::memory_order_relaxed) >=
          watermark_bytes(memory_limit_bytes_, options_.watermark->high)) {
    swap_scheduler_.wake();
  }

  return make_page(data, bytes, allocated_size, page_idx, false);
}

StoragePage TieredPageStorage::allocate_spill(size_t bytes,
                                              size_t allocated_size) {
  const size_t offset = spill_allocator_.acquire_extent(allocated_size);
  const int spill_fd = spill_allocator_.file_descriptor();
  void *mapping = mmap(nullptr, allocated_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, spill_fd, static_cast<off_t>(offset));
  if (mapping == MAP_FAILED) {
    spill_allocator_.release_extent(offset, allocated_size);
    throw std::runtime_error("failed to map spill page: " +
                             std::string(std::strerror(errno)));
  }
  spill_bytes_.fetch_add(allocated_size, std::memory_order_relaxed);
  spill_count_.fetch_add(1, std::memory_order_relaxed);
  auto *addr = static_cast<std::byte *>(mapping);
  const size_t page_idx = page_registry_.allocate_page_index();
  try {
    page_registry_.insert({addr, allocated_size, offset, page_idx,
                           PageBacking::kSpillMapping, false});
  } catch (...) {
    munmap(addr, allocated_size);
    spill_bytes_.fetch_sub(allocated_size, std::memory_order_relaxed);
    --spill_count_;
    spill_allocator_.release_extent(offset, allocated_size);
    throw;
  }

  emit_spill_warning(allocated_size);

  return make_page(addr, bytes, allocated_size, page_idx, false);
}

// =============================================================================
// Release path — removes PageRecord, then frees the backing storage
// =============================================================================

void TieredPageStorage::release(std::byte *data, size_t allocated_size,
                                bool heap_backed) noexcept {
  if (heap_backed) {
    std::free(data);
    memory_bytes_.fetch_sub(allocated_size, std::memory_order_relaxed);
    return;
  }
  std::unique_lock lock(mutex_);

  PageRecord *registered = page_registry_.find(data);
  while (registered != nullptr &&
         registered->backing == PageBacking::kSwapClaimed) {
    page_transition_cv_.wait(lock);
    registered = page_registry_.find(data);
  }
  if (registered != nullptr) {
    const PageRecord record = *registered;
    if (record.backing == PageBacking::kAnonymousMapping)
      page_registry_.unregister_evictable(record.page_idx);
    page_registry_.erase(data);
    if (record.evaluator_registered)
      unregister_page_evaluator_unlocked(record.page_idx);

    if (record.backing == PageBacking::kSpillMapping) {
      munmap(data, record.allocated_size);
      spill_bytes_.fetch_sub(record.allocated_size, std::memory_order_relaxed);
      spill_allocator_.release_extent(record.offset, record.allocated_size);
    } else {
      munmap(data, record.allocated_size);
      evictable_memory_bytes_.fetch_sub(record.allocated_size,
                                        std::memory_order_relaxed);
      memory_bytes_.fetch_sub(record.allocated_size, std::memory_order_relaxed);
      if (record.offset != SIZE_MAX) {
        spill_allocator_.release_extent(record.offset, record.allocated_size);
      }
    }
  } else {
    std::terminate();
  }
}

// =============================================================================
// Swap-out: msync (for disk-backed pages) or write memory to disk, then
// MAP_FIXED remap the same virtual address to the file mapping.
// Caller must hold mutex_.
// =============================================================================

TieredPageStorage::ClaimedSwap
TieredPageStorage::claim_swap_out_page(PageRecord &record,
                                       uint64_t proxy_generation) {
  if (record.backing != PageBacking::kAnonymousMapping)
    throw std::logic_error("heap-backed pages cannot be swapped");

  if (record.offset == SIZE_MAX) {
    record.offset = spill_allocator_.acquire_extent(record.allocated_size);
  }
  record.backing = PageBacking::kSwapClaimed;
  return {record.addr, record.allocated_size, record.offset, record.page_idx,
          proxy_generation};
}

void TieredPageStorage::write_claimed_page(const ClaimedSwap &claim) {
  const ssize_t written =
      pwrite(spill_allocator_.file_descriptor(), claim.addr,
             claim.allocated_size, static_cast<off_t>(claim.offset));
  if (written < 0 || static_cast<size_t>(written) != claim.allocated_size) {
    throw std::runtime_error("failed to write page during swap: " +
                             std::string(std::strerror(errno)));
  }
}

bool TieredPageStorage::commit_claimed_page(const ClaimedSwap &claim) {
  const uint64_t proxy_generation = proxy_generation_;
  if (proxy_generation != claim.proxy_generation)
    return false;
  PageRecord *record = page_registry_.find(claim.addr);
  if (record == nullptr)
    throw std::logic_error("claimed swap page is no longer registered");
  if (record->addr != claim.addr ||
      record->allocated_size != claim.allocated_size ||
      record->offset != claim.offset || record->page_idx != claim.page_idx ||
      record->backing != PageBacking::kSwapClaimed) {
    throw std::logic_error("claimed swap page identity changed before commit");
  }
  void *result =
      mmap(claim.addr, claim.allocated_size, PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_FIXED, spill_allocator_.file_descriptor(),
           static_cast<off_t>(claim.offset));
  if (result == MAP_FAILED) {
    throw std::runtime_error("failed to remap page during swap: " +
                             std::string(std::strerror(errno)));
  }

  memory_bytes_.fetch_sub(claim.allocated_size, std::memory_order_relaxed);
  evictable_memory_bytes_.fetch_sub(claim.allocated_size,
                                    std::memory_order_relaxed);
  spill_bytes_.fetch_add(claim.allocated_size, std::memory_order_relaxed);
  swap_out_count_.fetch_add(1, std::memory_order_relaxed);
  page_registry_.unregister_evictable(claim.page_idx);
  record->backing = PageBacking::kSpillMapping;
  if (record->evaluator_registered) {
    unregister_page_evaluator_unlocked(claim.page_idx);
    record->evaluator_registered = false;
  }
  emit_spill_warning(claim.allocated_size);
  page_transition_cv_.notify_all();
  return true;
}

void TieredPageStorage::rollback_claimed_page(
    const ClaimedSwap &claim) noexcept {
  PageRecord *record = page_registry_.find(claim.addr);
  if (record != nullptr) {
    if (record->addr == claim.addr && record->page_idx == claim.page_idx &&
        record->backing == PageBacking::kSwapClaimed) {
      record->backing = PageBacking::kAnonymousMapping;
    }
  }
  page_transition_cv_.notify_all();
}

// =============================================================================
// Coldest-page selection: score-guided only.
// =============================================================================

std::vector<TieredPageStorage::EvictionCandidate>
TieredPageStorage::select_coldest_pages(
    const std::vector<EvictionPageSnapshot> &pages, size_t bytes_to_evict,
    PageEvictionProxy &proxy) {
  const auto selection_started = std::chrono::steady_clock::now();
  storage_detail::EvictionSelectionResult selection =
      storage_detail::select_coldest_pages(
          std::span<const EvictionPageSnapshot>(pages.data(), pages.size()),
          bytes_to_evict, page_size_, proxy);
  eviction_selection_count_.fetch_add(1, std::memory_order_relaxed);
  eviction_evaluated_pages_.fetch_add(selection.evaluated_pages,
                                      std::memory_order_relaxed);
  const auto selection_elapsed =
      std::chrono::steady_clock::now() - selection_started;
  eviction_selection_ns_.fetch_add(
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              selection_elapsed)
              .count()),
      std::memory_order_relaxed);
  return std::move(selection.candidates);
}

TieredPageStorage::EvictionDrainStatus
TieredPageStorage::drain_eviction_backlog_until(
    size_t low_threshold, std::chrono::steady_clock::time_point deadline) {
  size_t no_progress_attempts = 0;
  std::unique_lock lock(mutex_);
  while (memory_bytes_.load(std::memory_order_relaxed) > low_threshold &&
         evictable_memory_bytes_.load(std::memory_order_relaxed) > 0) {
    if (std::chrono::steady_clock::now() >= deadline)
      return EvictionDrainStatus::kTimedOut;
    if (swap_scheduler_.stop_requested())
      return EvictionDrainStatus::kProxyChanged;
    auto proxy = proxy_;
    if (!proxy || !proxy->score_enabled())
      throw std::logic_error("page eviction drain requires a score evaluator");
    const uint64_t proxy_generation = proxy_generation_;
    const size_t current_memory = memory_bytes_.load(std::memory_order_relaxed);
    const size_t current_evictable =
        evictable_memory_bytes_.load(std::memory_order_relaxed);
    const size_t bytes_to_evict =
        std::min(current_evictable, current_memory - low_threshold);
    std::vector<EvictionPageSnapshot> page_snapshot =
        page_registry_.snapshot_evictable_pages();
    lock.unlock();
    bool proxy_changed = false;
    bool swapped = false;
    bool candidate_invalidated = false;
    std::vector<EvictionCandidate> candidates;
    lock.lock();
    if (proxy_generation_ != proxy_generation) {
      proxy_changed = true;
      lock.unlock();
    } else {
      lock.unlock();
      candidates = select_coldest_pages(page_snapshot, bytes_to_evict, proxy);
    }
    if (!proxy_changed) {
      for (const EvictionCandidate &candidate : candidates) {
        if (std::chrono::steady_clock::now() >= deadline) {
          lock.lock();
          return EvictionDrainStatus::kTimedOut;
        }
        if (swap_scheduler_.stop_requested()) {
          lock.lock();
          return EvictionDrainStatus::kProxyChanged;
        }
        lock.lock();
        if (proxy_generation_ != proxy_generation) {
          proxy_changed = true;
          lock.unlock();
          break;
        }
        if (memory_bytes_.load(std::memory_order_relaxed) <= low_threshold ||
            evictable_memory_bytes_.load(std::memory_order_relaxed) == 0) {
          lock.unlock();
          break;
        }
        PageRecord *record = page_registry_.find(candidate.addr);
        if (record == nullptr || record->addr != candidate.addr ||
            record->page_idx != candidate.page_idx ||
            record->backing != PageBacking::kAnonymousMapping) {
          candidate_invalidated = true;
          eviction_candidate_invalidated_count_.fetch_add(
              1, std::memory_order_relaxed);
          lock.unlock();
          continue;
        }
        const ClaimedSwap claim =
            claim_swap_out_page(*record, proxy_generation);
        lock.unlock();
        try {
          write_claimed_page(claim);
          lock.lock();
          if (!commit_claimed_page(claim)) {
            rollback_claimed_page(claim);
            proxy_changed = true;
            lock.unlock();
            break;
          }
          swapped = true;
          no_progress_attempts = 0;
          lock.unlock();
        } catch (...) {
          if (!lock.owns_lock())
            lock.lock();
          rollback_claimed_page(claim);
          lock.unlock();
          throw;
        }
      }
    }
    lock.lock();
    if (proxy_changed)
      return EvictionDrainStatus::kProxyChanged;
    if (candidate_invalidated && !swapped) {
      if (++no_progress_attempts >= kMaxEvictionNoProgressAttempts)
        throw std::runtime_error(
            "page eviction made no progress after candidate invalidation");
      std::this_thread::yield();
      continue;
    }
    if (!swapped &&
        memory_bytes_.load(std::memory_order_relaxed) > low_threshold &&
        evictable_memory_bytes_.load(std::memory_order_relaxed) > 0)
      throw std::runtime_error("no evictable page has a score evaluator");
  }
  return EvictionDrainStatus::kDrained;
}

void TieredPageStorage::run_swap_cycle(void *context) {
  if (context == nullptr)
    throw std::invalid_argument("swap cycle context must not be null");
  static_cast<TieredPageStorage *>(context)->swap_cycle();
}

void TieredPageStorage::swap_cycle() {
  const WatermarkConfig &watermark = *options_.watermark;
  const size_t high_threshold =
      watermark_bytes(memory_limit_bytes_, watermark.high);
  const size_t low_threshold =
      watermark_bytes(memory_limit_bytes_, watermark.low);
  {
    std::lock_guard lock(mutex_);
    if (!proxy_ || !proxy_->score_enabled() ||
        evictable_memory_bytes_.load(std::memory_order_relaxed) == 0 ||
        memory_bytes_.load(std::memory_order_relaxed) < high_threshold) {
      return;
    }
  }
  std::unique_lock drain_lock(eviction_drain_mutex_);
  const EvictionDrainStatus status = drain_eviction_backlog_until(
      low_threshold, std::chrono::steady_clock::time_point::max());
  if (status == EvictionDrainStatus::kTimedOut)
    throw std::logic_error("unbounded background eviction timed out");
  if (status == EvictionDrainStatus::kProxyChanged &&
      !swap_scheduler_.stop_requested())
    swap_scheduler_.wake();
}

void TieredPageStorage::emit_spill_warning(size_t requested_bytes) noexcept {
  if (!spill_warning_emitted_.exchange(true, std::memory_order_acq_rel)) {
    spill_warning_pending_.store(true, std::memory_order_release);
    const std::string directory = spill_allocator_.directory();
    std::fprintf(stderr,
                 "SparseEmbeddingSpillWarning: requested=%zu "
                 "memory_bytes=%zu memory_limit_bytes=%zu directory=%s\n",
                 requested_bytes, memory_bytes_.load(std::memory_order_relaxed),
                 memory_limit_bytes_, directory.c_str());
  }
}

StorageStats TieredPageStorage::stats() const noexcept {
  const RetirementPressureStats pressure = retirement_pressure_->stats();
  return {memory_bytes_.load(std::memory_order_relaxed),
          evictable_memory_bytes_.load(std::memory_order_relaxed),
          spill_bytes_.load(std::memory_order_relaxed),
          memory_limit_bytes_,
          spill_allocator_.limit_bytes(),
          spill_count_.load(std::memory_order_relaxed),
          spill_warning_pending_.load(std::memory_order_acquire),
          swap_out_count_.load(std::memory_order_relaxed),
          eviction_selection_count_.load(std::memory_order_relaxed),
          eviction_evaluated_pages_.load(std::memory_order_relaxed),
          eviction_selection_ns_.load(std::memory_order_relaxed),
          eviction_candidate_invalidated_count_.load(std::memory_order_relaxed),
          pressure.retired_bytes,
          pressure.state,
          pressure.backpressure_count,
          pressure.sink_hit_count,
          pressure.oldest_unreclaimed_age_ns,
          pressure.epoch_stall_count,
          retirement_pressure_};
}

std::string TieredPageStorage::spill_directory() const {
  return spill_allocator_.directory();
}

bool TieredPageStorage::consume_spill_warning() noexcept {
  return spill_warning_pending_.exchange(false, std::memory_order_acq_rel);
}

} // namespace se
