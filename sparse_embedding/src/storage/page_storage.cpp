#include "sparse_embedding/src/storage/page_storage.hpp"

#include <fcntl.h>
#include <linux/magic.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>

namespace se {
namespace {

constexpr size_t kAutoMemoryNumerator = 4;
constexpr size_t kAutoMemoryDenominator = 5;
constexpr size_t kCacheLineSize = 64;
constexpr size_t kExactEvictionPageThreshold = 1024;
constexpr size_t kSampledEvictionPageCount = 256;
constexpr size_t kMaxEvictionNoProgressAttempts = 8;
constexpr uint64_t kEvictionRandomSeed = 0x9e3779b97f4a7c15ULL;

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

size_t available_bytes(const std::filesystem::path &path) {
  struct statvfs state {};
  if (statvfs(path.c_str(), &state) != 0) {
    throw std::runtime_error("failed to inspect spill directory " +
                             path.string() + ": " + std::strerror(errno));
  }
  return static_cast<size_t>(state.f_bavail) * state.f_frsize;
}

bool is_memory_backed(const std::filesystem::path &path) {
  struct statfs state {};
  if (statfs(path.c_str(), &state) != 0) {
    throw std::runtime_error("failed to inspect spill filesystem " +
                             path.string() + ": " + std::strerror(errno));
  }
  return state.f_type == TMPFS_MAGIC || state.f_type == RAMFS_MAGIC;
}

void validate_spill_directory(const std::filesystem::path &path) {
  std::error_code ec;
  if (!std::filesystem::is_directory(path, ec) || ec ||
      access(path.c_str(), W_OK | X_OK) != 0) {
    throw std::runtime_error("spill directory is not writable: " +
                             path.string());
  }
  if (is_memory_backed(path)) {
    throw std::runtime_error("spill directory is memory-backed: " +
                             path.string());
  }
}

int storage_speed_rank(const std::filesystem::path &path) {
  struct stat state {};
  if (stat(path.c_str(), &state) != 0)
    return 0;
  std::filesystem::path device = std::filesystem::path("/sys/dev/block") /
                                 (std::to_string(major(state.st_dev)) + ":" +
                                  std::to_string(minor(state.st_dev)));
  std::error_code ec;
  device = std::filesystem::canonical(device, ec);
  if (ec)
    return 0;
  while (!device.empty() && device != device.root_path()) {
    std::ifstream rotational(device / "queue/rotational");
    int value = 0;
    if (rotational >> value)
      return value == 0 ? 2 : 1;
    device = device.parent_path();
  }
  return 0;
}

} // namespace

// =============================================================================
// PageRegistry: Tracks every live page allocation so the background swap
// thread can enumerate in-memory pages and execute MAP_FIXED swap-out/swap-in.
//
// In-memory pages use either page-aligned heap storage or anonymous mmap.
// Only anonymous mappings can be remapped to spill storage.
// =============================================================================

enum class PageBacking { kHeap, kAnonymousMapping, kSpillMapping };

struct PageRecord {
  std::byte *addr;       // virtual address handed to upper layer (stable)
  size_t allocated_size; // rounded-up size of the mapping
  size_t offset;         // offset within spill file (valid even when in-memory,
                         // because swap-out writes there)
  size_t page_idx;       // index used by PageEvictionProxy::evaluate_page
  PageBacking backing;
  bool evaluator_registered;
};

class TieredPageStorage final
    : public std::enable_shared_from_this<TieredPageStorage> {
public:
  explicit TieredPageStorage(StorageOptions options);
  ~TieredPageStorage();

  StoragePage
  allocate(size_t bytes, size_t alignment = 64, PageEvaluator evaluator = {},
           PageEvictionMode eviction_mode = PageEvictionMode::kDisabled);
  StorageStats stats() const noexcept;
  std::string spill_directory() const;
  bool consume_spill_warning() noexcept;
  bool supports_background_page_eviction() const noexcept;
  std::shared_ptr<std::shared_mutex> lifecycle_mutex() const;
  PageEvictionRegistration set_page_eviction_proxy(PageEvictionProxy proxy);
  void clear_page_eviction_proxy(PageEvictionRegistration registration);
  void release(std::byte *data, size_t allocated_size,
               bool heap_backed) noexcept;
  StoragePage make_page(std::byte *data, size_t size, size_t allocated_size,
                        size_t page_idx, bool heap_backed);

private:
  static constexpr size_t kInitialExtentCapacity = 64;

  struct Extent {
    size_t offset;
    size_t size;
  };

  StoragePage allocate_memory(size_t bytes, size_t allocated_size,
                              size_t alignment, PageBacking backing,
                              PageEvaluator evaluator);
  StoragePage allocate_spill(size_t bytes, size_t allocated_size);
  void ensure_spill_file();
  void emit_spill_warning(size_t requested_bytes) noexcept;
  std::string select_spill_directory() const;
  size_t acquire_extent(size_t bytes);
  void reserve_extent_capacity();
  void release_extent(size_t offset, size_t bytes) noexcept;
  void unregister_page_evaluator_unlocked(size_t page_idx) noexcept;
  bool try_reserve_memory(size_t bytes) noexcept;
  PageBacking
  select_memory_backing(PageEvictionMode eviction_mode) const noexcept;

  // Background swap-thread lifetime
  void start_swap_thread();
  void swap_thread_loop();

  // Atomic swap-out: flush page to spill file, then MAP_FIXED remap to disk.
  // Returns true on success. Caller must hold mutex_.
  bool swap_out_page(PageRecord &rec);

  // Builds one ordered eviction batch from a mutex-protected page snapshot.
  // Caller must hold the proxy lifecycle lock while evaluators run.
  struct EvictionPageSnapshot {
    std::byte *addr;
    size_t page_idx;
    size_t allocated_size;
  };

  struct EvictionCandidate {
    std::byte *addr;
    size_t page_idx;
  };

  void register_evictable_page_unlocked(const EvictionPageSnapshot &page);
  void unregister_evictable_page_unlocked(size_t page_idx) noexcept;
  std::vector<EvictionPageSnapshot> snapshot_evictable_pages_unlocked();

  std::vector<EvictionCandidate>
  select_coldest_pages(const std::vector<EvictionPageSnapshot> &pages,
                       size_t bytes_to_evict, PageEvictionProxy &proxy);

  StorageOptions options_;
  size_t page_size_;
  size_t memory_limit_bytes_;
  std::atomic<size_t> spill_limit_bytes_{0};
  int spill_fd_ = -1;
  std::string spill_directory_;
  size_t spill_file_size_ = 0;
  size_t outstanding_spill_pages_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable swap_cv_;
  std::vector<Extent> free_extents_;
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
  std::mt19937_64 eviction_rng_{kEvictionRandomSeed};

  // Registry of all live pages (both in-memory and disk-backed).
  // Indexed by the stable virtual address of the page.
  std::vector<PageRecord> page_registry_;
  std::unordered_map<std::byte *, size_t> page_positions_;
  std::vector<EvictionPageSnapshot> evictable_pages_;
  std::unordered_map<size_t, size_t> evictable_page_positions_;

  // Next page_idx to assign. Monotonically increasing.
  std::atomic<size_t> next_page_idx_{0};

  PageEvictionProxy proxy_;
  std::unordered_set<uint64_t> proxy_clients_;
  uint64_t next_proxy_registration_ = 1;
  uint64_t proxy_generation_ = 0;
  std::unordered_map<size_t, PageEvaluator> page_evaluators_;
  std::shared_ptr<std::shared_mutex> lifecycle_mutex_ =
      std::make_shared<std::shared_mutex>();
  // Background thread lifecycle
  std::thread swap_thread_;
  std::mutex swap_lifecycle_mutex_;
  std::atomic<bool> stop_swap_thread_{false};
  std::exception_ptr swap_error_;
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
          options_.mode == StorageMode::kDiskOnly
              ? 0
              : options_.memory_limit_bytes.value_or(detect_memory_limit())) {
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
}

TieredPageStorage::~TieredPageStorage() {
  std::lock_guard thread_lock(swap_lifecycle_mutex_);
  stop_swap_thread_.store(true, std::memory_order_release);
  swap_cv_.notify_all();
  if (swap_thread_.joinable())
    swap_thread_.join();

  if (spill_fd_ >= 0) {
    close(spill_fd_);
  }
}

PageEvictionRegistration
TieredPageStorage::set_page_eviction_proxy(PageEvictionProxy proxy) {
  if (!proxy)
    throw std::invalid_argument("page eviction proxy must not be empty");
  if (!proxy->score_enabled())
    throw std::invalid_argument("page eviction proxy must enable scoring");

  std::lock_guard thread_lock(swap_lifecycle_mutex_);
  if (stop_swap_thread_.load(std::memory_order_acquire) &&
      swap_thread_.joinable()) {
    swap_cv_.notify_all();
    swap_thread_.join();
  }
  stop_swap_thread_.store(false, std::memory_order_release);
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
  swap_error_ = nullptr;
  if (install_proxy) {
    for (auto &[page_idx, evaluator] : page_evaluators_)
      proxy_->register_page(page_idx, evaluator);
  }
  const size_t high_threshold =
      options_.watermark
          ? watermark_bytes(memory_limit_bytes_, options_.watermark->high)
          : 0;
  if (supports_background_page_eviction() && high_threshold > 0 &&
      proxy_->score_enabled() && !swap_thread_.joinable()) {
    start_swap_thread();
  }
  swap_cv_.notify_one();
  return registration;
}

std::shared_ptr<std::shared_mutex> TieredPageStorage::lifecycle_mutex() const {
  return lifecycle_mutex_;
}

bool TieredPageStorage::supports_background_page_eviction() const noexcept {
  return options_.mode == StorageMode::kAuto && options_.watermark &&
         watermark_bytes(memory_limit_bytes_, options_.watermark->high) > 0;
}

void TieredPageStorage::clear_page_eviction_proxy(
    PageEvictionRegistration registration) {
  if (registration.value == 0)
    throw std::invalid_argument("page eviction registration must be valid");

  std::lock_guard thread_lock(swap_lifecycle_mutex_);
  std::unique_lock lock(mutex_);
  if (proxy_clients_.erase(registration.value) == 0)
    throw std::logic_error("page eviction registration was not active");
  if (!proxy_clients_.empty())
    return;
  if (proxy_generation_ == std::numeric_limits<uint64_t>::max())
    throw std::overflow_error("page eviction proxy generation exhausted");
  proxy_.reset();
  ++proxy_generation_;
  stop_swap_thread_.store(true, std::memory_order_release);
  swap_cv_.notify_all();
  lock.unlock();
}

void TieredPageStorage::unregister_page_evaluator_unlocked(
    size_t page_idx) noexcept {
  page_evaluators_.erase(page_idx);
  if (proxy_)
    proxy_->unregister_page(page_idx);
}

void TieredPageStorage::register_evictable_page_unlocked(
    const EvictionPageSnapshot &page) {
  const size_t position = evictable_pages_.size();
  evictable_pages_.push_back(page);
  try {
    auto [_, inserted] =
        evictable_page_positions_.emplace(page.page_idx, position);
    if (!inserted)
      throw std::logic_error("duplicate evictable page index");
  } catch (...) {
    evictable_pages_.pop_back();
    throw;
  }
}

void TieredPageStorage::unregister_evictable_page_unlocked(
    size_t page_idx) noexcept {
  auto position = evictable_page_positions_.find(page_idx);
  if (position == evictable_page_positions_.end())
    std::terminate();
  const size_t removed_position = position->second;
  const size_t last_position = evictable_pages_.size() - 1;
  if (removed_position != last_position) {
    evictable_pages_[removed_position] = evictable_pages_[last_position];
    auto moved = evictable_page_positions_.find(
        evictable_pages_[removed_position].page_idx);
    if (moved == evictable_page_positions_.end())
      std::terminate();
    moved->second = removed_position;
  }
  evictable_pages_.pop_back();
  evictable_page_positions_.erase(position);
}

std::vector<TieredPageStorage::EvictionPageSnapshot>
TieredPageStorage::snapshot_evictable_pages_unlocked() {
  if (evictable_pages_.size() <= kExactEvictionPageThreshold)
    return evictable_pages_;

  const size_t sampled_count =
      std::min(kSampledEvictionPageCount, evictable_pages_.size());
  std::unordered_set<size_t> sampled_positions;
  sampled_positions.reserve(sampled_count * 2);
  for (size_t position = evictable_pages_.size() - sampled_count;
       position < evictable_pages_.size(); ++position) {
    std::uniform_int_distribution<size_t> distribution(0, position);
    const size_t sampled_position = distribution(eviction_rng_);
    if (!sampled_positions.insert(sampled_position).second)
      sampled_positions.insert(position);
  }

  std::vector<EvictionPageSnapshot> snapshot;
  snapshot.reserve(sampled_count);
  for (size_t position : sampled_positions)
    snapshot.push_back(evictable_pages_[position]);
  return snapshot;
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
                                        PageEvictionMode eviction_mode) {
  if (bytes == 0) {
    throw std::invalid_argument("storage allocation must not be empty");
  }
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("storage alignment must be a power of two");
  }
  if (alignment > page_size_) {
    throw std::invalid_argument("storage alignment exceeds system page size");
  }
  if ((eviction_mode == PageEvictionMode::kDisabled ||
       options_.mode == StorageMode::kMemoryOnly) &&
      options_.mode != StorageMode::kDiskOnly) {
    const size_t fast_alignment = std::max(alignment, kCacheLineSize);
    const size_t fast_size = checked_round_up(bytes, fast_alignment);
    if (try_reserve_memory(fast_size)) {
      return allocate_memory(bytes, fast_size, fast_alignment,
                             PageBacking::kHeap, {});
    }
  }
  std::unique_lock lock(mutex_);
  if (eviction_mode == PageEvictionMode::kEnabled && swap_error_)
    std::rethrow_exception(swap_error_);
  const PageBacking memory_backing = select_memory_backing(eviction_mode);
  const size_t memory_alignment = memory_backing == PageBacking::kHeap
                                      ? std::max(alignment, kCacheLineSize)
                                      : page_size_;
  const size_t memory_size = checked_round_up(bytes, memory_alignment);
  if (options_.mode != StorageMode::kDiskOnly &&
      try_reserve_memory(memory_size)) {
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
                     next_page_idx_.fetch_add(1, std::memory_order_relaxed),
                     true);
  }

  // Register this page so the swap thread can find and potentially swap it out.
  // offset is pre-assigned via acquire_extent so that swap-out can write to
  // disk without re-allocating.  We defer the actual file extent allocation
  // to the first actual swap-out of this page to avoid touching the disk on
  // every in-memory allocation.  We use SIZE_MAX as a sentinel for "not yet
  // assigned".
  const size_t page_idx =
      next_page_idx_.fetch_add(1, std::memory_order_relaxed);
  bool registry_inserted = false;
  bool position_inserted = false;
  bool evictable_page_registered = false;
  try {
    page_registry_.push_back({data, allocated_size, SIZE_MAX, page_idx, backing,
                              register_evaluator});
    registry_inserted = true;
    auto result = page_positions_.emplace(data, page_registry_.size() - 1);
    if (!result.second)
      throw std::logic_error("duplicate storage page address");
    position_inserted = true;
    register_evictable_page_unlocked({data, page_idx, allocated_size});
    evictable_page_registered = true;
    if (register_evaluator) {
      page_evaluators_[page_idx] = evaluator;
      if (proxy_)
        proxy_->register_page(page_idx, std::move(evaluator));
    }
  } catch (...) {
    if (register_evaluator)
      unregister_page_evaluator_unlocked(page_idx);
    if (evictable_page_registered)
      unregister_evictable_page_unlocked(page_idx);
    if (position_inserted)
      page_positions_.erase(data);
    if (registry_inserted)
      page_registry_.pop_back();
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
    swap_cv_.notify_one();
  }

  return make_page(data, bytes, allocated_size, page_registry_.back().page_idx,
                   false);
}

StoragePage TieredPageStorage::allocate_spill(size_t bytes,
                                              size_t allocated_size) {
  ensure_spill_file();
  reserve_extent_capacity();
  size_t offset = acquire_extent(allocated_size);
  void *mapping = mmap(nullptr, allocated_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, spill_fd_, static_cast<off_t>(offset));
  if (mapping == MAP_FAILED) {
    release_extent(offset, allocated_size);
    throw std::runtime_error("failed to map spill page: " +
                             std::string(std::strerror(errno)));
  }
  spill_bytes_.fetch_add(allocated_size, std::memory_order_relaxed);
  spill_count_.fetch_add(1, std::memory_order_relaxed);
  ++outstanding_spill_pages_;
  auto *addr = static_cast<std::byte *>(mapping);
  const size_t registry_size = page_registry_.size();
  try {
    page_registry_.push_back(
        {addr, allocated_size, offset,
         next_page_idx_.fetch_add(1, std::memory_order_relaxed),
         PageBacking::kSpillMapping, false});
    auto result = page_positions_.emplace(addr, page_registry_.size() - 1);
    if (!result.second)
      throw std::logic_error("duplicate storage page address");
  } catch (...) {
    if (page_registry_.size() > registry_size)
      page_registry_.pop_back();
    munmap(addr, allocated_size);
    spill_bytes_.fetch_sub(allocated_size, std::memory_order_relaxed);
    --outstanding_spill_pages_;
    --spill_count_;
    release_extent(offset, allocated_size);
    throw;
  }

  emit_spill_warning(allocated_size);

  return make_page(addr, bytes, allocated_size, page_registry_.back().page_idx,
                   false);
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
  std::lock_guard lock(mutex_);

  auto position = page_positions_.find(data);
  if (position != page_positions_.end()) {
    const size_t registry_position = position->second;
    const PageRecord record = page_registry_[registry_position];
    if (record.backing == PageBacking::kAnonymousMapping)
      unregister_evictable_page_unlocked(record.page_idx);
    page_positions_.erase(position);
    const size_t last_position = page_registry_.size() - 1;
    if (registry_position != last_position) {
      page_registry_[registry_position] = page_registry_[last_position];
      page_positions_[page_registry_[registry_position].addr] =
          registry_position;
    }
    page_registry_.pop_back();
    if (record.evaluator_registered)
      unregister_page_evaluator_unlocked(record.page_idx);

    if (record.backing == PageBacking::kSpillMapping) {
      munmap(data, record.allocated_size);
      spill_bytes_.fetch_sub(record.allocated_size, std::memory_order_relaxed);
      release_extent(record.offset, record.allocated_size);
      --outstanding_spill_pages_;
    } else {
      munmap(data, record.allocated_size);
      evictable_memory_bytes_.fetch_sub(record.allocated_size,
                                        std::memory_order_relaxed);
      memory_bytes_.fetch_sub(record.allocated_size, std::memory_order_relaxed);
      if (record.offset != SIZE_MAX) {
        release_extent(record.offset, record.allocated_size);
        --outstanding_spill_pages_;
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

bool TieredPageStorage::swap_out_page(PageRecord &rec) {
  if (rec.backing == PageBacking::kSpillMapping)
    return false; // already on disk
  if (rec.backing != PageBacking::kAnonymousMapping)
    throw std::logic_error("heap-backed pages cannot be swapped");

  // Step 1: ensure there is a disk extent for this page
  if (rec.offset == SIZE_MAX) {
    ensure_spill_file();
    reserve_extent_capacity();
    rec.offset = acquire_extent(rec.allocated_size);
    ++outstanding_spill_pages_;
  }

  // Step 2: write memory contents to the file at rec.offset
  ssize_t written = pwrite(spill_fd_, rec.addr, rec.allocated_size,
                           static_cast<off_t>(rec.offset));
  if (written < 0 || static_cast<size_t>(written) != rec.allocated_size) {
    throw std::runtime_error("failed to write page during swap: " +
                             std::string(std::strerror(errno)));
  }

  // Step 3: replace the anonymous mapping at the stable virtual address. The
  // store-level lifecycle lock serializes this remap with upper-layer access.
  void *result =
      mmap(rec.addr, rec.allocated_size, PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_FIXED, spill_fd_, static_cast<off_t>(rec.offset));
  if (result == MAP_FAILED) {
    throw std::runtime_error("failed to remap page during swap: " +
                             std::string(std::strerror(errno)));
  }

  memory_bytes_.fetch_sub(rec.allocated_size, std::memory_order_relaxed);
  evictable_memory_bytes_.fetch_sub(rec.allocated_size,
                                    std::memory_order_relaxed);
  spill_bytes_.fetch_add(rec.allocated_size, std::memory_order_relaxed);
  swap_out_count_.fetch_add(1, std::memory_order_relaxed);
  unregister_evictable_page_unlocked(rec.page_idx);
  rec.backing = PageBacking::kSpillMapping;
  if (rec.evaluator_registered) {
    unregister_page_evaluator_unlocked(rec.page_idx);
    rec.evaluator_registered = false;
  }
  emit_spill_warning(rec.allocated_size);
  return true;
}

// =============================================================================
// Coldest-page selection: score-guided only.
// =============================================================================

std::vector<TieredPageStorage::EvictionCandidate>
TieredPageStorage::select_coldest_pages(
    const std::vector<EvictionPageSnapshot> &pages, size_t bytes_to_evict,
    PageEvictionProxy &proxy) {
  struct ScoredCandidate {
    uint64_t score;
    EvictionCandidate candidate;
    size_t allocated_size;
  };

  std::vector<ScoredCandidate> scored_pages;
  if (!proxy || !proxy->score_enabled())
    throw std::runtime_error("page eviction requires a score evaluator");
  const auto selection_started = std::chrono::steady_clock::now();
  scored_pages.reserve(pages.size());
  for (const auto &page : pages) {
    uint64_t score = 0;
    if (!proxy->evaluate_page(page.page_idx, score))
      continue;
    scored_pages.push_back({score, EvictionCandidate{page.addr, page.page_idx},
                            page.allocated_size});
  }
  auto colder = [](const ScoredCandidate &left, const ScoredCandidate &right) {
    if (left.score != right.score)
      return left.score > right.score;
    return left.candidate.page_idx > right.candidate.page_idx;
  };
  std::make_heap(scored_pages.begin(), scored_pages.end(), colder);
  std::vector<EvictionCandidate> result;
  const size_t required_pages =
      bytes_to_evict / page_size_ + (bytes_to_evict % page_size_ != 0);
  result.reserve(std::min(scored_pages.size(), required_pages));
  size_t selected_bytes = 0;
  while (!scored_pages.empty() && selected_bytes < bytes_to_evict) {
    std::pop_heap(scored_pages.begin(), scored_pages.end(), colder);
    ScoredCandidate candidate = scored_pages.back();
    scored_pages.pop_back();
    result.push_back(candidate.candidate);
    if (candidate.allocated_size >= bytes_to_evict - selected_bytes)
      selected_bytes = bytes_to_evict;
    else
      selected_bytes += candidate.allocated_size;
  }
  eviction_selection_count_.fetch_add(1, std::memory_order_relaxed);
  eviction_evaluated_pages_.fetch_add(pages.size(), std::memory_order_relaxed);
  const auto selection_elapsed =
      std::chrono::steady_clock::now() - selection_started;
  eviction_selection_ns_.fetch_add(
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              selection_elapsed)
              .count()),
      std::memory_order_relaxed);
  return result;
}

// =============================================================================
// Background swap thread
// =============================================================================

void TieredPageStorage::start_swap_thread() {
  swap_thread_ = std::thread([this] { swap_thread_loop(); });
}

void TieredPageStorage::swap_thread_loop() {
  try {
    const WatermarkConfig &wm = *options_.watermark;
    const size_t high_threshold = watermark_bytes(memory_limit_bytes_, wm.high);
    const size_t low_threshold = watermark_bytes(memory_limit_bytes_, wm.low);

    while (!stop_swap_thread_.load(std::memory_order_acquire)) {
      std::unique_lock lock(mutex_);
      swap_cv_.wait(lock, [&] {
        return stop_swap_thread_.load(std::memory_order_relaxed) ||
               (proxy_ && proxy_->score_enabled() &&
                evictable_memory_bytes_.load(std::memory_order_relaxed) > 0 &&
                memory_bytes_.load(std::memory_order_relaxed) >=
                    high_threshold);
      });

      if (stop_swap_thread_.load(std::memory_order_relaxed))
        break;

      size_t no_progress_attempts = 0;
      while (memory_bytes_.load(std::memory_order_relaxed) > low_threshold &&
             evictable_memory_bytes_.load(std::memory_order_relaxed) > 0) {
        auto proxy = proxy_;
        const uint64_t proxy_generation = proxy_generation_;
        const size_t current_memory =
            memory_bytes_.load(std::memory_order_relaxed);
        const size_t current_evictable =
            evictable_memory_bytes_.load(std::memory_order_relaxed);
        const size_t bytes_to_evict =
            std::min(current_evictable, current_memory - low_threshold);
        std::vector<EvictionPageSnapshot> page_snapshot =
            snapshot_evictable_pages_unlocked();
        lock.unlock();
        bool proxy_changed = false;
        bool swapped = false;
        bool candidate_invalidated = false;
        std::vector<EvictionCandidate> candidates;
        {
          std::shared_lock page_access_lock(*proxy);
          lock.lock();
          if (proxy_generation_ != proxy_generation) {
            proxy_changed = true;
            lock.unlock();
          } else {
            lock.unlock();
            candidates =
                select_coldest_pages(page_snapshot, bytes_to_evict, proxy);
          }
        }
        if (!proxy_changed) {
          for (const EvictionCandidate &candidate : candidates) {
            if (stop_swap_thread_.load(std::memory_order_relaxed))
              break;
            std::unique_lock candidate_lock(*proxy);
            lock.lock();
            if (proxy_generation_ != proxy_generation) {
              proxy_changed = true;
              lock.unlock();
              break;
            }
            if (memory_bytes_.load(std::memory_order_relaxed) <=
                    low_threshold ||
                evictable_memory_bytes_.load(std::memory_order_relaxed) == 0) {
              lock.unlock();
              break;
            }
            auto position = page_positions_.find(candidate.addr);
            if (position == page_positions_.end() ||
                page_registry_[position->second].addr != candidate.addr ||
                page_registry_[position->second].page_idx !=
                    candidate.page_idx ||
                page_registry_[position->second].backing !=
                    PageBacking::kAnonymousMapping) {
              candidate_invalidated = true;
              eviction_candidate_invalidated_count_.fetch_add(
                  1, std::memory_order_relaxed);
              lock.unlock();
              continue;
            }
            swap_out_page(page_registry_[position->second]);
            swapped = true;
            no_progress_attempts = 0;
            lock.unlock();
          }
        }
        lock.lock();
        if (stop_swap_thread_.load(std::memory_order_relaxed))
          break;
        if (proxy_changed)
          break;
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
    }
  } catch (...) {
    std::lock_guard lock(mutex_);
    swap_error_ = std::current_exception();
    stop_swap_thread_.store(true, std::memory_order_release);
    swap_cv_.notify_all();
  }
}

// =============================================================================
// Spill-file management (unchanged from original, extracted into helpers)
// =============================================================================

void TieredPageStorage::ensure_spill_file() {
  if (spill_fd_ >= 0) {
    return;
  }
  std::filesystem::path directory(select_spill_directory());
  spill_directory_ = directory.string();
  spill_limit_bytes_.store(
      options_.max_spill_bytes.value_or(available_bytes(directory)),
      std::memory_order_release);
  std::string pattern = (directory / "sparse-embedding-XXXXXX").string();
  std::vector<char> path(pattern.begin(), pattern.end());
  path.push_back('\0');
  spill_fd_ = mkstemp(path.data());
  if (spill_fd_ < 0) {
    throw std::runtime_error("failed to create spill file in " +
                             directory.string() + ": " + std::strerror(errno));
  }
  if (unlink(path.data()) != 0) {
    int saved_errno = errno;
    close(spill_fd_);
    spill_fd_ = -1;
    throw std::runtime_error("failed to unlink spill file: " +
                             std::string(std::strerror(saved_errno)));
  }
}

void TieredPageStorage::emit_spill_warning(size_t requested_bytes) noexcept {
  if (!spill_warning_emitted_.exchange(true, std::memory_order_acq_rel)) {
    spill_warning_pending_.store(true, std::memory_order_release);
    std::fprintf(stderr,
                 "SparseEmbeddingSpillWarning: requested=%zu "
                 "memory_bytes=%zu memory_limit_bytes=%zu directory=%s\n",
                 requested_bytes, memory_bytes_.load(std::memory_order_relaxed),
                 memory_limit_bytes_, spill_directory_.c_str());
  }
}

std::string TieredPageStorage::select_spill_directory() const {
  if (options_.spill_directory) {
    std::filesystem::path configured(*options_.spill_directory);
    validate_spill_directory(configured);
    return configured.string();
  }
  const char *configured = std::getenv("SPARSE_EMBEDDING_SPILL_DIR");
  if (configured && *configured) {
    validate_spill_directory(configured);
    return configured;
  }
  std::array<const char *, 3> environment = {std::getenv("SLURM_TMPDIR"),
                                             std::getenv("LOCAL_SCRATCH"),
                                             std::getenv("TMPDIR")};
  std::vector<std::filesystem::path> candidates;
  for (const char *value : environment) {
    if (value && *value) {
      std::filesystem::path candidate(value);
      std::error_code ec;
      if (std::filesystem::is_directory(candidate, ec) && !ec &&
          access(candidate.c_str(), W_OK | X_OK) == 0 &&
          !is_memory_backed(candidate))
        return candidate.string();
    }
  }
  candidates.emplace_back("/var/tmp");
  candidates.emplace_back("/tmp");

  std::optional<std::filesystem::path> selected;
  int selected_speed_rank = -1;
  size_t selected_available = 0;
  for (const auto &candidate : candidates) {
    std::error_code ec;
    if (!std::filesystem::is_directory(candidate, ec) || ec ||
        access(candidate.c_str(), W_OK | X_OK) != 0 ||
        is_memory_backed(candidate)) {
      continue;
    }
    size_t candidate_available = available_bytes(candidate);
    int candidate_speed_rank = storage_speed_rank(candidate);
    if (!selected || candidate_speed_rank > selected_speed_rank ||
        (candidate_speed_rank == selected_speed_rank &&
         candidate_available > selected_available)) {
      selected = candidate;
      selected_speed_rank = candidate_speed_rank;
      selected_available = candidate_available;
    }
  }
  if (!selected) {
    throw std::runtime_error("no writable spill directory is available");
  }
  return selected->string();
}

void TieredPageStorage::reserve_extent_capacity() {
  if (outstanding_spill_pages_ == std::numeric_limits<size_t>::max() ||
      free_extents_.size() >
          std::numeric_limits<size_t>::max() - outstanding_spill_pages_ - 1) {
    throw std::overflow_error("spill extent count overflow");
  }
  size_t required_extent_capacity =
      free_extents_.size() + outstanding_spill_pages_ + 1;
  if (free_extents_.capacity() >= required_extent_capacity)
    return;
  size_t current_capacity = free_extents_.capacity();
  size_t next_capacity =
      current_capacity <= std::numeric_limits<size_t>::max() / 2
          ? std::max(kInitialExtentCapacity, current_capacity * 2)
          : std::numeric_limits<size_t>::max();
  free_extents_.reserve(std::max(required_extent_capacity, next_capacity));
}

size_t TieredPageStorage::acquire_extent(size_t bytes) {
  auto fit = std::ranges::find_if(free_extents_, [bytes](const Extent &extent) {
    return extent.size >= bytes;
  });
  if (fit != free_extents_.end()) {
    size_t offset = fit->offset;
    fit->offset += bytes;
    fit->size -= bytes;
    if (fit->size == 0) {
      free_extents_.erase(fit);
    }
    return offset;
  }
  size_t spill_limit = spill_limit_bytes_.load(std::memory_order_acquire);
  if (bytes > spill_limit - std::min(spill_limit, spill_file_size_)) {
    throw std::runtime_error("spill storage capacity exceeded");
  }
  if (available_bytes(spill_directory_) < bytes) {
    throw std::runtime_error("spill filesystem has insufficient free space");
  }
  int result = posix_fallocate(spill_fd_, static_cast<off_t>(spill_file_size_),
                               static_cast<off_t>(bytes));
  if (result != 0) {
    throw std::runtime_error("failed to reserve spill storage: " +
                             std::string(std::strerror(result)));
  }
  size_t offset = spill_file_size_;
  spill_file_size_ += bytes;
  return offset;
}

void TieredPageStorage::release_extent(size_t offset, size_t bytes) noexcept {
  free_extents_.push_back({offset, bytes});
  std::ranges::sort(free_extents_, [](const Extent &left, const Extent &right) {
    return left.offset < right.offset;
  });
  size_t write = 0;
  for (const auto &extent : free_extents_) {
    if (write > 0 &&
        free_extents_[write - 1].offset + free_extents_[write - 1].size ==
            extent.offset) {
      free_extents_[write - 1].size += extent.size;
    } else {
      free_extents_[write++] = extent;
    }
  }
  free_extents_.resize(write);
}

StorageStats TieredPageStorage::stats() const noexcept {
  return {
      memory_bytes_.load(std::memory_order_relaxed),
      spill_bytes_.load(std::memory_order_relaxed),
      memory_limit_bytes_,
      spill_limit_bytes_.load(std::memory_order_acquire),
      spill_count_.load(std::memory_order_relaxed),
      spill_warning_pending_.load(std::memory_order_acquire),
      swap_out_count_.load(std::memory_order_relaxed),
      eviction_selection_count_.load(std::memory_order_relaxed),
      eviction_evaluated_pages_.load(std::memory_order_relaxed),
      eviction_selection_ns_.load(std::memory_order_relaxed),
      eviction_candidate_invalidated_count_.load(std::memory_order_relaxed)};
}

std::string TieredPageStorage::spill_directory() const {
  std::lock_guard lock(mutex_);
  return spill_directory_;
}

bool TieredPageStorage::consume_spill_warning() noexcept {
  return spill_warning_pending_.exchange(false, std::memory_order_acq_rel);
}

} // namespace se
