#include "sparse_embedding/src/storage/page_storage.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <linux/magic.h>

namespace se {
namespace {

constexpr size_t kAutoMemoryNumerator = 4;
constexpr size_t kAutoMemoryDenominator = 5;

size_t checked_round_up(size_t value, size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("storage alignment must be a power of two");
  }
  if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    throw std::overflow_error("storage allocation size overflow");
  }
  return (value + alignment - 1) & ~(alignment - 1);
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
      return std::filesystem::path("/sys/fs/cgroup") /
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
      return std::filesystem::path("/sys/fs/cgroup/memory") / relative;
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

class TieredPageStorage final : public PageStorage {
public:
  explicit TieredPageStorage(StorageOptions options);
  ~TieredPageStorage() override;

  StoragePage allocate(size_t bytes, size_t alignment) override;
  StorageStats stats() const noexcept override;
  std::string spill_directory() const override;
  bool consume_spill_warning() noexcept override;

private:
  static constexpr size_t kInitialExtentCapacity = 64;

  struct Extent {
    size_t offset;
    size_t size;
  };

  void release(std::byte *data, size_t allocated_size, size_t offset,
               bool spilled) noexcept override;
  StoragePage allocate_memory(size_t bytes, size_t allocated_size,
                              size_t alignment);
  StoragePage allocate_spill(size_t bytes, size_t allocated_size);
  void ensure_spill_file();
  std::string select_spill_directory() const;
  size_t acquire_extent(size_t bytes);
  void release_extent(size_t offset, size_t bytes) noexcept;

  StorageOptions options_;
  size_t page_size_;
  size_t memory_limit_bytes_;
  size_t spill_limit_bytes_ = 0;
  int spill_fd_ = -1;
  std::string spill_directory_;
  size_t spill_file_size_ = 0;
  size_t outstanding_spill_pages_ = 0;
  mutable std::mutex mutex_;
  std::vector<Extent> free_extents_;
  std::atomic<size_t> memory_bytes_{0};
  std::atomic<size_t> spill_bytes_{0};
  std::atomic<uint64_t> spill_count_{0};
  std::atomic<bool> spill_warning_pending_{false};
  std::atomic<bool> spill_warning_emitted_{false};
};

StoragePage::StoragePage(std::shared_ptr<PageStorage> owner, std::byte *data,
                         size_t size, size_t allocated_size, size_t offset,
                         bool spilled) noexcept
    : owner_(std::move(owner)), data_(data), size_(size),
      allocated_size_(allocated_size), offset_(offset), spilled_(spilled) {}

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
    offset_ = std::exchange(other.offset_, 0);
    spilled_ = std::exchange(other.spilled_, false);
  }
  return *this;
}

StoragePage::~StoragePage() { reset(); }

void StoragePage::reset() noexcept {
  if (owner_ && data_) {
    owner_->release(data_, allocated_size_, offset_, spilled_);
  }
  owner_.reset();
  data_ = nullptr;
  size_ = 0;
  allocated_size_ = 0;
  offset_ = 0;
  spilled_ = false;
}

std::shared_ptr<PageStorage> PageStorage::create(StorageOptions options) {
  return std::make_shared<TieredPageStorage>(std::move(options));
}

StoragePage PageStorage::make_page(std::byte *data, size_t size,
                                   size_t allocated_size, size_t offset,
                                   bool spilled) {
  return StoragePage(shared_from_this(), data, size, allocated_size, offset,
                     spilled);
}

TieredPageStorage::TieredPageStorage(StorageOptions options)
    : options_(std::move(options)),
      page_size_(static_cast<size_t>(sysconf(_SC_PAGESIZE))),
      memory_limit_bytes_(
          options_.mode == StorageMode::kDiskOnly
              ? 0
              : options_.memory_limit_bytes.value_or(detect_memory_limit())) {
  if (page_size_ == 0 || (page_size_ & (page_size_ - 1)) != 0) {
    throw std::runtime_error("invalid operating system page size");
  }
}

TieredPageStorage::~TieredPageStorage() {
  if (spill_fd_ >= 0) {
    close(spill_fd_);
  }
}

StoragePage TieredPageStorage::allocate(size_t bytes, size_t alignment) {
  if (bytes == 0) {
    throw std::invalid_argument("storage allocation must not be empty");
  }
  if (alignment > page_size_) {
    throw std::invalid_argument("storage alignment exceeds system page size");
  }
  std::lock_guard lock(mutex_);
  size_t heap_size = checked_round_up(bytes, std::max<size_t>(alignment, 64));
  size_t memory_bytes = memory_bytes_.load(std::memory_order_relaxed);
  bool use_memory = options_.mode != StorageMode::kDiskOnly &&
                    memory_bytes <= memory_limit_bytes_ &&
                    heap_size <= memory_limit_bytes_ - memory_bytes;
  if (use_memory) {
    try {
      return allocate_memory(bytes, heap_size, std::max<size_t>(alignment, 64));
    } catch (const std::bad_alloc &) {
      if (options_.mode == StorageMode::kMemoryOnly) {
        throw;
      }
    }
  }
  if (options_.mode == StorageMode::kMemoryOnly) {
    throw std::bad_alloc();
  }
  return allocate_spill(bytes, checked_round_up(bytes, page_size_));
}

StoragePage TieredPageStorage::allocate_memory(size_t bytes,
                                               size_t allocated_size,
                                               size_t alignment) {
  auto *data =
      static_cast<std::byte *>(aligned_alloc(alignment, allocated_size));
  if (!data) {
    throw std::bad_alloc();
  }
  memory_bytes_.fetch_add(allocated_size, std::memory_order_relaxed);
  return make_page(data, bytes, allocated_size, 0, false);
}

StoragePage TieredPageStorage::allocate_spill(size_t bytes,
                                              size_t allocated_size) {
  ensure_spill_file();
  if (outstanding_spill_pages_ == std::numeric_limits<size_t>::max() ||
      free_extents_.size() >
          std::numeric_limits<size_t>::max() - outstanding_spill_pages_ - 1) {
    throw std::overflow_error("spill extent count overflow");
  }
  size_t required_extent_capacity =
      free_extents_.size() + outstanding_spill_pages_ + 1;
  if (free_extents_.capacity() < required_extent_capacity) {
    size_t current_capacity = free_extents_.capacity();
    size_t next_capacity =
        current_capacity <= std::numeric_limits<size_t>::max() / 2
            ? std::max(kInitialExtentCapacity, current_capacity * 2)
            : std::numeric_limits<size_t>::max();
    free_extents_.reserve(std::max(required_extent_capacity, next_capacity));
  }
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
  if (!spill_warning_emitted_.exchange(true, std::memory_order_acq_rel)) {
    spill_warning_pending_.store(true, std::memory_order_release);
    std::fprintf(stderr,
                 "SparseEmbeddingSpillWarning: requested=%zu "
                 "memory_bytes=%zu memory_limit_bytes=%zu directory=%s\n",
                 allocated_size, memory_bytes_.load(std::memory_order_relaxed),
                 memory_limit_bytes_, spill_directory_.c_str());
  }
  return make_page(static_cast<std::byte *>(mapping), bytes, allocated_size,
                   offset, true);
}

void TieredPageStorage::ensure_spill_file() {
  if (spill_fd_ >= 0) {
    return;
  }
  std::filesystem::path directory(select_spill_directory());
  spill_directory_ = directory.string();
  spill_limit_bytes_ =
      options_.max_spill_bytes.value_or(available_bytes(directory));
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

size_t TieredPageStorage::acquire_extent(size_t bytes) {
  auto fit = std::find_if(
      free_extents_.begin(), free_extents_.end(),
      [bytes](const Extent &extent) { return extent.size >= bytes; });
  if (fit != free_extents_.end()) {
    size_t offset = fit->offset;
    fit->offset += bytes;
    fit->size -= bytes;
    if (fit->size == 0) {
      free_extents_.erase(fit);
    }
    return offset;
  }
  if (bytes >
      spill_limit_bytes_ - std::min(spill_limit_bytes_, spill_file_size_)) {
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

void TieredPageStorage::release(std::byte *data, size_t allocated_size,
                                size_t offset, bool spilled) noexcept {
  std::lock_guard lock(mutex_);
  if (spilled) {
    munmap(data, allocated_size);
    spill_bytes_.fetch_sub(allocated_size, std::memory_order_relaxed);
    release_extent(offset, allocated_size);
    --outstanding_spill_pages_;
  } else {
    free(data);
    memory_bytes_.fetch_sub(allocated_size, std::memory_order_relaxed);
  }
}

void TieredPageStorage::release_extent(size_t offset, size_t bytes) noexcept {
  free_extents_.push_back({offset, bytes});
  std::sort(free_extents_.begin(), free_extents_.end(),
            [](const Extent &left, const Extent &right) {
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
  return {memory_bytes_.load(std::memory_order_relaxed),
          spill_bytes_.load(std::memory_order_relaxed),
          memory_limit_bytes_,
          spill_limit_bytes_,
          spill_count_.load(std::memory_order_relaxed),
          spill_warning_pending_.load(std::memory_order_acquire)};
}

std::string TieredPageStorage::spill_directory() const {
  std::lock_guard lock(mutex_);
  return spill_directory_;
}

bool TieredPageStorage::consume_spill_warning() noexcept {
  return spill_warning_pending_.exchange(false, std::memory_order_acq_rel);
}

} // namespace se
