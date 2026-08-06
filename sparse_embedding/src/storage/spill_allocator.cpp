#include "sparse_embedding/src/storage/spill_allocator.hpp"

#include <fcntl.h>
#include <linux/magic.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace se::storage_detail {
namespace {

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

void validate_directory(const std::filesystem::path &path) {
  std::error_code error;
  if (!std::filesystem::is_directory(path, error) || error ||
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
  std::error_code error;
  device = std::filesystem::canonical(device, error);
  if (error)
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

SpillAllocator::SpillAllocator(std::optional<std::string> configured_directory,
                               std::optional<size_t> configured_limit_bytes)
    : configured_directory_(std::move(configured_directory)),
      configured_limit_bytes_(configured_limit_bytes) {}

SpillAllocator::~SpillAllocator() {
  if (fd_ >= 0)
    close(fd_);
}

size_t SpillAllocator::acquire_extent(size_t bytes) {
  if (bytes == 0)
    throw std::invalid_argument("spill extent size must be nonzero");
  std::lock_guard lock(mutex_);
  ensure_file_locked();
  reserve_extent_capacity_locked();
  auto fit = std::ranges::find_if(free_extents_, [bytes](const Extent &extent) {
    return extent.size >= bytes;
  });
  if (fit != free_extents_.end()) {
    const size_t offset = fit->offset;
    fit->offset += bytes;
    fit->size -= bytes;
    if (fit->size == 0)
      free_extents_.erase(fit);
    ++active_extents_;
    return offset;
  }
  if (bytes > limit_bytes_ - std::min(limit_bytes_, file_size_))
    throw std::runtime_error("spill storage capacity exceeded");
  if (available_bytes(directory_) < bytes)
    throw std::runtime_error("spill filesystem has insufficient free space");
  const int result = posix_fallocate(fd_, static_cast<off_t>(file_size_),
                                     static_cast<off_t>(bytes));
  if (result != 0) {
    throw std::runtime_error("failed to reserve spill storage: " +
                             std::string(std::strerror(result)));
  }
  const size_t offset = file_size_;
  file_size_ += bytes;
  ++active_extents_;
  return offset;
}

void SpillAllocator::release_extent(size_t offset, size_t bytes) noexcept {
  if (bytes == 0)
    std::terminate();
  std::lock_guard lock(mutex_);
  if (active_extents_ == 0)
    std::terminate();
  free_extents_.push_back({offset, bytes});
  --active_extents_;
  std::ranges::sort(free_extents_, [](const Extent &left, const Extent &right) {
    return left.offset < right.offset;
  });
  size_t write = 0;
  for (const Extent &extent : free_extents_) {
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

int SpillAllocator::file_descriptor() {
  std::lock_guard lock(mutex_);
  ensure_file_locked();
  return fd_;
}

std::string SpillAllocator::directory() const {
  std::lock_guard lock(mutex_);
  return directory_;
}

size_t SpillAllocator::limit_bytes() const noexcept {
  std::lock_guard lock(mutex_);
  return limit_bytes_;
}

void SpillAllocator::ensure_file_locked() {
  if (fd_ >= 0)
    return;
  std::filesystem::path directory(select_directory());
  directory_ = directory.string();
  limit_bytes_ = configured_limit_bytes_.value_or(available_bytes(directory));
  std::string pattern = (directory / "sparse-embedding-XXXXXX").string();
  std::vector<char> path(pattern.begin(), pattern.end());
  path.push_back('\0');
  fd_ = mkstemp(path.data());
  if (fd_ < 0) {
    throw std::runtime_error("failed to create spill file in " +
                             directory.string() + ": " + std::strerror(errno));
  }
  if (unlink(path.data()) != 0) {
    const int saved_errno = errno;
    close(fd_);
    fd_ = -1;
    throw std::runtime_error("failed to unlink spill file: " +
                             std::string(std::strerror(saved_errno)));
  }
}

void SpillAllocator::reserve_extent_capacity_locked() {
  if (active_extents_ == std::numeric_limits<size_t>::max() ||
      free_extents_.size() >
          std::numeric_limits<size_t>::max() - active_extents_ - 1) {
    throw std::overflow_error("spill extent count overflow");
  }
  const size_t required = free_extents_.size() + active_extents_ + 1;
  if (free_extents_.capacity() >= required)
    return;
  const size_t current = free_extents_.capacity();
  const size_t next = current <= std::numeric_limits<size_t>::max() / 2
                          ? std::max(kInitialExtentCapacity, current * 2)
                          : std::numeric_limits<size_t>::max();
  free_extents_.reserve(std::max(required, next));
}

std::string SpillAllocator::select_directory() const {
  if (configured_directory_) {
    std::filesystem::path configured(*configured_directory_);
    validate_directory(configured);
    return configured.string();
  }
  const char *configured = std::getenv("SPARSE_EMBEDDING_SPILL_DIR");
  if (configured && *configured) {
    validate_directory(configured);
    return configured;
  }
  const std::array<const char *, 3> environment = {std::getenv("SLURM_TMPDIR"),
                                                   std::getenv("LOCAL_SCRATCH"),
                                                   std::getenv("TMPDIR")};
  for (const char *value : environment) {
    if (value && *value) {
      std::filesystem::path candidate(value);
      std::error_code error;
      if (std::filesystem::is_directory(candidate, error) && !error &&
          access(candidate.c_str(), W_OK | X_OK) == 0 &&
          !is_memory_backed(candidate)) {
        return candidate.string();
      }
    }
  }
  const std::array<std::filesystem::path, 2> candidates = {"/var/tmp", "/tmp"};
  std::optional<std::filesystem::path> selected;
  int selected_speed_rank = -1;
  size_t selected_available = 0;
  for (const auto &candidate : candidates) {
    std::error_code error;
    if (!std::filesystem::is_directory(candidate, error) || error ||
        access(candidate.c_str(), W_OK | X_OK) != 0 ||
        is_memory_backed(candidate)) {
      continue;
    }
    const size_t candidate_available = available_bytes(candidate);
    const int candidate_speed_rank = storage_speed_rank(candidate);
    if (!selected || candidate_speed_rank > selected_speed_rank ||
        (candidate_speed_rank == selected_speed_rank &&
         candidate_available > selected_available)) {
      selected = candidate;
      selected_speed_rank = candidate_speed_rank;
      selected_available = candidate_available;
    }
  }
  if (!selected)
    throw std::runtime_error("no writable spill directory is available");
  return selected->string();
}

} // namespace se::storage_detail
