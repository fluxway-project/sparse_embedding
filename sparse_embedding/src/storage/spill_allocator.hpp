#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace se::storage_detail {

class SpillAllocator {
public:
  SpillAllocator(std::optional<std::string> configured_directory,
                 std::optional<size_t> configured_limit_bytes);
  ~SpillAllocator();

  SpillAllocator(const SpillAllocator &) = delete;
  SpillAllocator &operator=(const SpillAllocator &) = delete;

  size_t acquire_extent(size_t bytes);
  void release_extent(size_t offset, size_t bytes) noexcept;
  int file_descriptor();
  std::string directory() const;
  size_t limit_bytes() const noexcept;

private:
  struct Extent {
    size_t offset;
    size_t size;
  };

  static constexpr size_t kInitialExtentCapacity = 64;

  void ensure_file_locked();
  void reserve_extent_capacity_locked();
  std::string select_directory() const;

  std::optional<std::string> configured_directory_;
  std::optional<size_t> configured_limit_bytes_;
  mutable std::mutex mutex_;
  int fd_ = -1;
  std::string directory_;
  size_t limit_bytes_ = 0;
  size_t file_size_ = 0;
  size_t active_extents_ = 0;
  std::vector<Extent> free_extents_;
};

} // namespace se::storage_detail
