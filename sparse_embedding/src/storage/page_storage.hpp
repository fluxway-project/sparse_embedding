#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace se {

enum class StorageMode : uint8_t { kAuto, kMemoryOnly, kDiskOnly };

struct StorageOptions {
  StorageMode mode = StorageMode::kAuto;
  std::optional<size_t> memory_limit_bytes;
  std::optional<std::string> spill_directory;
  std::optional<size_t> max_spill_bytes;
};

struct StorageStats {
  size_t memory_bytes;
  size_t spill_bytes;
  size_t memory_limit_bytes;
  size_t spill_limit_bytes;
  uint64_t spill_count;
  bool spill_warning_pending;
};

class PageStorage;

class StoragePage {
public:
  StoragePage() = default;
  StoragePage(const StoragePage &) = delete;
  StoragePage &operator=(const StoragePage &) = delete;
  StoragePage(StoragePage &&other) noexcept;
  StoragePage &operator=(StoragePage &&other) noexcept;
  ~StoragePage();

  std::byte *data() const noexcept { return data_; }
  size_t size() const noexcept { return size_; }
  explicit operator bool() const noexcept { return data_ != nullptr; }

private:
  friend class PageStorage;

  StoragePage(std::shared_ptr<PageStorage> owner, std::byte *data, size_t size,
              size_t allocated_size, size_t offset, bool spilled) noexcept;
  void reset() noexcept;

  std::shared_ptr<PageStorage> owner_;
  std::byte *data_ = nullptr;
  size_t size_ = 0;
  size_t allocated_size_ = 0;
  size_t offset_ = 0;
  bool spilled_ = false;
};

class PageStorage : public std::enable_shared_from_this<PageStorage> {
public:
  static std::shared_ptr<PageStorage> create(StorageOptions options = {});

  PageStorage(const PageStorage &) = delete;
  PageStorage &operator=(const PageStorage &) = delete;
  virtual ~PageStorage() = default;

  virtual StoragePage allocate(size_t bytes, size_t alignment = 64) = 0;
  virtual StorageStats stats() const noexcept = 0;
  virtual std::string spill_directory() const = 0;
  virtual bool consume_spill_warning() noexcept = 0;

protected:
  PageStorage() = default;
  StoragePage make_page(std::byte *data, size_t size, size_t allocated_size,
                        size_t offset, bool spilled);

private:
  friend class StoragePage;
  virtual void release(std::byte *data, size_t allocated_size, size_t offset,
                       bool spilled) noexcept = 0;
};

} // namespace se
