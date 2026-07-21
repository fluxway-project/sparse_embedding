#pragma once

#include <proxy/proxy.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>

namespace se {

enum class StorageMode : uint8_t { kAuto, kMemoryOnly, kDiskOnly };

enum class PageEvictionMode : uint8_t { kEnabled, kDisabled };

// Watermark thresholds controlling when background swap-out is triggered.
// high: fraction of memory_limit at which eviction begins [0.0, 1.0].
// low:  fraction of memory_limit at which eviction stops  [0.0, high).
struct WatermarkConfig {
  double high = 0.8;
  double low = 0.6;
};

struct StorageOptions {
  StorageMode mode = StorageMode::kAuto;
  std::optional<size_t> memory_limit_bytes;
  std::optional<std::string> spill_directory;
  std::optional<size_t> max_spill_bytes;
  std::optional<WatermarkConfig> watermark;
};

struct StorageStats {
  size_t memory_bytes;
  size_t spill_bytes;
  size_t memory_limit_bytes;
  size_t spill_limit_bytes;
  uint64_t spill_count;
  bool spill_warning_pending;
  uint64_t swap_out_count;
  uint64_t eviction_selection_count;
  uint64_t eviction_evaluated_pages;
  uint64_t eviction_selection_ns;
  uint64_t eviction_candidate_invalidated_count;
};

class StoragePage;
class TieredPageStorage;
using PageEvaluator = std::function<uint64_t()>;

PRO_DEF_MEM_DISPATCH(MemEvaluatePage, evaluate_page);
PRO_DEF_MEM_DISPATCH(MemRegisterPage, register_page);
PRO_DEF_MEM_DISPATCH(MemUnregisterPage, unregister_page);
PRO_DEF_MEM_DISPATCH(MemScoreEnabled, score_enabled);
PRO_DEF_MEM_DISPATCH(MemLockShared, lock_shared);
PRO_DEF_MEM_DISPATCH(MemUnlockShared, unlock_shared);
PRO_DEF_MEM_DISPATCH(MemLock, lock);
PRO_DEF_MEM_DISPATCH(MemUnlock, unlock);

struct PageEvictionProxyFacade
    : pro::facade_builder ::add_convention<
          MemEvaluatePage,
          bool(size_t, uint64_t &)>::add_convention<MemRegisterPage,
                                                    void(size_t,
                                                         PageEvaluator)>::
          add_convention<MemUnregisterPage, void(size_t) noexcept>::
              add_convention<MemScoreEnabled, bool() const noexcept>::
                  add_convention<MemLockShared, void()>::add_convention<
                      MemUnlockShared,
                      void() noexcept>::add_convention<MemLock, void()>::
                      add_convention<MemUnlock, void() noexcept>::support_copy<
                          pro::constraint_level::nontrivial>::
                          support_relocation<
                              pro::constraint_level::nontrivial>::build {};

using PageEvictionProxy = pro::proxy<PageEvictionProxyFacade>;

struct PageEvictionRegistration {
  uint64_t value = 0;

  friend bool operator==(const PageEvictionRegistration &,
                         const PageEvictionRegistration &) = default;
};

PRO_DEF_MEM_DISPATCH(MemAllocatePage, allocate);
PRO_DEF_MEM_DISPATCH(MemStorageStats, stats);
PRO_DEF_MEM_DISPATCH(MemSpillDirectory, spill_directory);
PRO_DEF_MEM_DISPATCH(MemConsumeSpillWarning, consume_spill_warning);
PRO_DEF_MEM_DISPATCH(MemSupportsPageEviction,
                     supports_background_page_eviction);
PRO_DEF_MEM_DISPATCH(MemLifecycleMutex, lifecycle_mutex);
PRO_DEF_MEM_DISPATCH(MemSetPageEvictionProxy, set_page_eviction_proxy);
PRO_DEF_MEM_DISPATCH(MemClearPageEvictionProxy, clear_page_eviction_proxy);
PRO_DEF_MEM_DISPATCH(MemReleasePage, release);

struct PageStorageFacade
    : pro              //
      ::facade_builder //
      ::add_convention<MemAllocatePage, StoragePage(size_t,
                                                    size_t)> //
      ::add_convention<MemAllocatePage,
                       StoragePage(size_t, size_t, PageEvaluator,
                                   PageEvictionMode)>                  //
      ::add_convention<MemStorageStats, StorageStats() const noexcept> //
      ::add_convention<MemSpillDirectory, std::string() const>         //
      ::add_convention<MemConsumeSpillWarning,
                       bool() noexcept>                                //
      ::add_convention<MemSupportsPageEviction, bool() const noexcept> //
      ::add_convention<MemLifecycleMutex,
                       std::shared_ptr<std::shared_mutex>() const> //
      ::add_convention<MemSetPageEvictionProxy,
                       PageEvictionRegistration(PageEvictionProxy)> //
      ::add_convention<MemClearPageEvictionProxy,
                       void(PageEvictionRegistration)> //
      ::add_convention<MemReleasePage,
                       void(std::byte *, size_t, bool) noexcept> //
      ::support_copy<pro::constraint_level::nontrivial>          //
      ::support_relocation<pro::constraint_level::nontrivial>    //
      ::build {};

using PageStorage = pro::proxy<PageStorageFacade>;

PageStorage create_page_storage(StorageOptions options = {});

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
  size_t page_index() const noexcept { return page_index_; }
  explicit operator bool() const noexcept { return data_ != nullptr; }

private:
  friend class TieredPageStorage;

  StoragePage(PageStorage owner, std::byte *data, size_t size,
              size_t allocated_size, size_t page_index,
              bool heap_backed) noexcept;
  void reset() noexcept;

  PageStorage owner_;
  std::byte *data_ = nullptr;
  size_t size_ = 0;
  size_t allocated_size_ = 0;
  size_t page_index_ = 0;
  bool heap_backed_ = false;
};

} // namespace se
