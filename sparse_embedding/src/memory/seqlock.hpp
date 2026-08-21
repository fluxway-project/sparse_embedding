#pragma once

#if !(defined(__GNUC__) || defined(__clang__))
#error "SeqLock requires GCC or Clang"
#endif

#if !(defined(__x86_64__) || defined(__i386__) || defined(__aarch64__) ||      \
      defined(__arm64__))
#error "SeqLock requires x86 or ARM architecture"
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace se::memory {

inline void compiler_barrier() noexcept { asm volatile("" ::: "memory"); }

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#elif defined(__aarch64__) || defined(__arm64__)
  __builtin_arm_yield();
#endif
}

static_assert(std::atomic<uint32_t>::is_always_lock_free);
static_assert(std::atomic<uint64_t>::is_always_lock_free);

class SeqLock {
public:
  explicit SeqLock(std::atomic<uint32_t> &sequence) noexcept
      : sequence_(sequence) {}
  SeqLock(const SeqLock &) = delete;
  SeqLock &operator=(const SeqLock &) = delete;
  void lock() noexcept {
    uint32_t observed = sequence_.load(std::memory_order_relaxed);
    while (true) {
      while ((observed & 1U) != 0U) {
        cpu_relax();
        observed = sequence_.load(std::memory_order_relaxed);
      }
      if (sequence_.compare_exchange_weak(observed, observed + 1U,
                                          std::memory_order_acquire,
                                          std::memory_order_relaxed))
        return;
    }
  }
  void unlock() noexcept {
    const uint32_t held = sequence_.load(std::memory_order_relaxed);
    if ((held & 1U) == 0U)
      __builtin_trap();
    sequence_.store(held + 1U, std::memory_order_release);
  }

private:
  std::atomic<uint32_t> &sequence_;
};

class SeqWriteGuard {
public:
  explicit SeqWriteGuard(std::atomic<uint32_t> &sequence) noexcept
      : lock_(sequence) {
    lock_.lock();
  }
  ~SeqWriteGuard() { lock_.unlock(); }
  SeqWriteGuard(const SeqWriteGuard &) = delete;
  SeqWriteGuard &operator=(const SeqWriteGuard &) = delete;

private:
  SeqLock lock_;
};

class PackedWriteLock {
public:
  PackedWriteLock(std::atomic<uint64_t> &packed,
                  std::atomic<uint32_t> &readers) noexcept
      : packed_(packed), readers_(readers) {}
  PackedWriteLock(const PackedWriteLock &) = delete;
  PackedWriteLock &operator=(const PackedWriteLock &) = delete;
  void lock() noexcept {
    uint64_t observed = packed_.load(std::memory_order_relaxed);
    while (true) {
      const uint32_t version = static_cast<uint32_t>(observed >> 32);
      if ((version & 1U) != 0U) {
        cpu_relax();
        observed = packed_.load(std::memory_order_relaxed);
        continue;
      }
      const uint64_t held = observed | (uint64_t{1} << 32);
      if (packed_.compare_exchange_weak(observed, held,
                                        std::memory_order_acquire,
                                        std::memory_order_relaxed)) {
        while (readers_.load(std::memory_order_acquire) != 0U)
          cpu_relax();
        return;
      }
    }
  }
  void unlock() noexcept {
    const uint64_t held = packed_.load(std::memory_order_relaxed);
    const uint32_t version = static_cast<uint32_t>(held >> 32);
    if ((version & 1U) == 0U)
      __builtin_trap();
    packed_.store(held + (uint64_t{1} << 32), std::memory_order_release);
  }

private:
  std::atomic<uint64_t> &packed_;
  std::atomic<uint32_t> &readers_;
};

class PackedWriteGuard {
public:
  PackedWriteGuard(std::atomic<uint64_t> &packed,
                   std::atomic<uint32_t> &readers) noexcept
      : lock_(packed, readers) {
    lock_.lock();
  }
  ~PackedWriteGuard() { lock_.unlock(); }
  PackedWriteGuard(const PackedWriteGuard &) = delete;
  PackedWriteGuard &operator=(const PackedWriteGuard &) = delete;

private:
  PackedWriteLock lock_;
};

struct PackedWriteTarget {
  std::atomic<uint64_t> *packed;
  std::atomic<uint32_t> *readers;
};

class PackedWriteGuardBatch {
public:
  explicit PackedWriteGuardBatch(std::span<const PackedWriteTarget> targets)
      : targets_(targets), locked_(0) {
    if (targets_.empty())
      throw std::invalid_argument("packed write guard batch is empty");
    for (; locked_ < targets_.size(); ++locked_)
      PackedWriteLock(*targets_[locked_].packed, *targets_[locked_].readers)
          .lock();
  }

  ~PackedWriteGuardBatch() {
    while (locked_ != 0) {
      --locked_;
      PackedWriteLock(*targets_[locked_].packed, *targets_[locked_].readers)
          .unlock();
    }
  }

  PackedWriteGuardBatch(const PackedWriteGuardBatch &) = delete;
  PackedWriteGuardBatch &operator=(const PackedWriteGuardBatch &) = delete;

private:
  std::span<const PackedWriteTarget> targets_;
  size_t locked_;
};

class PackedReadGuard {
public:
  PackedReadGuard(std::atomic<uint64_t> &packed,
                  std::atomic<uint32_t> &readers) noexcept
      : packed_(packed), readers_(readers), active_(false) {
    for (;;) {
      const uint64_t before = packed_.load(std::memory_order_acquire);
      if ((static_cast<uint32_t>(before >> 32) & 1U) != 0U) {
        cpu_relax();
        continue;
      }
      readers_.fetch_add(1, std::memory_order_acquire);
      if (packed_.load(std::memory_order_acquire) == before) {
        active_ = true;
        return;
      }
      readers_.fetch_sub(1, std::memory_order_release);
    }
  }

  ~PackedReadGuard() {
    if (active_)
      readers_.fetch_sub(1, std::memory_order_release);
  }

  PackedReadGuard(const PackedReadGuard &) = delete;
  PackedReadGuard &operator=(const PackedReadGuard &) = delete;

private:
  std::atomic<uint64_t> &packed_;
  std::atomic<uint32_t> &readers_;
  bool active_;
};

template <typename Read>
decltype(auto) read_seqlock(std::atomic<uint32_t> &sequence, Read &&read) {
  Read &reader = read;
  while (true) {
    const uint32_t before = sequence.load(std::memory_order_acquire);
    if ((before & 1U) != 0U) {
      cpu_relax();
      continue;
    }
    compiler_barrier();
    if constexpr (std::is_void_v<std::invoke_result_t<Read>>) {
      reader();
      compiler_barrier();
      if (before == sequence.load(std::memory_order_acquire))
        return;
    } else {
      auto result = reader();
      compiler_barrier();
      if (before == sequence.load(std::memory_order_acquire))
        return result;
    }
  }
}

template <typename Read>
decltype(auto) read_packed_seqlock(std::atomic<uint64_t> &packed, Read &&read) {
  Read &reader = read;
  while (true) {
    const uint64_t before = packed.load(std::memory_order_acquire);
    if ((static_cast<uint32_t>(before >> 32) & 1U) != 0U) {
      cpu_relax();
      continue;
    }
    compiler_barrier();
    if constexpr (std::is_void_v<std::invoke_result_t<Read>>) {
      reader();
      compiler_barrier();
      if (before == packed.load(std::memory_order_acquire))
        return;
    } else {
      auto result = reader();
      compiler_barrier();
      if (before == packed.load(std::memory_order_acquire))
        return result;
    }
  }
}

} // namespace se::memory
