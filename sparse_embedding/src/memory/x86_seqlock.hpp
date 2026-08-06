#pragma once

#if !defined(__linux__) || !defined(__x86_64__) ||                             \
    !(defined(__GNUC__) || defined(__clang__))
#error "X86 SeqLock requires Linux x86_64 with GCC or Clang"
#endif

#include <atomic>
#include <cstdint>
#include <immintrin.h>
#include <type_traits>

namespace se::memory {

inline void compiler_barrier() noexcept { asm volatile("" ::: "memory"); }

class X86SeqLock {
public:
  explicit X86SeqLock(std::atomic<uint32_t> &sequence) noexcept
      : sequence_(sequence) {}
  X86SeqLock(const X86SeqLock &) = delete;
  X86SeqLock &operator=(const X86SeqLock &) = delete;
  void lock() noexcept {
    uint32_t observed = sequence_.load(std::memory_order_relaxed);
    while (true) {
      while ((observed & 1U) != 0U) {
        _mm_pause();
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

class X86SeqWriteGuard {
public:
  explicit X86SeqWriteGuard(std::atomic<uint32_t> &sequence) noexcept
      : lock_(sequence) {
    lock_.lock();
  }
  ~X86SeqWriteGuard() { lock_.unlock(); }
  X86SeqWriteGuard(const X86SeqWriteGuard &) = delete;
  X86SeqWriteGuard &operator=(const X86SeqWriteGuard &) = delete;

private:
  X86SeqLock lock_;
};

class X86PackedWriteLock {
public:
  explicit X86PackedWriteLock(std::atomic<uint64_t> &packed) noexcept
      : packed_(packed) {}
  X86PackedWriteLock(const X86PackedWriteLock &) = delete;
  X86PackedWriteLock &operator=(const X86PackedWriteLock &) = delete;
  void lock() noexcept {
    uint64_t observed = packed_.load(std::memory_order_relaxed);
    while (true) {
      const uint32_t version = static_cast<uint32_t>(observed >> 32);
      if ((version & 1U) != 0U) {
        _mm_pause();
        observed = packed_.load(std::memory_order_relaxed);
        continue;
      }
      const uint64_t held = observed | (uint64_t{1} << 32);
      if (packed_.compare_exchange_weak(observed, held,
                                        std::memory_order_acquire,
                                        std::memory_order_relaxed))
        return;
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
};

class X86PackedWriteGuard {
public:
  explicit X86PackedWriteGuard(std::atomic<uint64_t> &packed) noexcept
      : lock_(packed) {
    lock_.lock();
  }
  ~X86PackedWriteGuard() { lock_.unlock(); }
  X86PackedWriteGuard(const X86PackedWriteGuard &) = delete;
  X86PackedWriteGuard &operator=(const X86PackedWriteGuard &) = delete;

private:
  X86PackedWriteLock lock_;
};

template <typename Read>
decltype(auto) read_x86_seqlock(std::atomic<uint32_t> &sequence, Read &&read) {
  Read &reader = read;
  while (true) {
    const uint32_t before = sequence.load(std::memory_order_acquire);
    if ((before & 1U) != 0U) {
      _mm_pause();
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
decltype(auto) read_x86_packed_seqlock(std::atomic<uint64_t> &packed,
                                       Read &&read) {
  Read &reader = read;
  while (true) {
    const uint64_t before = packed.load(std::memory_order_acquire);
    if ((static_cast<uint32_t>(before >> 32) & 1U) != 0U) {
      _mm_pause();
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
