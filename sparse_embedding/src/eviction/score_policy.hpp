// Copyright 2026 The Fluxway Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#pragma once

#ifndef SPARSE_EMBEDDING_SCORE_POLICY_HPP_
#define SPARSE_EMBEDDING_SCORE_POLICY_HPP_

#include <atomic>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>

namespace se {

enum class ScoreAccessResult : uint8_t {
  kUnchanged = 0,
  kUpdated = 1,
  kClockExhausted = 2,
};

enum class ScorePolicyKind : uint8_t { kNone, kLfu, kLru, kManual };

namespace detail {

inline ScoreAccessResult increment_lfu(uint64_t *score_ptr) noexcept {
  std::atomic_ref<uint64_t> score(*score_ptr);
  uint64_t observed = score.load(std::memory_order_relaxed);
  while (observed != std::numeric_limits<uint64_t>::max()) {
    if (score.compare_exchange_weak(observed, observed + 1,
                                    std::memory_order_relaxed,
                                    std::memory_order_relaxed))
      return ScoreAccessResult::kUpdated;
  }
  return ScoreAccessResult::kUnchanged;
}

inline ScoreAccessResult publish_lru(std::atomic<uint64_t> &clock,
                                     uint64_t *score_ptr) noexcept {
  uint64_t observed_clock = clock.load(std::memory_order_relaxed);
  uint64_t timestamp = 0;
  for (;;) {
    if (observed_clock == std::numeric_limits<uint64_t>::max()) {
      return ScoreAccessResult::kClockExhausted;
    }
    if (clock.compare_exchange_weak(observed_clock, observed_clock + 1,
                                    std::memory_order_relaxed,
                                    std::memory_order_relaxed)) {
      timestamp = observed_clock;
      break;
    }
  }
  std::atomic_ref<uint64_t> score(*score_ptr);
  uint64_t observed = score.load(std::memory_order_relaxed);
  while (observed < timestamp) {
    if (score.compare_exchange_weak(observed, timestamp,
                                    std::memory_order_relaxed,
                                    std::memory_order_relaxed))
      return ScoreAccessResult::kUpdated;
  }
  return ScoreAccessResult::kUnchanged;
}

} // namespace detail

class ScorePolicyRuntime {
public:
  explicit ScorePolicyRuntime(ScorePolicyKind kind)
      : kind_(kind), clock_(kind == ScorePolicyKind::kLru
                                ? std::make_unique<std::atomic<uint64_t>>(1)
                                : nullptr) {}

  bool enabled() const noexcept { return kind_ != ScorePolicyKind::kNone; }

  ScoreAccessResult on_access(uint64_t *score_ptr) noexcept {
    switch (kind_) {
    case ScorePolicyKind::kNone:
    case ScorePolicyKind::kManual:
      return ScoreAccessResult::kUnchanged;
    case ScorePolicyKind::kLfu:
      return detail::increment_lfu(score_ptr);
    case ScorePolicyKind::kLru:
      return detail::publish_lru(*clock_, score_ptr);
    }
    std::terminate();
  }

#ifdef SPARSE_EMBEDDING_TEST_HOOKS
  void set_lru_clock_for_testing(uint64_t value) {
    if (!clock_)
      throw std::logic_error(
          "LRU clock injection requires the LRU score policy");
    clock_->store(value, std::memory_order_relaxed);
  }
#endif

private:
  ScorePolicyKind kind_;
  std::unique_ptr<std::atomic<uint64_t>> clock_;
};

// ---------------------------------------------------------------------------
// Built-in policies (plain structs — zero virtual overhead, designed for
// template-based static polymorphism in the hot path).
// ---------------------------------------------------------------------------

struct NoScorePolicy {
  static constexpr bool kHasScore = false;
  inline ScoreAccessResult on_access(uint64_t *) noexcept {
    return ScoreAccessResult::kUnchanged;
  }
};

// LFU: monotonically incrementing, saturating access counter. Relaxed ordering
// is sufficient because the score is a standalone heuristic/training input;
// the atomic read-modify-write preserves every concurrent access.
struct LfuScorePolicy {
  static constexpr bool kHasScore = true;
  inline ScoreAccessResult on_access(uint64_t *score_ptr) noexcept {
    return detail::increment_lfu(score_ptr);
  }
};

// LRU: logical clock per-store shared across all threads via atomic.
// Publication is monotonic even when a delayed thread observes an older
// timestamp.
struct LruScorePolicy {
  static constexpr bool kHasScore = true;
  std::unique_ptr<std::atomic<uint64_t>> clock;
  LruScorePolicy() : clock(std::make_unique<std::atomic<uint64_t>>(1)) {}
  inline ScoreAccessResult on_access(uint64_t *score_ptr) noexcept {
    return detail::publish_lru(*clock, score_ptr);
  }
};

// Manual: on_access is a no-op; score set exclusively via assign_scores().
struct ManualScorePolicy {
  static constexpr bool kHasScore = true;
  inline ScoreAccessResult on_access(uint64_t *) noexcept {
    return ScoreAccessResult::kUnchanged;
  }
};

inline constexpr ScorePolicyKind score_policy_kind(NoScorePolicy) noexcept {
  return ScorePolicyKind::kNone;
}

inline constexpr ScorePolicyKind score_policy_kind(LfuScorePolicy) noexcept {
  return ScorePolicyKind::kLfu;
}

inline constexpr ScorePolicyKind
score_policy_kind(const LruScorePolicy &) noexcept {
  return ScorePolicyKind::kLru;
}

inline constexpr ScorePolicyKind score_policy_kind(ManualScorePolicy) noexcept {
  return ScorePolicyKind::kManual;
}

inline constexpr ScorePolicyKind
score_policy_kind(ScorePolicyKind kind) noexcept {
  return kind;
}

} // namespace se

#endif // SPARSE_EMBEDDING_SCORE_POLICY_HPP_
