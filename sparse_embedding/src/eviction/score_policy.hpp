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
#include <memory>

namespace se {

// ---------------------------------------------------------------------------
// Built-in policies (plain structs — zero virtual overhead, designed for
// template-based static polymorphism in the hot path).
// ---------------------------------------------------------------------------

struct NoScorePolicy {
  static constexpr bool kHasScore = false;
  inline void on_access(uint64_t *) {}
};

// LFU: monotonically incrementing access counter.
// NOTE: We use std::atomic_ref with memory_order_relaxed to safely increment
// the score without incurring hardware lock prefix overhead (which is critical
// for Hogwild performance), while strictly avoiding C++ Undefined Behavior (UB)
// caused by data races.
struct LfuScorePolicy {
  static constexpr bool kHasScore = true;
  inline void on_access(uint64_t *score_ptr) {
    std::atomic_ref<uint64_t> a(*score_ptr);
    a.store(a.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
  }
};

// LRU: logical clock per-store shared across all threads via atomic.
// Store passes a pointer to its own logical_clock_ field.
// NOTE: Similar to LFU, we use std::atomic_ref with memory_order_relaxed to
// write the score, ensuring maximum performance without undefined behavior.
struct LruScorePolicy {
  static constexpr bool kHasScore = true;
  std::unique_ptr<std::atomic<uint64_t>> clock;
  LruScorePolicy() : clock(std::make_unique<std::atomic<uint64_t>>(1)) {}
  inline void on_access(uint64_t *score_ptr) {
    std::atomic_ref<uint64_t>(*score_ptr)
        .store(clock->fetch_add(1, std::memory_order_relaxed),
               std::memory_order_relaxed);
  }
};

// Manual: on_access is a no-op; score set exclusively via assign_scores().
struct ManualScorePolicy {
  static constexpr bool kHasScore = true;
  inline void on_access(uint64_t *) {}
};

} // namespace se

#endif // SPARSE_EMBEDDING_SCORE_POLICY_HPP_
