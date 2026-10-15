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

// Tests for ScorePolicy strategy: kHasScore, on_access() semantics.

#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>

#include "sparse_embedding/src/eviction/score_policy.hpp"

using namespace se;

static void check(bool cond, const char *msg) {
  if (!cond) {
    std::cerr << "[FAIL] " << msg << "\n";
    std::abort();
  }
}
#define CHECK(cond) check((cond), #cond)

static void test_no_score_policy() {
  NoScorePolicy p;
  CHECK(!NoScorePolicy::kHasScore);
  uint64_t score = 0;
  CHECK(p.on_access(&score) == ScoreAccessResult::kUnchanged);
  CHECK(score == 0);
  std::cout << "  [OK] NoScorePolicy\n";
}

static void test_lfu_policy() {
  LfuScorePolicy p;
  CHECK(LfuScorePolicy::kHasScore);
  uint64_t score = 0;
  CHECK(p.on_access(&score) == ScoreAccessResult::kUpdated);
  CHECK(score == 1);
  CHECK(p.on_access(&score) == ScoreAccessResult::kUpdated);
  CHECK(p.on_access(&score) == ScoreAccessResult::kUpdated);
  CHECK(score == 3);
  std::cout << "  [OK] LfuScorePolicy\n";
}

static void test_lfu_saturates() {
  LfuScorePolicy p;
  uint64_t score = std::numeric_limits<uint64_t>::max();
  CHECK(p.on_access(&score) == ScoreAccessResult::kUnchanged);
  CHECK(score == std::numeric_limits<uint64_t>::max());
  std::cout << "  [OK] LfuScorePolicy saturation\n";
}

static void test_lru_policy() {
  LruScorePolicy p;
  CHECK(LruScorePolicy::kHasScore);
  uint64_t score1 = 0;
  CHECK(p.on_access(&score1) == ScoreAccessResult::kUpdated);
  CHECK(score1 == 1);
  uint64_t score2 = 0;
  CHECK(p.on_access(&score2) == ScoreAccessResult::kUpdated);
  CHECK(score2 == 2);
  score1 = std::numeric_limits<uint64_t>::max();
  CHECK(p.on_access(&score1) == ScoreAccessResult::kUnchanged);
  CHECK(score1 == std::numeric_limits<uint64_t>::max());
  std::cout << "  [OK] LruScorePolicy\n";
}

static void test_lru_clock_exhaustion() {
  LruScorePolicy p;
  *p.clock = std::numeric_limits<uint64_t>::max();
  uint64_t score = 10;
  CHECK(p.on_access(&score) == ScoreAccessResult::kClockExhausted);
  CHECK(score == 10);
  std::cout << "  [OK] LruScorePolicy clock exhaustion\n";
}

static void test_lru_concurrent_publication() {
  LruScorePolicy policy;
  uint64_t score = 0;
  constexpr size_t clients = 8;
  constexpr size_t accesses = 1024;
  std::barrier start(static_cast<std::ptrdiff_t>(clients));
  std::vector<std::jthread> readers;
  readers.reserve(clients);
  for (size_t client = 0; client < clients; ++client) {
    readers.emplace_back([&] {
      start.arrive_and_wait();
      for (size_t access = 0; access < accesses; ++access)
        policy.on_access(&score);
    });
  }
  readers.clear();
  CHECK(score == clients * accesses);
  std::cout << "  [OK] concurrent LruScorePolicy publication\n";
}

static void test_manual_policy() {
  ManualScorePolicy p;
  CHECK(ManualScorePolicy::kHasScore);
  uint64_t score = 42;
  CHECK(p.on_access(&score) == ScoreAccessResult::kUnchanged);
  CHECK(score == 42);
  std::cout << "  [OK] ManualScorePolicy\n";
}

int main() {
  std::cout << "=== score_policy_test ===\n";
  test_no_score_policy();
  test_lfu_policy();
  test_lfu_saturates();
  test_lru_policy();
  test_lru_clock_exhaustion();
  test_lru_concurrent_publication();
  test_manual_policy();
  std::cout << "=== ALL PASSED ===\n";
  return 0;
}
