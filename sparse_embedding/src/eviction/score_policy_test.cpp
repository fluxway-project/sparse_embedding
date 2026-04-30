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
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
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
  p.on_access(&score);
  CHECK(score == 0);
  std::cout << "  [OK] NoScorePolicy\n";
}

static void test_lfu_policy() {
  LfuScorePolicy p;
  CHECK(LfuScorePolicy::kHasScore);
  uint64_t score = 0;
  p.on_access(&score);
  CHECK(score == 1);
  p.on_access(&score);
  p.on_access(&score);
  CHECK(score == 3);
  std::cout << "  [OK] LfuScorePolicy\n";
}

static void test_lru_policy() {
  LruScorePolicy p;
  CHECK(LruScorePolicy::kHasScore);
  uint64_t score1 = 0;
  p.on_access(&score1);
  CHECK(score1 == 1);
  uint64_t score2 = 0;
  p.on_access(&score2);
  CHECK(score2 == 2);
  std::cout << "  [OK] LruScorePolicy\n";
}

static void test_manual_policy() {
  ManualScorePolicy p;
  CHECK(ManualScorePolicy::kHasScore);
  uint64_t score = 42;
  p.on_access(&score);
  CHECK(score == 42);
  std::cout << "  [OK] ManualScorePolicy\n";
}

int main() {
  std::cout << "=== score_policy_test ===\n";
  test_no_score_policy();
  test_lfu_policy();
  test_lru_policy();
  test_manual_policy();
  std::cout << "=== ALL PASSED ===\n";
  return 0;
}