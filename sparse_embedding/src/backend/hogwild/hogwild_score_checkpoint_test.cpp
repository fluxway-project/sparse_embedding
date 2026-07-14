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

// Tests for HogwildEmbeddingStore: score integration, assign_scores,
// export_batch, evict_by_score, checkpoint save/load via proxy facades.

#include <cassert>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/checkpoint/checkpoint_interface.hpp"
#include "sparse_embedding/src/eviction/score_policy.hpp"

using namespace se;

// ============================================================================
// Helpers
// ============================================================================

static void check(bool cond, const char *msg) {
  if (!cond) {
    std::cerr << "[FAIL] " << msg << "\n";
    std::abort();
  }
}
#define CHECK(cond) check((cond), #cond)
#define CHECK_EQ(a, b) check((a) == (b), #a " == " #b)

// ============================================================================
// In-memory snapshot buffer — filled by export_batch visitor
// ============================================================================

struct MemBuffer {
  std::vector<uint64_t> keys;
  std::vector<float> values;
  std::vector<uint64_t> scores;
};

// Saves store contents into a MemBuffer using export_batch visitor.
template <typename StoreT>
static MemBuffer capture_export(StoreT &store, size_t batch_size = 65536) {
  MemBuffer buf;
  store.template export_batch<uint64_t>(
      pro::make_proxy<se::FunctionFacade<bool(size_t, std::span<const uint64_t>,
                                              std::span<const float>,
                                              std::span<const uint64_t>)>>(
          [&](size_t dim, std::span<const uint64_t> keys,
              std::span<const float> vals,
              std::span<const uint64_t> scores) -> bool {
            buf.keys.insert(buf.keys.end(), keys.begin(), keys.end());
            buf.values.insert(buf.values.end(), vals.begin(), vals.end());
            buf.scores.insert(buf.scores.end(), scores.begin(), scores.end());
            return true;
          }),
      batch_size);
  return buf;
}

// Loads a MemBuffer into a store using assign + assign_scores.
template <typename StoreT>
static void apply_load(StoreT &store, const MemBuffer &buf, size_t dim) {
  size_t n = buf.keys.size();
  std::vector<int64_t> row_lengths(n, static_cast<int64_t>(dim));
  store.template assign<uint64_t>(buf.keys, buf.values, row_lengths);
  if (!buf.scores.empty() && store.has_score_policy()) {
    store.template assign_scores<uint64_t>(buf.keys, buf.scores);
  }
}

// ============================================================================
// Factories
// ============================================================================

template <typename Policy>
static HogwildEmbeddingStore<uint64_t, float, float, Policy>
make_store(size_t dim = 4, size_t threads = 2) {
  return HogwildEmbeddingStore<uint64_t, float, float, Policy>(
      {dim}, threads, {}, 1 << 16, nullptr, Policy{});
}

// ============================================================================
// Test: score disabled (none) — score arena not allocated
// ============================================================================
static void test_no_score_memory() {
  auto store = make_store<NoScorePolicy>(4);
  CHECK(!store.has_score_policy());

  std::vector<uint64_t> keys = {0, 1, 2};
  std::vector<float> vals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int64_t> lens = {4, 4, 4};
  store.assign<uint64_t>(keys, vals, lens);
  CHECK_EQ(store.size(), 3u);
  std::cout
      << "  [OK] score_policy=none: arena allocated without score pages\n";
}

// ============================================================================
// Test: LFU score increments on forward_lookup
// ============================================================================
static void test_lfu_score_increments_on_lookup() {
  auto store = make_store<LfuScorePolicy>(4, 2);
  CHECK(store.has_score_policy());

  std::vector<uint64_t> keys = {10, 11};
  std::vector<float> vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> lens = {4, 4};
  store.assign<uint64_t>(keys, vals, lens);

  std::vector<float> out(8);
  std::vector<int64_t> out_lens = {4, 4};
  // 3 lookups on key=10
  for (int i = 0; i < 3; ++i)
    store.template forward_lookup<uint64_t>(
        std::span<const uint64_t>({keys[0]}), std::span<float>(out.data(), 4),
        std::span<int64_t>(out_lens.data(), 1));
  // 1 lookup on key=11
  store.template forward_lookup<uint64_t>(
      std::span<const uint64_t>({keys[1]}), std::span<float>(out.data(), 4),
      std::span<int64_t>(out_lens.data(), 1));

  auto buf = capture_export(store);
  CHECK(buf.keys.size() == 2u);

  std::unordered_map<uint64_t, uint64_t> key_score;
  for (size_t i = 0; i < buf.keys.size(); ++i)
    key_score[buf.keys[i]] = buf.scores[i];

  CHECK(key_score.at(10) == 3u);
  CHECK(key_score.at(11) == 1u);
  std::cout << "  [OK] LFU score increments on forward_lookup\n";
}

// ============================================================================
// Test: assign_scores sets scores
// ============================================================================
static void test_assign_scores() {
  auto store = make_store<ManualScorePolicy>(4, 2);

  std::vector<uint64_t> keys = {100, 200, 300};
  std::vector<float> vals(12, 1.0f);
  std::vector<int64_t> lens = {4, 4, 4};
  store.assign<uint64_t>(keys, vals, lens);

  std::vector<uint64_t> scores = {50, 100, 25};
  store.template assign_scores<uint64_t>(keys, scores);

  auto buf = capture_export(store);
  CHECK(buf.keys.size() == 3u);

  std::unordered_map<uint64_t, uint64_t> ks;
  for (size_t i = 0; i < buf.keys.size(); ++i)
    ks[buf.keys[i]] = buf.scores[i];

  CHECK(ks.at(100) == 50u);
  CHECK(ks.at(200) == 100u);
  CHECK(ks.at(300) == 25u);
  std::cout << "  [OK] assign_scores sets scores correctly\n";
}

// ============================================================================
// Test: assign_scores throws for no-score policy
// ============================================================================
static void test_assign_scores_requires_enabled() {
  auto store = make_store<NoScorePolicy>(4, 1);
  std::vector<uint64_t> keys = {1};
  std::vector<uint64_t> scores = {99};
  bool threw = false;
  try {
    store.template assign_scores<uint64_t>(keys, scores);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  CHECK(threw);
  std::cout << "  [OK] assign_scores throws for policy=none\n";
}

// ============================================================================
// Test: export_batch + assign roundtrip
// ============================================================================
static void test_checkpoint_roundtrip() {
  constexpr size_t kDim = 4;
  auto store = make_store<LfuScorePolicy>(kDim, 2);

  std::vector<uint64_t> keys = {1, 2, 3};
  std::vector<float> vals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int64_t> lens = {4, 4, 4};
  store.assign<uint64_t>(keys, vals, lens);

  std::vector<float> out(kDim);
  std::vector<int64_t> out_lens = {static_cast<int64_t>(kDim)};
  for (int i = 0; i < 2; ++i)
    store.template forward_lookup<uint64_t>(
        std::span<const uint64_t>({keys[0]}), std::span<float>(out),
        std::span<int64_t>(out_lens));

  auto buf = capture_export(store);
  CHECK(buf.keys.size() == 3u);
  CHECK(buf.scores.size() == 3u);
  CHECK(buf.values.size() == 3u * kDim);

  HogwildEmbeddingStore<uint64_t, float, float, LfuScorePolicy> store2(
      {kDim}, 2, {}, 1 << 16, nullptr, LfuScorePolicy{});
  apply_load(store2, buf, kDim);
  CHECK_EQ(store2.size(), 3u);

  std::vector<float> restored(12);
  std::vector<int64_t> lens2 = {4, 4, 4};
  store2.template forward_lookup<uint64_t>(keys, restored, lens2);
  for (size_t i = 0; i < 12; ++i)
    CHECK(restored[i] == vals[i]);

  std::cout << "  [OK] checkpoint roundtrip: 3 entries saved and restored\n";
}

// ============================================================================
// Test: incremental export — only export entries above min_score
// ============================================================================
static void test_incremental_export() {
  constexpr size_t kDim = 4;
  auto store = make_store<LfuScorePolicy>(kDim, 2);

  std::vector<uint64_t> keys = {10, 11, 12};
  std::vector<float> vals(12, 1.0f);
  std::vector<int64_t> lens = {4, 4, 4};
  store.assign<uint64_t>(keys, vals, lens);

  std::vector<float> out(kDim);
  std::vector<int64_t> out_lens = {static_cast<int64_t>(kDim)};
  for (int i = 0; i < 5; ++i)
    store.template forward_lookup<uint64_t>(
        std::span<const uint64_t>({keys[0]}), std::span<float>(out),
        std::span<int64_t>(out_lens));
  store.template forward_lookup<uint64_t>(std::span<const uint64_t>({keys[1]}),
                                          std::span<float>(out),
                                          std::span<int64_t>(out_lens));

  // Export only entries with score >= 3
  constexpr uint64_t kMinScore = 3;
  MemBuffer buf;
  store.template export_batch<uint64_t>(
      pro::make_proxy<se::FunctionFacade<bool(size_t, std::span<const uint64_t>,
                                              std::span<const float>,
                                              std::span<const uint64_t>)>>(
          [&](size_t dim, std::span<const uint64_t> ks,
              std::span<const float> vs, std::span<const uint64_t> ss) -> bool {
            for (size_t i = 0; i < ks.size(); ++i) {
              if (ss[i] >= kMinScore) {
                buf.keys.push_back(ks[i]);
                buf.values.insert(buf.values.end(), vs.data() + i * kDim,
                                  vs.data() + (i + 1) * kDim);
                buf.scores.push_back(ss[i]);
              }
            }
            return true;
          }),
      65536);

  CHECK(buf.keys.size() == 1u); // only key=10 (score=5)
  CHECK(buf.keys[0] == 10u);
  std::cout << "  [OK] incremental export filters by min_score\n";
}

// ============================================================================
// Test: evict_by_score removes cold entries
// ============================================================================
static void test_evict_by_score() {
  constexpr size_t kDim = 4;
  auto store = make_store<ManualScorePolicy>(kDim, 2);

  std::vector<uint64_t> keys = {1, 2, 3, 4, 5};
  std::vector<float> vals(5 * kDim, 1.0f);
  std::vector<int64_t> lens(5, kDim);
  store.assign<uint64_t>(keys, vals, lens);

  // Assign scores: key=1 has score=100, others have score=1
  std::vector<uint64_t> scores = {100, 1, 1, 1, 1};
  store.template assign_scores<uint64_t>(keys, scores);

  CHECK_EQ(store.size(), 5u);

  // Evict entries with score <= 1, max 3
  std::vector<uint64_t> evicted_keys(3);
  std::vector<float> evicted_vals(3 * kDim);
  size_t evicted_count = 0;
  auto ec = store.template evict_by_score<uint64_t>(
      1, 3, evicted_keys, evicted_vals, evicted_count);
  CHECK(!ec);
  CHECK(evicted_count == 3u);
  CHECK(store.size() == 2u);

  // key=1 must survive (score=100 > max_score=1)
  std::vector<uint8_t> mask(1);
  store.template contains<uint64_t>(std::span<const uint64_t>({1ULL}), mask);
  CHECK(mask[0] == 1);
  std::cout << "  [OK] evict_by_score evicts cold entries, spares hot key\n";
}

int main() {
  std::cout << "=== hogwild_score_checkpoint_test ===\n";
  test_no_score_memory();
  test_lfu_score_increments_on_lookup();
  test_assign_scores();
  test_assign_scores_requires_enabled();
  test_checkpoint_roundtrip();
  test_incremental_export();
  test_evict_by_score();
  std::cout << "=== ALL PASSED ===\n";
  return 0;
}
