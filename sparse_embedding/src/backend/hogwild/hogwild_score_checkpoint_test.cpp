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
// export_batch, evict_by_score, checkpoint save/load via export/import
// roundtrip.

#include <barrier>
#include <cassert>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <latch>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
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
      {dim}, {}, nullptr, Policy{},
      HogwildStoreOptions{.thread_budget = threads});
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

static void test_lru_clock_exhaustion_terminalizes_store() {
  auto store = make_store<LruScorePolicy>(4, 1);
  const std::vector<uint64_t> keys = {7};
  const std::vector<float> values = {1, 2, 3, 4};
  const std::vector<int64_t> lengths = {4};
  store.assign<uint64_t>(keys, values, lengths);
  testing::set_lru_clock_for_testing(store,
                                     std::numeric_limits<uint64_t>::max());

  std::vector<float> output(4);
  bool exhausted = false;
  try {
    store.forward_lookup<uint64_t>(keys, output, lengths);
  } catch (const std::overflow_error &) {
    exhausted = true;
  }
  CHECK(exhausted);
  CHECK(!store.is_operational());

  bool rejected = false;
  try {
    store.assign<uint64_t>(keys, values, lengths);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(rejected);
  rejected = false;
  try {
    (void)store.checkpoint_mutations();
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  CHECK(rejected);
  std::cout << "  [OK] LRU clock exhaustion terminalizes the store\n";
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
      {kDim}, {}, nullptr, LfuScorePolicy{},
      HogwildStoreOptions{.thread_budget = 2});
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
  std::vector<float> vals(5 * kDim);
  for (size_t row = 0; row < keys.size(); ++row)
    std::fill_n(vals.data() + row * kDim, kDim, static_cast<float>(keys[row]));
  std::vector<int64_t> lens(5, kDim);
  store.assign<uint64_t>(keys, vals, lens);

  // Assign scores: key=1 has score=100, others have score=1
  std::vector<uint64_t> scores = {100, 1, 1, 1, 1};
  store.template assign_scores<uint64_t>(keys, scores);

  CHECK_EQ(store.size(), 5u);

  const auto cut = store.capture_checkpoint(
      false, "", false, store.checkpoint_page_layout_generation(),
      pro::make_proxy<se::FunctionFacade<void(
          std::span<const se::CheckpointMutation>, uint64_t)>>(
          [](std::span<const se::CheckpointMutation>, uint64_t) {}),
      pro::make_proxy<se::FunctionFacade<bool(
          size_t, se::PageImageExport<float, float> &&)>>(
          [](size_t, se::PageImageExport<float, float> &&) { return true; }));
  store.advance_checkpoint_baseline(cut.mutation_watermark, "eviction-base",
                                    cut.capture_id, cut.journal_identity);

  // Evict entries with score <= 1, max 3
  std::vector<uint64_t> evicted_keys(3);
  std::vector<float> evicted_vals(3 * kDim);
  size_t evicted_count = 0;
  auto ec = store.template evict_by_score<uint64_t>(
      1, 3, evicted_keys, evicted_vals, evicted_count);
  CHECK(!ec);
  CHECK(evicted_count == 3u);
  CHECK(store.size() == 2u);
  const auto mutations = store.checkpoint_mutations();
  CHECK(mutations.size() == evicted_count);
  for (const auto &mutation : mutations)
    CHECK(mutation.deleted);
  for (size_t row = 0; row < evicted_count; ++row) {
    CHECK(std::ranges::all_of(
        std::span<const float>(evicted_vals.data() + row * kDim, kDim),
        [&](float value) {
          return value == static_cast<float>(evicted_keys[row]);
        }));
  }

  // key=1 must survive (score=100 > max_score=1)
  std::vector<uint8_t> mask(1);
  store.template contains<uint64_t>(std::span<const uint64_t>({1ULL}), mask);
  CHECK(mask[0] == 1);
  std::cout << "  [OK] evict_by_score evicts cold entries, spares hot key\n";
}

static void test_capture_serializes_mutations() {
  auto store = make_store<ManualScorePolicy>(4, 2);
  std::vector<uint64_t> keys{1};
  std::vector<float> values(4, 1);
  std::vector<int64_t> lengths{4};
  store.assign<uint64_t>(keys, values, lengths);
  std::latch exporting(1), resume_export(1), writing(1);
  std::vector<float> captured_values;
  auto capture = std::async(std::launch::async, [&] {
    return store.capture_checkpoint(
        false, "", false, store.checkpoint_page_layout_generation(),
        pro::make_proxy<se::FunctionFacade<void(
            std::span<const se::CheckpointMutation>, uint64_t)>>(
            [](std::span<const se::CheckpointMutation>, uint64_t) {}),
        pro::make_proxy<se::FunctionFacade<bool(
            size_t, se::PageImageExport<float, float> &&)>>(
            [&](size_t, se::PageImageExport<float, float> &&image) {
              captured_values = std::move(image.values);
              exporting.count_down();
              resume_export.wait();
              return true;
            }));
  });
  exporting.wait();
  auto mutation = std::async(std::launch::async, [&] {
    writing.count_down();
    std::vector<float> updated(4, 2);
    store.assign<uint64_t>(keys, updated, lengths);
  });
  writing.wait();
  CHECK(mutation.wait_for(std::chrono::milliseconds(50)) ==
        std::future_status::timeout);
  resume_export.count_down();
  const auto cut = capture.get();
  mutation.get();
  CHECK(std::ranges::all_of(captured_values,
                            [](float value) { return value == 1; }));
  store.advance_checkpoint_baseline(cut.mutation_watermark, "serialized-cut",
                                    cut.capture_id, cut.journal_identity);
  const auto dirty = store.checkpoint_mutations();
  CHECK(dirty.size() == 1);
  CHECK(dirty.front().mutation_sequence > cut.mutation_watermark);
}

struct BlockingCheckpointInitializer {
  std::latch *started;
  std::latch *resume;
  void initialize(std::span<float> values, uint64_t) {
    started->count_down();
    resume->wait();
    std::ranges::fill(values, 7);
  }
};

static void test_capture_drains_inflight_insert() {
  std::latch initializing(1), resume_insert(1), capturing(1);
  HogwildEmbeddingStore<uint64_t, float, float, ManualScorePolicy> store(
      {4},
      pro::make_proxy<InitializerFacade<float>>(
          BlockingCheckpointInitializer{&initializing, &resume_insert}),
      nullptr, ManualScorePolicy{}, HogwildStoreOptions{.thread_budget = 2});
  std::vector<uint64_t> keys{1};
  std::vector<float> output(4);
  std::vector<int64_t> lengths{4};
  auto insert = std::async(std::launch::async, [&] {
    store.lookup_or_insert<uint64_t>(keys, output, lengths);
  });
  initializing.wait();
  std::vector<float> captured_values;
  auto capture = std::async(std::launch::async, [&] {
    capturing.count_down();
    return store.capture_checkpoint(
        false, "", false, store.checkpoint_page_layout_generation(),
        pro::make_proxy<se::FunctionFacade<void(
            std::span<const se::CheckpointMutation>, uint64_t)>>(
            [](std::span<const se::CheckpointMutation>, uint64_t) {}),
        pro::make_proxy<se::FunctionFacade<bool(
            size_t, se::PageImageExport<float, float> &&)>>(
            [&](size_t, se::PageImageExport<float, float> &&image) {
              captured_values = std::move(image.values);
              return true;
            }));
  });
  capturing.wait();
  CHECK(capture.wait_for(std::chrono::milliseconds(50)) ==
        std::future_status::timeout);
  resume_insert.count_down();
  insert.get();
  const auto cut = capture.get();
  CHECK(captured_values.size() == 4);
  CHECK(std::ranges::all_of(captured_values,
                            [](float value) { return value == 7; }));
  store.advance_checkpoint_baseline(cut.mutation_watermark, "drained-cut",
                                    cut.capture_id, cut.journal_identity);
  CHECK(store.checkpoint_mutations().empty());
}

static void test_dirty_capture_visits_only_requested_keys() {
  HogwildEmbeddingStore<uint64_t, float, float, ManualScorePolicy> store(
      {4}, {}, nullptr, ManualScorePolicy{},
      HogwildStoreOptions{
          .thread_budget = 2,
          .storage_tuning = {.entries_per_page = 4096,
                             .max_pages = 1024,
                             .calibrate_page_geometry = false}});
  std::vector<uint64_t> keys(20000);
  std::iota(keys.begin(), keys.end(), uint64_t{0});
  std::vector<float> values(keys.size() * 4, 1);
  std::vector<int64_t> lengths(keys.size(), 4);
  store.assign<uint64_t>(keys, values, lengths);
  auto prepare = pro::make_proxy<se::FunctionFacade<void(
      std::span<const se::CheckpointMutation>, uint64_t)>>(
      [](std::span<const se::CheckpointMutation>, uint64_t) {});
  auto visitor = pro::make_proxy<
      se::FunctionFacade<bool(size_t, se::PageImageExport<float, float> &&)>>(
      [](size_t, se::PageImageExport<float, float> &&) { return true; });
  const auto full = store.capture_checkpoint(
      false, "", false, store.checkpoint_page_layout_generation(),
      std::move(prepare), std::move(visitor));
  store.advance_checkpoint_baseline(full.mutation_watermark, "dirty-base",
                                    full.capture_id, full.journal_identity);
  std::vector<uint64_t> changed{keys.back()};
  std::vector<float> updated(4, 2);
  std::vector<int64_t> changed_lengths{4};
  store.assign<uint64_t>(changed, updated, changed_lengths);
  size_t pages = 0, entries = 0;
  const auto delta = store.capture_checkpoint(
      true, "dirty-base", false, store.checkpoint_page_layout_generation(),
      pro::make_proxy<se::FunctionFacade<void(
          std::span<const se::CheckpointMutation>, uint64_t)>>(
          [&](std::span<const se::CheckpointMutation> mutations, uint64_t) {
            CHECK(mutations.size() == 1);
            CHECK(mutations.front().key == changed.front());
          }),
      pro::make_proxy<se::FunctionFacade<bool(
          size_t, se::PageImageExport<float, float> &&)>>(
          [&](size_t, se::PageImageExport<float, float> &&image) {
            ++pages;
            entries += image.keys.size();
            CHECK(image.keys.front() == changed.front());
            return true;
          }));
  CHECK(pages == 1);
  CHECK(entries == 1);
  store.cancel_checkpoint_capture(delta.capture_id, delta.journal_identity);
  CHECK(store.checkpoint_mutations().size() == 1);
}

static void test_journal_identity_and_acknowledgement_lifecycle() {
  auto store = make_store<ManualScorePolicy>(4, 4);
  const std::vector<uint64_t> keys{1, 2};
  const std::vector<int64_t> lengths{4, 4};
  const std::vector<float> values(8, 1);
  store.assign<uint64_t>(keys, values, lengths);
  auto capture = [&](bool incremental, std::string parent) {
    return store.capture_checkpoint(
        incremental, std::move(parent), false,
        store.checkpoint_page_layout_generation(),
        pro::make_proxy<se::FunctionFacade<void(
            std::span<const se::CheckpointMutation>, uint64_t)>>(
            [](std::span<const se::CheckpointMutation>, uint64_t) {}),
        pro::make_proxy<se::FunctionFacade<bool(
            size_t, se::PageImageExport<float, float> &&)>>(
            [](size_t, se::PageImageExport<float, float> &&) { return true; }));
  };
  auto acknowledge = [&](const CheckpointCaptureCut &cut,
                         std::string baseline) {
    store.advance_checkpoint_baseline(cut.mutation_watermark,
                                      std::move(baseline), cut.capture_id,
                                      cut.journal_identity);
  };
  acknowledge(capture(false, ""), "base");
  for (size_t generation = 0; generation < 32; ++generation) {
    store.erase_dimension<uint64_t>(4, keys);
    store.assign<uint64_t>(keys, values, lengths);
    const auto live = store.checkpoint_mutations();
    CHECK(live.size() == keys.size());
    CHECK(std::ranges::none_of(live, &CheckpointMutation::deleted));
    const auto cut = capture(true, "base");
    const std::vector<uint64_t> changed{1};
    const std::vector<int64_t> changed_lengths{4};
    const std::vector<float> changed_values(4, 2);
    store.assign<uint64_t>(changed, changed_values, changed_lengths);
    acknowledge(cut, "next");
    const auto retained = store.checkpoint_mutations();
    CHECK(retained.size() == 1);
    CHECK(retained.front().key == 1);
    CHECK(retained.front().mutation_sequence > cut.mutation_watermark);
    const auto cancelled = capture(true, "next");
    store.cancel_checkpoint_capture(cancelled.capture_id,
                                    cancelled.journal_identity);
    CHECK(store.checkpoint_mutations().size() == 1);
    acknowledge(capture(true, "next"), "base");
    CHECK(store.checkpoint_mutations().empty());
  }
  store.clear();
  CHECK(store.checkpoint_mutations().size() == keys.size());
  CHECK(std::ranges::all_of(store.checkpoint_mutations(),
                            &CheckpointMutation::deleted));
  acknowledge(capture(true, "base"), "empty");
  CHECK(store.checkpoint_mutations().empty());
  store.assign<uint64_t>(keys, values, lengths);
  CHECK(store.checkpoint_mutations().size() == keys.size());
}

static void test_shared_key_score_publication() {
  auto store = make_store<LfuScorePolicy>(4, 4);
  const std::vector<uint64_t> keys{1};
  const std::vector<int64_t> lengths{4};
  const std::vector<float> values(4, 1);
  store.assign<uint64_t>(keys, values, lengths);
  constexpr size_t clients = 8, accesses = 1024;
  std::barrier start(static_cast<std::ptrdiff_t>(clients));
  std::vector<std::future<void>> readers;
  for (size_t client = 0; client < clients; ++client) {
    readers.push_back(std::async(std::launch::async, [&] {
      std::vector<float> output(4);
      start.arrive_and_wait();
      for (size_t access = 0; access < accesses; ++access)
        store.forward_lookup<uint64_t>(keys, output, lengths);
    }));
  }
  for (auto &reader : readers)
    reader.get();
  const auto state = capture_export(store);
  CHECK_EQ(state.scores.front(), clients * accesses);
  const auto mutations = store.checkpoint_mutations();
  CHECK_EQ(mutations.size(), 1u);
  CHECK_EQ(mutations.front().mutation_sequence,
           store.checkpoint_mutation_watermark());
}

int main() {
  std::cout << "=== hogwild_score_checkpoint_test ===\n";
  test_no_score_memory();
  test_lfu_score_increments_on_lookup();
  test_lru_clock_exhaustion_terminalizes_store();
  test_assign_scores();
  test_assign_scores_requires_enabled();
  test_checkpoint_roundtrip();
  test_incremental_export();
  test_evict_by_score();
  test_capture_serializes_mutations();
  test_capture_drains_inflight_insert();
  test_dirty_capture_visits_only_requested_keys();
  test_journal_identity_and_acknowledgement_lifecycle();
  test_shared_key_score_publication();
  std::cout << "=== ALL PASSED ===\n";
  return 0;
}
