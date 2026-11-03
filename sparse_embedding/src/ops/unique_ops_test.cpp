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

#include "sparse_embedding/src/ops/unique_ops.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define TEST_ASSERT(cond)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "Assertion failed: " #cond " at " << __FILE__ << ":"        \
                << __LINE__ << std::endl;                                      \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

struct ReferenceEntry {
  int32_t first_idx = -1;
  int32_t count = 0;
};

template <typename KeyT>
void verify_unique_result(const std::vector<KeyT> &keys,
                          const se::ops::UniqueResult<KeyT> &res,
                          bool check_first, bool check_counts, bool check_inv) {
  const size_t n = keys.size();
  if (n == 0) {
    TEST_ASSERT(res.unique_count == 0);
    TEST_ASSERT(res.unique_keys == nullptr);
    return;
  }
  TEST_ASSERT(res.unique_keys != nullptr);

  // 1. Build independent linear O(n) reference map
  std::unordered_map<KeyT, ReferenceEntry> ref;
  for (size_t i = 0; i < n; ++i) {
    auto &entry = ref[keys[i]];
    if (entry.count == 0) {
      entry.first_idx = static_cast<int32_t>(i);
    }
    ++entry.count;
  }
  TEST_ASSERT(res.unique_count == ref.size());

  // 2. Exact reconstruction from inverse indices
  if (check_inv) {
    TEST_ASSERT(res.inverse_count == n);
    TEST_ASSERT(res.inverse_indices != nullptr);
    for (size_t i = 0; i < n; ++i) {
      const int32_t uid = res.inverse_indices[i];
      TEST_ASSERT(uid >= 0 && static_cast<size_t>(uid) < res.unique_count);
      TEST_ASSERT(res.unique_keys[uid] == keys[i]);
    }
  }

  // 3. Verify first indices and counts in linear O(unique_count)
  if (check_first) {
    TEST_ASSERT(res.first_indices != nullptr);
  }
  if (check_counts) {
    TEST_ASSERT(res.counts != nullptr);
  }

  for (size_t u = 0; u < res.unique_count; ++u) {
    const KeyT k = res.unique_keys[u];
    auto it = ref.find(k);
    TEST_ASSERT(it != ref.end());
    if (check_first) {
      TEST_ASSERT(res.first_indices[u] == it->second.first_idx);
    }
    if (check_counts) {
      TEST_ASSERT(res.counts[u] == it->second.count);
    }
  }
}

template <typename KeyT> void run_suite_for_type() {
  std::mt19937_64 rng(1337);

  // 1. Edge sizes covering small, boundary, and large radix paths
  for (size_t n : {0UL, 1UL, 2UL, 17UL, 256UL, 16384UL, 32768UL, 65536UL}) {
    std::vector<KeyT> keys(n);
    const size_t card = std::max(size_t{1}, n / 4);
    for (size_t i = 0; i < n; ++i) {
      keys[i] = static_cast<KeyT>(rng() % card);
    }
    for (size_t threads : {0UL, 1UL, 2UL, 4UL, 8UL}) {
      auto res = se::ops::unique_keys_inverse<KeyT>(
          keys, /*compute_first_indices=*/true, /*compute_counts=*/true,
          threads, /*compute_inverse=*/true);
      verify_unique_result(keys, res, true, true, true);
    }
  }

  // 2. All identical keys
  {
    std::vector<KeyT> keys(40000, static_cast<KeyT>(42));
    for (size_t threads : {0UL, 1UL, 2UL, 4UL}) {
      auto res =
          se::ops::unique_keys_inverse<KeyT>(keys, true, true, threads, true);
      TEST_ASSERT(res.unique_count == 1);
      TEST_ASSERT(res.unique_keys[0] == static_cast<KeyT>(42));
      verify_unique_result(keys, res, true, true, true);
    }
  }

  // 3. Alternating patterns (e.g. 2, 1, 2, 1, ...)
  {
    std::vector<KeyT> keys(32768);
    for (size_t i = 0; i < keys.size(); ++i) {
      keys[i] = (i % 2 == 0) ? static_cast<KeyT>(2) : static_cast<KeyT>(1);
    }
    for (size_t threads : {0UL, 1UL, 2UL}) {
      auto res =
          se::ops::unique_keys_inverse<KeyT>(keys, true, true, threads, true);
      TEST_ASSERT(res.unique_count == 2);
      verify_unique_result(keys, res, true, true, true);
    }
  }

  // 4. Flags combinations
  {
    std::vector<KeyT> keys(35000);
    for (size_t i = 0; i < keys.size(); ++i) {
      keys[i] = static_cast<KeyT>(rng() % 500);
    }
    // inverse=false, first=true, counts=true
    auto r1 = se::ops::unique_keys_inverse<KeyT>(keys, true, true, 4, false);
    TEST_ASSERT(r1.inverse_indices == nullptr);
    verify_unique_result(keys, r1, true, true, false);

    // inverse=true, first=false, counts=false
    auto r2 = se::ops::unique_keys_inverse<KeyT>(keys, false, false, 4, true);
    TEST_ASSERT(r2.first_indices == nullptr);
    TEST_ASSERT(r2.counts == nullptr);
    verify_unique_result(keys, r2, false, false, true);

    // inverse=false, first=false, counts=false
    auto r3 = se::ops::unique_keys_inverse<KeyT>(keys, false, false, 4, false);
    TEST_ASSERT(r3.inverse_indices == nullptr);
    TEST_ASSERT(r3.first_indices == nullptr);
    TEST_ASSERT(r3.counts == nullptr);
    TEST_ASSERT(r3.unique_count == 500);
  }
}

int main() {
  std::cout << "Testing unique_ops for int32_t...\n";
  run_suite_for_type<int32_t>();

  std::cout << "Testing unique_ops for uint32_t...\n";
  run_suite_for_type<uint32_t>();

  std::cout << "Testing unique_ops for int64_t...\n";
  run_suite_for_type<int64_t>();

  std::cout << "Testing unique_ops for uint64_t...\n";
  run_suite_for_type<uint64_t>();

  // Size limit validation
  {
    std::cout << "Testing size contract limit exception...\n";
    se::ops::validate_unique_input_size(INT32_MAX);
    bool caught = false;
    try {
      const size_t fake_huge_size = static_cast<size_t>(INT32_MAX) + 100UL;
      se::ops::validate_unique_input_size(fake_huge_size);
    } catch (const std::invalid_argument &) {
      caught = true;
    }
    TEST_ASSERT(caught);
  }

  std::cout
      << "All unique_ops tests passed successfully with active assertions!\n";
  return 0;
}
