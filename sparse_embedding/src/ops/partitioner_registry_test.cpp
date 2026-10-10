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

#include "sparse_embedding/src/ops/partitioner_registry.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct CustomRoundRobinPartitioner {
  uint32_t num_shards_;

  template <typename KeyT>
  void partition_span(std::span<const KeyT> keys,
                      std::span<uint32_t> out_shard_ids) const {
    for (size_t i = 0; i < keys.size(); ++i) {
      out_shard_ids[i] = static_cast<uint32_t>(i % num_shards_);
    }
  }

  template <typename KeyT>
  void filter_span(std::span<const KeyT> keys, uint32_t target_shard,
                   std::span<bool> out_mask) const {
    for (size_t i = 0; i < keys.size(); ++i) {
      out_mask[i] = (i % num_shards_) == target_shard;
    }
  }
};

} // namespace

int main() {
  auto &registry = se::ops::PartitionerRegistry::instance();

  // Test 1: Builtins are registered
  require(registry.is_registered(se::ops::kModuloPartitionerName),
          "modulo partitioner must be registered");
  require(registry.is_registered(se::ops::kHashPartitionerName),
          "hash partitioner must be registered");
  require(registry.is_registered(se::ops::kFastRangePartitionerName),
          "fast_range partitioner must be registered");
  require(!registry.is_registered("non_existent_partitioner"),
          "non existent partitioner must return false");

  // Test 2: Create builtins
  auto modulo_p = registry.create(se::ops::kModuloPartitionerName, 4);
  require(modulo_p.has_value(), "modulo proxy must have value");
  std::array<uint64_t, 4> keys{0, 1, 2, 3};
  std::array<uint32_t, 4> shards{};
  modulo_p->partition_span(std::span<const uint64_t>(keys),
                           std::span<uint32_t>(shards));
  require(shards[0] == 0 && shards[1] == 1 && shards[2] == 2 && shards[3] == 3,
          "modulo partition values must match");

  auto hash_p = registry.create(se::ops::kHashPartitionerName, 4);
  require(hash_p.has_value(), "hash proxy must have value");

  auto fast_range_p = registry.create(se::ops::kFastRangePartitionerName, 4);
  require(fast_range_p.has_value(), "fast_range proxy must have value");

  // Test 3: Invalid arguments throw
  bool caught_zero = false;
  try {
    registry.create(se::ops::kModuloPartitionerName, 0);
  } catch (const std::invalid_argument &) {
    caught_zero = true;
  }
  require(caught_zero, "num_partitions=0 must throw");

  bool caught_unknown = false;
  try {
    registry.create("unknown_partitioner", 4);
  } catch (const std::invalid_argument &) {
    caught_unknown = true;
  }
  require(caught_unknown, "unknown partitioner must throw");

  bool caught_null = false;
  try {
    registry.register_factory("null_factory", nullptr);
  } catch (const std::invalid_argument &) {
    caught_null = true;
  }
  require(caught_null, "null factory must throw");

  // Test 4: Dynamic custom registration
  const std::string custom_name = "test.custom_rr";
  registry.register_factory(
      custom_name,
      [](uint32_t num_shards,
         const std::unordered_map<std::string, std::string> &)
          -> se::ops::PartitionerProxy {
        return pro::make_proxy<se::ops::PartitionerFacade,
                               CustomRoundRobinPartitioner>(num_shards);
      });
  require(registry.is_registered(custom_name),
          "custom partitioner must be registered");

  auto custom_p = registry.create(custom_name, 2);
  require(custom_p.has_value(), "custom partitioner proxy must have value");

  std::array<int64_t, 4> custom_keys{100, 200, 300, 400};
  std::array<uint32_t, 4> custom_shards{};
  custom_p->partition_span(std::span<const int64_t>(custom_keys),
                           std::span<uint32_t>(custom_shards));
  require(custom_shards[0] == 0 && custom_shards[1] == 1 &&
              custom_shards[2] == 0 && custom_shards[3] == 1,
          "custom partition values must match round-robin");

  std::array<bool, 4> mask{};
  custom_p->filter_span(std::span<const int64_t>(custom_keys), 1,
                        std::span<bool>(mask));
  require(!mask[0] && mask[1] && !mask[2] && mask[3],
          "custom filter values must match target shard");

  std::cout << "All PartitionerRegistry C++ unit tests passed!" << std::endl;
  return 0;
}
