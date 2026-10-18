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

#ifndef SPARSE_EMBEDDING_PARTITIONER_REGISTRY_HPP_
#define SPARSE_EMBEDDING_PARTITIONER_REGISTRY_HPP_

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include "sparse_embedding/src/ops/partitioner.hpp"

namespace se::ops {

inline constexpr std::string_view kModuloPartitionerName =
    "sparse_embedding.modulo";
inline constexpr std::string_view kHashPartitionerName =
    "sparse_embedding.hash";
inline constexpr std::string_view kFastRangePartitionerName =
    "sparse_embedding.fast_range";

inline uint32_t checked_partition_count(uint64_t count) {
  if (count == 0 || count > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument(
        "partition_count must be within [1, UINT32_MAX]");
  return static_cast<uint32_t>(count);
}

using PartitionerProxy = pro::proxy<PartitionerFacade>;
using PartitionerFactory = std::function<PartitionerProxy(
    uint32_t num_partitions,
    const std::unordered_map<std::string, std::string> &parameters)>;

class PartitionerRegistry {
  std::unordered_map<std::string, PartitionerFactory> factories_;
  mutable std::shared_mutex mutex_;

  PartitionerRegistry();

public:
  static PartitionerRegistry &instance();

  void register_factory(std::string_view name, PartitionerFactory factory);

  PartitionerProxy create(std::string_view name, uint32_t num_partitions,
                          const std::unordered_map<std::string, std::string>
                              &parameters = {}) const;

  bool is_registered(std::string_view name) const;
};

} // namespace se::ops

#endif // SPARSE_EMBEDDING_PARTITIONER_REGISTRY_HPP_
