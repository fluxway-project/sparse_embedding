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

#include <mutex>
#include <stdexcept>
#include <utility>

namespace se::ops {

PartitionerRegistry::PartitionerRegistry() {
  register_factory(
      kModuloPartitionerName,
      [](uint32_t num_partitions,
         const std::unordered_map<std::string, std::string> &)
          -> PartitionerProxy {
        return pro::make_proxy<PartitionerFacade, ModuloPartitioner>(
            num_partitions);
      });

  register_factory(kHashPartitionerName,
                   [](uint32_t num_partitions,
                      const std::unordered_map<std::string, std::string> &)
                       -> PartitionerProxy {
                     return pro::make_proxy<PartitionerFacade, HashPartitioner>(
                         num_partitions);
                   });

  register_factory(
      kFastRangePartitionerName,
      [](uint32_t num_partitions,
         const std::unordered_map<std::string, std::string> &)
          -> PartitionerProxy {
        return pro::make_proxy<PartitionerFacade, FastRangePartitioner>(
            num_partitions);
      });
}

PartitionerRegistry &PartitionerRegistry::instance() {
  static PartitionerRegistry registry;
  return registry;
}

void PartitionerRegistry::register_factory(std::string_view name,
                                           PartitionerFactory factory) {
  if (!factory) {
    throw std::invalid_argument("partitioner factory cannot be empty");
  }
  std::unique_lock lock(mutex_);
  factories_[std::string(name)] = std::move(factory);
}

PartitionerProxy PartitionerRegistry::create(
    std::string_view name, uint32_t num_partitions,
    const std::unordered_map<std::string, std::string> &parameters) const {
  if (num_partitions == 0) {
    throw std::invalid_argument("num_partitions must be > 0");
  }
  std::shared_lock lock(mutex_);
  auto it = factories_.find(std::string(name));
  if (it == factories_.end()) {
    throw std::invalid_argument("unknown partitioner: " + std::string(name));
  }
  return it->second(num_partitions, parameters);
}

bool PartitionerRegistry::is_registered(std::string_view name) const {
  std::shared_lock lock(mutex_);
  return factories_.find(std::string(name)) != factories_.end();
}

} // namespace se::ops
