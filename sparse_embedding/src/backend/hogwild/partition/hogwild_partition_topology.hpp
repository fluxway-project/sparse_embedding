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

#include "sparse_embedding/src/storage/topology/storage_resource_tuner.hpp"
#include "sparse_embedding/src/storage/topology/storage_topology.hpp"

namespace se {

class HogwildPartitionTopology {
public:
  static HogwildPartitionTopology derive(size_t thread_budget,
                                         LocalityPolicy locality) {
    auto topo =
        storage::StorageResourceTuner::derive(storage::TopologyTuneRequest{
            .thread_budget = thread_budget,
            .locality = locality,
        });
    return HogwildPartitionTopology(topo);
  }

  explicit HogwildPartitionTopology(const storage::StorageTopology &topo)
      : topology_(topo) {}

  size_t execution_workers() const noexcept {
    return topology_.execution_workers();
  }
  size_t partition_count() const noexcept { return topology_.shard_count(); }
  size_t shard_mask() const noexcept { return topology_.shard_mask(); }
  bool numa_execution() const noexcept { return topology_.numa_execution(); }
  bool numa_memory() const noexcept { return topology_.numa_memory(); }
  const storage::StorageTopology &storage_topology() const noexcept {
    return topology_;
  }

private:
  storage::StorageTopology topology_;
};

} // namespace se
