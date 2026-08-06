#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>

#include "sparse_embedding/src/execution/dispatch_budget.hpp"
#include "sparse_embedding/src/execution/locality_policy.hpp"
#include "sparse_embedding/src/execution/topology/thread_affinity.hpp"

namespace se {

class HogwildPartitionTopology {
public:
  static HogwildPartitionTopology derive(size_t thread_budget,
                                         LocalityPolicy locality) {
    const size_t automatic = std::max(1u, std::thread::hardware_concurrency());
    const size_t execution_workers =
        normalize_thread_budget(thread_budget, automatic);
    if (execution_workers > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("execution thread budget exceeds uint32");
    const bool numa_execution = locality.execution == LocalityMode::kNuma;
    const bool numa_memory = locality.memory == LocalityMode::kNuma;
    const bool numa = numa_execution || numa_memory;
    if (numa && !NumaTopology::available())
      throw std::runtime_error("NUMA locality requires libnuma support");
    const size_t nodes = numa ? NumaTopology::node_count() : 1;
    if (nodes == 0 || !std::has_single_bit(nodes))
      throw std::invalid_argument(
          "NUMA topology node count must be a power of two");

    // Partitions are a storage topology resource. They are derived once from
    // the execution budget and topology, then remain immutable on the hot path.
    size_t partitions = 1;
    while (partitions < execution_workers) {
      if (partitions > std::numeric_limits<size_t>::max() / 2)
        throw std::overflow_error("partition count overflows");
      partitions <<= 1;
    }
    if (partitions > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("partition count exceeds uint32");
    if (numa && (partitions < nodes || partitions % nodes != 0))
      throw std::invalid_argument(
          "NUMA locality requires partitions divisible by NUMA nodes");
    if (partitions > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("partition count exceeds uint32");
    return HogwildPartitionTopology(execution_workers, partitions,
                                    numa_execution, numa_memory,
                                    partitions - 1);
  }

  size_t execution_workers() const noexcept { return execution_workers_; }
  size_t partition_count() const noexcept { return partition_count_; }
  size_t shard_mask() const noexcept { return shard_mask_; }
  bool numa_execution() const noexcept { return numa_execution_; }
  bool numa_memory() const noexcept { return numa_memory_; }

private:
  HogwildPartitionTopology(size_t execution_workers, size_t partition_count,
                           bool numa_execution, bool numa_memory,
                           size_t shard_mask)
      : execution_workers_(execution_workers),
        partition_count_(partition_count), numa_execution_(numa_execution),
        numa_memory_(numa_memory), shard_mask_(shard_mask) {}

  size_t execution_workers_;
  size_t partition_count_;
  bool numa_execution_;
  bool numa_memory_;
  size_t shard_mask_;
};

} // namespace se
