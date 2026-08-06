#include "sparse_embedding/src/backend/hogwild/partition/hogwild_partition_topology.hpp"

#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
} // namespace

int main() {
  const se::HogwildPartitionTopology topology =
      se::HogwildPartitionTopology::derive(3, se::LocalityPolicy{});
  require(topology.execution_workers() == 3, "execution budget changed");
  require(topology.partition_count() == 4, "partition derivation changed");
  require(topology.shard_mask() == 3, "partition mask changed");

  const se::HogwildPartitionTopology single =
      se::HogwildPartitionTopology::derive(1, se::LocalityPolicy{});
  require(single.execution_workers() == 1, "single execution budget changed");
  require(single.partition_count() == 1, "single partition count changed");
  require(single.shard_mask() == 0, "single partition mask changed");
  return 0;
}
