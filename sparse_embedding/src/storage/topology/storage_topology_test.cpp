#include <bit>
#include <cassert>
#include <iostream>
#include <stdexcept>

#include "sparse_embedding/src/storage/topology/storage_resource_tuner.hpp"
#include "sparse_embedding/src/storage/topology/storage_topology.hpp"
#include "sparse_embedding/src/storage/topology/system_resource_probe.hpp"

void require(bool condition, const char *msg) {
  if (!condition)
    throw std::runtime_error(msg);
}

int main() {
  // Test 1: SystemResourceProbe invariants
  size_t hw = se::storage::SystemResourceProbe::hardware_concurrency();
  require(hw >= 1, "hardware concurrency must be >= 1");
  size_t ram = se::storage::SystemResourceProbe::available_ram_bytes();
  require(ram > 0, "available ram must be > 0");

  // Test 2: StorageTopology Coordinate Encoding & Invariants
  se::storage::StorageTopology topo(16, 64, 4096, 8192, 16, false, false);
  require(topo.execution_workers() == 16, "workers mismatch");
  require(topo.shard_count() == 64, "shards mismatch");
  require(topo.shard_mask() == 63, "shard mask mismatch");
  require(topo.entries_per_page() == 4096, "entries per page mismatch");
  require(topo.entries_shift() == 12, "entries shift mismatch");
  require(topo.slot_mask() == 4095, "slot mask mismatch");
  require(topo.max_pages() == 8192, "max pages mismatch");
  require(topo.cohort_size() == 16, "cohort size mismatch");
  require(topo.cohort_count() == 4, "cohort count mismatch");
  require(topo.max_capacity_keys() == 8192 * 4096, "capacity mismatch");

  require(topo.page_id(0) == 0, "page_id(0) failed");
  require(topo.page_slot(0) == 0, "page_slot(0) failed");
  require(topo.page_id(4095) == 0, "page_id(4095) failed");
  require(topo.page_slot(4095) == 4095, "page_slot(4095) failed");
  require(topo.page_id(4096) == 1, "page_id(4096) failed");
  require(topo.page_slot(4096) == 0, "page_slot(4096) failed");
  require(topo.page_id(8193) == 2, "page_id(8193) failed");
  require(topo.page_slot(8193) == 1, "page_slot(8193) failed");
  const auto &geometry = topo.page_geometry();
  require(geometry.page_id(8193) == 2, "geometry page id failed");
  require(geometry.slot(8193) == 1, "geometry slot failed");

  // Test 3: StorageResourceTuner derivation
  se::storage::TopologyTuneRequest req_auto{
      .thread_budget = 0,
      .embedding_dim = 32,
  };
  auto tuned_auto = se::storage::StorageResourceTuner::derive(req_auto);
  require(std::has_single_bit(tuned_auto.shard_count()),
          "auto shards must be power of two");
  require(std::has_single_bit(tuned_auto.entries_per_page()),
          "auto entries_per_page must be power of two");
  require(tuned_auto.entries_per_page() >= 4096 &&
              tuned_auto.entries_per_page() <= 16384,
          "entries_per_page bounds violated");
  require(tuned_auto.max_pages() >= 8192, "auto max_pages too small");

  se::storage::TopologyTuneRequest calibrated_request{
      .thread_budget = 8,
      .embedding_dim = 128,
      .tuning = {.workload = se::storage::StorageWorkload::kLookupInsert,
                 .calibrate_page_geometry = true,
                 .calibration_samples = 3}};
  const auto calibrated =
      se::storage::StorageResourceTuner::derive(calibrated_request);
  require(calibrated.entries_per_page() == 4096 ||
              calibrated.entries_per_page() == 8192 ||
              calibrated.entries_per_page() == 16384,
          "calibrated page geometry is not a supported candidate");

  calibrated_request.tuning.entries_per_page = 4096;
  calibrated_request.tuning.calibration_samples = 0;
  const auto explicit_page =
      se::storage::StorageResourceTuner::derive(calibrated_request);
  require(explicit_page.entries_per_page() == 4096,
          "explicit page geometry was not preserved");

  // Test 4: Custom Constraints & Sizing
  se::storage::TopologyTuneRequest req_custom{
      .thread_budget = 4,
      .embedding_dim = 512,
      .tuning = {.target_capacity_keys = 2000000,
                 .entries_per_page = 4096,
                 .max_pages = 4096},
  };
  auto tuned_custom = se::storage::StorageResourceTuner::derive(req_custom);
  require(tuned_custom.shard_count() == 4, "custom shard count failed");
  require(tuned_custom.entries_per_page() == 4096,
          "custom entries_per_page failed");
  require(tuned_custom.max_pages() == 4096, "custom max_pages failed");

  bool invalid_page_rejected = false;
  try {
    se::storage::StorageResourceTuner::derive(se::storage::TopologyTuneRequest{
        .thread_budget = 1, .tuning = {.entries_per_page = 256}});
  } catch (const std::invalid_argument &) {
    invalid_page_rejected = true;
  }
  require(invalid_page_rejected, "tuner accepted an undersized page");

  std::cout << "All StorageTopology tests passed successfully!" << std::endl;
  return 0;
}
