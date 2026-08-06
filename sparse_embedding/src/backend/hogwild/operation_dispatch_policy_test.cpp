#include "sparse_embedding/src/backend/hogwild/operation_dispatch_policy.hpp"

#include <cstdint>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Function> void require_throws(Function &&function) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected operation to throw");
}

} // namespace

int main() {
  require(se::OperationDispatchConfig::kDefaultMinimumBytesPerThread > 0,
          "dispatch config default grain must be explicit");
  require(se::OperationDispatchPolicy::normalize_thread_count(1) == 1,
          "dispatch normalization changed one worker");
  require(se::OperationDispatchPolicy::normalize_thread_count(17) == 32,
          "dispatch normalization did not round to the next power of two");
  require_throws(
      [] { se::OperationDispatchPolicy::normalize_thread_count(0); });
  constexpr se::DispatchTopology topology{64, 32, 32, 256 * 1024};
  constexpr size_t batch_size = 65'536;
  constexpr size_t bytes_per_entry = 128 * sizeof(float);

  const se::DispatchTopology default_topology =
      se::OperationDispatchPolicy::make_topology(8, std::nullopt);
  require(default_topology.available_threads == 8 &&
              default_topology.physical_threads == 8 &&
              default_topology.memory_parallelism == 8,
          "default dispatch topology did not mirror the worker count");
  const se::DispatchTopology configured_topology =
      se::OperationDispatchPolicy::make_topology(
          64, se::OperationDispatchConfig{24, 32, 512});
  require(configured_topology.available_threads == 64 &&
              configured_topology.physical_threads == 24 &&
              configured_topology.memory_parallelism == 32 &&
              configured_topology.minimum_bytes_per_thread == 512,
          "configured dispatch topology was not preserved");
  require_throws([] {
    se::OperationDispatchPolicy::make_topology(
        8, se::OperationDispatchConfig{0, 1, 1});
  });
  require_throws([] {
    se::OperationDispatchPolicy::make_topology(
        8, se::OperationDispatchConfig{16, 1, 1});
  });
  require_throws([] {
    se::OperationDispatchPolicy::make_topology(
        8, se::OperationDispatchConfig{1, 16, 1});
  });

  require(se::OperationDispatchPolicy::thread_count(
              se::HogwildOperation::kForwardLookup, batch_size, bytes_per_entry,
              topology) == 32,
          "lookup did not stop at physical worker capacity");
  require(se::OperationDispatchPolicy::thread_count(
              se::HogwildOperation::kLookupOrInsert, batch_size,
              bytes_per_entry, topology) == 32,
          "lookup-or-insert did not stop at physical worker capacity");
  require(se::OperationDispatchPolicy::thread_count(
              se::HogwildOperation::kAccumulateGradients, batch_size,
              bytes_per_entry, topology) == 32,
          "gradient accumulation did not stop at physical worker capacity");
  require(se::OperationDispatchPolicy::thread_count(
              se::HogwildOperation::kOptimizerStep, batch_size, bytes_per_entry,
              topology) == 32,
          "optimizer did not stop at physical worker capacity");
  require(se::OperationDispatchPolicy::thread_count(
              se::HogwildOperation::kAssign, batch_size, bytes_per_entry,
              topology) == 32,
          "assignment exceeded physical worker capacity");
  require(se::OperationDispatchPolicy::thread_count(
              se::HogwildOperation::kContains, batch_size, sizeof(uint64_t),
              topology) == 32,
          "contains exceeded physical worker capacity");
  require(se::OperationDispatchPolicy::thread_count(
              se::HogwildOperation::kErase, batch_size / 10, sizeof(uint64_t),
              topology) == 32,
          "erase exceeded physical worker capacity");
  require(se::OperationDispatchPolicy::thread_count(
              se::HogwildOperation::kContains, 7, sizeof(uint64_t), topology) ==
              7,
          "small contains exceeded its independent key count");
  require(se::OperationDispatchPolicy::thread_count(
              se::HogwildOperation::kForwardLookup, 128, bytes_per_entry,
              topology) == 1,
          "small lookup did not preserve serial execution");
  require(se::OperationDispatchPolicy::concurrency_limited_thread_count(
              16, 128, 1) == 16,
          "isolated dispatch changed its requested worker count");
  require(se::OperationDispatchPolicy::concurrency_limited_thread_count(
              16, 128, 8) == 16,
          "dispatch shrank before active calls exhausted worker capacity");
  require(se::OperationDispatchPolicy::concurrency_limited_thread_count(
              16, 128, 16) == 8,
          "dispatch did not assign a power-of-two fair share");
  require(se::OperationDispatchPolicy::concurrency_limited_thread_count(
              16, 128, 128) == 1,
          "fully concurrent dispatch did not become caller-local");
  require(se::OperationDispatchPolicy::concurrency_limited_thread_count(
              16, 128, 256) == 1,
          "oversubscribed dispatch did not remain caller-local");
  require(se::OperationDispatchPolicy::task_count(
              se::HogwildOperation::kForwardLookup, batch_size, 32, topology) ==
              32,
          "parallel lookup did not expose full task decomposition");
  require(se::OperationDispatchPolicy::task_count(
              se::HogwildOperation::kForwardLookup, 128, 1, topology) == 1,
          "serial lookup created parallel tasks");
  require(se::OperationDispatchPolicy::task_count(
              se::HogwildOperation::kLookupOrInsert, batch_size, 32,
              topology) == 32,
          "mutation task count diverged from its logical worker count");
  require_throws([] {
    se::OperationDispatchPolicy::concurrency_limited_thread_count(0, 8, 1);
  });
  require_throws([] {
    se::OperationDispatchPolicy::concurrency_limited_thread_count(8, 0, 1);
  });
  require_throws([] {
    se::OperationDispatchPolicy::concurrency_limited_thread_count(8, 8, 0);
  });
  require_throws([] {
    se::OperationDispatchPolicy::concurrency_limited_thread_count(16, 8, 1);
  });
  require_throws([] {
    se::OperationDispatchPolicy::task_count(
        se::HogwildOperation::kForwardLookup, 0, 1,
        se::DispatchTopology{8, 8, 8, 1});
  });
  require_throws([] {
    se::OperationDispatchPolicy::task_count(
        se::HogwildOperation::kForwardLookup, 1, 16,
        se::DispatchTopology{8, 8, 8, 1});
  });
  require_throws([] {
    se::OperationDispatchPolicy::thread_count(
        se::HogwildOperation::kAssign, 1, 1, se::DispatchTopology{0, 0, 0, 1});
  });
  require_throws([] {
    se::OperationDispatchPolicy::thread_count(
        se::HogwildOperation::kAssign, 1, 1, se::DispatchTopology{8, 16, 8, 1});
  });
  require_throws([] {
    se::OperationDispatchPolicy::thread_count(
        se::HogwildOperation::kAssign, 0, 1, se::DispatchTopology{1, 1, 1, 1});
  });
}
