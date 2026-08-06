#pragma once

#ifndef SPARSE_EMBEDDING_OPERATION_DISPATCH_POLICY_HPP_
#define SPARSE_EMBEDDING_OPERATION_DISPATCH_POLICY_HPP_

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

namespace se {

enum class HogwildOperation {
  kForwardLookup,
  kLookupOrInsert,
  kAssign,
  kAccumulateGradients,
  kOptimizerStep,
  kContains,
  kErase,
};

struct DispatchTopology {
  size_t available_threads;
  size_t physical_threads;
  size_t memory_parallelism;
  size_t minimum_bytes_per_thread;
  size_t forward_prefetch_distance = 16;
};

struct OperationDispatchConfig {
  static constexpr size_t kDefaultMinimumBytesPerThread = 256 * 1024;
  static constexpr size_t kNumaMinimumBytesPerThread = 16 * 1024;

  size_t physical_threads;
  size_t memory_parallelism;
  size_t minimum_bytes_per_thread = kDefaultMinimumBytesPerThread;
  bool bind_worker_affinity = false;
  size_t forward_prefetch_distance = 16;
};

class OperationDispatchPolicy {
public:
  static size_t normalize_thread_count(size_t count);
  static DispatchTopology
  make_topology(size_t available_threads,
                std::optional<OperationDispatchConfig> dispatch_config);
  static size_t thread_count(HogwildOperation operation, size_t batch_size,
                             size_t bytes_per_entry,
                             DispatchTopology topology) {
    validate(topology, batch_size, bytes_per_entry);
    if (operation == HogwildOperation::kContains ||
        operation == HogwildOperation::kErase) {
      return std::min(batch_size, topology.physical_threads);
    }
    return std::min(memory_parallelism(batch_size, bytes_per_entry, topology),
                    operation_limit(operation, topology));
  }

  static size_t concurrency_limited_thread_count(size_t requested_threads,
                                                 size_t available_threads,
                                                 size_t active_invocations) {
    if (requested_threads == 0 || available_threads == 0 ||
        active_invocations == 0) {
      throw std::invalid_argument("concurrent dispatch capacity contains zero");
    }
    if (requested_threads > available_threads) {
      throw std::invalid_argument(
          "requested dispatch threads exceed available threads");
    }
    const size_t fair_share =
        std::max(size_t{1}, available_threads / active_invocations);
    return std::min(requested_threads, std::bit_floor(fair_share));
  }

  static size_t task_count(HogwildOperation operation, size_t batch_size,
                           size_t logical_threads, DispatchTopology topology) {
    validate(topology, batch_size, 1);
    if (logical_threads == 0) {
      throw std::invalid_argument("dispatch task capacity contains zero");
    }
    const size_t capacity = operation_limit(operation, topology);
    if (logical_threads > capacity) {
      throw std::invalid_argument(
          "logical dispatch threads exceed operation capacity");
    }
    if (operation == HogwildOperation::kForwardLookup && logical_threads > 1) {
      return std::min(batch_size, capacity);
    }
    return logical_threads;
  }

private:
  static size_t memory_parallelism(size_t batch_size, size_t bytes_per_entry,
                                   DispatchTopology topology) {
    return ceil_divide(checked_product(batch_size, bytes_per_entry),
                       topology.minimum_bytes_per_thread);
  }

  static void validate(DispatchTopology topology, size_t batch_size,
                       size_t bytes_per_entry) {
    if (topology.available_threads == 0 || topology.physical_threads == 0 ||
        topology.memory_parallelism == 0 ||
        topology.minimum_bytes_per_thread == 0 ||
        topology.forward_prefetch_distance > 64) {
      throw std::invalid_argument("dispatch topology contains zero capacity");
    }
    if (topology.physical_threads > topology.available_threads) {
      throw std::invalid_argument(
          "dispatch topology physical threads exceed available threads");
    }
    if (topology.memory_parallelism > topology.available_threads) {
      throw std::invalid_argument(
          "dispatch topology memory parallelism exceeds available threads");
    }
    if (batch_size == 0 || bytes_per_entry == 0) {
      throw std::invalid_argument("dispatch work must not be empty");
    }
  }

  static size_t operation_limit(HogwildOperation operation,
                                DispatchTopology topology) {
    switch (operation) {
    case HogwildOperation::kForwardLookup:
    case HogwildOperation::kLookupOrInsert:
    case HogwildOperation::kAccumulateGradients:
    case HogwildOperation::kOptimizerStep:
      return std::min(topology.memory_parallelism, topology.physical_threads);
    case HogwildOperation::kAssign:
    case HogwildOperation::kContains:
    case HogwildOperation::kErase:
      return topology.physical_threads;
    }
    throw std::invalid_argument("dispatch operation is unsupported");
  }

  static size_t checked_product(size_t left, size_t right) {
    if (left > std::numeric_limits<size_t>::max() / right)
      throw std::overflow_error("dispatch work size overflows size_t");
    return left * right;
  }

  static size_t ceil_divide(size_t numerator, size_t denominator) {
    return numerator / denominator + (numerator % denominator != 0);
  }
};

} // namespace se

#endif
