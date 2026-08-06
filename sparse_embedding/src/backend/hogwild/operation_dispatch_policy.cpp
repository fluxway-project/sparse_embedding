#include "sparse_embedding/src/backend/hogwild/operation_dispatch_policy.hpp"

namespace se {
namespace {

constexpr size_t kMaximumThreadCount = std::numeric_limits<uint32_t>::max();

}

size_t OperationDispatchPolicy::normalize_thread_count(size_t count) {
  if (count == 0)
    throw std::invalid_argument("thread count is zero");
  size_t normalized = 1;
  while (normalized < count) {
    if (normalized > std::numeric_limits<size_t>::max() / 2)
      throw std::overflow_error("thread count overflows");
    normalized <<= 1;
  }
  if (normalized > kMaximumThreadCount)
    throw std::overflow_error("thread count exceeds uint32");
  return normalized;
}

DispatchTopology OperationDispatchPolicy::make_topology(
    size_t available_threads,
    std::optional<OperationDispatchConfig> dispatch_config) {
  if (!dispatch_config.has_value()) {
    return {available_threads, available_threads, available_threads,
            OperationDispatchConfig::kDefaultMinimumBytesPerThread, 16};
  }
  if (dispatch_config->physical_threads == 0 ||
      dispatch_config->memory_parallelism == 0 ||
      dispatch_config->minimum_bytes_per_thread == 0) {
    throw std::invalid_argument(
        "operation dispatch config contains zero capacity");
  }
  if (dispatch_config->physical_threads > available_threads) {
    throw std::invalid_argument(
        "operation dispatch physical threads exceed store threads");
  }
  if (dispatch_config->memory_parallelism > available_threads) {
    throw std::invalid_argument(
        "operation dispatch memory parallelism exceeds store threads");
  }
  return {available_threads, dispatch_config->physical_threads,
          dispatch_config->memory_parallelism,
          dispatch_config->minimum_bytes_per_thread,
          dispatch_config->forward_prefetch_distance};
}

} // namespace se
