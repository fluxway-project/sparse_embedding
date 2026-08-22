#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <thread>

#include "sparse_embedding/src/backend/hogwild/operation_dispatch_policy.hpp"
#include "sparse_embedding/src/execution/dispatch_budget.hpp"
#include "sparse_embedding/src/execution/locality_policy.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"
#include "sparse_embedding/src/storage/topology/storage_topology.hpp"

namespace se {

struct HogwildStoreOptions {
  size_t thread_budget = 0;
  LocalityPolicy locality;
  PageStorage storage = create_page_storage();
  storage::StorageTuningOptions storage_tuning;

  std::optional<OperationDispatchConfig> dispatch_config() const {
    if (locality.execution == LocalityMode::kAuto &&
        locality.memory == LocalityMode::kAuto)
      return std::nullopt;
    const size_t automatic = std::max(1u, std::thread::hardware_concurrency());
    const size_t budget = normalize_thread_budget(thread_budget, automatic);
    const bool bind = locality.execution == LocalityMode::kNuma;
    const size_t minimum_bytes_per_thread =
        locality.execution == LocalityMode::kNuma ||
                locality.memory == LocalityMode::kNuma
            ? OperationDispatchConfig::kNumaMinimumBytesPerThread
            : OperationDispatchConfig::kDefaultMinimumBytesPerThread;
    return OperationDispatchConfig{budget, budget, minimum_bytes_per_thread,
                                   bind, 16};
  }
};

} // namespace se
