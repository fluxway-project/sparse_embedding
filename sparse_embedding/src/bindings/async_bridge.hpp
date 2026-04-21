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

#ifndef SPARSE_EMBEDDING_ASYNC_BRIDGE_HPP_
#define SPARSE_EMBEDDING_ASYNC_BRIDGE_HPP_

#include <nanobind/nanobind.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

#include "python_wake_manager.hpp"

namespace se::py {

namespace nb = nanobind;

// =============================================================================
// AsyncTracker: tracks outstanding async operations for safe shutdown
//
// Each async operation is dispatched on a detached std::thread. The tracker
// counts outstanding operations and blocks in wait_all() / destructor until
// all complete.
// =============================================================================

class AsyncTracker {
public:
  void spawn(std::function<void()> fn) {
    outstanding_.fetch_add(1, std::memory_order_relaxed);
    std::thread([this, fn = std::move(fn)]() mutable {
      fn();
      if (outstanding_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard lock(mutex_);
        cv_.notify_all();
      }
    }).detach();
  }

  void wait_all() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this]() {
      return outstanding_.load(std::memory_order_acquire) == 0;
    });
  }

  ~AsyncTracker() { wait_all(); }

private:
  std::atomic<size_t> outstanding_{0};
  std::mutex mutex_;
  std::condition_variable cv_;
};

// =============================================================================
// create_asyncio_future: create a Future from the running event loop (GIL held)
// =============================================================================

inline nb::object create_asyncio_future() {
  nb::module_ asyncio = nb::module_::import_("asyncio");
  nb::object loop = asyncio.attr("get_running_loop")();
  return loop.attr("create_future")();
}

// =============================================================================
// async_dispatch: bridge a blocking C++ operation to asyncio.Future
//
// 1. Creates Future (with GIL, on caller thread)
// 2. Spawns background thread where:
//    a. work() runs WITHOUT GIL (background thread never holds GIL)
//    b. Result callback is enqueued to PythonWakeManager (edge-triggered)
// 3. Returns Future immediately to Python
//
// The on_complete callback runs with GIL held inside PythonWakeManager::Drain.
// The nb::object (future_ref) is moved into the Drain callback so its
// destructor runs with GIL held — no ref count manipulation without GIL.
// =============================================================================

template <typename WorkFn, typename CompleteFn>
nb::object async_dispatch(AsyncTracker &tracker, WorkFn &&work,
                          CompleteFn &&on_complete) {
  nb::object future = create_asyncio_future();

  tracker.spawn([work = std::forward<WorkFn>(work),
                 on_complete = std::forward<CompleteFn>(on_complete),
                 future_ref = nb::borrow(future)]() mutable {
    // Background thread: never holds GIL.
    // No nb::gil_scoped_release needed (we never acquired it).

    auto result_or_error = work();

    // Enqueue callback to run with GIL inside Drain().
    // Move future_ref into the callback so it is destroyed with GIL held.
    GetPythonWakeManager().Enqueue(
        [on_complete = std::move(on_complete),
         future_ref = std::move(future_ref),
         result = std::move(result_or_error)]() mutable {
          on_complete(std::move(future_ref), std::move(result));
        });
  });

  return future;
}

} // namespace se::py

#endif // SPARSE_EMBEDDING_ASYNC_BRIDGE_HPP_
