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

#ifndef SPARSE_EMBEDDING_PYTHON_WAKE_MANAGER_HPP_
#define SPARSE_EMBEDDING_PYTHON_WAKE_MANAGER_HPP_

#include <sys/eventfd.h>
#include <unistd.h>

#include <functional>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sparse_embedding/src/common/function.hpp"

namespace se::py {

// =============================================================================
// PythonWakeManager: edge-triggered GIL callback batching
//
// Instead of acquiring the GIL for every async completion (N GIL round-trips
// for N completions), callbacks are enqueued into a lock-free-ish queue and a
// single eventfd write wakes the Python event loop.  The event loop handler
// (Drain) runs with GIL held and processes ALL pending callbacks in one shot.
//
//   edge-triggered:   N completions  →  1 eventfd write  →  1 GIL acquire
//   level-triggered:  N completions  →  N GIL acquires    (bad)
// =============================================================================

class PythonWakeManager {
public:
  PythonWakeManager() {
    event_fd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (event_fd_ < 0) {
      throw std::runtime_error("PythonWakeManager: failed to create eventfd");
    }
  }

  ~PythonWakeManager() {
    if (event_fd_ >= 0) {
      ::close(event_fd_);
    }
  }

  PythonWakeManager(const PythonWakeManager &) = delete;
  PythonWakeManager &operator=(const PythonWakeManager &) = delete;

  void Enqueue(se::Function<void()> task) {
    bool was_empty = false;
    {
      std::lock_guard lock(mutex_);
      was_empty = pending_.empty();
      pending_.push_back(std::move(task));
    }
    if (was_empty) {
      uint64_t val = 1;
      [[maybe_unused]] auto _ = ::write(event_fd_, &val, sizeof(val));
    }
  }

  void Drain() {
    uint64_t val;
    [[maybe_unused]] auto _ = ::read(event_fd_, &val, sizeof(val));

    std::lock_guard gil_lock(gil_lock_);
    std::vector<se::Function<void()>> tasks;
    {
      std::lock_guard lock(mutex_);
      std::swap(tasks, pending_);
    }
    for (auto &task : tasks) {
      (*task)();
    }
  }

  void Clear() {
    std::lock_guard gil_lock(gil_lock_);
    std::vector<se::Function<void()>> tasks;
    {
      std::lock_guard lock(mutex_);
      std::swap(tasks, pending_);
    }
    // Destruction of tasks will decref the captured PyObjects.
    // Must be called with the GIL held.
  }

  int fd() const noexcept { return event_fd_; }
  std::mutex &gil_lock() noexcept { return gil_lock_; }

private:
  std::mutex mutex_;
  std::mutex gil_lock_;
  std::vector<se::Function<void()>> pending_;
  int event_fd_ = -1;
};

inline PythonWakeManager &GetPythonWakeManager() {
  static PythonWakeManager instance;
  return instance;
}

} // namespace se::py

#endif // SPARSE_EMBEDDING_PYTHON_WAKE_MANAGER_HPP_
