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
#include <proxy/proxy.h>

#include <exception>
#include <utility>

#include <exec/start_detached.hpp>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include "sparse_embedding/src/bindings/python_wake_manager.hpp"

namespace se::py {

namespace nb = nanobind;

// =============================================================================
// AsyncTask: pro::proxy-backed type-erased move-only callable (void())
// =============================================================================

PRO_DEF_MEM_DISPATCH(MemInvoke, operator());

struct AsyncTaskFacade
    : pro::facade_builder ::add_convention<MemInvoke, void()>::support_copy<
          pro::constraint_level::none>::build {};

// =============================================================================

inline nb::object create_asyncio_future() {
  nb::module_ asyncio = nb::module_::import_("asyncio");
  nb::object loop = asyncio.attr("get_running_loop")();
  return loop.attr("create_future")();
}

// =============================================================================
// async_dispatch: bridge exec::static_thread_pool to asyncio.Future
//
// 1. Creates Future (with GIL, on caller thread)
// 2. Type-erases work lambda via pro::proxy<AsyncTaskFacade>
// 3. Builds sender: schedule(pool) | then(task)
// 4. start_detached: schedules the execution without blocking
// 5. Pool worker thread executes task, enqueues result to wake manager
// 6. Returns Future immediately to Python
//
// All execution on exec::static_thread_pool worker threads. Zero extra threads.
// =============================================================================

template <typename WorkFn, typename CompleteFn>
nb::object async_dispatch(exec::static_thread_pool &pool, WorkFn &&work,
                          CompleteFn &&on_complete) {
  nb::object future = create_asyncio_future();

  pro::proxy<AsyncTaskFacade> erased_task = pro::make_proxy<AsyncTaskFacade>(
      [work = std::forward<WorkFn>(work),
       on_complete = std::forward<CompleteFn>(on_complete),
       future_ref = nb::borrow(future)]() mutable {
        try {
          auto result = work();
          GetPythonWakeManager().Enqueue(
              [on_complete = std::move(on_complete),
               future_ref = std::move(future_ref),
               result = std::move(result)]() mutable {
                on_complete(std::move(future_ref), std::move(result));
              });
        } catch (...) {
          auto eptr = std::current_exception();
          GetPythonWakeManager().Enqueue([future_ref = std::move(future_ref),
                                          eptr = std::move(eptr)]() mutable {
            try {
              std::rethrow_exception(eptr);
            } catch (const std::exception &e) {
              nb::module_ builtins = nb::module_::import_("builtins");
              future_ref.attr("set_exception")(
                  builtins.attr("RuntimeError")(e.what()));
            }
          });
        }
      });

  exec::start_detached(
      stdexec::schedule(pool.get_scheduler()) |
      stdexec::then([t = std::move(erased_task)]() mutable { (*t)(); }));

  return future;
}

} // namespace se::py

#endif // SPARSE_EMBEDDING_ASYNC_BRIDGE_HPP_
