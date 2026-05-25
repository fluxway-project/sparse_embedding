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
#include <memory>
#include <utility>

#include <exec/start_detached.hpp>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include "sparse_embedding/src/bindings/python_wake_manager.hpp"

namespace se::py {

#if defined(Py_LIMITED_API)
extern "C" PyAPI_FUNC(int) PyGILState_Check(void);
#endif

namespace nb = nanobind;

// AsyncPyObjectKeeper: safely defers Py_DECREF to the GIL-holding wake thread
//
// Storing nb::object in a lambda captured by a worker thread is unsafe because
// the destructor calls Py_DECREF without holding the GIL. This wrapper holds a
// raw PyObject* and enqueues the decref via PythonWakeManager so it executes on
// the main asyncio thread (which holds the GIL).
// =============================================================================

struct AsyncPyObjectKeeper {
  PyObject *ptr;
  explicit AsyncPyObjectKeeper(nb::handle obj) : ptr(obj.ptr()) {
    if (ptr) {
      nb::handle(ptr).inc_ref();
    }
  }
  ~AsyncPyObjectKeeper() {
    if (ptr) {
      if (PyGILState_Check()) {
        nb::handle(ptr).dec_ref();
      } else {
        GetPythonWakeManager().Enqueue(
            pro::make_proxy_shared<se::FunctionFacade<void()>>(
                [p = ptr]() { nb::handle(p).dec_ref(); }));
      }
    }
  }
  AsyncPyObjectKeeper(const AsyncPyObjectKeeper &) = delete;
  AsyncPyObjectKeeper &operator=(const AsyncPyObjectKeeper &) = delete;
};

// =============================================================================

inline nb::object create_asyncio_future() {
  nb::module_ asyncio = nb::module_::import_("asyncio");
  nb::object loop = asyncio.attr("get_running_loop")();

  if (!nb::hasattr(loop, "__se_wake_registered__")) {
    int fd = GetPythonWakeManager().fd();
    nb::object drain =
        nb::cpp_function([]() { GetPythonWakeManager().Drain(); });
    loop.attr("add_reader")(fd, drain);
    loop.attr("__se_wake_registered__") = nb::bool_(true);
  }

  return loop.attr("create_future")();
}

// =============================================================================
// async_dispatch: bridge exec::static_thread_pool to asyncio.Future
//
// 1. Creates Future (with GIL, on caller thread)
// 2. Wraps future in AsyncPyObjectKeeper — never stored as nb::object on worker
// 3. Type-erases work lambda via pro::proxy<AsyncTaskFacade>
// 4. Builds sender: schedule(pool) | then(task)
// 5. start_detached: schedules the execution without blocking
// 6. Pool worker thread executes task, enqueues result to wake manager
// 7. Returns Future immediately to Python
//
// All execution on exec::static_thread_pool worker threads. Zero extra threads.
// =============================================================================

template <typename WorkFn, typename CompleteFn>
nb::object async_dispatch(exec::static_thread_pool &pool, WorkFn &&work,
                          CompleteFn &&on_complete) {
  nb::object future = create_asyncio_future();

  auto future_keeper =
      std::make_shared<AsyncPyObjectKeeper>(nb::borrow(future));

  se::Function<void()> erased_task =
      pro::make_proxy_shared<se::FunctionFacade<void()>>(
          [work = std::forward<WorkFn>(work),
           on_complete = std::forward<CompleteFn>(on_complete),
           future_keeper]() mutable {
            try {
              auto result = work();
              GetPythonWakeManager().Enqueue(
                  pro::make_proxy_shared<se::FunctionFacade<void()>>(
                      [on_complete, future_keeper, result]() mutable {
                        nb::object fut = nb::borrow(future_keeper->ptr);
                        on_complete(std::move(fut), std::move(result));
                      }));
            } catch (...) {
              auto eptr = std::current_exception();
              GetPythonWakeManager().Enqueue(
                  pro::make_proxy_shared<se::FunctionFacade<void()>>(
                      [future_keeper, eptr]() mutable {
                        nb::object fut = nb::borrow(future_keeper->ptr);
                        try {
                          std::rethrow_exception(eptr);
                        } catch (const std::exception &e) {
                          nb::module_ builtins =
                              nb::module_::import_("builtins");
                          fut.attr("set_exception")(
                              builtins.attr("RuntimeError")(e.what()));
                        }
                      }));
            }
          });

  exec::start_detached(stdexec::schedule(pool.get_scheduler()) |
                       stdexec::then([t = erased_task]() mutable { (*t)(); }));

  return future;
}

} // namespace se::py

#endif // SPARSE_EMBEDDING_ASYNC_BRIDGE_HPP_
