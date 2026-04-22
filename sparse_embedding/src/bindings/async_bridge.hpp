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
// fire_and_forget_receiver: custom receiver for manually connecting to
// a pool scheduler sender. Self-deletes operation state on completion.
//
// This bypasses start_detached's __constant_completion_signatures_v check
// that is incompatible with static_thread_pool in this stdexec version.
// =============================================================================

namespace detail {

struct op_base {
  virtual ~op_base() = default;
};

struct fire_forget_receiver {
  using receiver_concept = stdexec::receiver_t;

  op_base *op_;

  template <class... As> void set_value(As &&...) noexcept { delete op_; }

  template <class E> [[noreturn]] void set_error(E &&) noexcept {
    std::terminate();
  }

  void set_stopped() noexcept { delete op_; }

  auto get_env() const noexcept { return stdexec::empty_env{}; }
};

template <typename OpState> struct op_impl : op_base {
  OpState state_;

  template <typename Sender>
  explicit op_impl(Sender &&sndr)
      : state_(stdexec::connect(std::forward<Sender>(sndr),
                                fire_forget_receiver{this})) {}

  void start() { stdexec::start(state_); }
};

template <typename Sender> void fire_and_forget(Sender &&sndr) {
  using op_t = op_impl<stdexec::connect_result_t<Sender, fire_forget_receiver>>;
  auto *op = new op_t(std::forward<Sender>(sndr));
  op->start();
}

} // namespace detail

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
// 4. fire_and_forget: manually connect + start (bypasses start_detached
//    concept limitations, same mechanism, custom receiver)
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

  detail::fire_and_forget(
      stdexec::schedule(pool.get_scheduler()) |
      stdexec::then([t = std::move(erased_task)]() mutable { (*t)(); }));

  return future;
}

} // namespace se::py

#endif // SPARSE_EMBEDDING_ASYNC_BRIDGE_HPP_
