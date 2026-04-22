#pragma once
#include <cstddef>
#include <cstdint>

#include "exec/static_thread_pool.hpp"

namespace se {

struct TaskClosure {
  void *ctx;
  void (*invoke)(void *ctx, size_t begin, size_t end);
};

void dispatch_parallel_tasks(exec::static_thread_pool &pool,
                             uint32_t num_threads, size_t n,
                             TaskClosure closure);

void dispatch_batch_00(exec::static_thread_pool &pool, uint32_t num_threads,
                       size_t n, TaskClosure closure);
void dispatch_batch_01(exec::static_thread_pool &pool, uint32_t num_threads,
                       size_t n, TaskClosure closure);
void dispatch_batch_02(exec::static_thread_pool &pool, uint32_t num_threads,
                       size_t n, TaskClosure closure);
void dispatch_batch_03(exec::static_thread_pool &pool, uint32_t num_threads,
                       size_t n, TaskClosure closure);
void dispatch_batch_04(exec::static_thread_pool &pool, uint32_t num_threads,
                       size_t n, TaskClosure closure);
void dispatch_batch_05(exec::static_thread_pool &pool, uint32_t num_threads,
                       size_t n, TaskClosure closure);

} // namespace se
