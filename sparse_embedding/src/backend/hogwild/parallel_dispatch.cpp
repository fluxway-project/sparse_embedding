#include "sparse_embedding/src/backend/hogwild/parallel_dispatch.hpp"

namespace se {

void dispatch_parallel_tasks(exec::static_thread_pool &pool,
                             uint32_t num_threads, size_t n,
                             TaskClosure closure) {
  if (num_threads <= 1) {
    closure.invoke(closure.ctx, 0, n);
    return;
  }
  if (num_threads <= 2) {
    dispatch_batch_00(pool, 2, n, closure);
    return;
  }
  if (num_threads <= 4) {
    dispatch_batch_00(pool, 4, n, closure);
    return;
  }
  if (num_threads <= 8) {
    dispatch_batch_00(pool, 8, n, closure);
    return;
  }
  if (num_threads <= 12) {
    dispatch_batch_00(pool, 12, n, closure);
    return;
  }
  if (num_threads <= 16) {
    dispatch_batch_01(pool, 16, n, closure);
    return;
  }
  if (num_threads <= 24) {
    dispatch_batch_01(pool, 24, n, closure);
    return;
  }
  if (num_threads <= 32) {
    dispatch_batch_01(pool, 32, n, closure);
    return;
  }
  if (num_threads <= 48) {
    dispatch_batch_01(pool, 48, n, closure);
    return;
  }
  if (num_threads <= 64) {
    dispatch_batch_02(pool, 64, n, closure);
    return;
  }
  if (num_threads <= 80) {
    dispatch_batch_02(pool, 80, n, closure);
    return;
  }
  if (num_threads <= 96) {
    dispatch_batch_02(pool, 96, n, closure);
    return;
  }
  if (num_threads <= 112) {
    dispatch_batch_02(pool, 112, n, closure);
    return;
  }
  if (num_threads <= 128) {
    dispatch_batch_03(pool, 128, n, closure);
    return;
  }
  if (num_threads <= 144) {
    dispatch_batch_03(pool, 144, n, closure);
    return;
  }
  if (num_threads <= 160) {
    dispatch_batch_03(pool, 160, n, closure);
    return;
  }
  if (num_threads <= 176) {
    dispatch_batch_03(pool, 176, n, closure);
    return;
  }
  if (num_threads <= 192) {
    dispatch_batch_04(pool, 192, n, closure);
    return;
  }
  if (num_threads <= 208) {
    dispatch_batch_04(pool, 208, n, closure);
    return;
  }
  if (num_threads <= 224) {
    dispatch_batch_04(pool, 224, n, closure);
    return;
  }
  if (num_threads <= 240) {
    dispatch_batch_04(pool, 240, n, closure);
    return;
  }
  if (num_threads <= 256) {
    dispatch_batch_05(pool, 256, n, closure);
    return;
  }
  dispatch_batch_05(pool, 256, n, closure);
}

} // namespace se
