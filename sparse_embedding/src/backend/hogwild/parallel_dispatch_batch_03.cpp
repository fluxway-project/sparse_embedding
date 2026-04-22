#include "sparse_embedding/src/backend/hogwild/parallel_dispatch_internal.hpp"

namespace se {
void dispatch_batch_03(exec::static_thread_pool &pool, uint32_t num_threads,
                       size_t n, TaskClosure closure) {
  switch (num_threads) {
  case 128:
    dispatch_when_all<128>(pool, n, closure);
    break;
  case 144:
    dispatch_when_all<144>(pool, n, closure);
    break;
  case 160:
    dispatch_when_all<160>(pool, n, closure);
    break;
  case 176:
    dispatch_when_all<176>(pool, n, closure);
    break;
  }
}
} // namespace se
