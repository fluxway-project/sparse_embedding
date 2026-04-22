#include "sparse_embedding/src/backend/hogwild/parallel_dispatch_internal.hpp"

namespace se {
void dispatch_batch_05(exec::static_thread_pool &pool, uint32_t num_threads,
                       size_t n, TaskClosure closure) {
  switch (num_threads) {
  case 256:
    dispatch_when_all<256>(pool, n, closure);
    break;
  }
}
} // namespace se
