#include "sparse_embedding/src/backend/hogwild/parallel_dispatch_internal.hpp"

namespace se {
void dispatch_batch_02(exec::static_thread_pool &pool, uint32_t num_threads,
                       size_t n, TaskClosure closure) {
  switch (num_threads) {
  case 64:
    dispatch_when_all<64>(pool, n, closure);
    break;
  case 80:
    dispatch_when_all<80>(pool, n, closure);
    break;
  case 96:
    dispatch_when_all<96>(pool, n, closure);
    break;
  case 112:
    dispatch_when_all<112>(pool, n, closure);
    break;
  }
}
} // namespace se
