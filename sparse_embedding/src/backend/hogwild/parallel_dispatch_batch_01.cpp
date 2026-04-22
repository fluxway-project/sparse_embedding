#include "sparse_embedding/src/backend/hogwild/parallel_dispatch_internal.hpp"

namespace se {
void dispatch_batch_01(exec::static_thread_pool &pool, uint32_t num_threads,
                       size_t n, TaskClosure closure) {
  switch (num_threads) {
  case 16:
    dispatch_when_all<16>(pool, n, closure);
    break;
  case 24:
    dispatch_when_all<24>(pool, n, closure);
    break;
  case 32:
    dispatch_when_all<32>(pool, n, closure);
    break;
  case 48:
    dispatch_when_all<48>(pool, n, closure);
    break;
  }
}
} // namespace se
