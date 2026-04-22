#include "sparse_embedding/src/backend/hogwild/parallel_dispatch_internal.hpp"

namespace se {
void dispatch_batch_00(exec::static_thread_pool &pool, uint32_t num_threads,
                       size_t n, TaskClosure closure) {
  switch (num_threads) {
  case 2:
    dispatch_when_all<2>(pool, n, closure);
    break;
  case 4:
    dispatch_when_all<4>(pool, n, closure);
    break;
  case 8:
    dispatch_when_all<8>(pool, n, closure);
    break;
  case 12:
    dispatch_when_all<12>(pool, n, closure);
    break;
  }
}
} // namespace se
