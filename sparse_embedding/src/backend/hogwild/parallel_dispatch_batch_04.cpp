#include "sparse_embedding/src/backend/hogwild/parallel_dispatch_internal.hpp"

namespace se {
void dispatch_batch_04(exec::static_thread_pool &pool, uint32_t num_threads,
                       size_t n, TaskClosure closure) {
  switch (num_threads) {
  case 192:
    dispatch_when_all<192>(pool, n, closure);
    break;
  case 208:
    dispatch_when_all<208>(pool, n, closure);
    break;
  case 224:
    dispatch_when_all<224>(pool, n, closure);
    break;
  case 240:
    dispatch_when_all<240>(pool, n, closure);
    break;
  }
}
} // namespace se
