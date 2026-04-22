#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "exec/static_thread_pool.hpp"
#include "sparse_embedding/src/backend/hogwild/parallel_dispatch.hpp"
#include "stdexec/execution.hpp"

namespace se {

template <size_t N>
void dispatch_when_all(exec::static_thread_pool &pool, size_t n,
                       TaskClosure closure) {
  size_t chunk_size = (n + N - 1) / N;
  auto make_tasks = [&]<size_t... Is>(std::index_sequence<Is...>) {
    return stdexec::when_all(
        stdexec::then(stdexec::schedule(pool.get_scheduler()),
                      [closure, n, chunk_size, i = Is]() {
                        size_t begin = std::min(i * chunk_size, n);
                        size_t end = std::min(begin + chunk_size, n);
                        if (begin < end) {
                          closure.invoke(closure.ctx, begin, end);
                        }
                      })...);
  };
  stdexec::sync_wait(make_tasks(std::make_index_sequence<N>{}));
}

} // namespace se
