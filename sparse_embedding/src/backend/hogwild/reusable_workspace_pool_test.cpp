#include "sparse_embedding/src/execution/reusable_workspace_pool.hpp"

#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

std::atomic<bool> track_allocations{false};
std::atomic<size_t> allocation_count{0};

template <typename T> struct CountingAllocator {
  using value_type = T;

  CountingAllocator() = default;

  template <typename U>
  CountingAllocator(const CountingAllocator<U> &) noexcept {}

  T *allocate(size_t count) {
    if (track_allocations.load(std::memory_order_relaxed))
      allocation_count.fetch_add(1, std::memory_order_relaxed);
    return std::allocator<T>{}.allocate(count);
  }

  void deallocate(T *memory, size_t count) noexcept {
    std::allocator<T>{}.deallocate(memory, count);
  }

  template <typename U>
  bool operator==(const CountingAllocator<U> &) const noexcept {
    return true;
  }
};

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Workspace {
  std::vector<size_t, CountingAllocator<size_t>> scratch;
};

} // namespace

int main() {
  se::execution::ReusableWorkspacePool<Workspace> pool;
  Workspace *retained = nullptr;
  {
    auto lease = pool.acquire();
    retained = lease.get();
    lease->scratch.reserve(4096);
  }
  {
    auto lease = pool.acquire();
    require(lease.get() == retained && lease->scratch.capacity() >= 4096,
            "workspace pool did not retain scratch capacity");
  }
  allocation_count.store(0, std::memory_order_relaxed);
  track_allocations.store(true, std::memory_order_relaxed);
  {
    auto lease = pool.acquire();
    lease->scratch.resize(4096);
  }
  track_allocations.store(false, std::memory_order_relaxed);
  require(allocation_count.load(std::memory_order_relaxed) == 0,
          "warm workspace lease performed a heap allocation");
  try {
    auto lease = pool.acquire();
    retained = lease.get();
    throw 17;
  } catch (int value) {
    require(value == 17, "workspace exception changed");
  }
  {
    auto lease = pool.acquire();
    require(lease.get() == retained,
            "exceptional scope did not return its workspace lease");
  }

  constexpr size_t worker_count = 8;
  std::array<Workspace *, worker_count> active{};
  std::barrier acquired(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier release(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&, worker] {
      auto lease = pool.acquire();
      active[worker] = lease.get();
      acquired.arrive_and_wait();
      release.arrive_and_wait();
    });
  }
  acquired.arrive_and_wait();
  std::unordered_set<Workspace *> unique(active.begin(), active.end());
  require(unique.size() == worker_count,
          "workspace pool shared one workspace between active leases");
  release.arrive_and_wait();
  for (auto &worker : workers)
    worker.join();
}
