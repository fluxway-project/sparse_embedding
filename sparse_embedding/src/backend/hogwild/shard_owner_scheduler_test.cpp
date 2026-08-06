#include "sparse_embedding/src/backend/hogwild/partition/shard_owner_scheduler.hpp"

#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Function> void require_throws(Function &&function) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected operation to throw");
}

} // namespace

int main() {
  constexpr size_t shard_count = 64;
  constexpr size_t worker_count = 16;
  constexpr size_t rounds = 8;
  std::array<std::unique_ptr<se::ShardOwnerScheduler>, shard_count> schedulers;
  std::array<std::atomic<size_t>, shard_count> active{};
  std::array<std::atomic<size_t>, shard_count> executions{};
  for (auto &scheduler : schedulers)
    scheduler = std::make_unique<se::ShardOwnerScheduler>();
  std::barrier start(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&, worker] {
      start.arrive_and_wait();
      for (size_t round = 0; round < rounds; ++round) {
        for (size_t offset = 0; offset < shard_count; ++offset) {
          const size_t shard = (worker + offset) % shard_count;
          schedulers[shard]->submit([&, shard] {
            require(active[shard].fetch_add(1, std::memory_order_acq_rel) == 0,
                    "two owners published one shard concurrently");
            std::this_thread::yield();
            executions[shard].fetch_add(1, std::memory_order_relaxed);
            require(active[shard].fetch_sub(1, std::memory_order_acq_rel) == 1,
                    "shard owner activity underflowed");
          });
        }
      }
    });
  }
  start.arrive_and_wait();
  for (auto &worker : workers)
    worker.join();
  for (const auto &count : executions) {
    require(count.load(std::memory_order_relaxed) == worker_count * rounds,
            "shard-owner scheduler lost a publication request");
  }

  se::ShardOwnerScheduler scheduler;
  require_throws([&] { scheduler.submit([&] { scheduler.submit([] {}); }); });
  require_throws([&] {
    scheduler.submit([] { throw std::runtime_error("publication failed"); });
  });
  size_t recovered = 0;
  scheduler.submit([&] { ++recovered; });
  require(recovered == 1,
          "shard-owner scheduler did not recover after callback failure");
}
