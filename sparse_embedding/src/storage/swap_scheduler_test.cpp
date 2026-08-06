#include "sparse_embedding/src/storage/swap_scheduler.hpp"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace {

struct Counter {
  std::atomic<unsigned> value{0};
};

void increment(void *context) {
  auto &counter = *static_cast<Counter *>(context);
  counter.value.fetch_add(1, std::memory_order_release);
  counter.value.notify_all();
}

void fail(void *) { throw std::runtime_error("expected scheduler failure"); }

void wait_for_value(Counter &counter, unsigned expected) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (counter.value.load(std::memory_order_acquire) < expected) {
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error("swap scheduler task timed out");
    const unsigned observed = counter.value.load(std::memory_order_relaxed);
    counter.value.wait(observed, std::memory_order_acquire);
  }
}

} // namespace

int main() {
  Counter counter;
  se::storage_detail::SwapScheduler scheduler;
  scheduler.ensure_started(&counter, increment);
  scheduler.wake();
  wait_for_value(counter, 1);
  scheduler.request_stop();
  scheduler.stop_and_join();

  scheduler.ensure_started(&counter, increment);
  scheduler.wake();
  wait_for_value(counter, 2);
  scheduler.stop_and_join();

  scheduler.ensure_started(&counter, fail);
  scheduler.wake();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!scheduler.stop_requested()) {
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error("swap scheduler failure was not captured");
    std::this_thread::yield();
  }
  bool observed_failure = false;
  try {
    scheduler.rethrow_if_failed();
  } catch (const std::runtime_error &) {
    observed_failure = true;
  }
  if (!observed_failure)
    throw std::runtime_error("swap scheduler failure was not propagated");
  scheduler.stop_and_join();
}
