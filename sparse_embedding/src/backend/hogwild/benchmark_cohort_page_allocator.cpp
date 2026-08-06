#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/core/cohort_page_allocator.hpp"

namespace {

constexpr size_t kAllocationsPerWorker = 500000;
constexpr size_t kShardCount = 64;

struct Result {
  double seconds;
  uint64_t checksum;
};

Result run_central(size_t workers) {
  std::atomic<uint32_t> next_index{0};
  std::barrier start(workers + 1);
  std::vector<uint64_t> checksums(workers);
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (size_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      uint64_t checksum = 0;
      start.arrive_and_wait();
      for (size_t allocation = 0; allocation < kAllocationsPerWorker;
           ++allocation) {
        checksum += next_index.fetch_add(1, std::memory_order_relaxed);
      }
      checksums[worker] = checksum;
    });
  }
  const auto began = std::chrono::steady_clock::now();
  start.arrive_and_wait();
  for (std::thread &thread : threads)
    thread.join();
  uint64_t checksum = 0;
  for (uint64_t worker_checksum : checksums)
    checksum += worker_checksum;
  return {
      std::chrono::duration<double>(std::chrono::steady_clock::now() - began)
          .count(),
      checksum};
}

Result run_cohort(size_t workers) {
  se::CohortPageAllocator allocator(kShardCount, 16, 4096, 4096);
  std::barrier start(workers + 1);
  std::vector<uint64_t> checksums(workers);
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (size_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      uint64_t checksum = 0;
      start.arrive_and_wait();
      for (size_t allocation = 0; allocation < kAllocationsPerWorker;
           ++allocation) {
        const uint32_t shard = static_cast<uint32_t>(
            (worker * 131 + allocation) & (kShardCount - 1));
        checksum += allocator.allocate(shard).index;
      }
      checksums[worker] = checksum;
    });
  }
  const auto began = std::chrono::steady_clock::now();
  start.arrive_and_wait();
  for (std::thread &thread : threads)
    thread.join();
  uint64_t checksum = 0;
  for (uint64_t worker_checksum : checksums)
    checksum += worker_checksum;
  return {
      std::chrono::duration<double>(std::chrono::steady_clock::now() - began)
          .count(),
      checksum};
}

size_t parse_workers(std::string_view value) {
  size_t workers = 0;
  for (char digit : value) {
    if (digit < '0' || digit > '9')
      throw std::invalid_argument(
          "allocator benchmark worker count is invalid");
    workers = workers * 10 + static_cast<size_t>(digit - '0');
  }
  if (workers == 0)
    throw std::invalid_argument("allocator benchmark worker count is zero");
  return workers;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3)
    throw std::invalid_argument("expected mode and worker count");
  const std::string_view mode(argv[1]);
  const size_t workers = parse_workers(argv[2]);
  Result result{};
  if (mode == "central") {
    result = run_central(workers);
  } else if (mode == "cohort") {
    result = run_cohort(workers);
  } else {
    throw std::invalid_argument("allocator benchmark mode is invalid");
  }
  if (result.seconds <= 0.0 || result.checksum == 0)
    throw std::runtime_error("allocator benchmark produced an invalid result");
  const double throughput = workers * kAllocationsPerWorker / result.seconds;
  std::cout << "allocation_benchmark mode=" << mode << " workers=" << workers
            << " seconds=" << result.seconds << " throughput=" << throughput
            << " checksum=" << result.checksum << '\n';
}
