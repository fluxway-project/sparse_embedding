#include <array>
#include <barrier>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

namespace {

constexpr size_t kOwnerCount =
    16 * 4 * se::EmbeddingArena<float>::kEntriesPerPage;
constexpr size_t kRepetitions = 32;
constexpr size_t kSamples = 5;
constexpr std::array<size_t, 5> kWorkerCounts{1, 2, 4, 8, 16};

enum class Mode { kPublish, kClear, kPublishClear };

std::string_view mode_name(Mode mode) {
  switch (mode) {
  case Mode::kPublish:
    return "publish";
  case Mode::kClear:
    return "clear";
  case Mode::kPublishClear:
    return "publish_clear";
  }
  throw std::logic_error("owner benchmark mode is invalid");
}

Mode parse_mode(std::string_view value) {
  if (value == "publish")
    return Mode::kPublish;
  if (value == "clear")
    return Mode::kClear;
  if (value == "publish_clear")
    return Mode::kPublishClear;
  throw std::invalid_argument("owner benchmark mode is invalid");
}

size_t parse_workers(std::string_view value) {
  size_t workers = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), workers);
  if (error != std::errc{} || end != value.data() + value.size() ||
      workers == 0 || kOwnerCount % workers != 0) {
    throw std::invalid_argument("owner benchmark worker count is invalid");
  }
  return workers;
}

double run_once(se::EmbeddingArena<float> &arena, Mode mode, size_t workers) {
  const size_t owners_per_worker = kOwnerCount / workers;
  std::barrier start(workers + 1);
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(workers);
  threads.reserve(workers);
  for (size_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      start.arrive_and_wait();
      try {
        const uint32_t begin =
            static_cast<uint32_t>(worker * owners_per_worker);
        const uint32_t end = static_cast<uint32_t>(begin + owners_per_worker);
        for (size_t repetition = 0; repetition < kRepetitions; ++repetition) {
          for (uint32_t index = begin; index < end; ++index) {
            if (mode == Mode::kPublish || mode == Mode::kPublishClear)
              arena.publish_owner(index, static_cast<uint64_t>(index) + 1,
                                  index & 63U);
            if (mode == Mode::kClear || mode == Mode::kPublishClear)
              arena.mark_owner_not_live(index);
          }
        }
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  }
  const auto began = std::chrono::steady_clock::now();
  start.arrive_and_wait();
  for (auto &thread : threads)
    thread.join();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - began)
          .count();
  for (const auto &error : errors) {
    if (error != nullptr)
      std::rethrow_exception(error);
  }
  if (seconds <= 0.0)
    throw std::runtime_error("owner benchmark timer did not advance");
  return seconds;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3)
    throw std::invalid_argument("expected owner mode and worker count");
  const Mode mode = parse_mode(argv[1]);
  const size_t workers = parse_workers(argv[2]);
  se::StorageOptions options;
  options.mode = se::StorageMode::kMemoryOnly;
  options.memory_limit_bytes = size_t{1} << 30;
  auto storage = se::create_page_storage(std::move(options));
  se::EmbeddingArena<float> arena(1, false, storage);
  arena.reserve_entries(kOwnerCount);
  for (uint32_t index = 0; index < kOwnerCount; ++index) {
    arena.publish_owner(index, static_cast<uint64_t>(index) + 1, index & 63U);
    arena.mark_owner_not_live(index);
  }
  std::vector<double> elapsed;
  elapsed.reserve(kSamples);
  for (size_t sample = 0; sample < kSamples; ++sample)
    elapsed.push_back(run_once(arena, mode, workers));
  std::ranges::sort(elapsed);
  const double median_seconds = elapsed[elapsed.size() / 2];
  const uint64_t operations = kOwnerCount * kRepetitions;
  std::cout << "owner_mode=" << mode_name(mode) << " workers=" << workers
            << " samples=" << kSamples << " owners=" << kOwnerCount
            << " repetitions=" << kRepetitions << " operations=" << operations
            << " elapsed_seconds=" << median_seconds
            << " operations_per_second=" << operations / median_seconds << '\n';
}
