#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

#include <algorithm>
#include <array>
#include <barrier>
#include <cmath>
#include <exception>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void run_concurrent_erase_reinsert_regression() {
  se::StorageOptions options;
  options.memory_limit_bytes = size_t{1} << 30;
  options.spill_directory = std::filesystem::temp_directory_path();
  options.max_spill_bytes = 256 << 20;
  auto storage = se::create_page_storage(std::move(options));
  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  Store store({4}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = 4, .storage = storage});
  constexpr size_t key_count = 256;
  constexpr size_t dim = 4;
  constexpr size_t rounds = 32;
  std::vector<uint64_t> keys(key_count);
  std::iota(keys.begin(), keys.end(), uint64_t{1});
  std::vector<int64_t> lengths(key_count, dim);
  std::vector<float> values(key_count * dim);
  for (size_t row = 0; row < key_count; ++row)
    std::fill_n(values.data() + row * dim, dim, static_cast<float>(keys[row]));
  store.assign<uint64_t>(keys, values, lengths);

  std::barrier start(3);
  std::array<std::exception_ptr, 3> errors;
  std::array<std::thread, 3> workers;
  workers[0] = std::thread([&] {
    try {
      start.arrive_and_wait();
      for (size_t round = 0; round < rounds; ++round)
        store.erase<uint64_t>(keys);
    } catch (...) {
      errors[0] = std::current_exception();
    }
  });
  workers[1] = std::thread([&] {
    try {
      start.arrive_and_wait();
      for (size_t round = 0; round < rounds; ++round)
        store.assign<uint64_t>(keys, values, lengths);
    } catch (...) {
      errors[1] = std::current_exception();
    }
  });
  workers[2] = std::thread([&] {
    try {
      start.arrive_and_wait();
      std::vector<float> output(values.size());
      for (size_t round = 0; round < rounds; ++round)
        store.lookup_or_insert<uint64_t>(keys, output, lengths);
    } catch (...) {
      errors[2] = std::current_exception();
    }
  });
  for (auto &worker : workers)
    worker.join();
  for (const auto &error : errors)
    if (error != nullptr)
      std::rethrow_exception(error);

  store.assign<uint64_t>(keys, values, lengths);
  require(store.size() == key_count,
          "concurrent erase/reinsert produced an invalid size");
  std::vector<uint8_t> present(key_count);
  store.contains<uint64_t>(keys, present);
  require(
      std::ranges::all_of(present, [](uint8_t value) { return value == 1; }),
      "concurrent erase/reinsert lost a key");
  std::vector<float> output(values.size());
  store.lookup<uint64_t>(keys, output, lengths);
  require(std::ranges::all_of(output,
                              [](float value) { return std::isfinite(value); }),
          "concurrent erase/reinsert produced a non-finite value");
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1) {
    if (argc != 2 || std::string_view(argv[1]) != "concurrent_erase_reinsert")
      throw std::invalid_argument("unknown mutation contract regression");
    run_concurrent_erase_reinsert_regression();
    return 0;
  }
  run_concurrent_erase_reinsert_regression();
  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  constexpr std::array<uint64_t, 2> keys{7, 7};
  constexpr std::array<int64_t, 2> dims{4, 4};
  std::array<float, 8> values{};
  Store lookup_store({4}, {}, nullptr, se::NoScorePolicy{},
                     se::HogwildStoreOptions{.thread_budget = 1});
  values.fill(0.0F);
  lookup_store.lookup_or_insert<uint64_t>(keys, values, dims);
  const bool outputs_match = values[0] == values[4] && values[1] == values[5] &&
                             values[2] == values[6] && values[3] == values[7];
  return outputs_match && lookup_store.size() == 1 ? 0 : 1;
}
