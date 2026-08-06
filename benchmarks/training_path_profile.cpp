#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/common/function.hpp"
#include "sparse_embedding/src/interface/optimizer_interface.hpp"

namespace {

constexpr size_t kDimension = 128;
constexpr size_t kRows = 65'536;

struct ProfileSgd {
  void allocate(uint32_t) {}
  void deallocate(uint32_t) {}
  void clear() {}
  void set_page_eviction_enabled(bool) {}
  size_t capacity() const { return std::numeric_limits<size_t>::max(); }
  void reserve(size_t) {}
  void state_data(uint32_t, std::span<float>) const {}
  size_t state_dim() const { return 0; }
  size_t allocation_bytes_per_entry() const { return 0; }
  void set_state(uint32_t, std::span<const float>) {}
  uint32_t step_data(uint32_t) const { return 0; }
  void set_step(uint32_t, uint32_t) {}
  bool requires_score() const { return false; }
  void step(uint32_t, std::span<float> parameter,
            std::span<const float> gradient, uint64_t, uint64_t) {
    for (size_t index = 0; index < parameter.size(); ++index)
      parameter[index] -= 0.01F * gradient[index];
  }
};

double elapsed_seconds(const std::chrono::steady_clock::time_point started) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       started)
      .count();
}

void print_result(std::string_view name, size_t rows, double seconds) {
  if (seconds <= 0.0)
    throw std::runtime_error("profile elapsed time is non-positive");
  const double mops = static_cast<double>(rows) / seconds / 1e6;
  const double gib_per_second = static_cast<double>(rows) * kDimension *
                                sizeof(float) / seconds /
                                static_cast<double>(size_t{1} << 30);
  std::cout << name << " rows=" << rows << " seconds=" << std::fixed
            << std::setprecision(6) << seconds
            << " mops=" << std::setprecision(3) << mops
            << " payload_gib_s=" << gib_per_second << '\n';
}

} // namespace

size_t parse_positive(std::string_view text, std::string_view option) {
  size_t value = 0;
  for (char character : text) {
    if (character < '0' || character > '9')
      throw std::invalid_argument("invalid " + std::string(option));
    const size_t digit = static_cast<size_t>(character - '0');
    if (value > (std::numeric_limits<size_t>::max() - digit) / 10)
      throw std::overflow_error("overflowing " + std::string(option));
    value = value * 10 + digit;
  }
  if (value == 0)
    throw std::invalid_argument("zero " + std::string(option));
  return value;
}

int main(int argc, char **argv) {
  if (argc < 2)
    throw std::invalid_argument("expected assign, accumulate, or optimizer");
  const std::string_view mode(argv[1]);
  if (mode != "assign" && mode != "accumulate" && mode != "optimizer")
    throw std::invalid_argument("unknown training profile mode");
  bool bind_affinity = true;
  size_t workers = 8;
  size_t repetitions = 100;
  for (int argument = 2; argument < argc; ++argument) {
    const std::string_view option(argv[argument]);
    if (option == "--no-affinity") {
      bind_affinity = false;
    } else if (option.starts_with("--workers=")) {
      workers = parse_positive(
          option.substr(std::string_view("--workers=").size()), "workers");
    } else if (option.starts_with("--repetitions=")) {
      repetitions = parse_positive(
          option.substr(std::string_view("--repetitions=").size()),
          "repetitions");
    } else {
      throw std::invalid_argument("unknown training profile option");
    }
  }

  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  auto optimizer_factory = pro::make_proxy<se::FunctionFacade<
      pro::proxy<se::OptimizerFacade<float, float>>(size_t)>>([](size_t) {
    return pro::make_proxy<se::OptimizerFacade<float, float>, ProfileSgd>();
  });
  Store store({kDimension}, {}, optimizer_factory, se::NoScorePolicy{},
              se::HogwildStoreOptions{
                  .thread_budget = workers,
                  .locality = se::LocalityPolicy{bind_affinity
                                                     ? se::LocalityMode::kNuma
                                                     : se::LocalityMode::kAuto,
                                                 se::LocalityMode::kAuto}});
  std::vector<uint64_t> keys(kRows);
  std::vector<int64_t> lengths(kRows, static_cast<int64_t>(kDimension));
  std::vector<float> values(kRows * kDimension, 0.25F);
  std::vector<float> output(kRows * kDimension);
  std::vector<float> gradients(kRows * kDimension, 0.001F);
  std::vector<uint8_t> inserted(kRows);
  for (size_t index = 0; index < kRows; ++index)
    keys[index] = index + 1;
  store.lookup_or_insert<uint64_t>(keys, output, lengths, inserted);

  if (mode == "assign") {
    store.assign<uint64_t>(keys, values, lengths);
    const auto started = std::chrono::steady_clock::now();
    for (size_t repetition = 0; repetition < repetitions; ++repetition)
      store.assign<uint64_t>(keys, values, lengths);
    print_result("assign", kRows * repetitions, elapsed_seconds(started));
    return 0;
  }
  if (mode == "accumulate") {
    const auto started = std::chrono::steady_clock::now();
    for (size_t repetition = 0; repetition < repetitions; ++repetition) {
      store.accumulate_gradients<uint64_t>(keys, gradients, lengths);
      store.optimizer_step();
    }
    print_result("accumulate", kRows * repetitions, elapsed_seconds(started));
    return 0;
  }
  std::chrono::nanoseconds optimizer_elapsed{};
  for (size_t repetition = 0; repetition < repetitions; ++repetition) {
    store.accumulate_gradients<uint64_t>(keys, gradients, lengths);
    const auto started = std::chrono::steady_clock::now();
    store.optimizer_step();
    optimizer_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
  }
  print_result("optimizer", kRows * repetitions,
               std::chrono::duration<double>(optimizer_elapsed).count());
}
