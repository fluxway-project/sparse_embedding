// Full benchmark for HogwildEmbeddingStore covering all public APIs under
// realistic recommendation system embedding table workloads.
//
// Scenario: production-scale embedding table with mixed hot/cold access,
//   dim=128, 500K-key vocabulary, one slot, multi-thread.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/common/function.hpp"
#include "sparse_embedding/src/execution/topology/thread_affinity.hpp"
#include "sparse_embedding/src/interface/optimizer_interface.hpp"
#include "sparse_embedding/src/optimizer/optimizer_contract.hpp"

using namespace se;

// ============================================================================
// Workload parameters
// ============================================================================
static constexpr size_t kDim =
    128; // embedding dimension (128 floats = 512 bytes)
static constexpr size_t kUniqueKeys = 500'000; // vocabulary size
static constexpr size_t kBatchSize =
    65'536; // batch size (typical training step)
static constexpr size_t kHotRatio =
    10; // top-10% keys get 80% traffic (power-law)

// ============================================================================
// Dummy SGD optimizer
// ============================================================================
struct SGDOptimizer {
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
  void prefetch_state(uint32_t) {}
  bool requires_score() const { return false; }
  void step(uint32_t, std::span<float> param, std::span<const float> grad,
            uint64_t, uint64_t /*score*/) {
    static constexpr float lr = 0.01f;
    for (size_t i = 0; i < param.size(); ++i) {
      param[i] -= lr * grad[i];
    }
  }
  void update(const se::OptimizerBatchView<float> &batch, uint64_t) {
    for (size_t row = 0; row < batch.size(); ++row)
      step(batch.rows[row].index, batch.rows[row].parameters,
           batch.rows[row].gradients, 0, batch.rows[row].score);
  }
};

// ============================================================================
// Timing helper
// ============================================================================
struct Timer {
  std::chrono::high_resolution_clock::time_point t0;
  void start() { t0 = std::chrono::high_resolution_clock::now(); }
  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - t0)
        .count();
  }
};

// ============================================================================
// Power-law key generator (simulates hot-feature skew)
// ============================================================================
std::vector<uint64_t> generate_power_law_batch(size_t batch_size,
                                               size_t vocab_size,
                                               std::mt19937_64 &gen) {
  size_t hot_vocab = vocab_size / kHotRatio;
  std::vector<uint64_t> batch(batch_size);
  std::uniform_int_distribution<size_t> hot_dist(0, hot_vocab - 1);
  std::uniform_int_distribution<size_t> cold_dist(hot_vocab, vocab_size - 1);
  std::bernoulli_distribution is_hot(0.8);
  for (auto &k : batch) {
    size_t idx = is_hot(gen) ? hot_dist(gen) : cold_dist(gen);
    k = idx; // slot 0: upper 16 bits = 0
  }
  return batch;
}

void deduplicate_mutation_batch(std::vector<uint64_t> &batch) {
  std::sort(batch.begin(), batch.end());
  batch.erase(std::unique(batch.begin(), batch.end()), batch.end());
}

struct MutationBatch {
  std::vector<uint64_t> keys;
  std::vector<int64_t> row_lengths;
};

std::vector<MutationBatch>
make_mutation_batches(size_t repetitions, std::mt19937_64 &generator,
                      std::span<const uint64_t> keys) {
  if (keys.empty())
    throw std::invalid_argument("mutation benchmark key pool is empty");
  std::uniform_int_distribution<size_t> pick(0, keys.size() - 1);
  std::vector<MutationBatch> batches;
  batches.reserve(repetitions);
  for (size_t repetition = 0; repetition < repetitions; ++repetition) {
    MutationBatch batch{std::vector<uint64_t>(kBatchSize), {}};
    for (uint64_t &key : batch.keys)
      key = keys[pick(generator)];
    deduplicate_mutation_batch(batch.keys);
    batch.row_lengths.assign(batch.keys.size(), static_cast<int64_t>(kDim));
    batches.push_back(std::move(batch));
  }
  return batches;
}

// ============================================================================
// Benchmark runner
// ============================================================================
void print_header(const std::string &name) {
  std::cout
      << "\n┌─────────────────────────────────────────────────────────────┐\n"
      << "│  " << std::left << std::setw(59) << name << "│\n"
      << "└─────────────────────────────────────────────────────────────┘\n";
}

struct Result {
  std::string label;
  double mops;
  double gbps;
};

void print_result(const Result &r) {
  std::cout << std::left << std::setw(28) << r.label << std::right
            << std::setw(10) << std::fixed << std::setprecision(2) << r.mops
            << " Mops/s" << std::setw(10) << r.gbps << " GB/s\n";
}

bool starts_with(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

size_t parse_positive_size(std::string_view value, std::string_view name) {
  if (value.empty())
    throw std::invalid_argument(std::string(name) + " must not be empty");
  size_t parsed_end = 0;
  const unsigned long long parsed =
      std::stoull(std::string(value), &parsed_end);
  if (parsed_end != value.size() || parsed == 0 ||
      parsed > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument(std::string(name) + " is invalid");
  }
  return static_cast<size_t>(parsed);
}

size_t parse_nonnegative_size(std::string_view value, std::string_view name) {
  if (value.empty())
    throw std::invalid_argument(std::string(name) + " is empty");
  size_t parsed = 0;
  for (char character : value) {
    if (character < '0' || character > '9')
      throw std::invalid_argument(std::string(name) + " is not numeric");
    if (parsed > (std::numeric_limits<size_t>::max() -
                  static_cast<size_t>(character - '0')) /
                     10)
      throw std::overflow_error(std::string(name) + " overflows");
    parsed = parsed * 10 + static_cast<size_t>(character - '0');
  }
  return parsed;
}

void run_storage_pressure_benchmark() {
  print_header("Storage pressure: LFU background eviction");
  se::StorageOptions options;
  options.memory_limit_bytes = 5 << 20;
  options.spill_directory = std::filesystem::temp_directory_path();
  options.max_spill_bytes = 1 << 30;
  options.watermark = se::WatermarkConfig{0.8, 0.6};
  auto storage = se::create_page_storage(std::move(options));
  using PressureStore =
      HogwildEmbeddingStore<uint64_t, float, float, LfuScorePolicy>;
  auto opt_factory = pro::make_proxy<
      se::FunctionFacade<OptimizerExtension<float, float>(size_t)>>([](size_t) {
    return pro::make_proxy<OptimizerExtensionFacade<float, float>,
                           SGDOptimizer>();
  });
  constexpr size_t kPressureKeys = 16'384;
  constexpr size_t kPressureBatch = 4'096;
  PressureStore store(
      {kDim}, {}, opt_factory, LfuScorePolicy{},
      HogwildStoreOptions{.thread_budget = 2, .storage = storage});
  if (store.reserve(kPressureKeys, kDim))
    throw std::runtime_error("pressure benchmark reserve failed");
  std::mt19937_64 gen(42);
  std::vector<int64_t> row_lengths(kPressureBatch, static_cast<int64_t>(kDim));
  std::vector<float> output(kPressureBatch * kDim);
  std::vector<float> values(kPressureBatch * kDim, 0.1F);
  std::vector<float> gradients(kPressureBatch * kDim, 0.001F);
  auto batch = generate_power_law_batch(kPressureBatch, kPressureKeys, gen);
  deduplicate_mutation_batch(batch);
  if (std::adjacent_find(batch.begin(), batch.end()) != batch.end())
    throw std::logic_error("pressure insertion batch remained duplicated");
  std::cout << "pressure_initial_unique_rows=" << batch.size() << '\n';
  std::vector<int64_t> insertion_lengths(batch.size(),
                                         static_cast<int64_t>(kDim));
  std::vector<float> insertion_output(batch.size() * kDim);
  store.lookup_or_insert<uint64_t>(batch, insertion_output, insertion_lengths);

  Timer timer;
  timer.start();
  constexpr size_t kRepetitions = 4;
  for (size_t repetition = 0; repetition < kRepetitions; ++repetition) {
    batch = generate_power_law_batch(kPressureBatch, kPressureKeys, gen);
    deduplicate_mutation_batch(batch);
    row_lengths.assign(batch.size(), static_cast<int64_t>(kDim));
    output.resize(batch.size() * kDim);
    values.resize(batch.size() * kDim, 0.1F);
    gradients.resize(batch.size() * kDim, 0.001F);
    store.lookup<uint64_t>(batch, output, row_lengths);
    store.assign<uint64_t>(batch, values, row_lengths);
    store.accumulate_gradients<uint64_t>(batch, gradients, row_lengths);
    store.optimizer_step(1);
  }
  const double elapsed_ms = timer.elapsed_ms();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const auto stats = storage->stats();
  if (stats.swap_out_count == 0 || stats.eviction_selection_count == 0)
    throw std::runtime_error(
        "storage pressure benchmark did not trigger background eviction");
  const double mops =
      static_cast<double>(kPressureBatch * kRepetitions * 4) / elapsed_ms / 1e3;
  std::cout << "pressure_e2e                         " << std::fixed
            << std::setprecision(2) << mops << " Mops/s\n"
            << "pressure_memory_bytes                 " << stats.memory_bytes
            << "\n"
            << "pressure_spill_bytes                  " << stats.spill_bytes
            << "\n"
            << "pressure_swap_out_count               " << stats.swap_out_count
            << "\n"
            << "pressure_eviction_selection_count     "
            << stats.eviction_selection_count << "\n"
            << "pressure_eviction_evaluated_pages     "
            << stats.eviction_evaluated_pages << "\n"
            << "pressure_eviction_selection_ns       "
            << stats.eviction_selection_ns << "\n"
            << "pressure_eviction_invalidated_count   "
            << stats.eviction_candidate_invalidated_count << "\n";
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "pressure") {
    run_storage_pressure_benchmark();
    return 0;
  }
  std::optional<size_t> only_threads;
  std::optional<OperationDispatchConfig> operation_dispatch_config;
  std::optional<size_t> configured_physical_workers;
  std::optional<size_t> configured_memory_parallelism;
  std::optional<size_t> configured_prefetch_distance;
  bool run_pressure = true;
  bool bind_affinity = false;
  bool numa_aware_sharding = false;
  for (int arg_index = 1; arg_index < argc; ++arg_index) {
    const std::string_view arg(argv[arg_index]);
    constexpr std::string_view threads_prefix = "--threads=";
    constexpr std::string_view physical_prefix =
        "--operation-physical-workers=";
    constexpr std::string_view memory_prefix =
        "--operation-memory-parallelism=";
    constexpr std::string_view prefetch_prefix = "--forward-prefetch-distance=";
    if (arg == "--skip-pressure") {
      run_pressure = false;
    } else if (arg == "--bind-affinity") {
      bind_affinity = true;
    } else if (arg == "--numa-aware-sharding") {
      numa_aware_sharding = true;
    } else if (starts_with(arg, threads_prefix)) {
      only_threads =
          parse_positive_size(arg.substr(threads_prefix.size()), "threads");
    } else if (starts_with(arg, physical_prefix)) {
      configured_physical_workers = parse_positive_size(
          arg.substr(physical_prefix.size()), "operation physical workers");
    } else if (starts_with(arg, memory_prefix)) {
      configured_memory_parallelism = parse_positive_size(
          arg.substr(memory_prefix.size()), "operation memory parallelism");
    } else if (starts_with(arg, prefetch_prefix)) {
      configured_prefetch_distance = parse_nonnegative_size(
          arg.substr(prefetch_prefix.size()), "forward prefetch distance");
    } else {
      throw std::invalid_argument("unknown benchmark argument: " +
                                  std::string(arg));
    }
  }
  if (configured_memory_parallelism.has_value() &&
      !configured_physical_workers.has_value()) {
    throw std::invalid_argument(
        "operation memory parallelism requires physical worker topology");
  }
  if (configured_physical_workers.has_value()) {
    operation_dispatch_config = OperationDispatchConfig{
        *configured_physical_workers,
        configured_memory_parallelism.value_or(*configured_physical_workers),
        OperationDispatchConfig::kDefaultMinimumBytesPerThread, bind_affinity,
        configured_prefetch_distance.value_or(16)};
  }
  if (numa_aware_sharding && !operation_dispatch_config.has_value()) {
    operation_dispatch_config = OperationDispatchConfig{
        std::max<size_t>(1, std::thread::hardware_concurrency()),
        std::max<size_t>(1, std::thread::hardware_concurrency()),
        OperationDispatchConfig::kDefaultMinimumBytesPerThread, true,
        configured_prefetch_distance.value_or(16)};
  }
  if (bind_affinity) {
    const auto allowed_cpus = ThreadAffinity::allowed_cpus();
    std::cout << "  affinity=code-bound allowed_cpus=";
    for (size_t index = 0; index < allowed_cpus.size(); ++index) {
      if (index != 0)
        std::cout << ",";
      std::cout << allowed_cpus[index];
    }
    std::cout << " main_cpu=" << ThreadAffinity::current_cpu() << "\n";
  }
  if (numa_aware_sharding) {
    std::cout << "  numa_aware_sharding=enabled nodes="
              << se::NumaTopology::node_count() << "\n";
  }
  std::cout << "\n=== HogwildEmbeddingStore Full Benchmark ===\n"
            << "  dim=" << kDim << "  vocab=" << kUniqueKeys
            << "  batch=" << kBatchSize << "  power-law access\n";

  std::vector<size_t> thread_counts = {1, 2, 4, 8, 16, 64};
  if (only_threads.has_value())
    thread_counts = {*only_threads};
  std::vector<std::pair<size_t, double>> forward_mops;
  forward_mops.reserve(thread_counts.size());

  auto opt_factory = pro::make_proxy<
      se::FunctionFacade<OptimizerExtension<float, float>(size_t)>>(
      [](size_t d) {
        return pro::make_proxy<OptimizerExtensionFacade<float, float>,
                               SGDOptimizer>();
      });

  for (size_t num_threads : thread_counts) {
    std::optional<OperationDispatchConfig> store_dispatch_config =
        operation_dispatch_config;
    if (bind_affinity && !store_dispatch_config.has_value()) {
      store_dispatch_config =
          OperationDispatchConfig{num_threads, num_threads, 1, true};
    }
    if (numa_aware_sharding) {
      store_dispatch_config =
          OperationDispatchConfig{num_threads, num_threads, 1, true,
                                  configured_prefetch_distance.value_or(16)};
    }
    print_header("Threads: " + std::to_string(num_threads));
    std::cout << std::left << std::setw(28) << "Operation" << std::right
              << std::setw(17) << "Throughput" << std::setw(10) << "Bandwidth"
              << "\n"
              << std::string(56, '-') << "\n";

    HogwildEmbeddingStore<uint64_t, float, float, NoScorePolicy> store(
        {kDim}, {}, opt_factory, NoScorePolicy{},
        HogwildStoreOptions{.thread_budget = num_threads,
                            .locality =
                                LocalityPolicy{store_dispatch_config.has_value()
                                                   ? LocalityMode::kNuma
                                                   : LocalityMode::kAuto,
                                               numa_aware_sharding
                                                   ? LocalityMode::kNuma
                                                   : LocalityMode::kAuto},
                            .storage = create_page_storage()});
    if (store.reserve(kUniqueKeys, kDim))
      throw std::runtime_error("benchmark reserve failed");

    std::mt19937_64 gen(42);
    Timer timer;

    // ── Pre-allocate output buffers ─────────────────────────────────────────
    std::vector<int64_t> row_lengths(kBatchSize, static_cast<int64_t>(kDim));
    std::vector<float> flat_values(kBatchSize * kDim, 0.1f);
    std::vector<float> flat_out(kBatchSize * kDim, 0.0f);
    std::vector<float> flat_grads(kBatchSize * kDim, 0.001f);
    std::vector<uint8_t> exists_mask(kBatchSize);
    std::vector<uint64_t> mutation_keys;

    // ── 1. lookup_or_insert (cold: all new keys) ───────────────────────────
    {
      auto batch = generate_power_law_batch(kBatchSize, kUniqueKeys, gen);
      deduplicate_mutation_batch(batch);
      mutation_keys = batch;
      // warmup
      store.lookup_or_insert<uint64_t>(
          batch, std::span<float>(flat_out).first(batch.size() * kDim),
          std::span<const int64_t>(row_lengths).first(batch.size()));

      size_t reps = 10;
      size_t operation_count = 0;
      timer.start();
      for (size_t r = 0; r < reps; ++r) {
        auto b = generate_power_law_batch(kBatchSize, kUniqueKeys, gen);
        deduplicate_mutation_batch(b);
        operation_count += b.size();
        store.lookup_or_insert<uint64_t>(
            b, std::span<float>(flat_out).first(b.size() * kDim),
            std::span<const int64_t>(row_lengths).first(b.size()));
      }
      double ms = timer.elapsed_ms() / reps;
      double mops = operation_count / (ms * reps) / 1e3;
      double gbps = operation_count * kDim * sizeof(float) / (ms * reps) / 1e6;
      print_result({"lookup_or_insert", mops, gbps});
    }

    // ── 2. forward_lookup (warm: keys already inserted) ───────────────────
    {
      size_t reps = 20;
      timer.start();
      for (size_t r = 0; r < reps; ++r) {
        auto b = generate_power_law_batch(kBatchSize, kUniqueKeys, gen);
        store.forward_lookup<uint64_t>(b, flat_out, row_lengths);
      }
      double ms = timer.elapsed_ms() / reps;
      double mops = kBatchSize / ms / 1e3;
      double gbps = kBatchSize * kDim * sizeof(float) / ms / 1e6;
      print_result({"forward_lookup", mops, gbps});
      forward_mops.emplace_back(num_threads, mops);
    }

    // ── 3. assign (write embedding values) ────────────────────────────────
    {
      size_t reps = 10;
      auto batches = make_mutation_batches(reps, gen, mutation_keys);
      size_t operation_count = 0;
      for (const auto &batch : batches)
        operation_count += batch.keys.size();
      timer.start();
      for (const auto &batch : batches)
        store.assign<uint64_t>(batch.keys, flat_values, batch.row_lengths);
      double ms = timer.elapsed_ms() / reps;
      double mops = operation_count / (ms * reps) / 1e3;
      double gbps = operation_count * kDim * sizeof(float) / (ms * reps) / 1e6;
      print_result({"assign", mops, gbps});
    }

    // ── 4. accumulate_gradients ───────────────────────────────────────────
    {
      size_t reps = 10;
      auto batches = make_mutation_batches(reps, gen, mutation_keys);
      size_t operation_count = 0;
      for (const auto &batch : batches)
        operation_count += batch.keys.size();
      timer.start();
      for (const auto &batch : batches)
        store.accumulate_gradients<uint64_t>(batch.keys, flat_grads,
                                             batch.row_lengths);
      double ms = timer.elapsed_ms() / reps;
      double mops = operation_count / (ms * reps) / 1e3;
      double gbps = operation_count * kDim * sizeof(float) / (ms * reps) / 1e6;
      print_result({"accumulate_gradients", mops, gbps});
    }

    // ── 5. optimizer_step ─────────────────────────────────────────────────
    {
      // Ensure there's something accumulated
      size_t reps = 10;
      auto batches = make_mutation_batches(reps + 1, gen, mutation_keys);
      store.accumulate_gradients<uint64_t>(batches.front().keys, flat_grads,
                                           batches.front().row_lengths);
      timer.start();
      for (size_t r = 0; r < reps; ++r) {
        store.optimizer_step(1);
        store.accumulate_gradients<uint64_t>(batches[r + 1].keys, flat_grads,
                                             batches[r + 1].row_lengths);
      }
      double ms = timer.elapsed_ms() / reps;
      // optimizer_step processes the accumulated unique keys (up to kBatchSize)
      double mops = store.size() / ms / 1e3;
      double gbps =
          store.size() * kDim * sizeof(float) * 2 / ms / 1e6; // read+write
      print_result({"optimizer_step", mops, gbps});
      store.optimizer_step(1);
    }

    // ── 6. contains ───────────────────────────────────────────────────────
    {
      size_t reps = 20;
      timer.start();
      for (size_t r = 0; r < reps; ++r) {
        auto b = generate_power_law_batch(kBatchSize, kUniqueKeys, gen);
        store.contains<uint64_t>(b, exists_mask);
      }
      double ms = timer.elapsed_ms() / reps;
      double mops = kBatchSize / ms / 1e3;
      print_result({"contains", mops, 0.0});
    }

    // ── 7. erase (10% of batch) ───────────────────────────────────────────
    {
      size_t erase_n = kBatchSize / 10;
      std::vector<uint64_t> erase_keys(erase_n);
      std::uniform_int_distribution<uint64_t> dist(0, kUniqueKeys - 1);
      for (auto &k : erase_keys)
        k = dist(gen);

      size_t reps = 10;
      timer.start();
      for (size_t r = 0; r < reps; ++r) {
        store.erase<uint64_t>(erase_keys);
      }
      double ms = timer.elapsed_ms() / reps;
      double mops = erase_n / ms / 1e3;
      print_result({"erase (10% batch)", mops, 0.0});
    }

    std::cout << "  [store.size() = " << store.size() << " embeddings]\n";
  }

  if (!only_threads.has_value()) {
    const auto forward_for = [&forward_mops](size_t threads) {
      const auto result = std::ranges::find(forward_mops, threads,
                                            &std::pair<size_t, double>::first);
      if (result == forward_mops.end())
        throw std::logic_error(
            "benchmark forward result is missing a thread count");
      return result->second;
    };
    const double baseline = forward_for(1);
    if (baseline <= 0.0)
      throw std::runtime_error(
          "single-thread forward throughput is non-positive");
    for (const auto [threads, minimum_ratio] :
         std::array<std::pair<size_t, double>, 3>{
             {{2, 1.6}, {4, 2.5}, {8, 3.5}}}) {
      const double ratio = forward_for(threads) / baseline;
      std::cout << "forward_scaling_" << threads << "_ratio=" << ratio << '\n';
      if (ratio < minimum_ratio)
        throw std::runtime_error("forward scaling target was not met");
    }
  }

  if (run_pressure)
    run_storage_pressure_benchmark();
  std::cout << "\n=== Benchmark complete ===\n";
  return 0;
}
