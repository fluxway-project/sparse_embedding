// Full benchmark for HogwildEmbeddingStore covering all public APIs under
// realistic recommendation system embedding table workloads.
//
// Scenario: production-scale embedding table with mixed hot/cold access,
//   dim=128, 500K-key vocabulary, one slot, multi-thread.

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/common/function.hpp"
#include "sparse_embedding/src/interface/optimizer_interface.hpp"

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
  void set_state(uint32_t, std::span<const float>) {}
  uint32_t step_data(uint32_t) const { return 0; }
  void set_step(uint32_t, uint32_t) {}
  void step(uint32_t, std::span<float> param, std::span<const float> grad,
            uint64_t) {
    static constexpr float lr = 0.01f;
    for (size_t i = 0; i < param.size(); ++i) {
      param[i] -= lr * grad[i];
    }
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
      se::FunctionFacade<pro::proxy<OptimizerFacade<float, float>>(size_t)>>(
      [](size_t) {
        return pro::make_proxy<OptimizerFacade<float, float>, SGDOptimizer>();
      });
  constexpr size_t kPressureKeys = 16'384;
  constexpr size_t kPressureBatch = 4'096;
  PressureStore store({kDim}, 2, {}, opt_factory, LfuScorePolicy{}, 16,
                      storage);
  if (store.reserve(kPressureKeys, kDim))
    throw std::runtime_error("pressure benchmark reserve failed");
  std::mt19937_64 gen(42);
  std::vector<int64_t> row_lengths(kPressureBatch, static_cast<int64_t>(kDim));
  std::vector<float> output(kPressureBatch * kDim);
  std::vector<float> values(kPressureBatch * kDim, 0.1F);
  std::vector<float> gradients(kPressureBatch * kDim, 0.001F);
  std::vector<uint8_t> inserted(kPressureBatch);
  auto batch = generate_power_law_batch(kPressureBatch, kPressureKeys, gen);
  store.lookup_or_insert<uint64_t>(batch, output, row_lengths, inserted);

  Timer timer;
  timer.start();
  constexpr size_t kRepetitions = 4;
  for (size_t repetition = 0; repetition < kRepetitions; ++repetition) {
    batch = generate_power_law_batch(kPressureBatch, kPressureKeys, gen);
    store.lookup<uint64_t>(batch, output, row_lengths);
    store.assign<uint64_t>(batch, values, row_lengths);
    store.accumulate_gradients<uint64_t>(batch, gradients, row_lengths);
    store.optimizer_step();
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
  std::cout << "\n=== HogwildEmbeddingStore Full Benchmark ===\n"
            << "  dim=" << kDim << "  vocab=" << kUniqueKeys
            << "  batch=" << kBatchSize << "  power-law access\n";

  std::vector<size_t> thread_counts = {1, 4, 8, 16, 32, 64};

  auto opt_factory = pro::make_proxy<
      se::FunctionFacade<pro::proxy<OptimizerFacade<float, float>>(size_t)>>(
      [](size_t d) {
        return pro::make_proxy<OptimizerFacade<float, float>, SGDOptimizer>();
      });

  for (size_t num_threads : thread_counts) {
    print_header("Threads: " + std::to_string(num_threads));
    std::cout << std::left << std::setw(28) << "Operation" << std::right
              << std::setw(17) << "Throughput" << std::setw(10) << "Bandwidth"
              << "\n"
              << std::string(56, '-') << "\n";

    HogwildEmbeddingStore<uint64_t, float, float, NoScorePolicy> store(
        {kDim}, num_threads, {}, opt_factory, NoScorePolicy{});
    if (store.reserve(kUniqueKeys, kDim))
      throw std::runtime_error("benchmark reserve failed");

    std::mt19937_64 gen(42);
    Timer timer;

    // ── Pre-allocate output buffers ─────────────────────────────────────────
    std::vector<int64_t> row_lengths(kBatchSize, static_cast<int64_t>(kDim));
    std::vector<float> flat_values(kBatchSize * kDim, 0.1f);
    std::vector<float> flat_out(kBatchSize * kDim, 0.0f);
    std::vector<float> flat_grads(kBatchSize * kDim, 0.001f);
    std::vector<uint8_t> inserted_mask(kBatchSize);
    std::vector<uint8_t> exists_mask(kBatchSize);

    // ── 1. lookup_or_insert (cold: all new keys) ───────────────────────────
    {
      auto batch = generate_power_law_batch(kBatchSize, kUniqueKeys, gen);
      // warmup
      store.lookup_or_insert<uint64_t>(batch, flat_out, row_lengths,
                                       inserted_mask);

      size_t reps = 10;
      timer.start();
      for (size_t r = 0; r < reps; ++r) {
        auto b = generate_power_law_batch(kBatchSize, kUniqueKeys, gen);
        store.lookup_or_insert<uint64_t>(b, flat_out, row_lengths,
                                         inserted_mask);
      }
      double ms = timer.elapsed_ms() / reps;
      double mops = kBatchSize / ms / 1e3;
      double gbps = kBatchSize * kDim * sizeof(float) / ms / 1e6;
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
    }

    // ── 3. assign (write embedding values) ────────────────────────────────
    {
      size_t reps = 10;
      timer.start();
      for (size_t r = 0; r < reps; ++r) {
        auto b = generate_power_law_batch(kBatchSize, kUniqueKeys, gen);
        store.assign<uint64_t>(b, flat_values, row_lengths);
      }
      double ms = timer.elapsed_ms() / reps;
      double mops = kBatchSize / ms / 1e3;
      double gbps = kBatchSize * kDim * sizeof(float) / ms / 1e6;
      print_result({"assign", mops, gbps});
    }

    // ── 4. accumulate_gradients ───────────────────────────────────────────
    {
      size_t reps = 10;
      timer.start();
      for (size_t r = 0; r < reps; ++r) {
        auto b = generate_power_law_batch(kBatchSize, kUniqueKeys, gen);
        store.accumulate_gradients<uint64_t>(b, flat_grads, row_lengths);
      }
      double ms = timer.elapsed_ms() / reps;
      double mops = kBatchSize / ms / 1e3;
      double gbps = kBatchSize * kDim * sizeof(float) / ms / 1e6;
      print_result({"accumulate_gradients", mops, gbps});
    }

    // ── 5. optimizer_step ─────────────────────────────────────────────────
    {
      // Ensure there's something accumulated
      {
        auto b = generate_power_law_batch(kBatchSize, kUniqueKeys, gen);
        store.accumulate_gradients<uint64_t>(b, flat_grads, row_lengths);
      }
      size_t reps = 10;
      timer.start();
      for (size_t r = 0; r < reps; ++r) {
        store.optimizer_step();
        // Re-accumulate for next rep
        auto b = generate_power_law_batch(kBatchSize, kUniqueKeys, gen);
        store.accumulate_gradients<uint64_t>(b, flat_grads, row_lengths);
      }
      double ms = timer.elapsed_ms() / reps;
      // optimizer_step processes the accumulated unique keys (up to kBatchSize)
      double mops = store.size() / ms / 1e3;
      double gbps =
          store.size() * kDim * sizeof(float) * 2 / ms / 1e6; // read+write
      print_result({"optimizer_step", mops, gbps});
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

  run_storage_pressure_benchmark();
  std::cout << "\n=== Benchmark complete ===\n";
  return 0;
}
