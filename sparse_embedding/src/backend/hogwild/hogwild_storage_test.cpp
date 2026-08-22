#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/optimizer/adam_optimizer.hpp"
#include "sparse_embedding/src/optimizer/sgd_optimizer.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function> void require_throws(Function &&function) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected operation to throw");
}

void run_optimizer_step_dim1_regression(std::string_view mode) {
  const bool run_compaction = mode == "all" || mode == "compact";
  const bool run_lookup = mode == "all" || mode == "lookup";
  if (mode != "step" && mode != "lookup" && mode != "compact" && mode != "all")
    throw std::invalid_argument("invalid optimizer_step_dim1 mode");

  se::StorageOptions options;
  options.memory_limit_bytes = 32 << 20;
  options.spill_directory = std::filesystem::temp_directory_path();
  options.max_spill_bytes = 256 << 20;
  auto storage = se::create_page_storage(std::move(options));
  se::AdamConfig config;
  auto factory = pro::make_proxy<
      se::FunctionFacade<se::OptimizerExtension<float, float>(size_t)>>(
      [storage, config](size_t dim) mutable {
        using Adam = se::AdamOptimizer<float, float, se::StaticLearningRate>;
        return pro::make_proxy<se::OptimizerExtensionFacade<float, float>,
                               Adam>(dim, config, storage,
                                     se::StaticLearningRate{1.0e-3F});
      });
  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  Store store({1}, {}, std::move(factory), se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = 4, .storage = storage});
  constexpr size_t key_count = 1024;
  constexpr size_t rounds = 64;
  std::vector<uint64_t> keys(key_count);
  std::iota(keys.begin(), keys.end(), uint64_t{1});
  std::vector<int64_t> lengths(key_count, 1);
  std::vector<float> values(key_count, 0.0F);
  store.lookup_or_insert<uint64_t>(keys, values, lengths);

  const size_t participants = 1 + run_compaction + run_lookup;
  std::barrier start(static_cast<std::ptrdiff_t>(participants));
  std::vector<std::exception_ptr> errors(participants);
  std::vector<std::thread> workers;
  workers.emplace_back([&] {
    try {
      start.arrive_and_wait();
      std::vector<float> gradients(key_count, 1.0F);
      for (size_t round = 0; round < rounds; ++round) {
        store.accumulate_gradients<uint64_t>(keys, gradients, lengths);
        store.optimizer_step();
      }
    } catch (...) {
      errors[0] = std::current_exception();
    }
  });
  if (run_compaction) {
    const size_t error_index = workers.size();
    workers.emplace_back([&, error_index] {
      try {
        start.arrive_and_wait();
        require(store.compact_page(1, 0) == key_count,
                "dim=1 Adam compaction lost a key");
      } catch (...) {
        errors[error_index] = std::current_exception();
      }
    });
  }
  if (run_lookup) {
    const size_t error_index = workers.size();
    workers.emplace_back([&, error_index] {
      try {
        start.arrive_and_wait();
        std::vector<float> output(key_count);
        for (size_t round = 0; round < rounds; ++round)
          store.lookup<uint64_t>(keys, output, lengths);
      } catch (...) {
        errors[error_index] = std::current_exception();
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  for (const auto &error : errors)
    if (error != nullptr)
      std::rethrow_exception(error);

  std::vector<float> output(key_count);
  store.lookup<uint64_t>(keys, output, lengths);
  require(store.size() == key_count, "dim=1 Adam regression lost a key");
  require(std::ranges::all_of(output,
                              [](float value) { return std::isfinite(value); }),
          "dim=1 Adam regression produced a non-finite value");
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1) {
    const std::string_view regression = argv[1];
    if (regression == "optimizer_step_dim1") {
      if (argc != 3)
        throw std::invalid_argument(
            "optimizer_step_dim1 requires an explicit mode");
      run_optimizer_step_dim1_regression(argv[2]);
      return 0;
    }
    throw std::invalid_argument("unknown hogwild storage regression");
  }
  run_optimizer_step_dim1_regression("all");

  se::StorageOptions options;
  options.memory_limit_bytes = 0;
  options.spill_directory = std::filesystem::temp_directory_path();
  options.max_spill_bytes = 256 << 20;
  auto storage = se::create_page_storage(std::move(options));

  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  using ScoredStore =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::ManualScorePolicy>;
  using AdamStore =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  se::SgdConfig training_config;
  training_config.lr = 0.01F;
  Store store({8}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = 1, .storage = storage});
  require_throws([&] { store.promote_hot_keys(8, 0, 1, 0); });

  constexpr std::array<uint64_t, 3> keys{11, 22, 33};
  constexpr std::array<int64_t, 3> lengths{8, 8, 8};
  std::array<float, 24> values{};
  store.lookup_or_insert<uint64_t>(keys, values, lengths);
  require(store.size() == keys.size(), "store size mismatch");
  require(storage->stats().spill_bytes > 0,
          "store did not use shared spill storage");

  {
    Store duplicate_gradient_store(
        {8}, {}, nullptr, se::NoScorePolicy{},
        se::HogwildStoreOptions{.thread_budget = 1, .storage = storage});
    constexpr std::array<uint64_t, 1> duplicate_key{9911};
    constexpr std::array<int64_t, 1> duplicate_length{8};
    std::array<float, 8> duplicate_values{};
    duplicate_gradient_store.lookup_or_insert<uint64_t>(
        duplicate_key, duplicate_values, duplicate_length);
    constexpr std::array<uint64_t, 3> repeated_keys{9911, 9911, 9911};
    constexpr std::array<int64_t, 3> repeated_lengths{8, 8, 8};
    std::array<float, 24> repeated_gradients{};
    for (size_t row = 0; row < 3; ++row)
      std::fill_n(repeated_gradients.data() + row * 8, 8,
                  static_cast<float>(row + 1));
    duplicate_gradient_store.accumulate_gradients<uint64_t>(
        repeated_keys, repeated_gradients, repeated_lengths);
    duplicate_gradient_store.optimizer_step();
  }
  constexpr se::AdaptivePageSelectorConfig sealing_config{0.0F, 0.0F, 0.75F, 1,
                                                          4096 * 64};
  auto sealed_pages = store.seal_adaptive_pages(
      8, 1.0F, sealing_config, se::PageThermalTier::kCold, 17, 1);
  require(sealed_pages.size() == 1 &&
              sealed_pages[0]->layout().entry_count == keys.size() &&
              sealed_pages[0]->layout().tier == se::PageThermalTier::kCold &&
              sealed_pages[0]->layout().cluster_id == 17,
          "live arena sealing did not publish the expected page layout");
  require(std::ranges::equal(sealed_pages[0]->keys(), keys) &&
              std::ranges::equal(sealed_pages[0]->value(0),
                                 std::span<const float>(values).first(8)),
          "live arena sealing did not preserve stable payload snapshots");

  constexpr size_t residency_page_key_count =
      se::EmbeddingArena<float>::kEntriesPerPage;
  Store residency_store(
      {8}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{
          .thread_budget = 1,
          .storage = storage,
          .storage_tuning = {.entries_per_page =
                                 se::EmbeddingArena<float>::kEntriesPerPage,
                             .calibrate_page_geometry = false}});
  std::vector<uint64_t> residency_old_keys(residency_page_key_count);
  std::iota(residency_old_keys.begin(), residency_old_keys.end(),
            uint64_t{3000000});
  std::vector<int64_t> residency_lengths(residency_page_key_count, 8);
  std::vector<float> residency_old_values(residency_page_key_count * 8, 3.0F);
  residency_store.assign<uint64_t>(residency_old_keys, residency_old_values,
                                   residency_lengths);
  const auto residency_seed = residency_store.seal_adaptive_pages(
      8, 1.0F, se::AdaptivePageSelectorConfig{0.0F, 0.0F, 0.75F, 1, 4096 * 64},
      se::PageThermalTier::kCold, 21, 1);
  require(residency_seed.size() == 1,
          "residency control plane did not seal the initial page");
  std::vector<uint64_t> residency_new_keys(residency_page_key_count);
  std::iota(residency_new_keys.begin(), residency_new_keys.end(),
            uint64_t{3000000 + residency_page_key_count});
  std::vector<float> residency_new_values(residency_page_key_count * 8, 4.0F);
  residency_store.assign<uint64_t>(residency_new_keys, residency_new_values,
                                   residency_lengths);
  const auto residency_selection = residency_store.seal_adaptive_pages(
      8, 1.0F,
      se::AdaptivePageSelectorConfig{2.0F, 0.0F, 0.75F, 2, 2 * 4096 * 64},
      se::PageThermalTier::kCold, 22, 2);
  require(residency_selection.size() == 1 &&
              std::ranges::equal(residency_selection[0]->keys(),
                                 residency_old_keys),
          "adaptive residency age did not exclude the newly allocated page");

  bool page_image_seen = false;
  store.export_page_images(pro::make_proxy<se::FunctionFacade<bool(
                               size_t, se::PageImageExport<float, float> &&)>>(
      [&](size_t dim, se::PageImageExport<float, float> &&image) {
        if (dim == 8 && image.physical_page_id == 0 &&
            image.embedding_dim == 8 && image.page_slot_stride_bytes == 64) {
          page_image_seen =
              std::ranges::equal(image.keys, keys) &&
              std::ranges::equal(image.slot_ordinals,
                                 std::array<uint32_t, 3>{0, 1, 2}) &&
              std::ranges::equal(std::span<const float>(image.values).first(8),
                                 std::span<const float>(values).first(8));
        }
        return true;
      }));
  require(page_image_seen,
          "physical page image did not preserve locator and payload layout");
  Store page_image_restore_store(
      {8}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 1, .storage = storage});
  std::vector<se::PageImageImport<float, float>> page_image_imports;
  store.export_page_images(pro::make_proxy<se::FunctionFacade<bool(
                               size_t, se::PageImageExport<float, float> &&)>>(
      [&](size_t dim, se::PageImageExport<float, float> &&image) {
        page_image_imports.push_back({dim, std::move(image)});
        return true;
      }));
  std::array<float, 24> mutated_values{};
  mutated_values.fill(91.0F);
  store.assign<uint64_t>(keys, mutated_values, lengths);
  std::array<float, 24> mutated_lookup{};
  store.lookup<uint64_t>(keys, mutated_lookup, lengths);
  require(std::ranges::equal(mutated_lookup, mutated_values),
          "source mutation did not publish");
  require(std::ranges::equal(page_image_imports.front().image.values, values),
          "page image export aliased mutable arena payload");
  page_image_restore_store.import_page_images(page_image_imports);
  std::vector<float> page_image_restored_values(values.size());
  page_image_restore_store.lookup<uint64_t>(keys, page_image_restored_values,
                                            lengths);
  require(std::ranges::equal(page_image_restored_values, values),
          "page image import did not preserve payload");
  bool page_image_layout_restored = false;
  page_image_restore_store.export_page_images(
      pro::make_proxy<se::FunctionFacade<bool(
          size_t, se::PageImageExport<float, float> &&)>>(
          [&](size_t dim, se::PageImageExport<float, float> &&image) {
            page_image_layout_restored =
                dim == 8 && image.physical_page_id == 0 &&
                image.slot_ordinals == std::vector<uint32_t>{0, 1, 2} &&
                image.keys == std::vector<uint64_t>{11, 22, 33} &&
                image.seq_counts == std::vector<uint64_t>{2, 2, 2};
            return true;
          }));
  require(page_image_layout_restored,
          "page image import did not preserve physical layout");
  auto duplicate_page_image_imports = page_image_imports;
  duplicate_page_image_imports.push_back(page_image_imports.front());
  Store rejected_page_image_store(
      {8}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 1, .storage = storage});
  require_throws([&] {
    rejected_page_image_store.import_page_images(duplicate_page_image_imports);
  });
  require(rejected_page_image_store.size() == 0,
          "rejected page-image import published partial state");

  constexpr size_t page_compaction_keys = 4096;
  Store page_compaction_store(
      {8}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 4, .storage = storage});
  std::vector<uint64_t> page_compaction_key_values(page_compaction_keys);
  std::iota(page_compaction_key_values.begin(),
            page_compaction_key_values.end(), uint64_t{50000});
  std::vector<int64_t> page_compaction_lengths(page_compaction_keys, 8);
  std::vector<float> page_compaction_values(page_compaction_keys * 8);
  for (size_t key_index = 0; key_index < page_compaction_keys; ++key_index) {
    std::fill_n(page_compaction_values.data() + key_index * 8, 8,
                static_cast<float>(key_index + 1));
  }
  page_compaction_store.lookup_or_insert<uint64_t>(page_compaction_key_values,
                                                   page_compaction_values,
                                                   page_compaction_lengths);
  require(page_compaction_store.compact_page(8, 0) == page_compaction_keys,
          "page compaction did not relocate every live source entry");
  std::vector<float> page_compaction_result(page_compaction_values.size());
  page_compaction_store.lookup<uint64_t>(page_compaction_key_values,
                                         page_compaction_result,
                                         page_compaction_lengths);
  require(page_compaction_result == page_compaction_values,
          "page compaction changed a relocated payload");
  require_throws([&] { page_compaction_store.compact_page(8, 0); });
  require(page_compaction_store.compact_adaptive_pages(
              8, 1.0F, sealing_config) == page_compaction_keys,
          "adaptive compaction did not select the remaining physical page");
  page_compaction_store.lookup<uint64_t>(page_compaction_key_values,
                                         page_compaction_result,
                                         page_compaction_lengths);
  require(page_compaction_result == page_compaction_values,
          "adaptive page compaction changed a relocated payload");
  page_compaction_store.clear();
  page_compaction_store.assign<uint64_t>(page_compaction_key_values,
                                         page_compaction_values,
                                         page_compaction_lengths);
  require(page_compaction_store.compact_page(8, 0) == page_compaction_keys,
          "clear did not reset a retired page for a later compaction");
  page_compaction_store.lookup<uint64_t>(page_compaction_key_values,
                                         page_compaction_result,
                                         page_compaction_lengths);
  require(page_compaction_result == page_compaction_values,
          "clear and later compaction changed a payload");

  constexpr size_t tight_compaction_dim = 128;
  constexpr size_t tight_compaction_page_count = 2;
  constexpr size_t tight_compaction_key_count =
      tight_compaction_page_count * se::EmbeddingArena<float>::kEntriesPerPage;
  Store tight_compaction_store(
      {tight_compaction_dim}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{
          .thread_budget = 8,
          .storage = storage,
          .storage_tuning = {.entries_per_page =
                                 se::EmbeddingArena<float>::kEntriesPerPage,
                             .calibrate_page_geometry = false}});
  std::vector<uint64_t> tight_compaction_keys(tight_compaction_key_count);
  std::iota(tight_compaction_keys.begin(), tight_compaction_keys.end(),
            uint64_t{600000});
  std::vector<int64_t> tight_compaction_lengths(tight_compaction_key_count,
                                                tight_compaction_dim);
  std::vector<float> tight_compaction_values(tight_compaction_key_count *
                                             tight_compaction_dim);
  for (size_t key_index = 0; key_index < tight_compaction_key_count;
       ++key_index) {
    std::fill_n(tight_compaction_values.data() +
                    key_index * tight_compaction_dim,
                tight_compaction_dim, static_cast<float>(key_index + 1));
  }
  tight_compaction_store.lookup_or_insert<uint64_t>(
      tight_compaction_keys, tight_compaction_values, tight_compaction_lengths);
  std::array<uint64_t, tight_compaction_page_count> tight_compaction_pages{};
  std::iota(tight_compaction_pages.begin(), tight_compaction_pages.end(),
            uint64_t{0});
  require(tight_compaction_store.compact_pages(tight_compaction_dim,
                                               tight_compaction_pages) ==
              tight_compaction_key_count,
          "tight page compaction did not relocate every source entry");
  std::vector<float> tight_compaction_result(tight_compaction_values.size());
  tight_compaction_store.lookup<uint64_t>(
      tight_compaction_keys, tight_compaction_result, tight_compaction_lengths);
  require(tight_compaction_result == tight_compaction_values,
          "tight page compaction changed a payload");

  using HotPromotionStore =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::ManualScorePolicy>;
  auto hot_training_factory = pro::make_proxy<
      se::FunctionFacade<se::OptimizerExtension<float, float>(size_t)>>(
      [storage, training_config](size_t dim) mutable {
        using Sgd = se::SgdOptimizer<float, float, se::StaticLearningRate>;
        return pro::make_proxy<se::OptimizerExtensionFacade<float, float>, Sgd>(
            dim, training_config, storage,
            se::StaticLearningRate{training_config.lr});
      });
  HotPromotionStore hot_promotion_store(
      {8}, {}, std::move(hot_training_factory), se::ManualScorePolicy{},
      se::HogwildStoreOptions{
          .thread_budget = 4,
          .storage = storage,
          .storage_tuning = {.entries_per_page =
                                 se::EmbeddingArena<float>::kEntriesPerPage,
                             .calibrate_page_geometry = false}});
  constexpr size_t hot_promotion_keys =
      se::EmbeddingArena<float>::kEntriesPerPage;
  std::vector<uint64_t> hot_keys(hot_promotion_keys);
  std::iota(hot_keys.begin(), hot_keys.end(), uint64_t{90000});
  std::vector<int64_t> hot_lengths(hot_promotion_keys, 8);
  std::vector<float> hot_values(hot_promotion_keys * 8, 1.0F);
  hot_promotion_store.lookup_or_insert<uint64_t>(hot_keys, hot_values,
                                                 hot_lengths);
  std::vector<uint64_t> hot_scores(hot_promotion_keys, 1);
  hot_scores[0] = 1000;
  hot_scores[1] = 900;
  hot_promotion_store.assign_scores<uint64_t>(hot_keys, hot_scores);
  require(hot_promotion_store.promote_hot_keys(8, 0, 2, 100) == 2,
          "hot-key promotion did not relocate the selected keys");
  require(hot_promotion_store.promoted_hot_page_size(8) == 2,
          "hot-key promotion did not publish a sealed hot page");
  std::vector<uint8_t> promoted_exists(2);
  hot_promotion_store.contains<uint64_t>(
      std::span<const uint64_t>(hot_keys.data(), 2), promoted_exists);
  require(std::ranges::equal(promoted_exists, std::array<uint8_t, 2>{1, 1}),
          "promoted keys disappeared from the directory");
  size_t source_page_entries = 0;
  hot_promotion_store.export_page_images(
      pro::make_proxy<se::FunctionFacade<bool(
          size_t, se::PageImageExport<float, float> &&)>>(
          [&](size_t dim, se::PageImageExport<float, float> &&image) {
            if (dim == 8 && image.physical_page_id == 0) {
              source_page_entries = image.keys.size();
              require(std::ranges::find(image.keys, hot_keys[0]) ==
                              image.keys.end() &&
                          std::ranges::find(image.keys, hot_keys[1]) ==
                              image.keys.end(),
                      "promoted keys remained in the source page");
            }
            return true;
          }));
  require(source_page_entries == hot_promotion_keys - 2,
          "cold source page was not retained after hot-key promotion");
  const std::array<uint64_t, 2> promoted_training_keys{hot_keys[0],
                                                       hot_keys[2]};
  constexpr std::array<int64_t, 2> promoted_training_lengths{8, 8};
  std::array<float, 16> promoted_training_before{};
  hot_promotion_store.lookup<uint64_t>(promoted_training_keys,
                                       promoted_training_before,
                                       promoted_training_lengths);
  std::array<float, 16> promoted_training_gradients{};
  promoted_training_gradients.fill(1.0F);
  hot_promotion_store.accumulate_gradients<uint64_t>(
      promoted_training_keys, promoted_training_gradients,
      promoted_training_lengths);
  hot_promotion_store.optimizer_step();
  std::array<float, 16> promoted_training_after{};
  hot_promotion_store.lookup<uint64_t>(promoted_training_keys,
                                       promoted_training_after,
                                       promoted_training_lengths);
  require(
      std::ranges::none_of(promoted_training_after,
                           [](float value) { return !std::isfinite(value); }) &&
          !std::ranges::equal(
              std::span<const float>(promoted_training_before).first(8),
              std::span<const float>(promoted_training_after).first(8)),
      "promoted hot key was not trainable");
  require(!std::ranges::equal(
              std::span<const float>(promoted_training_before).last(8),
              std::span<const float>(promoted_training_after).last(8)),
          "retained cold key was not trainable");

  Store coaccess_compaction_store(
      {8}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 4, .storage = storage});
  constexpr size_t coaccess_keys = page_compaction_keys * 2;
  std::vector<uint64_t> coaccess_key_values(coaccess_keys);
  std::iota(coaccess_key_values.begin(), coaccess_key_values.end(),
            uint64_t{200000});
  std::vector<int64_t> coaccess_lengths(coaccess_keys, 8);
  std::vector<float> coaccess_values(coaccess_keys * 8, 3.0F);
  coaccess_compaction_store.lookup_or_insert<uint64_t>(
      coaccess_key_values, coaccess_values, coaccess_lengths);
  std::vector<float> coaccess_result(coaccess_values.size());
  for (size_t repeat = 0; repeat < 8; ++repeat) {
    std::array<uint64_t, 2> pair{coaccess_key_values[0],
                                 coaccess_key_values[page_compaction_keys]};
    std::array<int64_t, 2> pair_lengths{8, 8};
    std::array<float, 16> pair_values{};
    coaccess_compaction_store.forward_lookup<uint64_t>(pair, pair_values,
                                                       pair_lengths);
  }
  require(coaccess_compaction_store.compact_adaptive_pages(
              8, 1.0F,
              se::AdaptivePageSelectorConfig{0.0F, 0.0F, 0.75F, 2,
                                             2 * 4096 * 64}) == coaccess_keys,
          "adaptive co-access compaction did not relocate both source pages");
  coaccess_compaction_store.lookup<uint64_t>(coaccess_key_values,
                                             coaccess_result, coaccess_lengths);
  require(coaccess_result == coaccess_values,
          "adaptive co-access compaction changed payloads");
  bool coaccess_pair_is_compact = false;
  coaccess_compaction_store.export_page_images(
      pro::make_proxy<se::FunctionFacade<bool(
          size_t, se::PageImageExport<float, float> &&)>>(
          [&](size_t, se::PageImageExport<float, float> &&image) {
            const auto first =
                std::ranges::find(image.keys, coaccess_key_values[0]);
            const auto second = std::ranges::find(
                image.keys, coaccess_key_values[page_compaction_keys]);
            coaccess_pair_is_compact |=
                first != image.keys.end() && second != image.keys.end();
            return true;
          }));
  require(coaccess_pair_is_compact,
          "adaptive co-access compaction did not co-locate the hot pair");

  constexpr size_t parallel_page_sources = 16;
  constexpr size_t parallel_page_key_count =
      parallel_page_sources * se::EmbeddingArena<float>::kEntriesPerPage;
  Store parallel_page_store(
      {8}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{
          .thread_budget = 16,
          .storage = storage,
          .storage_tuning = {.entries_per_page =
                                 se::EmbeddingArena<float>::kEntriesPerPage,
                             .calibrate_page_geometry = false}});
  std::vector<uint64_t> parallel_page_keys(parallel_page_key_count);
  std::iota(parallel_page_keys.begin(), parallel_page_keys.end(),
            uint64_t{400000});
  std::vector<int64_t> parallel_page_lengths(parallel_page_key_count, 8);
  std::vector<float> parallel_page_values(parallel_page_key_count * 8);
  for (size_t key_index = 0; key_index < parallel_page_key_count; ++key_index) {
    std::fill_n(parallel_page_values.data() + key_index * 8, 8,
                static_cast<float>(key_index + 1));
  }
  parallel_page_store.lookup_or_insert<uint64_t>(
      parallel_page_keys, parallel_page_values, parallel_page_lengths);
  std::barrier parallel_page_start(parallel_page_sources + 1);
  std::array<size_t, parallel_page_sources> parallel_page_migrated{};
  std::array<std::exception_ptr, parallel_page_sources> parallel_page_errors{};
  std::array<std::thread, parallel_page_sources> parallel_page_workers;
  for (size_t source_page = 0; source_page < parallel_page_sources;
       ++source_page) {
    parallel_page_workers[source_page] = std::thread([&, source_page] {
      parallel_page_start.arrive_and_wait();
      try {
        parallel_page_migrated[source_page] =
            parallel_page_store.compact_page(8, source_page);
      } catch (...) {
        parallel_page_errors[source_page] = std::current_exception();
      }
    });
  }
  parallel_page_start.arrive_and_wait();
  for (auto &worker : parallel_page_workers)
    worker.join();
  for (size_t source_page = 0; source_page < parallel_page_sources;
       ++source_page) {
    require(parallel_page_errors[source_page] == nullptr,
            "independent page compaction raised an exception");
    require(
        parallel_page_migrated[source_page] ==
            se::EmbeddingArena<float>::kEntriesPerPage,
        "independent page compaction did not drain exactly one source page");
  }
  std::vector<float> parallel_page_result(parallel_page_values.size());
  parallel_page_store.lookup<uint64_t>(parallel_page_keys, parallel_page_result,
                                       parallel_page_lengths);
  require(parallel_page_result == parallel_page_values,
          "independent page compaction changed a payload");
  std::vector<uint8_t> parallel_page_seen(parallel_page_key_count);
  bool parallel_source_image_seen = false;
  parallel_page_store.export_page_images(
      pro::make_proxy<se::FunctionFacade<bool(
          size_t, se::PageImageExport<float, float> &&)>>(
          [&](size_t, se::PageImageExport<float, float> &&image) {
            if (image.physical_page_id < parallel_page_sources)
              parallel_source_image_seen = true;
            for (uint64_t key : image.keys) {
              if (key < parallel_page_keys.front() ||
                  key > parallel_page_keys.back())
                continue;
              const size_t offset =
                  static_cast<size_t>(key - parallel_page_keys.front());
              require(image.physical_page_id >= parallel_page_sources,
                      "page compaction destination aliases a source page");
              require(
                  parallel_page_seen[offset] == 0,
                  "page compaction published a duplicate directory locator");
              parallel_page_seen[offset] = 1;
            }
            return true;
          }));
  require(!parallel_source_image_seen,
          "independent page compaction retained a source page image");
  require(std::all_of(parallel_page_seen.begin(), parallel_page_seen.end(),
                      [](uint8_t seen) { return seen == 1; }),
          "independent page compaction lost a relocated key");

  Store same_page_store(
      {8}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{
          .thread_budget = 16,
          .storage = storage,
          .storage_tuning = {.entries_per_page =
                                 se::EmbeddingArena<float>::kEntriesPerPage,
                             .calibrate_page_geometry = false}});
  std::vector<uint64_t> same_page_keys(
      se::EmbeddingArena<float>::kEntriesPerPage);
  std::iota(same_page_keys.begin(), same_page_keys.end(), uint64_t{500000});
  std::vector<int64_t> same_page_lengths(same_page_keys.size(), 8);
  std::vector<float> same_page_values(same_page_keys.size() * 8, 17.0F);
  same_page_store.lookup_or_insert<uint64_t>(same_page_keys, same_page_values,
                                             same_page_lengths);
  std::barrier same_page_start(3);
  std::array<size_t, 2> same_page_migrated{};
  std::array<std::exception_ptr, 2> same_page_errors{};
  std::array<std::thread, 2> same_page_workers;
  for (size_t worker = 0; worker < same_page_workers.size(); ++worker) {
    same_page_workers[worker] = std::thread([&, worker] {
      same_page_start.arrive_and_wait();
      try {
        same_page_migrated[worker] = same_page_store.compact_page(8, 0);
      } catch (...) {
        same_page_errors[worker] = std::current_exception();
      }
    });
  }
  same_page_start.arrive_and_wait();
  for (auto &worker : same_page_workers)
    worker.join();
  const size_t same_page_successes =
      static_cast<size_t>(same_page_errors[0] == nullptr) +
      static_cast<size_t>(same_page_errors[1] == nullptr);
  require(same_page_successes == 1,
          "same-page concurrent compaction did not reject one claimant");
  require(same_page_migrated[0] + same_page_migrated[1] ==
              se::EmbeddingArena<float>::kEntriesPerPage,
          "same-page concurrent compaction migrated a source more than once");
  std::vector<float> same_page_result(same_page_values.size());
  same_page_store.lookup<uint64_t>(same_page_keys, same_page_result,
                                   same_page_lengths);
  require(same_page_result == same_page_values,
          "same-page concurrent compaction changed a payload");
  bool same_source_image_seen = false;
  same_page_store.export_page_images(
      pro::make_proxy<se::FunctionFacade<bool(
          size_t, se::PageImageExport<float, float> &&)>>(
          [&](size_t, se::PageImageExport<float, float> &&image) {
            same_source_image_seen |= image.physical_page_id == 0;
            return true;
          }));
  require(!same_source_image_seen,
          "same-page concurrent compaction did not retire the source page");

  Store concurrent_page_store(
      {8}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 4, .storage = storage});
  std::vector<uint64_t> concurrent_page_keys(page_compaction_keys);
  std::iota(concurrent_page_keys.begin(), concurrent_page_keys.end(),
            uint64_t{100000});
  std::vector<int64_t> concurrent_page_lengths(page_compaction_keys, 8);
  std::vector<float> concurrent_page_values(page_compaction_keys * 8, 1.0F);
  concurrent_page_store.lookup_or_insert<uint64_t>(
      concurrent_page_keys, concurrent_page_values, concurrent_page_lengths);
  std::atomic<bool> concurrent_page_ok{true};
  std::exception_ptr concurrent_page_writer_error;
  std::exception_ptr concurrent_page_reader_error;
  std::exception_ptr concurrent_checkpoint_error;
  std::thread page_writer([&] {
    try {
      for (size_t iteration = 0; iteration < 32; ++iteration) {
        std::fill(concurrent_page_values.begin(), concurrent_page_values.end(),
                  static_cast<float>(iteration + 1));
        concurrent_page_store.assign<uint64_t>(concurrent_page_keys,
                                               concurrent_page_values,
                                               concurrent_page_lengths);
      }
    } catch (...) {
      concurrent_page_writer_error = std::current_exception();
      concurrent_page_ok.store(false, std::memory_order_release);
    }
  });
  std::thread page_reader([&] {
    try {
      std::vector<float> readback(concurrent_page_values.size());
      for (size_t iteration = 0; iteration < 32; ++iteration) {
        concurrent_page_store.lookup<uint64_t>(concurrent_page_keys, readback,
                                               concurrent_page_lengths);
        if (!std::all_of(readback.begin(), readback.end(),
                         [](float value) { return std::isfinite(value); }))
          throw std::runtime_error(
              "page compaction lookup produced non-finite data");
      }
    } catch (...) {
      concurrent_page_reader_error = std::current_exception();
      concurrent_page_ok.store(false, std::memory_order_release);
    }
  });
  std::thread checkpoint_exporter([&] {
    try {
      for (size_t iteration = 0; iteration < 8; ++iteration) {
        std::unordered_set<uint64_t> exported_keys;
        std::unordered_map<uint64_t, uint64_t> exported_pages;
        concurrent_page_store.export_page_images(
            pro::make_proxy<se::FunctionFacade<bool(
                size_t, se::PageImageExport<float, float> &&)>>(
                [&](size_t dim, se::PageImageExport<float, float> &&image) {
                  require(dim == 8 && image.embedding_dim == 8 &&
                              image.values.size() == image.keys.size() * 8 &&
                              image.slot_ordinals.size() == image.keys.size() &&
                              image.seq_counts.size() == image.keys.size(),
                          "concurrent checkpoint image layout is invalid");
                  require(
                      std::ranges::all_of(
                          image.values,
                          [](float value) { return std::isfinite(value); }),
                      "concurrent checkpoint image contains non-finite data");
                  for (uint64_t key : image.keys) {
                    if (!exported_keys.insert(key).second) {
                      throw std::runtime_error(
                          "concurrent checkpoint duplicated key " +
                          std::to_string(key) + " on pages " +
                          std::to_string(exported_pages.at(key)) + " and " +
                          std::to_string(image.physical_page_id));
                    }
                    exported_pages.emplace(key, image.physical_page_id);
                  }
                  return true;
                }));
        require(exported_keys.size() == concurrent_page_keys.size(),
                "concurrent checkpoint lost a mapped key");
      }
    } catch (...) {
      concurrent_checkpoint_error = std::current_exception();
      concurrent_page_ok.store(false, std::memory_order_release);
    }
  });
  const size_t concurrent_page_migrated =
      concurrent_page_store.compact_page(8, 0);
  page_writer.join();
  page_reader.join();
  checkpoint_exporter.join();
  if (concurrent_page_writer_error != nullptr)
    std::rethrow_exception(concurrent_page_writer_error);
  if (concurrent_page_reader_error != nullptr)
    std::rethrow_exception(concurrent_page_reader_error);
  if (concurrent_checkpoint_error != nullptr)
    std::rethrow_exception(concurrent_checkpoint_error);
  require(concurrent_page_ok.load(std::memory_order_acquire) &&
              concurrent_page_migrated == page_compaction_keys,
          "concurrent page compaction lost a source locator");
  std::fill(concurrent_page_values.begin(), concurrent_page_values.end(),
            1234.0F);
  concurrent_page_store.assign<uint64_t>(
      concurrent_page_keys, concurrent_page_values, concurrent_page_lengths);
  concurrent_page_store.lookup<uint64_t>(
      concurrent_page_keys, page_compaction_result, concurrent_page_lengths);
  require(page_compaction_result == concurrent_page_values,
          "retired page compaction lost the final assignment");

  std::atomic<bool> compaction_ok{true};
  std::atomic<size_t> compaction_count{0};
  std::thread compactor([&] {
    try {
      for (size_t iteration = 0; iteration < 128; ++iteration) {
        if (store.compact_one(8, 11))
          compaction_count.fetch_add(1, std::memory_order_relaxed);
      }
    } catch (...) {
      compaction_ok.store(false, std::memory_order_release);
    }
  });
  for (size_t iteration = 0; iteration < 128; ++iteration) {
    std::fill(values.begin(), values.begin() + 8,
              static_cast<float>(iteration + 1));
    store.assign<uint64_t>(std::span<const uint64_t>(keys).subspan(0, 1),
                           std::span<const float>(values).subspan(0, 8),
                           std::span<const int64_t>(lengths).subspan(0, 1));
  }
  compactor.join();
  require(compaction_ok.load(std::memory_order_acquire),
          "concurrent compaction raised an exception");
  require(compaction_count.load(std::memory_order_relaxed) != 0,
          "compaction did not publish a destination locator");
  std::fill(values.begin(), values.begin() + 8, 777.0F);
  store.assign<uint64_t>(std::span<const uint64_t>(keys).subspan(0, 1),
                         std::span<const float>(values).subspan(0, 8),
                         std::span<const int64_t>(lengths).subspan(0, 1));
  values.fill(0.0F);
  store.lookup<uint64_t>(std::span<const uint64_t>(keys).subspan(0, 1),
                         std::span<float>(values).subspan(0, 8),
                         std::span<const int64_t>(lengths).subspan(0, 1));
  require(std::all_of(values.begin(), values.begin() + 8,
                      [](float value) { return value == 777.0F; }),
          "concurrent compaction lost the final assignment");
  require(!store.compact_one(8, 999999),
          "compaction reported success for an absent key");

  se::AdamConfig adam_config;
  auto adam_factory = pro::make_proxy<
      se::FunctionFacade<se::OptimizerExtension<float, float>(size_t)>>(
      [storage, adam_config](size_t dim) mutable {
        using Adam = se::AdamOptimizer<float, float, se::StaticLearningRate>;
        return pro::make_proxy<se::OptimizerExtensionFacade<float, float>,
                               Adam>(dim, adam_config, storage,
                                     se::StaticLearningRate{0.01F});
      });
  AdamStore adam_store(
      {8}, {}, std::move(adam_factory), se::NoScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 1, .storage = storage});
  constexpr std::array<uint64_t, 1> adam_key{71};
  constexpr std::array<int64_t, 1> adam_length{8};
  std::array<float, 8> adam_values{};
  adam_store.lookup_or_insert<uint64_t>(adam_key, adam_values, adam_length);
  std::array<float, 16> adam_state{};
  std::iota(adam_state.begin(), adam_state.end(), 1.0F);
  constexpr std::array<uint32_t, 1> adam_step{37};
  adam_store.assign_opt_states(8, adam_key, adam_state, adam_step);
  require(adam_store.compact_one(8, 71),
          "optimizer-backed compaction did not publish a destination");
  bool adam_seen = false;
  adam_store.export_opt_state_batch(
      pro::make_proxy<se::FunctionFacade<bool(
          size_t, std::span<const uint64_t>, std::span<const float>,
          std::span<const uint64_t>, std::span<const float>,
          std::span<const uint32_t>)>>(
          [&](size_t dim, std::span<const uint64_t> exported_keys,
              std::span<const float> /*exported_values*/,
              std::span<const uint64_t> /*exported_scores*/,
              std::span<const float> exported_state,
              std::span<const uint32_t> exported_steps) {
            if (dim == 8 && exported_keys.size() == 1 &&
                exported_keys[0] == 71) {
              adam_seen = std::ranges::equal(exported_state, adam_state) &&
                          exported_steps.size() == 1 && exported_steps[0] == 37;
            }
            return true;
          }),
      16);
  require(adam_seen, "compaction did not preserve optimizer state and step");

  std::atomic<bool> adam_concurrent_ok{true};
  std::array<std::thread, 4> adam_workers;
  adam_workers[0] = std::thread([&] {
    try {
      for (size_t iteration = 0; iteration < 256; ++iteration) {
        std::array<float, 8> row{};
        row.fill(static_cast<float>(iteration + 1));
        adam_store.assign<uint64_t>(adam_key, row, adam_length);
      }
    } catch (...) {
      adam_concurrent_ok.store(false, std::memory_order_release);
    }
  });
  adam_workers[1] = std::thread([&] {
    try {
      for (size_t iteration = 0; iteration < 256; ++iteration) {
        std::array<float, 16> state{};
        state.fill(static_cast<float>(iteration + 1));
        std::array<uint32_t, 1> step{static_cast<uint32_t>(iteration + 1)};
        adam_store.assign_opt_states(8, adam_key, state, step);
      }
    } catch (...) {
      adam_concurrent_ok.store(false, std::memory_order_release);
    }
  });
  adam_workers[2] = std::thread([&] {
    try {
      for (size_t iteration = 0; iteration < 256; ++iteration) {
        std::array<float, 8> row{};
        adam_store.lookup<uint64_t>(adam_key, row, adam_length);
        if (!std::all_of(row.begin(), row.end(),
                         [](float value) { return std::isfinite(value); }))
          throw std::runtime_error(
              "concurrent lookup produced non-finite data");
      }
    } catch (...) {
      adam_concurrent_ok.store(false, std::memory_order_release);
    }
  });
  adam_workers[3] = std::thread([&] {
    try {
      for (size_t iteration = 0; iteration < 256; ++iteration) {
        if (!adam_store.compact_one(8, 71))
          throw std::runtime_error("concurrent optimizer compaction lost key");
      }
    } catch (...) {
      adam_concurrent_ok.store(false, std::memory_order_release);
    }
  });
  for (auto &worker : adam_workers)
    worker.join();
  require(adam_concurrent_ok.load(std::memory_order_acquire),
          "lookup, assignment, optimizer update, and compaction conflicted");

  constexpr std::array<float, 8> final_adam_value{
      101.0F, 102.0F, 103.0F, 104.0F, 105.0F, 106.0F, 107.0F, 108.0F};
  constexpr std::array<float, 16> final_adam_state{
      201.0F, 202.0F, 203.0F, 204.0F, 205.0F, 206.0F, 207.0F, 208.0F,
      209.0F, 210.0F, 211.0F, 212.0F, 213.0F, 214.0F, 215.0F, 216.0F};
  constexpr std::array<uint32_t, 1> final_adam_step{257};
  adam_store.assign<uint64_t>(adam_key, final_adam_value, adam_length);
  adam_store.assign_opt_states(8, adam_key, final_adam_state, final_adam_step);
  std::array<float, 8> verified_adam_value{};
  adam_store.lookup<uint64_t>(adam_key, verified_adam_value, adam_length);
  require(verified_adam_value == final_adam_value,
          "concurrent compaction lost the final optimizer value assignment");
  bool final_adam_seen = false;
  adam_store.export_opt_state_batch(
      pro::make_proxy<se::FunctionFacade<bool(
          size_t, std::span<const uint64_t>, std::span<const float>,
          std::span<const uint64_t>, std::span<const float>,
          std::span<const uint32_t>)>>(
          [&](size_t dim, std::span<const uint64_t> exported_keys,
              std::span<const float> /*exported_values*/,
              std::span<const uint64_t> /*exported_scores*/,
              std::span<const float> exported_state,
              std::span<const uint32_t> exported_steps) {
            if (dim == 8 && exported_keys.size() == 1 &&
                exported_keys[0] == 71) {
              final_adam_seen =
                  std::ranges::equal(exported_state, final_adam_state) &&
                  std::ranges::equal(exported_steps, final_adam_step);
            }
            return true;
          }),
      16);
  require(final_adam_seen,
          "concurrent compaction lost final optimizer state or step");

  auto full_page_adam_factory = pro::make_proxy<
      se::FunctionFacade<se::OptimizerExtension<float, float>(size_t)>>(
      [storage, adam_config](size_t dim) mutable {
        using Adam = se::AdamOptimizer<float, float, se::StaticLearningRate>;
        return pro::make_proxy<se::OptimizerExtensionFacade<float, float>,
                               Adam>(dim, adam_config, storage,
                                     se::StaticLearningRate{0.01F});
      });
  AdamStore full_page_adam_store(
      {8}, {}, std::move(full_page_adam_factory), se::NoScorePolicy{},
      se::HogwildStoreOptions{
          .thread_budget = 16,
          .storage = storage,
          .storage_tuning = {.entries_per_page =
                                 se::EmbeddingArena<float>::kEntriesPerPage,
                             .calibrate_page_geometry = false}});
  constexpr size_t full_page_adam_key_count =
      se::EmbeddingArena<float>::kEntriesPerPage;
  std::vector<uint64_t> full_page_adam_keys(full_page_adam_key_count);
  std::iota(full_page_adam_keys.begin(), full_page_adam_keys.end(),
            uint64_t{610000});
  std::vector<int64_t> full_page_adam_lengths(full_page_adam_key_count, 8);
  std::vector<float> full_page_adam_values(full_page_adam_key_count * 8);
  std::vector<float> full_page_adam_states(full_page_adam_key_count * 16);
  std::vector<uint32_t> full_page_adam_steps(full_page_adam_key_count);
  for (size_t index = 0; index < full_page_adam_key_count; ++index) {
    std::fill_n(full_page_adam_values.data() + index * 8, 8,
                static_cast<float>(index + 3));
    std::iota(full_page_adam_states.begin() + index * 16,
              full_page_adam_states.begin() + (index + 1) * 16,
              static_cast<float>(index * 16 + 1));
    full_page_adam_steps[index] = static_cast<uint32_t>(index + 41);
  }
  full_page_adam_store.assign<uint64_t>(
      full_page_adam_keys, full_page_adam_values, full_page_adam_lengths);
  full_page_adam_store.assign_opt_states(
      8, full_page_adam_keys, full_page_adam_states, full_page_adam_steps);
  require(full_page_adam_store.compact_page(8, 0) == full_page_adam_key_count,
          "full-page optimizer compaction did not drain every source key");
  std::vector<uint8_t> full_page_adam_seen(full_page_adam_key_count);
  full_page_adam_store.export_page_images(
      pro::make_proxy<se::FunctionFacade<bool(
          size_t, se::PageImageExport<float, float> &&)>>(
          [&](size_t dim, se::PageImageExport<float, float> &&image) {
            if (dim != 8 || image.keys.size() != full_page_adam_key_count)
              return true;
            require(image.physical_page_id != 0 &&
                        image.values.size() == full_page_adam_values.size() &&
                        image.optimizer_states.size() ==
                            full_page_adam_states.size() &&
                        image.steps.size() == full_page_adam_steps.size(),
                    "full-page optimizer image layout is incomplete");
            for (size_t slot = 0; slot < image.keys.size(); ++slot) {
              const size_t index = static_cast<size_t>(
                  image.keys[slot] - full_page_adam_keys.front());
              require(index < full_page_adam_key_count &&
                          full_page_adam_seen[index] == 0 &&
                          std::ranges::equal(
                              std::span<const float>(image.values)
                                  .subspan(slot * 8, 8),
                              std::span<const float>(full_page_adam_values)
                                  .subspan(index * 8, 8)) &&
                          std::ranges::equal(
                              std::span<const float>(image.optimizer_states)
                                  .subspan(slot * 16, 16),
                              std::span<const float>(full_page_adam_states)
                                  .subspan(index * 16, 16)) &&
                          image.steps[slot] == full_page_adam_steps[index],
                      "full-page optimizer compaction changed state");
              full_page_adam_seen[index] = 1;
            }
            return true;
          }));
  require(std::all_of(full_page_adam_seen.begin(), full_page_adam_seen.end(),
                      [](uint8_t seen) { return seen == 1; }),
          "full-page optimizer compaction lost a key");

  Store concurrent_store(
      {8}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 4, .storage = storage});
  constexpr size_t concurrent_key_count = 128;
  std::array<uint64_t, concurrent_key_count> concurrent_keys{};
  std::iota(concurrent_keys.begin(), concurrent_keys.end(), uint64_t{1000});
  std::array<int64_t, concurrent_key_count> concurrent_lengths{};
  concurrent_lengths.fill(8);
  std::array<float, concurrent_key_count * 8> concurrent_values{};
  concurrent_store.lookup_or_insert<uint64_t>(
      concurrent_keys, concurrent_values, concurrent_lengths);
  std::atomic<bool> concurrent_ok{true};
  std::array<std::thread, 4> concurrent_workers;
  for (size_t worker = 0; worker < concurrent_workers.size(); ++worker) {
    concurrent_workers[worker] = std::thread([&, worker] {
      try {
        for (size_t index = worker; index < concurrent_key_count; index += 4) {
          std::array<uint64_t, 1> key{concurrent_keys[index]};
          std::array<int64_t, 1> length{8};
          std::array<float, 8> row{};
          for (size_t iteration = 0; iteration < 64; ++iteration) {
            row.fill(static_cast<float>(iteration + 1));
            concurrent_store.assign<uint64_t>(key, row, length);
            if (!concurrent_store.compact_one(8, key[0]))
              throw std::runtime_error("concurrent compaction lost a key");
          }
        }
      } catch (...) {
        concurrent_ok.store(false, std::memory_order_release);
      }
    });
  }
  for (auto &worker : concurrent_workers)
    worker.join();
  require(concurrent_ok.load(std::memory_order_acquire),
          "multi-worker compaction raised or lost a locator");
  concurrent_values.fill(0.0F);
  concurrent_store.lookup<uint64_t>(concurrent_keys, concurrent_values,
                                    concurrent_lengths);
  require(std::all_of(concurrent_values.begin(), concurrent_values.end(),
                      [](float value) { return value == 64.0F; }),
          "multi-worker compaction produced an incorrect value");

  bool reserve_overflow_rejected = false;
  try {
    constexpr std::array<size_t, 1> excessive_reserve{
        std::numeric_limits<size_t>::max()};
    store.reserve_per_shard(excessive_reserve, 8);
  } catch (const std::overflow_error &error) {
    reserve_overflow_rejected =
        std::string_view(error.what()) == "key shard capacity overflow";
  }
  require(reserve_overflow_rejected,
          "reserve overflow was not rejected before allocation");

  se::StorageOptions duplicate_options;
  duplicate_options.mode = se::StorageMode::kMemoryOnly;
  duplicate_options.memory_limit_bytes = 64 << 20;
  auto duplicate_storage =
      se::create_page_storage(std::move(duplicate_options));
  Store duplicate_store({8}, {}, nullptr, se::NoScorePolicy{},
                        se::HogwildStoreOptions{.thread_budget = 32,
                                                .storage = duplicate_storage});
  require(duplicate_storage->stats().memory_bytes == 0,
          "empty store eagerly allocated transient pages");
  constexpr std::array<uint64_t, 1> duplicate_key{44};
  constexpr std::array<int64_t, 1> duplicate_length{8};
  std::barrier duplicate_start(3);
  std::array<std::exception_ptr, 2> duplicate_errors{};
  std::array<std::thread, 2> duplicate_workers;
  for (size_t worker = 0; worker < duplicate_workers.size(); ++worker) {
    duplicate_workers[worker] = std::thread([&, worker] {
      try {
        std::array<float, 8> values{};
        duplicate_start.arrive_and_wait();
        duplicate_store.lookup_or_insert<uint64_t>(duplicate_key, values,
                                                   duplicate_length);
      } catch (...) {
        duplicate_errors[worker] = std::current_exception();
      }
    });
  }
  duplicate_start.arrive_and_wait();
  for (auto &worker : duplicate_workers)
    worker.join();
  for (const auto &error : duplicate_errors) {
    if (error)
      std::rethrow_exception(error);
  }
  require(duplicate_store.size() == 1,
          "concurrent calls consumed active-key capacity twice");

  ScoredStore memory_scored_store(
      {8}, {}, nullptr, se::ManualScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 1,
                              .storage = duplicate_storage});
  constexpr std::array<uint64_t, 1> memory_scored_key{55};
  constexpr std::array<int64_t, 1> memory_scored_length{8};
  std::array<float, 8> memory_scored_values{};
  memory_scored_store.lookup_or_insert<uint64_t>(
      memory_scored_key, memory_scored_values, memory_scored_length);
  constexpr std::array<uint64_t, 1> compact_score{99};
  memory_scored_store.assign_scores<uint64_t>(memory_scored_key, compact_score);
  require(memory_scored_store.compact_one(8, 55),
          "score-backed compaction did not publish a destination");
  bool score_preserved = false;
  memory_scored_store.export_batch<uint64_t>(
      pro::make_proxy<se::FunctionFacade<bool(size_t, std::span<const uint64_t>,
                                              std::span<const float>,
                                              std::span<const uint64_t>)>>(
          [&](size_t dim, std::span<const uint64_t> exported_keys,
              std::span<const float> /*exported_values*/,
              std::span<const uint64_t> exported_scores) {
            if (dim == 8 && exported_keys.size() == 1 &&
                exported_keys[0] == 55 && exported_scores[0] == 99)
              score_preserved = true;
            return true;
          }),
      16);
  require(score_preserved, "compaction did not preserve score payload");

  constexpr size_t full_page_score_key_count =
      se::EmbeddingArena<float>::kEntriesPerPage;
  ScoredStore full_page_score_store(
      {8}, {}, nullptr, se::ManualScorePolicy{},
      se::HogwildStoreOptions{
          .thread_budget = 16,
          .storage = duplicate_storage,
          .storage_tuning = {.entries_per_page =
                                 se::EmbeddingArena<float>::kEntriesPerPage,
                             .calibrate_page_geometry = false}});
  std::vector<uint64_t> full_page_score_keys(full_page_score_key_count);
  std::iota(full_page_score_keys.begin(), full_page_score_keys.end(),
            uint64_t{620000});
  std::vector<int64_t> full_page_score_lengths(full_page_score_key_count, 8);
  std::vector<float> full_page_score_values(full_page_score_key_count * 8);
  std::vector<uint64_t> full_page_scores(full_page_score_key_count);
  for (size_t index = 0; index < full_page_score_key_count; ++index) {
    std::fill_n(full_page_score_values.data() + index * 8, 8,
                static_cast<float>(index + 5));
    full_page_scores[index] = index + 101;
  }
  full_page_score_store.assign<uint64_t>(
      full_page_score_keys, full_page_score_values, full_page_score_lengths);
  full_page_score_store.assign_scores<uint64_t>(full_page_score_keys,
                                                full_page_scores);
  require(full_page_score_store.compact_page(8, 0) == full_page_score_key_count,
          "full-page score compaction did not drain every source key");
  std::vector<uint8_t> full_page_score_seen(full_page_score_key_count);
  full_page_score_store.export_page_images(
      pro::make_proxy<se::FunctionFacade<bool(
          size_t, se::PageImageExport<float, float> &&)>>(
          [&](size_t dim, se::PageImageExport<float, float> &&image) {
            if (dim != 8 || image.keys.size() != full_page_score_key_count)
              return true;
            require(image.physical_page_id != 0 &&
                        image.values.size() == full_page_score_values.size() &&
                        image.scores.size() == full_page_scores.size(),
                    "full-page score image layout is incomplete");
            for (size_t slot = 0; slot < image.keys.size(); ++slot) {
              const size_t index = static_cast<size_t>(
                  image.keys[slot] - full_page_score_keys.front());
              require(index < full_page_score_key_count &&
                          full_page_score_seen[index] == 0 &&
                          std::ranges::equal(
                              std::span<const float>(image.values)
                                  .subspan(slot * 8, 8),
                              std::span<const float>(full_page_score_values)
                                  .subspan(index * 8, 8)) &&
                          image.scores[slot] == full_page_scores[index],
                      "full-page score compaction changed payload");
              full_page_score_seen[index] = 1;
            }
            return true;
          }));
  require(std::all_of(full_page_score_seen.begin(), full_page_score_seen.end(),
                      [](uint8_t seen) { return seen == 1; }),
          "full-page score compaction lost a key");

  std::array<uint8_t, 1> memory_scored_exists{};
  memory_scored_store.contains<uint64_t>(memory_scored_key,
                                         memory_scored_exists);

  se::StorageOptions reuse_options;
  reuse_options.memory_limit_bytes = 0;
  reuse_options.spill_directory = std::filesystem::temp_directory_path();
  reuse_options.max_spill_bytes = 1 << 20;
  auto reuse_storage = se::create_page_storage(std::move(reuse_options));
  Store reuse_store(
      {8}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 1, .storage = reuse_storage});
  std::array<uint64_t, 1> reuse_key{};
  constexpr std::array<int64_t, 1> reuse_length{8};
  std::array<float, 8> reuse_value{};
  for (uint64_t key = 1; key <= 10000; ++key) {
    reuse_key[0] = key;
    reuse_store.lookup_or_insert<uint64_t>(reuse_key, reuse_value,
                                           reuse_length);
    reuse_store.erase<uint64_t>(reuse_key);
  }
  require(reuse_storage->stats().spill_count == 1,
          "erased embedding indices were not reused");

  se::EmbeddingArena<float> owner_arena(
      8, true, duplicate_storage,
      duplicate_storage->stats().retirement_pressure_domain);
  const uint32_t owner_index = owner_arena.allocate();
  owner_arena.publish_owner(owner_index, 700, 2);
  owner_arena.mark_live(owner_index, true);
  const auto old_owner = owner_arena.owner_snapshot(owner_index);
  require(old_owner.has_value() && old_owner->key == 700 &&
              old_owner->shard_id == 2,
          "arena owner metadata was not published");
  owner_arena.mark_live(owner_index, false);
  require(!owner_arena.owner_snapshot(owner_index).has_value(),
          "retired arena index retained a live owner");
  owner_arena.publish_owner(owner_index, 701, 3);
  owner_arena.mark_live(owner_index, true);
  const auto new_owner = owner_arena.owner_snapshot(owner_index);
  require(new_owner.has_value() && new_owner->key == 701 &&
              new_owner->shard_id == 3 &&
              new_owner->seq_count != old_owner->seq_count,
          "reused arena index did not invalidate its stale owner record");

  se::TransientArena<float> transient(8, duplicate_storage);
  transient.get_or_allocate(1);
  transient.get_or_allocate(2);
  require(transient.active_slots() == 2,
          "transient arena did not grow dynamically");
  std::array<uint64_t, 2> transient_iteration{};
  size_t transient_iteration_count = 0;
  transient.for_each_accumulated([&](uint64_t key, const float *) {
    transient_iteration[transient_iteration_count++] = key;
  });
  require(transient_iteration == std::array<uint64_t, 2>{1, 2},
          "transient arena did not traverse gradient slots in insertion order");

  Store snapshot_store({8}, {}, nullptr, se::NoScorePolicy{},
                       se::HogwildStoreOptions{.thread_budget = 4,
                                               .storage = duplicate_storage});
  std::array<uint64_t, 32> snapshot_keys{};
  std::iota(snapshot_keys.begin(), snapshot_keys.end(), uint64_t{1});
  std::array<int64_t, 32> snapshot_lengths{};
  snapshot_lengths.fill(8);
  std::array<float, 32 * 8> snapshot_values{};
  snapshot_store.lookup_or_insert<uint64_t>(snapshot_keys, snapshot_values,
                                            snapshot_lengths);
  std::atomic<bool> reader_ok{true};
  std::thread reader([&] {
    try {
      std::array<float, 32 * 8> reader_values{};
      for (size_t i = 0; i < 200; ++i) {
        snapshot_store.lookup<uint64_t>(snapshot_keys, reader_values,
                                        snapshot_lengths);
      }
    } catch (...) {
      reader_ok.store(false, std::memory_order_relaxed);
    }
  });
  for (size_t i = 0; i < 50; ++i) {
    snapshot_store.clear();
    snapshot_store.lookup_or_insert<uint64_t>(snapshot_keys, snapshot_values,
                                              snapshot_lengths);
  }
  reader.join();
  require(reader_ok.load(std::memory_order_relaxed),
          "concurrent clear invalidated an active reader");

  se::StorageOptions watermark_options;
  watermark_options.memory_limit_bytes = 300 << 10;
  watermark_options.spill_directory = std::filesystem::temp_directory_path();
  watermark_options.max_spill_bytes = 1 << 20;
  auto watermark_storage =
      se::create_page_storage(std::move(watermark_options));
  auto watermark_training_factory = pro::make_proxy<
      se::FunctionFacade<se::OptimizerExtension<float, float>(size_t)>>(
      [watermark_storage, training_config](size_t dim) mutable {
        using Sgd = se::SgdOptimizer<float, float, se::StaticLearningRate>;
        return pro::make_proxy<se::OptimizerExtensionFacade<float, float>, Sgd>(
            dim, training_config, watermark_storage,
            se::StaticLearningRate{training_config.lr});
      });
  ScoredStore watermark_store(
      {8}, {}, std::move(watermark_training_factory), se::ManualScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 1,
                              .storage = watermark_storage});
  constexpr std::array<uint64_t, 1> watermark_key{91};
  constexpr std::array<int64_t, 1> watermark_length{8};
  std::array<float, 8> watermark_values{};
  watermark_store.lookup_or_insert<uint64_t>(watermark_key, watermark_values,
                                             watermark_length);
  for (size_t attempt = 0;
       attempt < 100 && watermark_storage->stats().swap_out_count == 0;
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  require(watermark_storage->stats().swap_out_count == 1,
          "scored arena page was not registered for background swap-out");
  watermark_values.fill(7.0F);
  watermark_store.assign<uint64_t>(watermark_key, watermark_values,
                                   watermark_length);
  watermark_values.fill(0.0F);
  watermark_store.lookup<uint64_t>(watermark_key, watermark_values,
                                   watermark_length);
  require(std::all_of(watermark_values.begin(), watermark_values.end(),
                      [](float value) { return value == 7.0F; }),
          "swapped page lost a post-swap assignment");
  const std::array<float, 8> spill_training_before = watermark_values;
  std::array<float, 8> spill_training_gradients{};
  spill_training_gradients.fill(1.0F);
  watermark_store.accumulate_gradients<uint64_t>(
      watermark_key, spill_training_gradients, watermark_length);
  watermark_store.optimizer_step();
  watermark_values.fill(0.0F);
  watermark_store.lookup<uint64_t>(watermark_key, watermark_values,
                                   watermark_length);
  require(
      std::ranges::none_of(watermark_values,
                           [](float value) { return !std::isfinite(value); }) &&
          watermark_values != spill_training_before,
      "spill-backed optimizer page was not trainable");

  se::StorageOptions shared_options;
  shared_options.memory_limit_bytes = 256 << 20;
  shared_options.spill_directory = std::filesystem::temp_directory_path();
  shared_options.max_spill_bytes = 4 << 20;
  auto shared_storage = se::create_page_storage(std::move(shared_options));
  constexpr std::array<uint64_t, 1> shared_key{123};
  constexpr std::array<int64_t, 1> shared_length{8};
  std::array<float, 8> shared_values{};
  {
    auto first = std::make_unique<ScoredStore>(
        std::vector<size_t>{8}, ScoredStore::Initializer{}, nullptr,
        se::ManualScorePolicy{},
        se::HogwildStoreOptions{.thread_budget = 1, .storage = shared_storage});
    auto second = std::make_unique<ScoredStore>(
        std::vector<size_t>{8}, ScoredStore::Initializer{}, nullptr,
        se::ManualScorePolicy{},
        se::HogwildStoreOptions{.thread_budget = 1, .storage = shared_storage});
    first->lookup_or_insert<uint64_t>(shared_key, shared_values, shared_length);
    second->lookup_or_insert<uint64_t>(shared_key, shared_values,
                                       shared_length);
    first.reset();
    shared_values.fill(3.0F);
    second->assign<uint64_t>(shared_key, shared_values, shared_length);
    shared_values.fill(0.0F);
    second->lookup<uint64_t>(shared_key, shared_values, shared_length);
    require(std::all_of(shared_values.begin(), shared_values.end(),
                        [](float value) { return value == 3.0F; }),
            "shared storage lost the surviving store after peer destruction");

    auto memory_store = std::make_unique<Store>(
        std::vector<size_t>{8}, Store::Initializer{}, nullptr,
        se::NoScorePolicy{},
        se::HogwildStoreOptions{.thread_budget = 1, .storage = shared_storage});
    std::array<uint64_t, 1> memory_key{456};
    std::array<float, 8> memory_values{};
    memory_store->lookup_or_insert<uint64_t>(memory_key, memory_values,
                                             shared_length);
    std::array<uint8_t, 1> memory_exists{};
    memory_store->contains<uint64_t>(memory_key, memory_exists);
    require(memory_exists[0] == 1,
            "pure memory store could not share a scored storage");
  }

  se::StorageOptions stress_options;
  stress_options.mode = se::StorageMode::kMemoryOnly;
  stress_options.memory_limit_bytes = 128 << 20;
  auto stress_storage = se::create_page_storage(std::move(stress_options));
  Store stress_store(
      {8}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 8, .storage = stress_storage});
  constexpr size_t stress_batch_size = 8192;
  constexpr size_t stress_rounds = 8;
  std::vector<uint64_t> stress_keys(stress_batch_size);
  std::vector<int64_t> stress_lengths(stress_batch_size, 8);
  std::vector<float> stress_values(stress_batch_size * 8);
  for (size_t round = 0; round < stress_rounds; ++round) {
    for (size_t i = 0; i < stress_batch_size; ++i)
      stress_keys[i] = round * stress_batch_size + i;
    stress_store.lookup_or_insert<uint64_t>(stress_keys, stress_values,
                                            stress_lengths);
  }
  const size_t expected_stress_size = stress_batch_size * stress_rounds;
  const size_t observed_stress_size = stress_store.size();
  if (observed_stress_size != expected_stress_size)
    throw std::runtime_error("8-thread unique insertion lost keys: got " +
                             std::to_string(observed_stress_size) +
                             " expected " +
                             std::to_string(expected_stress_size));

  ScoredStore scored_stress_store(
      {8}, {}, nullptr, se::ManualScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 8, .storage = stress_storage});
  for (size_t i = 0; i < stress_batch_size; ++i)
    stress_keys[i] = i;
  scored_stress_store.lookup_or_insert<uint64_t>(stress_keys, stress_values,
                                                 stress_lengths);
  require(scored_stress_store.size() == stress_batch_size,
          "admitted scored insertion batch lost keys");

  se::StorageOptions circuit_options;
  circuit_options.mode = se::StorageMode::kMemoryOnly;
  circuit_options.memory_limit_bytes = 1 << 20;
  auto circuit_storage = se::create_page_storage(std::move(circuit_options));
  Store circuit_store(
      {8}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 1, .storage = circuit_storage});
  auto pressure = circuit_storage->stats().retirement_pressure_domain;
  const uint64_t circuit_retirement = pressure->begin_retirement(1 << 18);
  constexpr std::array<uint64_t, 1> circuit_key{900001};
  constexpr std::array<int64_t, 1> circuit_length{8};
  std::array<float, 8> circuit_values;
  circuit_values.fill(7.0F);
  bool insertion_rejected = false;
  try {
    circuit_store.lookup_or_insert<uint64_t>(circuit_key, circuit_values,
                                             circuit_length);
  } catch (const std::runtime_error &error) {
    insertion_rejected =
        std::string_view(error.what()) ==
        "retirement pressure circuit is open for control-plane mutation";
  }
  require(insertion_rejected,
          "circuit lookup-or-insert did not reject the missing key");
  require(circuit_store.size() == 0,
          "rejected lookup-or-insert grew the key directory");
  bool assignment_rejected = false;
  try {
    circuit_store.assign<uint64_t>(circuit_key, circuit_values, circuit_length);
  } catch (const std::runtime_error &error) {
    assignment_rejected =
        std::string_view(error.what()) ==
        "retirement pressure circuit is open for control-plane mutation";
  }
  require(assignment_rejected, "circuit assign did not reject the missing key");
  require(circuit_store.size() == 0, "rejected assign grew the key directory");
  pressure->complete_retirement(circuit_retirement);
  circuit_store.lookup_or_insert<uint64_t>(circuit_key, circuit_values,
                                           circuit_length);
  require(circuit_store.size() == 1,
          "key insertion did not resume after circuit recovery");
}
