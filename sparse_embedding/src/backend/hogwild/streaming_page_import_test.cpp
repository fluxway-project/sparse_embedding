#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/checkpoint/checkpoint_test_support.hpp"
#include "sparse_embedding/src/optimizer/adam_optimizer.hpp"

#include <array>
#include <chrono>
#include <future>
#include <latch>

namespace {

using namespace se::checkpoint::test;
using Store =
    se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
using Import = se::PageImageImport<float, float>;

se::PageStorage storage() {
  se::StorageOptions options;
  options.memory_limit_bytes = 32 << 20;
  options.spill_directory = std::filesystem::temp_directory_path();
  options.max_spill_bytes = 256 << 20;
  return se::create_page_storage(std::move(options));
}

Import page(size_t dim, uint64_t id, uint64_t key, uint32_t ordinal = 0) {
  return {dim,
          {id,
           static_cast<uint32_t>(dim),
           64,
           {ordinal},
           {key},
           {2},
           std::vector<float>(dim, static_cast<float>(key)),
           {},
           {},
           {}}};
}

template <typename Producer> auto source(Producer &&producer) {
  return pro::make_proxy<
      se::FunctionFacade<void(se::PageImageVisitor<float, float>)>>(
      std::forward<Producer>(producer));
}

auto adam_factory(se::PageStorage pages) {
  return pro::make_proxy<
      se::FunctionFacade<se::OptimizerExtension<float, float>(size_t)>>(
      [pages](size_t dim) {
        return pro::make_proxy<
            se::OptimizerExtensionFacade<float, float>,
            se::AdamOptimizer<float, float, se::StaticLearningRate>>(
            dim, se::AdamConfig{}, pages, se::StaticLearningRate{1.0e-3F});
      });
}

void check_stream_and_generation_contracts() {
  auto pages = storage();
  Store store({2, 4}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = 1, .storage = pages});
  store.import_page_images(source([&](auto visitor) {
                             (*visitor)(page(2, 3, 7));
                             (*visitor)(page(4, 0, 7));
                             (*visitor)(page(2, 0, 8));
                           }),
                           9, true);
  require(store.size() == 3 && store.checkpoint_page_layout_generation() == 9,
          "stream did not publish all dimensions and generation");
  const std::array<uint64_t, 3> keys{7, 7, 8};
  const std::array<int64_t, 3> dims{2, 4, 2};
  std::array<float, 8> values{};
  store.lookup<uint64_t>(keys, values, dims);
  require(values == std::array<float, 8>{7, 7, 7, 7, 7, 7, 8, 8},
          "cross-dimension identity or out-of-order page state was lost");
  bool called = false;
  require_throws<std::logic_error>(
      [&] {
        store.import_page_images(source([&](auto) { called = true; }), 10,
                                 true);
      },
      "nonempty target accepted page import");
  require(!called && store.is_operational() && store.size() == 3 &&
              store.checkpoint_page_layout_generation() == 9,
          "failed empty-target precondition changed the store");

  Store empty({2}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = 1, .storage = pages});
  empty.import_page_images(source([](auto) {}), 11, true);
  for (uint64_t generation : {0, 10}) {
    require_throws<std::invalid_argument>(
        [&] {
          empty.import_page_images(source([&](auto) { called = true; }),
                                   generation, true);
        },
        "invalid generation was accepted");
  }
  require(!called && empty.is_operational() && empty.size() == 0 &&
              empty.checkpoint_page_layout_generation() == 11,
          "empty import did not commit generation atomically");
}

void check_duplicate_failure_domains() {
  auto pages = storage();
  for (size_t conflict = 0; conflict < 3; ++conflict) {
    Store store({2}, {}, nullptr, se::NoScorePolicy{},
                se::HogwildStoreOptions{.thread_budget = 1, .storage = pages});
    require_throws<std::invalid_argument>(
        [&] {
          store.import_page_images(source([&](auto visitor) {
                                     (*visitor)(page(2, 0, 7));
                                     if (conflict == 0)
                                       (*visitor)(page(2, 1, 7));
                                     else if (conflict == 1)
                                       (*visitor)(page(2, 0, 8));
                                     else
                                       (*visitor)(page(2, 0, 8, 1));
                                   }),
                                   8, true);
        },
        "duplicate key, slot or disjoint repeated page was accepted");
    require(!store.is_operational() &&
                store.checkpoint_page_layout_generation() == 1,
            "failed stream published generation or reopened partial state");
    require_throws<std::runtime_error>(
        [&] { (void)store.size(); }, "terminal target accepted an invocation");
  }
  for (bool duplicate_key : {false, true}) {
    Store store({2}, {}, nullptr, se::NoScorePolicy{},
                se::HogwildStoreOptions{.thread_budget = 1, .storage = pages});
    auto image = page(2, 0, 7);
    image.image.keys.push_back(duplicate_key ? 7 : 8);
    image.image.slot_ordinals.push_back(duplicate_key ? 1 : 0);
    image.image.seq_counts.push_back(2);
    image.image.values.insert(image.image.values.end(), {8, 8});
    require_throws<std::invalid_argument>(
        [&] {
          store.import_page_images(
              source([&](auto visitor) { (*visitor)(image); }), 8, true);
        },
        "within-page duplicate key or slot was accepted");
    require(!store.is_operational() &&
                store.checkpoint_page_layout_generation() == 1,
            "within-page failure reopened its target");
  }
}

void check_swallowed_emission_failure() {
  auto pages = storage();
  Store store({2}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = 1, .storage = pages});
  size_t failures = 0;
  require_throws<std::invalid_argument>(
      [&] {
        store.import_page_images(
            source([&](auto visitor) {
              (*visitor)(page(2, 0, 7));
              for (const auto &[id, key] : {std::pair{1, 7}, std::pair{2, 8}}) {
                try {
                  (*visitor)(page(2, id, key));
                } catch (const std::invalid_argument &) {
                  ++failures;
                }
              }
            }),
            8, true);
      },
      "source swallowed an emitter failure and committed generation");
  require(failures == 2 && !store.is_operational() &&
              store.checkpoint_page_layout_generation() == 1,
          "latched import failure accepted later emission");
}

void check_directory_capacity_growth() {
  auto pages = storage();
  Store store({2}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = 1, .storage = pages});
  const size_t page_rows = store.entries_per_page();
  constexpr size_t page_count = 16;
  store.import_page_images(source([&](auto visitor) {
                             auto image = page(2, 0, 0);
                             image.image.keys.resize(page_rows);
                             image.image.slot_ordinals.resize(page_rows);
                             image.image.seq_counts.assign(page_rows, 2);
                             image.image.values.assign(page_rows * 2, 1);
                             std::iota(image.image.slot_ordinals.begin(),
                                       image.image.slot_ordinals.end(), 0);
                             for (size_t id = 0; id < page_count; ++id) {
                               image.image.physical_page_id = id;
                               std::iota(image.image.keys.begin(),
                                         image.image.keys.end(),
                                         id * page_rows);
                               (*visitor)(image);
                             }
                           }),
                           8, true);
  require(store.size() == page_rows * page_count &&
              pages->stats().retired_bytes == 0 &&
              pages->stats().epoch_stall_count == 0,
          "directory growth lost keys or retained obsolete maps after import");
  const std::array<uint64_t, 2> keys{0, page_rows * page_count - 1};
  const std::array<int64_t, 2> dims{2, 2};
  std::array<float, 4> values{};
  store.lookup<uint64_t>(keys, values, dims);
  require(values == std::array<float, 4>{1, 1, 1, 1},
          "directory growth did not preserve early and late keys");
}

void check_waiting_reader_on_completion(bool fail) {
  auto pages = storage();
  Store store({2}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = 1, .storage = pages});
  std::latch first_page(1), finish_source(1);
  std::atomic<bool> first_imported{false};
  std::exception_ptr import_error;
  std::jthread importer([&] {
    try {
      store.import_page_images(
          source([&](auto visitor) {
            (*visitor)(page(2, 0, 7));
            first_imported.store(true, std::memory_order_release);
            first_page.count_down();
            finish_source.wait();
            if (fail)
              throw std::domain_error("injected late page-source failure");
            (*visitor)(page(2, 1, 8));
          }),
          8, true);
    } catch (...) {
      import_error = std::current_exception();
      if (!first_imported.load(std::memory_order_acquire))
        first_page.count_down();
    }
  });
  first_page.wait();
  if (!first_imported.load(std::memory_order_acquire)) {
    importer.join();
    std::rethrow_exception(import_error);
  }
  auto reader = std::async(std::launch::async, [&] { return store.size(); });
  const auto waiting = reader.wait_for(std::chrono::milliseconds(20));
  finish_source.count_down();
  importer.join();
  require(waiting == std::future_status::timeout,
          "reader observed an in-progress import");
  if (fail) {
    require(import_error != nullptr, "late source failure disappeared");
    require_throws<std::domain_error>(
        [&] { std::rethrow_exception(import_error); },
        "page import replaced the primary source error");
    require_throws<std::runtime_error>(
        [&] { (void)reader.get(); },
        "waiting reader observed partial failed state");
    require(!store.is_operational() &&
                store.checkpoint_page_layout_generation() == 1,
            "failed import reopened its admission or advanced generation");
  } else {
    if (import_error)
      std::rethrow_exception(import_error);
    require(reader.get() == 2 && store.checkpoint_page_layout_generation() == 8,
            "waiting reader did not observe the complete import");
  }
}

void check_optimizer_scope_and_sparse_watermark() {
  auto pages = storage();
  se::HogwildStoreOptions options{.thread_budget = 1, .storage = pages};
  Store restored({2}, {}, adam_factory(pages), se::NoScorePolicy{}, options);
  Store reference({2}, {}, adam_factory(pages), se::NoScorePolicy{}, options);
  restored.import_page_images(
      source([&](auto visitor) {
        for (const auto &[id, key] : {std::pair{3, 7}, std::pair{0, 8}}) {
          auto image = page(2, id, key);
          image.image.optimizer_states = {0.1F, 0.2F, 0.3F, 0.4F};
          image.image.steps = {7};
          (*visitor)(image);
        }
      }),
      8, true);
  const std::array<uint64_t, 3> keys{7, 8, 9};
  const std::array<int64_t, 3> dims{2, 2, 2};
  const std::array<float, 6> values{7, 7, 8, 8, 9, 9};
  const std::array<float, 8> states{0.1F, 0.2F, 0.3F, 0.4F,
                                    0.1F, 0.2F, 0.3F, 0.4F};
  const std::array<uint32_t, 2> steps{7, 7};
  reference.assign<uint64_t>(keys, values, dims);
  reference.assign_opt_states<uint64_t>(2, std::span(keys).first(2), states,
                                        steps);
  restored.assign<uint64_t>(std::span(keys).last(1), std::span(values).last(2),
                            std::span(dims).last(1));
  const std::array<float, 6> gradients{1, 1, 1, 1, 1, 1};
  reference.accumulate_gradients<uint64_t>(keys, gradients, dims);
  restored.accumulate_gradients<uint64_t>(keys, gradients, dims);
  reference.optimizer_step(1);
  restored.optimizer_step(1);
  std::array<float, 6> expected{}, actual{};
  reference.lookup<uint64_t>(keys, expected, dims);
  restored.lookup<uint64_t>(keys, actual, dims);
  require(expected == actual,
          "out-of-order physical import lost Adam state or watermark");

  Store missing({2}, {}, adam_factory(pages), se::NoScorePolicy{}, options);
  require_throws<std::invalid_argument>(
      [&] {
        missing.import_page_images(
            source([&](auto visitor) { (*visitor)(page(2, 0, 7)); }), 8, true);
      },
      "training page restore accepted missing Adam state");
  Store weights({2}, {}, adam_factory(pages), se::NoScorePolicy{}, options);
  weights.import_page_images(
      source([&](auto visitor) { (*visitor)(page(2, 0, 7)); }), 8, false);
  require(weights.size() == 1,
          "weight-only page restore required optimizer payload");
}

} // namespace

int main() {
  ScratchRoot scratch;
  check_stream_and_generation_contracts();
  check_duplicate_failure_domains();
  check_swallowed_emission_failure();
  check_directory_capacity_growth();
  check_waiting_reader_on_completion(false);
  check_waiting_reader_on_completion(true);
  check_optimizer_scope_and_sparse_watermark();
  require(std::filesystem::is_empty(scratch.path()),
          "page import left scratch files");
}
