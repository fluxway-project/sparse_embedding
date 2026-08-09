#include <algorithm>
#include <array>
#include <barrier>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <optional>
#include <sched.h>
#include <span>
#include <stdexcept>
#include <string_view>
#include <sys/resource.h>
#include <thread>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

namespace {

constexpr size_t kDimension = 128;
constexpr size_t kShardCount = 64;
constexpr size_t kPagesPerWorker = 4;
constexpr std::array<size_t, 5> kWorkerCounts{1, 2, 4, 8, 16};
constexpr size_t kSamples = 5;

using Store =
    se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;

struct Result {
  size_t workers;
  double seconds;
  double relocated_keys_per_second;
};

#if defined(PAGE_COMPACTION_THREAD_FAULT_PROFILE)
struct ThreadFaults {
  long minor = 0;
  long major = 0;
};

ThreadFaults thread_faults() {
  rusage usage{};
  if (getrusage(RUSAGE_THREAD, &usage) != 0)
    throw std::runtime_error("page-compaction thread fault sampling failed");
  return {usage.ru_minflt, usage.ru_majflt};
}
#endif

#if defined(PAGE_COMPACTION_PIN_WORKERS)
void pin_worker_to_allowed_cpu(size_t worker, size_t workers) {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
    throw std::runtime_error("page-compaction worker affinity query failed");
  std::vector<int> cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &allowed))
      cpus.push_back(cpu);
  }
  if (cpus.size() < workers)
    throw std::runtime_error("page-compaction worker affinity is undersized");
  cpu_set_t selected;
  CPU_ZERO(&selected);
  CPU_SET(cpus[worker], &selected);
  if (sched_setaffinity(0, sizeof(selected), &selected) != 0)
    throw std::runtime_error("page-compaction worker affinity update failed");
}
#endif

Result run_once(size_t workers, size_t store_threads,
                bool preallocate_destinations, bool profile_pause,
                bool strong_scaling) {
#if defined(PAGE_COMPACTION_PROFILE)
  if (profile_pause)
    se::reset_page_compaction_profile();
#endif
  const size_t source_worker_count =
      strong_scaling ? kWorkerCounts.back() : workers;
  const size_t source_page_count = source_worker_count * kPagesPerWorker;
  if (source_page_count % workers != 0)
    throw std::invalid_argument("strong scaling does not divide source pages");
  const size_t pages_per_worker = source_page_count / workers;
  const size_t key_count =
      source_page_count * se::EmbeddingArena<float>::kEntriesPerPage;
  se::StorageOptions options;
  options.mode = se::StorageMode::kMemoryOnly;
  options.memory_limit_bytes = size_t{1} << 30;
  auto storage = se::create_page_storage(std::move(options));
  Store store({kDimension}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = store_threads,
                                      .storage = storage});
  if (preallocate_destinations && store.reserve(key_count * 2, kDimension)) {
    throw std::runtime_error(
        "page-compaction destination preallocation failed");
  }
  std::vector<uint64_t> keys(key_count);
  std::iota(keys.begin(), keys.end(), uint64_t{700000});
  std::vector<int64_t> lengths(key_count, kDimension);
  std::vector<float> values(key_count * kDimension);
  for (size_t index = 0; index < key_count; ++index) {
    std::fill_n(values.data() + index * kDimension, kDimension,
                static_cast<float>(index + 1));
  }
  store.lookup_or_insert<uint64_t>(keys, values, lengths);
  std::vector<uint64_t> source_page_ids;
  std::vector<size_t> source_key_counts;
  store.export_page_images(pro::make_proxy<se::FunctionFacade<bool(
                               size_t, se::PageImageExport<float, float> &&)>>(
      [&](size_t, se::PageImageExport<float, float> &&image) {
        if (!image.keys.empty()) {
          source_page_ids.push_back(image.physical_page_id);
          source_key_counts.push_back(image.keys.size());
        }
        return true;
      }));
  if (source_page_ids.size() < source_page_count)
    throw std::runtime_error(
        "page-compaction benchmark lacks live cohort pages");
  source_page_ids.resize(source_page_count);
  source_key_counts.resize(source_page_count);

  std::barrier start(workers + 1);
  std::vector<size_t> migrated(workers);
  std::vector<std::exception_ptr> errors(workers);
#if defined(PAGE_COMPACTION_THREAD_FAULT_PROFILE)
  std::vector<ThreadFaults> fault_before(workers);
  std::vector<ThreadFaults> fault_after(workers);
#endif
#if defined(PAGE_COMPACTION_WORKER_PROFILE)
  std::vector<uint64_t> worker_elapsed_ns(workers);
  std::vector<int> worker_cpu_start(workers);
  std::vector<int> worker_cpu_end(workers);
#endif
  std::vector<std::vector<uint64_t>> source_page_batches(workers);
  std::vector<size_t> worker_expected_keys(workers, 0);
  for (size_t worker = 0; worker < workers; ++worker) {
    auto &source_pages = source_page_batches[worker];
    source_pages.resize(pages_per_worker);
    for (size_t page = 0; page < pages_per_worker; ++page) {
      const size_t source_index = worker * pages_per_worker + page;
      source_pages[page] = source_page_ids[source_index];
      worker_expected_keys[worker] += source_key_counts[source_index];
    }
  }
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (size_t page_id = 0; page_id < workers; ++page_id) {
    threads.emplace_back([&, page_id] {
#if defined(PAGE_COMPACTION_PIN_WORKERS)
      try {
        pin_worker_to_allowed_cpu(page_id, workers);
      } catch (...) {
        errors[page_id] = std::current_exception();
      }
#endif
      start.arrive_and_wait();
      try {
#if defined(PAGE_COMPACTION_PIN_WORKERS)
        if (errors[page_id] != nullptr)
          std::rethrow_exception(errors[page_id]);
#endif
#if defined(PAGE_COMPACTION_WORKER_PROFILE)
        worker_cpu_start[page_id] = sched_getcpu();
        if (worker_cpu_start[page_id] < 0)
          throw std::runtime_error(
              "page-compaction worker CPU sampling failed");
        const auto worker_started = std::chrono::steady_clock::now();
#endif
#if defined(PAGE_COMPACTION_THREAD_FAULT_PROFILE)
        fault_before[page_id] = thread_faults();
#endif
        migrated[page_id] =
            store.compact_pages(kDimension, source_page_batches[page_id]);
#if defined(PAGE_COMPACTION_THREAD_FAULT_PROFILE)
        fault_after[page_id] = thread_faults();
#endif
#if defined(PAGE_COMPACTION_WORKER_PROFILE)
        worker_elapsed_ns[page_id] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - worker_started)
                .count();
        worker_cpu_end[page_id] = sched_getcpu();
        if (worker_cpu_end[page_id] < 0)
          throw std::runtime_error(
              "page-compaction worker CPU sampling failed");
#endif
      } catch (...) {
        errors[page_id] = std::current_exception();
      }
    });
  }
  if (profile_pause) {
    std::cout << "page_compaction_profile_ready\n" << std::flush;
    if (std::raise(SIGSTOP) != 0)
      throw std::runtime_error("page-compaction profiler pause failed");
  }
  const auto began = std::chrono::steady_clock::now();
  start.arrive_and_wait();
  for (auto &thread : threads)
    thread.join();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - began)
          .count();
  if (seconds <= 0.0)
    throw std::runtime_error("page-compaction benchmark timer did not advance");
  for (size_t page_id = 0; page_id < workers; ++page_id) {
    if (errors[page_id] != nullptr) {
      try {
        std::rethrow_exception(errors[page_id]);
      } catch (const std::exception &error) {
        throw std::runtime_error("page-compaction worker failed: " +
                                 std::string(error.what()));
      }
    }
    if (migrated[page_id] != worker_expected_keys[page_id]) {
      throw std::runtime_error(
          "page-compaction worker did not drain every source page");
    }
  }
#if defined(PAGE_COMPACTION_PROFILE)
  if (profile_pause) {
    const se::PageCompactionProfileSnapshot profile =
        se::page_compaction_profile_snapshot();
    std::cout << "page_compaction_profile enumeration_ns="
              << profile.enumeration_ns
              << " reservation_ns=" << profile.reservation_ns
              << " payload_copy_ns=" << profile.payload_copy_ns
              << " staging_ns=" << profile.staging_ns
              << " preparation_ns=" << profile.preparation_ns
              << " publication_ns=" << profile.publication_ns
              << " retirement_ns=" << profile.retirement_ns
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
              << " destination_ready_ns=" << profile.destination_ready_ns
              << " destination_ready_calls=" << profile.destination_ready_calls
              << " destination_ready_pages=" << profile.destination_ready_pages
              << " owner_publish_ns=" << profile.owner_publish_ns
              << " owner_publish_calls=" << profile.owner_publish_calls
              << " owner_clear_ns=" << profile.owner_clear_ns
              << " owner_clear_calls=" << profile.owner_clear_calls
              << " locator_publish_ns=" << profile.locator_publish_ns
              << " locator_publish_calls=" << profile.locator_publish_calls
              << " locator_lookup_ns=" << profile.locator_lookup_ns
              << " locator_lookup_calls=" << profile.locator_lookup_calls
              << " owner_validate_ns=" << profile.owner_validate_ns
              << " owner_validate_calls=" << profile.owner_validate_calls
#endif
#if defined(PAGE_COMPACTION_ADMISSION_PROFILE)
              << " publication_admission_misses="
              << profile.publication_admission_misses
              << " publication_map_misses=" << profile.publication_map_misses
              << " publication_yields=" << profile.publication_yields
#endif
              << '\n';
#if defined(PAGE_COMPACTION_SHARD_PROFILE)
    for (size_t shard = 0; shard < profile.shard_publication.size(); ++shard) {
      const auto &shard_profile = profile.shard_publication[shard];
      if (shard_profile.acquisitions == 0 &&
          shard_profile.compaction_misses == 0 &&
          shard_profile.map_misses == 0) {
        continue;
      }
      std::cout << "page_compaction_shard_profile shard=" << shard
                << " acquisitions=" << shard_profile.acquisitions
                << " compaction_misses=" << shard_profile.compaction_misses
                << " map_misses=" << shard_profile.map_misses
                << " protected_keys=" << shard_profile.protected_keys
                << " hold_ns=" << shard_profile.hold_ns << '\n';
    }
#endif
  }
#endif
#if defined(PAGE_COMPACTION_THREAD_FAULT_PROFILE)
  long minor_faults = 0;
  long major_faults = 0;
  for (size_t worker = 0; worker < workers; ++worker) {
    minor_faults += fault_after[worker].minor - fault_before[worker].minor;
    major_faults += fault_after[worker].major - fault_before[worker].major;
  }
  std::cout << "page_compaction_thread_faults minor=" << minor_faults
            << " major=" << major_faults << '\n';
#endif
#if defined(PAGE_COMPACTION_WORKER_PROFILE)
  for (size_t worker = 0; worker < workers; ++worker) {
    std::cout << "page_compaction_worker_profile worker=" << worker
              << " source_first=" << source_page_batches[worker].front()
              << " source_pages=" << source_page_batches[worker].size()
              << " elapsed_ns=" << worker_elapsed_ns[worker]
              << " cpu_start=" << worker_cpu_start[worker]
              << " cpu_end=" << worker_cpu_end[worker] << '\n';
  }
#endif
  if (!profile_pause) {
    std::vector<float> actual(values.size());
    store.lookup<uint64_t>(keys, actual, lengths);
    if (actual != values)
      throw std::runtime_error("page-compaction benchmark changed a payload");
  }
  return {workers, seconds, static_cast<double>(key_count) / seconds};
}

size_t parse_worker_count(std::string_view argument) {
  size_t workers = 0;
  const auto [end, error] = std::from_chars(
      argument.data(), argument.data() + argument.size(), workers);
  if (error != std::errc{} || end != argument.data() + argument.size() ||
      workers == 0) {
    throw std::invalid_argument("worker count must be a positive integer");
  }
  return workers;
}

struct BenchmarkOptions {
  std::optional<size_t> workers;
  size_t store_threads = kShardCount;
  bool preallocate_destinations = false;
  bool profile_pause = false;
  bool strong_scaling = false;
};

BenchmarkOptions parse_options(int argc, char **argv) {
  BenchmarkOptions options;
  for (int argument_index = 1; argument_index < argc; ++argument_index) {
    const std::string_view argument(argv[argument_index]);
    constexpr std::string_view kStoreThreadsPrefix = "--store-threads=";
    if (argument.starts_with(kStoreThreadsPrefix)) {
      if (options.store_threads != kShardCount)
        throw std::invalid_argument("store thread count was repeated");
      options.store_threads =
          parse_worker_count(argument.substr(kStoreThreadsPrefix.size()));
      continue;
    }
    if (argument == "--preallocate-destinations") {
      if (options.preallocate_destinations)
        throw std::invalid_argument("destination preallocation was repeated");
      options.preallocate_destinations = true;
      continue;
    }
    if (argument == "--profile-pause") {
      if (options.profile_pause)
        throw std::invalid_argument("profiler pause was repeated");
      options.profile_pause = true;
      continue;
    }
    if (argument == "--strong-scaling") {
      if (options.strong_scaling)
        throw std::invalid_argument("strong scaling was repeated");
      options.strong_scaling = true;
      continue;
    }
    if (options.workers.has_value())
      throw std::invalid_argument("expected at most one worker-count argument");
    options.workers = parse_worker_count(argument);
  }
  return options;
}

} // namespace

int main(int argc, char **argv) {
  std::array<size_t, 1> requested_workers{};
  std::span<const size_t> worker_counts = kWorkerCounts;
  const BenchmarkOptions options = parse_options(argc, argv);
  if (options.profile_pause && !options.workers.has_value())
    throw std::invalid_argument("profiler pause requires one worker count");
  if (options.workers.has_value()) {
    requested_workers[0] = *options.workers;
    worker_counts = requested_workers;
  }
  const size_t samples = options.profile_pause ? 1 : kSamples;
  for (size_t workers : worker_counts) {
    std::vector<double> seconds;
    std::vector<double> throughputs;
    std::vector<size_t> compaction_runs;
    seconds.reserve(samples);
    throughputs.reserve(samples);
    compaction_runs.reserve(samples);
    for (size_t sample = 0; sample < samples; ++sample) {
      const Result result = run_once(
          workers, options.store_threads, options.preallocate_destinations,
          options.profile_pause, options.strong_scaling);
      seconds.push_back(result.seconds);
      throughputs.push_back(result.relocated_keys_per_second);
      compaction_runs.push_back(1);
    }
    std::ranges::sort(seconds);
    std::ranges::sort(throughputs);
    std::ranges::sort(compaction_runs);
    std::cout << "page_compaction_workers=" << workers << " samples=" << samples
              << " preallocated_destinations="
              << static_cast<int>(options.preallocate_destinations)
              << " store_threads=" << options.store_threads
              << " profile_pause=" << static_cast<int>(options.profile_pause)
              << " strong_scaling=" << static_cast<int>(options.strong_scaling)
              << " compaction_runs_per_sample="
              << compaction_runs[compaction_runs.size() / 2]
              << " elapsed_seconds=" << seconds[seconds.size() / 2]
              << " relocated_keys_per_second="
              << throughputs[throughputs.size() / 2] << '\n';
  }
  return 0;
}
