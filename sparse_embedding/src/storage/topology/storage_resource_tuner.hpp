#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "sparse_embedding/src/execution/dispatch_budget.hpp"
#include "sparse_embedding/src/execution/locality_policy.hpp"
#include "sparse_embedding/src/storage/topology/storage_topology.hpp"
#include "sparse_embedding/src/storage/topology/system_resource_probe.hpp"

namespace se::storage {

struct TopologyTuneRequest {
  size_t thread_budget = 0;
  LocalityPolicy locality{};
  size_t embedding_dim = 32;
  size_t param_bytes_per_element = sizeof(float);
  size_t opt_bytes_per_entry = 0;
  StorageTuningOptions tuning{};
};

class StorageResourceTuner {
public:
  static constexpr size_t kMinEntriesPerPage = 4096;
  static constexpr size_t kMaxEntriesPerPage = 16384;
  static constexpr size_t kTargetPagePayloadBytes = 512 * 1024;

  static StorageTopology derive(const TopologyTuneRequest &req) {
    if (req.embedding_dim == 0)
      throw std::invalid_argument("embedding dim must be positive");

    // 1. Thread & Shard Derivation
    const size_t automatic = SystemResourceProbe::hardware_concurrency();
    const size_t execution_workers =
        normalize_thread_budget(req.thread_budget, automatic);
    if (execution_workers > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("execution thread budget exceeds uint32");

    const bool numa_execution = req.locality.execution == LocalityMode::kNuma;
    const bool numa_memory = req.locality.memory == LocalityMode::kNuma;
    const bool numa = numa_execution || numa_memory;
    if (numa && !SystemResourceProbe::numa_available())
      throw std::runtime_error("NUMA locality requires libnuma support");
    const size_t nodes = numa ? SystemResourceProbe::numa_node_count() : 1;
    if (nodes == 0 || !std::has_single_bit(nodes))
      throw std::invalid_argument(
          "NUMA topology node count must be a power of two");

    size_t shards = std::bit_ceil(std::max(execution_workers, size_t{1}));
    if (numa) {
      while (shards < nodes || shards % nodes != 0)
        shards <<= 1;
    }

    // 2. Cohort Size
    const size_t cohort_size = numa_memory ? std::bit_floor(shards / nodes)
                                           : std::min(size_t{16}, shards);

    // 3. Entries per Page (power of 2)
    size_t entries_per_page = 0;
    if (req.tuning.entries_per_page != 0) {
      if (!std::has_single_bit(req.tuning.entries_per_page) ||
          req.tuning.entries_per_page < kMinEntriesPerPage ||
          req.tuning.entries_per_page > kMaxEntriesPerPage)
        throw std::invalid_argument(
            "user entries per page must be a power of two within tuner bounds");
      entries_per_page = req.tuning.entries_per_page;
    } else {
      entries_per_page = calibrate_entries_per_page(req, execution_workers);
    }

    // 4. Max Pages Derivation
    size_t max_pages = 8192;
    if (req.tuning.max_pages != 0) {
      max_pages = req.tuning.max_pages;
    } else if (req.tuning.target_capacity_keys != 0 ||
               req.tuning.memory_budget_bytes != 0) {
      size_t memory_budget = req.tuning.memory_budget_bytes;
      if (memory_budget == 0) {
        const size_t available_ram = SystemResourceProbe::available_ram_bytes();
        memory_budget = static_cast<size_t>(available_ram * 0.75);
      }
      const size_t entry_cost =
          req.embedding_dim * req.param_bytes_per_element +
          req.opt_bytes_per_entry + 64; // directory + locator overhead
      const size_t page_cost = checked_product(entries_per_page, entry_cost);
      max_pages = memory_budget / std::max(size_t{1}, page_cost);

      if (req.tuning.target_capacity_keys != 0) {
        if (req.tuning.target_capacity_keys >
            std::numeric_limits<size_t>::max() - entries_per_page + 1)
          throw std::overflow_error("target capacity keys overflows");
        const size_t needed_pages =
            (req.tuning.target_capacity_keys + entries_per_page - 1) /
            entries_per_page;
        max_pages = std::max(max_pages, needed_pages);
      }
      max_pages = std::max(max_pages, size_t{8192});
    }

    // Round max_pages up to a multiple of cohort_count
    const size_t cohort_count = shards / cohort_size;
    if (max_pages > std::numeric_limits<size_t>::max() - cohort_count + 1)
      throw std::overflow_error("storage page count alignment overflows");
    max_pages = ((max_pages + cohort_count - 1) / cohort_count) * cohort_count;

    const size_t max_safe_pages =
        std::numeric_limits<uint32_t>::max() / entries_per_page;
    if (max_pages > max_safe_pages)
      max_pages = (max_safe_pages / cohort_count) * cohort_count;

    return StorageTopology(execution_workers, shards, entries_per_page,
                           max_pages, cohort_size, numa_execution, numa_memory,
                           req.tuning.target_capacity_keys);
  }

private:
  struct CalibrationKey {
    size_t embedding_dim;
    size_t param_bytes_per_element;
    size_t opt_bytes_per_entry;
    size_t workers;
    size_t hardware_threads;
    size_t numa_nodes;
    StorageWorkload workload;
    LocalityMode execution_locality;
    LocalityMode memory_locality;

    bool operator==(const CalibrationKey &) const = default;
    bool operator<(const CalibrationKey &other) const noexcept {
      return std::tie(embedding_dim, param_bytes_per_element,
                      opt_bytes_per_entry, workers, hardware_threads,
                      numa_nodes, workload, execution_locality,
                      memory_locality) <
             std::tie(other.embedding_dim, other.param_bytes_per_element,
                      other.opt_bytes_per_entry, other.workers,
                      other.hardware_threads, other.numa_nodes, other.workload,
                      other.execution_locality, other.memory_locality);
    }
  };

  static size_t calibrate_entries_per_page(const TopologyTuneRequest &req,
                                           size_t workers) {
    if (!req.tuning.calibrate_page_geometry)
      return derive_entries_per_page(req);
    if (req.tuning.calibration_samples == 0)
      throw std::invalid_argument("page calibration samples must be positive");
    const CalibrationKey key{req.embedding_dim,
                             req.param_bytes_per_element,
                             req.opt_bytes_per_entry,
                             workers,
                             SystemResourceProbe::hardware_concurrency(),
                             SystemResourceProbe::numa_available()
                                 ? SystemResourceProbe::numa_node_count()
                                 : 1,
                             req.tuning.workload,
                             req.locality.execution,
                             req.locality.memory};
    static std::mutex mutex;
    static std::map<CalibrationKey, size_t> cache;
    {
      std::lock_guard lock(mutex);
      if (const auto found = cache.find(key); found != cache.end())
        return found->second;
    }

    constexpr std::array<size_t, 3> candidates{4096, 8192, 16384};
    std::array<std::vector<double>, candidates.size()> measurements;
    const size_t samples = std::min<size_t>(req.tuning.calibration_samples, 7);
    const size_t operations =
        std::max<size_t>(8192, 1u << 18) / std::max<size_t>(1, workers);
    const size_t row_bytes = checked_add(
        checked_product(req.embedding_dim, req.param_bytes_per_element),
        req.opt_bytes_per_entry);
    const size_t row_words = std::max<size_t>(1, (row_bytes + 63) / 64);
    const size_t working_set_entries = 1u << 18;
    uint64_t sink = 0;
    for (size_t candidate_index = 0; candidate_index < candidates.size();
         ++candidate_index) {
      const size_t entries_per_page = candidates[candidate_index];
      std::vector<std::unique_ptr<uint64_t[]>> pages;
      pages.reserve((working_set_entries + entries_per_page - 1) /
                    entries_per_page);
      for (size_t first = 0; first < working_set_entries;
           first += entries_per_page)
        pages.push_back(
            std::make_unique<uint64_t[]>(entries_per_page * row_words));
      for (size_t sample = 0; sample < samples; ++sample) {
        uint64_t state = 0x9e3779b97f4a7c15ULL + sample;
        const auto started = std::chrono::steady_clock::now();
        for (size_t operation = 0; operation < operations; ++operation) {
          if (req.tuning.workload == StorageWorkload::kCompaction) {
            const size_t index = operation % working_set_entries;
            const size_t page = index >> std::countr_zero(entries_per_page);
            const size_t slot = index & (entries_per_page - 1);
            sink += pages[page][slot * row_words] + page + slot;
          } else {
            state ^= state << 7;
            state ^= state >> 9;
            const size_t index =
                static_cast<size_t>(state) % working_set_entries;
            const size_t page = index >> std::countr_zero(entries_per_page);
            const size_t slot = index & (entries_per_page - 1);
            sink += pages[page][slot * row_words];
            if (req.tuning.workload == StorageWorkload::kTraining)
              sink +=
                  pages[page][slot * row_words + row_words - 1] + page + slot;
            if (req.tuning.workload == StorageWorkload::kLookupInsert)
              pages[page][slot * row_words] = sink;
          }
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
        if (!(seconds > 0.0))
          throw std::runtime_error("page calibration duration is non-positive");
        measurements[candidate_index].push_back(seconds);
      }
    }
    if (sink == std::numeric_limits<uint64_t>::max())
      throw std::runtime_error("page calibration sentinel reached");
    size_t selected = candidates.front();
    auto &baseline_samples = measurements.front();
    std::ranges::sort(baseline_samples);
    const double baseline_score = baseline_samples[baseline_samples.size() / 2];
    double selected_score = baseline_score;
    for (size_t candidate_index = 0; candidate_index < candidates.size();
         ++candidate_index) {
      auto &samples_for_candidate = measurements[candidate_index];
      std::ranges::sort(samples_for_candidate);
      const double score =
          samples_for_candidate[samples_for_candidate.size() / 2];
      if (candidate_index != 0 && score < baseline_score * 0.50 &&
          score < selected_score) {
        selected = candidates[candidate_index];
        selected_score = score;
      }
    }
    {
      std::lock_guard lock(mutex);
      cache.emplace(key, selected);
    }
    return selected;
  }

  static size_t derive_entries_per_page(const TopologyTuneRequest &req) {
    if (!req.tuning.calibrate_page_geometry)
      return kMinEntriesPerPage;
    const size_t row_bytes = checked_add(
        checked_product(req.embedding_dim, req.param_bytes_per_element),
        checked_add(req.opt_bytes_per_entry, 64));
    const size_t target_entries =
        std::max(size_t{1}, kTargetPagePayloadBytes / row_bytes);
    return std::clamp(std::bit_floor(target_entries), kMinEntriesPerPage,
                      kMaxEntriesPerPage);
  }

  static size_t checked_product(size_t left, size_t right) {
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
      throw std::overflow_error("storage tuning size overflows");
    return left * right;
  }

  static size_t checked_add(size_t left, size_t right) {
    if (right > std::numeric_limits<size_t>::max() - left)
      throw std::overflow_error("storage tuning size overflows");
    return left + right;
  }
};

} // namespace se::storage
