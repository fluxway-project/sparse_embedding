#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <latch>
#include <limits>
#include <memory_resource>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <sys/resource.h>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/common/function.hpp"
#include "sparse_embedding/src/interface/optimizer_interface.hpp"
#include "sparse_embedding/src/optimizer/optimizer_contract.hpp"

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
  void prefetch_state(uint32_t) {}
  bool requires_score() const { return false; }
  void step(uint32_t, std::span<float> parameter,
            std::span<const float> gradient, uint64_t, uint64_t) {
    for (size_t index = 0; index < parameter.size(); ++index)
      parameter[index] -= 0.01F * gradient[index];
  }
  void update(const se::OptimizerBatchView<float> &batch, uint64_t) {
    for (size_t row = 0; row < batch.size(); ++row)
      step(batch.rows[row].index, batch.rows[row].parameters,
           batch.rows[row].gradients, 0, batch.rows[row].score);
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

struct JournalAblationResult {
  std::string_view name;
  size_t mutations;
  double seconds;
  size_t allocations;
  size_t captured;
};

struct MutationSchedule {
  std::vector<std::vector<uint64_t>> keys;
  size_t touched;
};

size_t schedule_operation_count(const MutationSchedule &schedule) {
  size_t total = 0;
  for (const auto &worker_keys : schedule.keys) {
    if (worker_keys.size() > std::numeric_limits<size_t>::max() - total)
      throw std::overflow_error("benchmark schedule operation count overflows");
    total += worker_keys.size();
  }
  return total;
}

size_t checked_schedule_repeat_count(size_t count, size_t rounds) {
  if (count != 0 && rounds > std::numeric_limits<size_t>::max() / count)
    throw std::overflow_error("benchmark schedule size overflows");
  return count * rounds;
}

MutationSchedule make_schedule(size_t workers, size_t key_count, size_t rounds,
                               std::string_view distribution) {
  if (workers == 0 || key_count == 0 || rounds == 0)
    throw std::invalid_argument("journal schedule dimensions must be positive");
  std::vector<double> weights;
  if (distribution == "zipf") {
    weights.resize(key_count);
    for (size_t key = 0; key < key_count; ++key)
      weights[key] = 1.0 / std::pow(static_cast<double>(key + 1), 1.1);
  } else if (distribution != "uniform") {
    throw std::invalid_argument("expected uniform or zipf distribution");
  }
  std::discrete_distribution<size_t> zipf(weights.begin(), weights.end());
  std::uniform_int_distribution<size_t> uniform(0, key_count - 1);
  MutationSchedule schedule{std::vector<std::vector<uint64_t>>(workers), 0};
  std::vector<bool> touched(key_count, false);
  for (size_t worker = 0; worker < workers; ++worker) {
    std::mt19937_64 random(0x9e3779b97f4a7c15ULL + worker);
    const size_t count =
        worker < key_count ? (key_count - 1 - worker) / workers + 1 : 0;
    const size_t operation_count = checked_schedule_repeat_count(count, rounds);
    auto &keys = schedule.keys[worker];
    keys.reserve(operation_count);
    for (size_t operation = 0; operation < operation_count; ++operation) {
      const uint64_t key =
          distribution == "zipf" ? zipf(random) : uniform(random);
      keys.push_back(key);
      if (!touched[key]) {
        touched[key] = true;
        ++schedule.touched;
      }
    }
  }
  return schedule;
}

MutationSchedule
make_disjoint_training_schedule(size_t workers, size_t key_count, size_t rounds,
                                std::string_view distribution) {
  if (workers == 0 || key_count == 0 || rounds == 0)
    throw std::invalid_argument(
        "training schedule dimensions must be positive");
  MutationSchedule schedule{std::vector<std::vector<uint64_t>>(workers), 0};
  for (size_t worker = 0; worker < workers; ++worker) {
    std::vector<uint64_t> owned_keys;
    for (size_t key = worker; key < key_count; key += workers)
      owned_keys.push_back(key);
    if (owned_keys.empty())
      continue;
    std::mt19937_64 random(0xd1b54a32d192ed03ULL + worker);
    std::vector<double> weights;
    if (distribution == "zipf") {
      weights.resize(owned_keys.size());
      for (size_t index = 0; index < owned_keys.size(); ++index)
        weights[index] = 1.0 / std::pow(static_cast<double>(index + 1), 1.1);
    } else if (distribution != "uniform") {
      throw std::invalid_argument("expected uniform or zipf distribution");
    }
    std::discrete_distribution<size_t> zipf(weights.begin(), weights.end());
    std::uniform_int_distribution<size_t> uniform(0, owned_keys.size() - 1);
    const size_t operation_count =
        checked_schedule_repeat_count(owned_keys.size(), rounds);
    auto &keys = schedule.keys[worker];
    keys.reserve(operation_count);
    for (size_t operation = 0; operation < operation_count; ++operation)
      keys.push_back(
          owned_keys[distribution == "zipf" ? zipf(random) : uniform(random)]);
    schedule.touched += owned_keys.size();
  }
  return schedule;
}

class CountingMemoryResource final : public std::pmr::memory_resource {
  size_t allocations_ = 0;

  void *do_allocate(size_t bytes, size_t alignment) override {
    void *result = std::pmr::new_delete_resource()->allocate(bytes, alignment);
    ++allocations_;
    return result;
  }

  void do_deallocate(void *pointer, size_t bytes, size_t alignment) override {
    std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
  }

  bool
  do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }

public:
  size_t allocations() const noexcept { return allocations_; }
};

class MutexMapJournal {
  struct State {
    uint64_t sequence;
    bool deleted;
  };

  struct Shard {
    std::mutex mutex;
    CountingMemoryResource memory;
    std::pmr::unordered_map<uint64_t, State> entries{&memory};
  };

  std::vector<Shard> shards_;
  std::atomic<uint64_t> sequence_{0};

public:
  MutexMapJournal(size_t shard_count, size_t, bool sharded)
      : shards_(sharded ? shard_count : 1) {}

  void mutate(uint64_t key, bool deleted) {
    Shard &shard = shards_[key & (shards_.size() - 1)];
    std::lock_guard lock(shard.mutex);
    const uint64_t sequence =
        sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    shard.entries.insert_or_assign(key, State{sequence, deleted});
  }

  size_t capture_and_ack() {
    size_t captured = 0;
    const uint64_t watermark = sequence_.load(std::memory_order_acquire);
    for (Shard &shard : shards_) {
      std::lock_guard lock(shard.mutex);
      captured += shard.entries.size();
      std::erase_if(shard.entries, [watermark](const auto &entry) {
        return entry.second.sequence <= watermark;
      });
    }
    return captured;
  }

  size_t allocations() const noexcept {
    size_t result = shards_.capacity() == 0 ? 0 : 1;
    for (const Shard &shard : shards_)
      result += shard.memory.allocations();
    return result;
  }

  void reactivate(uint64_t key) { mutate(key, false); }

  bool validate_after_ack() const {
    return std::ranges::all_of(
        shards_, [](const Shard &shard) { return shard.entries.empty(); });
  }
};

class ReservedNodeJournal {
  struct State {
    uint64_t key = 0;
    uint64_t sequence = 0;
    bool deleted = false;
    bool occupied = false;
  };

  struct Shard {
    std::mutex mutex;
    CountingMemoryResource memory;
    std::pmr::unordered_map<uint64_t, size_t> entries{&memory};
    std::vector<State> nodes;
    std::vector<size_t> free_nodes;
  };

  std::vector<Shard> shards_;
  std::atomic<uint64_t> sequence_{0};

public:
  ReservedNodeJournal(size_t shard_count, size_t key_count, bool sharded)
      : shards_(shard_count) {
    if (!sharded)
      throw std::invalid_argument("reserved-state-map requires sharded mode");
    for (Shard &shard : shards_) {
      const size_t capacity = key_count / shard_count + 1;
      shard.entries.reserve(capacity);
      shard.nodes.resize(capacity);
      shard.free_nodes.reserve(capacity);
      for (size_t index = capacity; index != 0; --index)
        shard.free_nodes.push_back(index - 1);
    }
  }

  void mutate(uint64_t key, bool deleted) {
    Shard &shard = shards_[key & (shards_.size() - 1)];
    std::lock_guard lock(shard.mutex);
    const uint64_t sequence =
        sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    auto found = shard.entries.find(key);
    if (found == shard.entries.end()) {
      if (shard.free_nodes.empty())
        throw std::length_error("reserved journal capacity exhausted");
      const size_t node = shard.free_nodes.back();
      shard.free_nodes.pop_back();
      shard.nodes[node] = State{key, sequence, deleted, true};
      shard.entries.emplace(key, node);
      return;
    }
    State &state = shard.nodes[found->second];
    state.sequence = sequence;
    state.deleted = deleted;
  }

  size_t capture_and_ack() {
    size_t captured = 0;
    const uint64_t watermark = sequence_.load(std::memory_order_acquire);
    for (Shard &shard : shards_) {
      std::lock_guard lock(shard.mutex);
      captured += shard.entries.size();
      for (auto iterator = shard.entries.begin();
           iterator != shard.entries.end();) {
        State &state = shard.nodes[iterator->second];
        if (state.sequence <= watermark) {
          state.occupied = false;
          shard.free_nodes.push_back(iterator->second);
          iterator = shard.entries.erase(iterator);
        } else {
          ++iterator;
        }
      }
    }
    return captured;
  }

  size_t allocations() const noexcept {
    size_t result = shards_.capacity() == 0 ? 0 : 1;
    for (const Shard &shard : shards_)
      result += shard.memory.allocations() +
                (shard.nodes.capacity() == 0 ? 0 : 1) +
                (shard.free_nodes.capacity() == 0 ? 0 : 1);
    return result;
  }

  void reactivate(uint64_t key) { mutate(key, false); }

  bool validate_after_ack() const {
    return std::ranges::all_of(
        shards_, [](const Shard &shard) { return shard.entries.empty(); });
  }
};

struct StableJournalRecord {
  std::atomic<uint64_t> state{0};
  std::atomic<bool> queued{false};
  StableJournalRecord *next = nullptr;
};

class StableRecordJournal {
  static constexpr uint64_t kDeleted = uint64_t{1} << 63;

  std::vector<StableJournalRecord> records_;
  std::vector<std::atomic<StableJournalRecord *>> heads_;
  std::atomic<uint64_t> sequence_{0};
  bool sharded_;

  void admit(StableJournalRecord &record, size_t shard) {
    if (record.queued.exchange(true, std::memory_order_acq_rel))
      return;
    auto &head = heads_[sharded_ ? shard : 0];
    StableJournalRecord *current = head.load(std::memory_order_relaxed);
    do {
      record.next = current;
    } while (!head.compare_exchange_weak(current, &record,
                                         std::memory_order_release,
                                         std::memory_order_relaxed));
  }

public:
  StableRecordJournal(size_t shard_count, size_t key_count, bool sharded)
      : records_(key_count), heads_(sharded ? shard_count : 1),
        sharded_(sharded) {
    for (auto &head : heads_)
      head.store(nullptr, std::memory_order_relaxed);
  }

  void mutate(uint64_t key, bool deleted) {
    StableJournalRecord &record = records_[key];
    uint64_t sequence = sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t previous = record.state.load(std::memory_order_acquire);
    for (;;) {
      if ((previous & kDeleted) != 0)
        return;
      if (!deleted && previous >= sequence)
        break;
      const uint64_t next =
          deleted ? std::max(sequence, previous) | kDeleted : sequence;
      if (record.state.compare_exchange_weak(previous, next,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
        break;
    }
    admit(record, key & (heads_.size() - 1));
  }

  size_t capture_and_ack() {
    const uint64_t watermark = sequence_.load(std::memory_order_acquire);
    size_t captured = 0;
    for (auto &head : heads_) {
      StableJournalRecord *current = head.load(std::memory_order_acquire);
      StableJournalRecord *retained = nullptr;
      while (current != nullptr) {
        StableJournalRecord *next = current->next;
        const uint64_t state = current->state.load(std::memory_order_acquire);
        if ((state & ~kDeleted) != 0) {
          ++captured;
          if ((state & ~kDeleted) > watermark) {
            current->next = retained;
            retained = current;
          } else {
            current->queued.store(false, std::memory_order_release);
          }
        }
        current = next;
      }
      head.store(retained, std::memory_order_release);
    }
    return captured;
  }

  size_t allocations() const noexcept {
    return (records_.capacity() == 0 ? 0 : 1) +
           (heads_.capacity() == 0 ? 0 : 1);
  }

  void reactivate(uint64_t key) {
    StableJournalRecord &record = records_[key];
    record.state.store(0, std::memory_order_release);
    record.queued.store(false, std::memory_order_release);
    record.next = nullptr;
    mutate(key, false);
  }

  bool validate_after_ack() const {
    for (const auto &head : heads_)
      if (head.load(std::memory_order_acquire) != nullptr)
        return false;
    return std::ranges::all_of(records_, [](const StableJournalRecord &record) {
      return !record.queued.load(std::memory_order_acquire);
    });
  }
};

template <typename Journal>
JournalAblationResult
run_journal_ablation(std::string_view name, size_t workers, size_t key_count,
                     const MutationSchedule &schedule, bool sharded = true) {
  const size_t shard_count = std::bit_ceil(std::max(size_t{1}, workers * 2));
  Journal journal(shard_count, key_count, sharded);
  std::vector<uint64_t> keys(key_count);
  for (size_t index = 0; index < key_count; ++index)
    keys[index] = index;
  for (uint64_t key : keys)
    journal.mutate(key, false);
  journal.capture_and_ack();

  std::barrier start(static_cast<std::ptrdiff_t>(workers + 1));
  std::vector<std::jthread> threads;
  threads.reserve(workers);
  for (size_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      start.arrive_and_wait();
      for (uint64_t key : schedule.keys[worker])
        journal.mutate(key, false);
    });
  }
  const auto started = std::chrono::steady_clock::now();
  start.arrive_and_wait();
  threads.clear();
  const double seconds = elapsed_seconds(started);
  const size_t captured = journal.capture_and_ack();
  if (captured != schedule.touched || !journal.validate_after_ack())
    throw std::runtime_error("journal ablation state validation failed");
  return {name, schedule_operation_count(schedule), seconds,
          journal.allocations(), captured};
}

template <typename Journal>
JournalAblationResult run_journal_churn(std::string_view name, size_t workers,
                                        size_t key_count, size_t rounds,
                                        const MutationSchedule &schedule,
                                        bool sharded = true) {
  const size_t shard_count = std::bit_ceil(std::max(size_t{1}, workers * 2));
  Journal journal(shard_count, key_count, sharded);
  for (size_t key = 0; key < key_count; ++key)
    journal.mutate(key, false);
  journal.capture_and_ack();
  if (rounds == 0)
    throw std::invalid_argument("journal churn rounds must be positive");
  for (const auto &worker_keys : schedule.keys)
    if (worker_keys.size() % rounds != 0)
      throw std::logic_error("journal schedule is not round-partitioned");
  std::barrier phase(static_cast<std::ptrdiff_t>(workers + 1));
  std::vector<std::jthread> threads;
  threads.reserve(workers);
  for (size_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      const size_t count = schedule.keys[worker].size() / rounds;
      for (size_t round = 0; round < rounds; ++round) {
        phase.arrive_and_wait();
        for (uint64_t key :
             std::span(schedule.keys[worker]).subspan(round * count, count))
          journal.mutate(key, false);
        phase.arrive_and_wait();
      }
    });
  }
  const auto started = std::chrono::steady_clock::now();
  size_t mutations = schedule_operation_count(schedule);
  for (size_t round = 0; round < rounds; ++round) {
    phase.arrive_and_wait();
    phase.arrive_and_wait();
    journal.capture_and_ack();
    for (size_t key = round % 100; key < key_count; key += 100) {
      journal.mutate(key, true);
      ++mutations;
    }
    journal.capture_and_ack();
    for (size_t key = round % 100; key < key_count; key += 100) {
      journal.reactivate(key);
      ++mutations;
    }
  }
  threads.clear();
  const double seconds = elapsed_seconds(started);
  const size_t captured = journal.capture_and_ack();
  if (!journal.validate_after_ack())
    throw std::runtime_error("journal churn state validation failed");
  return {name, mutations, seconds, journal.allocations(), captured};
}

void run_journal_ablation_suite(size_t workers, size_t key_count, size_t rounds,
                                std::string_view distribution) {
  const auto schedule = make_schedule(workers, key_count, rounds, distribution);
  const size_t churn_rounds = std::max(size_t{2}, rounds / 4);
  const auto churn_schedule =
      make_schedule(workers, key_count, churn_rounds, distribution);
  const std::array<JournalAblationResult, 4> results{
      run_journal_ablation<MutexMapJournal>("mutex-map", workers, key_count,
                                            schedule),
      run_journal_ablation<ReservedNodeJournal>("reserved-state-map", workers,
                                                key_count, schedule),
      run_journal_ablation<StableRecordJournal>("stable-sharded", workers,
                                                key_count, schedule),
      run_journal_ablation<StableRecordJournal>("stable-single", workers,
                                                key_count, schedule, false),
  };
  for (const auto &result : results) {
    if (result.seconds <= 0.0 || result.captured == 0)
      throw std::runtime_error("journal ablation produced invalid result");
    std::cout << "journal variant=" << result.name << " workers=" << workers
              << " keys=" << key_count << " distribution=" << distribution
              << " rounds=" << rounds << " seconds=" << std::fixed
              << std::setprecision(6) << result.seconds
              << " mops=" << std::setprecision(3)
              << static_cast<double>(result.mutations) / result.seconds / 1e6
              << " allocator_calls=" << result.allocations
              << " captured=" << result.captured << '\n';
  }
  const std::array<JournalAblationResult, 4> churn_results{
      run_journal_churn<MutexMapJournal>("mutex-map", workers, key_count,
                                         churn_rounds, churn_schedule),
      run_journal_churn<ReservedNodeJournal>("reserved-state-map", workers,
                                             key_count, churn_rounds,
                                             churn_schedule),
      run_journal_churn<StableRecordJournal>(
          "stable-sharded", workers, key_count, churn_rounds, churn_schedule),
      run_journal_churn<StableRecordJournal>("stable-single", workers,
                                             key_count, churn_rounds,
                                             churn_schedule, false),
  };
  for (const auto &result : churn_results) {
    if (result.seconds <= 0.0 || result.captured == 0)
      throw std::runtime_error("journal churn produced invalid result");
    std::cout << "journal churn variant=" << result.name
              << " workers=" << workers << " keys=" << key_count
              << " distribution=" << distribution << " seconds=" << std::fixed
              << std::setprecision(6) << result.seconds
              << " mops=" << std::setprecision(3)
              << static_cast<double>(result.mutations) / result.seconds / 1e6
              << " allocator_calls=" << result.allocations
              << " captured=" << result.captured << '\n';
  }
}

double percentile_ms(std::vector<double> samples, double quantile) {
  if (samples.empty())
    throw std::invalid_argument("latency sample is empty");
  std::ranges::sort(samples);
  const size_t index = static_cast<size_t>(std::ceil(
                           quantile * static_cast<double>(samples.size()))) -
                       1;
  return samples.at(index) * 1000.0;
}

void run_training_checkpoint_profile(size_t workers, size_t key_count,
                                     size_t rounds, size_t batch_size,
                                     std::string_view distribution,
                                     bool bind_affinity, size_t ack_delay_us,
                                     size_t churn_stride) {
  using Store = se::HogwildEmbeddingStoreCore<float, float>;
  auto optimizer = pro::make_proxy<
      se::FunctionFacade<se::OptimizerExtension<float, float>(size_t)>>(
      [](size_t) {
        return pro::make_proxy<se::OptimizerExtensionFacade<float, float>,
                               ProfileSgd>();
      });
  Store store({kDimension}, {}, std::move(optimizer), se::LfuScorePolicy{},
              se::HogwildStoreOptions{
                  .thread_budget = workers,
                  .locality = se::LocalityPolicy{bind_affinity
                                                     ? se::LocalityMode::kNuma
                                                     : se::LocalityMode::kAuto,
                                                 se::LocalityMode::kAuto},
                  .storage_tuning = {
                      .workload = se::storage::StorageWorkload::kTraining,
                      .calibrate_page_geometry = true}});
  const auto schedule =
      make_disjoint_training_schedule(workers, key_count, rounds, distribution);
  std::vector<float> reference(key_count, 0.25F), replica(key_count, 0);
  std::vector<uint64_t> reference_scores(key_count, 0),
      replica_scores(key_count, 0);
  std::vector<bool> replica_live(key_count, false);
  std::vector<uint64_t> all_keys(key_count);
  std::iota(all_keys.begin(), all_keys.end(), uint64_t{0});
  std::vector<float> initial_values(key_count * kDimension, 0.25F);
  std::vector<int64_t> initial_lengths(key_count, kDimension);
  store.assign<uint64_t>(all_keys, initial_values, initial_lengths);
  initial_values.clear();
  initial_values.shrink_to_fit();
  size_t payload_rows = 0, tombstones = 0;
  auto capture = [&](bool incremental, const std::string &parent) {
    return store.capture_checkpoint(
        incremental, parent, false, store.checkpoint_page_layout_generation(),
        pro::make_proxy<se::FunctionFacade<void(
            std::span<const se::CheckpointMutation>, uint64_t)>>(
            [&](std::span<const se::CheckpointMutation> mutations,
                uint64_t watermark) {
              uint64_t previous_sequence = 0;
              for (const auto &mutation : mutations) {
                if (mutation.dimension != kDimension ||
                    mutation.key >= key_count ||
                    mutation.mutation_sequence <= previous_sequence ||
                    mutation.mutation_sequence > watermark)
                  throw std::runtime_error("invalid checkpoint mutation cut");
                previous_sequence = mutation.mutation_sequence;
                if (mutation.deleted) {
                  replica_live[mutation.key] = false;
                  ++tombstones;
                }
              }
            }),
        pro::make_proxy<se::FunctionFacade<bool(
            size_t, se::PageImageExport<float, float> &&)>>(
            [&](size_t dimension, se::PageImageExport<float, float> &&image) {
              if (dimension != kDimension ||
                  image.values.size() != image.keys.size() * kDimension ||
                  image.scores.size() != image.keys.size())
                throw std::runtime_error("invalid checkpoint payload shape");
              for (size_t row = 0; row < image.keys.size(); ++row) {
                const uint64_t key = image.keys[row];
                if (key >= key_count)
                  throw std::runtime_error("unexpected checkpoint key");
                const float value = image.values[row * kDimension];
                if (!std::ranges::all_of(
                        std::span(image.values)
                            .subspan(row * kDimension, kDimension),
                        [value](float lane) { return lane == value; }))
                  throw std::runtime_error("torn checkpoint vector");
                replica[key] = value;
                replica_scores[key] = image.scores[row];
                replica_live[key] = true;
              }
              payload_rows += image.keys.size();
              return true;
            }));
  };
  auto acknowledge = [&](const se::CheckpointCaptureCut &cut,
                         const std::string &baseline) {
    store.advance_checkpoint_baseline(cut.mutation_watermark, baseline,
                                      cut.capture_id, cut.journal_identity);
  };
  acknowledge(capture(false, ""), "training-base");
  payload_rows = 0;
  std::atomic<bool> abort_training{false};
  std::barrier batch_start(static_cast<std::ptrdiff_t>(workers + 1));
  std::barrier gradients_ready(static_cast<std::ptrdiff_t>(workers + 1));
  std::barrier optimizer_complete(static_cast<std::ptrdiff_t>(workers + 1));
  auto drop_after_abort = [&] {
    batch_start.arrive_and_drop();
    gradients_ready.arrive_and_drop();
    optimizer_complete.arrive_and_drop();
  };
  std::atomic<bool> finished{false};
  std::latch checkpoint_ready(1), begin(1);
  std::vector<double> capture_latency;
  auto checkpointer = std::async(std::launch::async, [&] {
    checkpoint_ready.count_down();
    begin.wait();
    std::string parent = "training-base";
    size_t generation = 0;
    do {
      const bool final = finished.load(std::memory_order_acquire);
      const auto started = std::chrono::steady_clock::now();
      const auto cut = capture(true, parent);
      capture_latency.push_back(elapsed_seconds(started));
      if (!final && ack_delay_us != 0)
        std::this_thread::sleep_for(std::chrono::microseconds(ack_delay_us));
      parent = "training-" + std::to_string(++generation);
      acknowledge(cut, parent);
      if (final)
        break;
    } while (true);
  });
  checkpoint_ready.wait();
  std::vector<std::vector<double>> producer_batch_latency(workers);
  std::vector<size_t> batch_unique_rows(workers, 0);
  std::vector<size_t> producer_training_rows(workers, 0);
  std::vector<size_t> producer_churn_rows(workers, 0);
  size_t batch_count = 0;
  for (const auto &worker_keys : schedule.keys)
    batch_count =
        std::max(batch_count, worker_keys.size() / batch_size +
                                  (worker_keys.size() % batch_size != 0));
  std::vector<std::jthread> producers;
  producers.reserve(workers);
  for (size_t worker = 0; worker < workers; ++worker) {
    producers.emplace_back([&, worker] {
      const auto &worker_schedule = schedule.keys[worker];
      std::vector<uint64_t> keys, churn;
      std::vector<int64_t> lengths;
      std::vector<float> values, output, gradients;
      size_t training_rows = 0;
      size_t churn_rows = 0;
      for (size_t batch = 0; batch < batch_count; ++batch) {
        const size_t offset = batch * batch_size;
        const size_t count =
            offset < worker_schedule.size()
                ? std::min(batch_size, worker_schedule.size() - offset)
                : 0;
        keys.assign(worker_schedule.begin() + offset,
                    worker_schedule.begin() + offset + count);
        std::ranges::sort(keys);
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        batch_unique_rows[worker] = keys.size();
        lengths.assign(keys.size(), kDimension);
        values.assign(keys.size() * kDimension, 0.25F);
        output.resize(values.size());
        gradients.assign(values.size(), 0.001F);
        const auto batch_started = std::chrono::steady_clock::now();
        batch_start.arrive_and_wait();
        if (abort_training.load(std::memory_order_acquire)) {
          drop_after_abort();
          return;
        }
        if (!keys.empty()) {
          store.forward_lookup<uint64_t>(keys, output, lengths);
          store.assign<uint64_t>(keys, values, lengths);
          store.accumulate_gradients<uint64_t>(keys, gradients, lengths);
          training_rows += keys.size();
        }
        gradients_ready.arrive_and_wait();
        if (abort_training.load(std::memory_order_acquire)) {
          drop_after_abort();
          return;
        }
        optimizer_complete.arrive_and_wait();
        if (abort_training.load(std::memory_order_acquire)) {
          drop_after_abort();
          return;
        }
        for (uint64_t key : keys) {
          reference[key] = 0.25F - 0.01F * 0.001F;
          ++reference_scores[key];
        }
        churn.clear();
        for (size_t row = 0; row < keys.size(); row += churn_stride)
          churn.push_back(keys[row]);
        if (!churn.empty()) {
          store.erase_dimension<uint64_t>(kDimension, churn);
          values.assign(churn.size() * kDimension, 0.25F);
          lengths.assign(churn.size(), kDimension);
          store.assign<uint64_t>(churn, values, lengths);
          for (uint64_t key : churn) {
            reference[key] = 0.25F;
            reference_scores[key] = 0;
          }
          churn_rows += churn.size();
        }
        producer_batch_latency[worker].push_back(
            elapsed_seconds(batch_started));
      }
      producer_training_rows[worker] = training_rows;
      producer_churn_rows[worker] = churn_rows;
    });
  }
  const auto started = std::chrono::steady_clock::now();
  begin.count_down();
  size_t optimizer_rows = 0;
  try {
    for (size_t batch = 0; batch < batch_count; ++batch) {
      batch_start.arrive_and_wait();
      gradients_ready.arrive_and_wait();
      store.optimizer_step(1);
      optimizer_complete.arrive_and_wait();
      const size_t updated = std::accumulate(
          batch_unique_rows.begin(), batch_unique_rows.end(), size_t{0});
      optimizer_rows += updated;
    }
  } catch (...) {
    finished.store(true, std::memory_order_release);
    abort_training.store(true, std::memory_order_release);
    drop_after_abort();
    producers.clear();
    checkpointer.wait();
    throw;
  }
  producers.clear();
  finished.store(true, std::memory_order_release);
  checkpointer.get();
  const double seconds = elapsed_seconds(started);
  std::vector<double> batch_latency;
  for (auto &latencies : producer_batch_latency)
    batch_latency.insert(batch_latency.end(), latencies.begin(),
                         latencies.end());
  const size_t training_rows = std::accumulate(
      producer_training_rows.begin(), producer_training_rows.end(), size_t{0});
  const size_t churn_rows = std::accumulate(
      producer_churn_rows.begin(), producer_churn_rows.end(), size_t{0});
  if (training_rows >
          (std::numeric_limits<size_t>::max() - optimizer_rows) / 3 ||
      churn_rows > (std::numeric_limits<size_t>::max() - training_rows * 3 -
                    optimizer_rows) /
                       2)
    throw std::overflow_error("benchmark row-operation count overflows");
  const size_t operation_count =
      training_rows * 3 + optimizer_rows + churn_rows * 2;
  for (size_t key = 0; key < key_count; ++key) {
    if (!replica_live[key] || std::abs(replica[key] - reference[key]) > 1e-6F ||
        replica_scores[key] != reference_scores[key])
      throw std::runtime_error(
          "incremental replica differs from training reference");
  }
  if (!store.checkpoint_mutations().empty())
    throw std::runtime_error("final acknowledgement retained dirty state");
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("cannot measure peak RSS");
  std::cout << "training workers=" << workers << " keys=" << key_count
            << " distribution=" << distribution << " rounds=" << rounds
            << " batch_size=" << batch_size << " affinity=" << bind_affinity
            << " row_operations=" << operation_count
            << " training_rows=" << training_rows
            << " optimizer_rows=" << optimizer_rows
            << " churn_rows=" << churn_rows << " producers=" << workers
            << " seconds=" << std::fixed << std::setprecision(6) << seconds
            << " mops=" << static_cast<double>(operation_count) / seconds / 1e6
            << " batch_p95_ms=" << percentile_ms(batch_latency, 0.95)
            << " batch_p99_ms=" << percentile_ms(batch_latency, 0.99)
            << " capture_p95_ms=" << percentile_ms(capture_latency, 0.95)
            << " capture_p99_ms=" << percentile_ms(capture_latency, 0.99)
            << " captures=" << capture_latency.size()
            << " payload_rows=" << payload_rows << " tombstones=" << tombstones
            << " peak_rss_kib=" << usage.ru_maxrss << " correctness=passed\n";
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
  if (mode != "assign" && mode != "accumulate" && mode != "optimizer" &&
      mode != "journal-ablation" && mode != "training-checkpoint")
    throw std::invalid_argument("unknown training profile mode");
  bool bind_affinity = true;
  size_t workers = 8;
  size_t repetitions = 100;
  size_t key_count = kRows;
  size_t batch_size = 4096, ack_delay_us = 250, churn_stride = 100;
  std::string_view distribution = "uniform";
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
    } else if (option.starts_with("--keys=")) {
      key_count = parse_positive(
          option.substr(std::string_view("--keys=").size()), "keys");
    } else if (option.starts_with("--distribution=")) {
      distribution = option.substr(std::string_view("--distribution=").size());
    } else if (option.starts_with("--batch-size=")) {
      batch_size = parse_positive(option.substr(13), "batch-size");
    } else if (option.starts_with("--ack-delay-us=")) {
      ack_delay_us = parse_positive(option.substr(15), "ack-delay-us");
    } else if (option.starts_with("--churn-stride=")) {
      churn_stride = parse_positive(option.substr(15), "churn-stride");
    } else {
      throw std::invalid_argument("unknown training profile option");
    }
  }

  if (mode == "journal-ablation") {
    run_journal_ablation_suite(workers, key_count, repetitions, distribution);
    return 0;
  }
  if (mode == "training-checkpoint") {
    run_training_checkpoint_profile(workers, key_count, repetitions, batch_size,
                                    distribution, bind_affinity, ack_delay_us,
                                    churn_stride);
    return 0;
  }

  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  auto optimizer_factory = pro::make_proxy<
      se::FunctionFacade<se::OptimizerExtension<float, float>(size_t)>>(
      [](size_t) {
        return pro::make_proxy<se::OptimizerExtensionFacade<float, float>,
                               ProfileSgd>();
      });
  Store store({kDimension}, {}, optimizer_factory, se::NoScorePolicy{},
              se::HogwildStoreOptions{
                  .thread_budget = workers,
                  .locality = se::LocalityPolicy{bind_affinity
                                                     ? se::LocalityMode::kNuma
                                                     : se::LocalityMode::kAuto,
                                                 se::LocalityMode::kAuto},
                  .storage_tuning = {
                      .workload = se::storage::StorageWorkload::kTraining,
                      .calibrate_page_geometry = true}});
  std::vector<uint64_t> keys(key_count);
  std::vector<int64_t> lengths(key_count, static_cast<int64_t>(kDimension));
  std::vector<float> values(key_count * kDimension, 0.25F);
  std::vector<float> output(key_count * kDimension);
  std::vector<float> gradients(key_count * kDimension, 0.001F);
  for (size_t index = 0; index < key_count; ++index)
    keys[index] = index + 1;
  store.lookup_or_insert<uint64_t>(keys, output, lengths);

  if (mode == "assign") {
    store.assign<uint64_t>(keys, values, lengths);
    const auto started = std::chrono::steady_clock::now();
    for (size_t repetition = 0; repetition < repetitions; ++repetition)
      store.assign<uint64_t>(keys, values, lengths);
    print_result("assign", key_count * repetitions, elapsed_seconds(started));
    return 0;
  }
  if (mode == "accumulate") {
    const auto started = std::chrono::steady_clock::now();
    for (size_t repetition = 0; repetition < repetitions; ++repetition) {
      store.accumulate_gradients<uint64_t>(keys, gradients, lengths);
      store.optimizer_step(1);
    }
    print_result("accumulate", key_count * repetitions,
                 elapsed_seconds(started));
    return 0;
  }
  std::chrono::nanoseconds optimizer_elapsed{};
  for (size_t repetition = 0; repetition < repetitions; ++repetition) {
    store.accumulate_gradients<uint64_t>(keys, gradients, lengths);
    const auto started = std::chrono::steady_clock::now();
    store.optimizer_step(1);
    optimizer_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
  }
  print_result("optimizer", key_count * repetitions,
               std::chrono::duration<double>(optimizer_elapsed).count());
}
