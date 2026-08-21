#include <parlay_hash/unordered_map.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "sparse_embedding/src/execution/topology/thread_affinity.hpp"

namespace {

struct SplitMixHash {
  size_t operator()(int64_t key) const noexcept {
    uint64_t value = static_cast<uint64_t>(key);
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<size_t>(value);
  }
};

using MapData =
    parlay::MapData<int64_t, uint32_t, SplitMixHash, std::equal_to<int64_t>>;
using Key = parlay::IndirectEntries<MapData>;
using KeyMap = parlay::parlay_hash<Key>;

struct ParlayBundle {
  Key entries;
  mutable KeyMap map;

  explicit ParlayBundle(size_t capacity)
      : entries(false), map(capacity, &entries, false) {}

  std::optional<uint32_t> find(int64_t key) const {
    return map.Find(Key::Entry::make_key(key),
                    [](const auto &entry) { return entry.get_entry().second; });
  }

  bool insert(int64_t key, uint32_t locator) {
    const auto map_key = Key::Entry::make_key(key);
    const auto existing = map.Insert(
        map_key,
        [&] {
          return entries.make_entry(map_key, std::make_pair(key, locator));
        },
        [](const auto &entry) { return entry.get_entry().second; });
    return !existing.has_value();
  }
};

class CurrentParlayDirectory {
public:
  explicit CurrentParlayDirectory(size_t capacity) : bundle_(capacity) {}

  void bulk_load(const std::vector<int64_t> &keys) {
    std::unique_lock lock(mutex_);
    for (size_t index = 0; index < keys.size(); ++index) {
      if (!bundle_.insert(keys[index], static_cast<uint32_t>(index)))
        throw std::logic_error("duplicate key in current directory bulk load");
    }
  }

  std::optional<uint32_t> find(int64_t key) const {
    std::shared_lock lock(mutex_);
    return bundle_.find(key);
  }

  bool insert(int64_t key, uint32_t locator) {
    std::unique_lock lock(mutex_);
    return bundle_.insert(key, locator);
  }

  bool update(int64_t key, uint32_t locator) {
    std::unique_lock lock(mutex_);
    bool updated = false;
    bundle_.map.Find(Key::Entry::make_key(key), [&](const auto &entry) {
      entry.get_entry().second = locator;
      updated = true;
      return locator;
    });
    return updated;
  }

  bool erase(int64_t key) {
    std::unique_lock lock(mutex_);
    return bundle_.map
        .Remove(Key::Entry::make_key(key),
                [](const auto &entry) { return entry.get_entry().second; })
        .has_value();
  }

private:
  mutable std::shared_mutex mutex_;
  ParlayBundle bundle_;
};

class SharedPtrCowDirectory {
public:
  explicit SharedPtrCowDirectory(size_t capacity)
      : capacity_(capacity), bundle_(std::make_shared<ParlayBundle>(capacity)) {
  }

  void bulk_load(const std::vector<int64_t> &keys) {
    auto replacement = std::make_shared<ParlayBundle>(capacity_);
    for (size_t index = 0; index < keys.size(); ++index) {
      if (!replacement->insert(keys[index], static_cast<uint32_t>(index)))
        throw std::logic_error("duplicate key in COW directory bulk load");
    }
    std::atomic_store_explicit(&bundle_, std::move(replacement),
                               std::memory_order_release);
  }

  std::optional<uint32_t> find(int64_t key) const {
    const auto snapshot =
        std::atomic_load_explicit(&bundle_, std::memory_order_acquire);
    return snapshot->find(key);
  }

  bool insert(int64_t key, uint32_t locator) {
    return rebuild(key, locator, Mutation::kInsert);
  }

  bool update(int64_t key, uint32_t locator) {
    return rebuild(key, locator, Mutation::kUpdate);
  }

  bool erase(int64_t key) { return rebuild(key, 0, Mutation::kErase); }

private:
  enum class Mutation { kInsert, kUpdate, kErase };

  bool rebuild(int64_t key, uint32_t locator, Mutation mutation) {
    std::lock_guard lock(writer_mutex_);
    const auto current =
        std::atomic_load_explicit(&bundle_, std::memory_order_acquire);
    const bool found = current->find(key).has_value();
    if ((mutation == Mutation::kInsert && found) ||
        (mutation != Mutation::kInsert && !found)) {
      return false;
    }
    auto replacement = std::make_shared<ParlayBundle>(capacity_);
    current->map.for_each([&](const auto &entry) {
      const auto [existing_key, existing_locator] = entry.get_entry();
      if (existing_key == key) {
        if (mutation == Mutation::kUpdate)
          replacement->insert(existing_key, locator);
        return;
      }
      replacement->insert(existing_key, existing_locator);
    });
    if (mutation == Mutation::kInsert)
      replacement->insert(key, locator);
    std::atomic_store_explicit(&bundle_, std::move(replacement),
                               std::memory_order_release);
    return true;
  }

  size_t capacity_;
  mutable std::mutex writer_mutex_;
  std::shared_ptr<ParlayBundle> bundle_;
};

class AtomicLinearDirectory {
public:
  explicit AtomicLinearDirectory(size_t minimum_capacity)
      : capacity_(next_power_of_two(minimum_capacity)), mask_(capacity_ - 1),
        buckets_(std::make_unique<Bucket[]>(capacity_)) {}

  void bulk_load(const std::vector<int64_t> &keys) {
    for (size_t index = 0; index < keys.size(); ++index) {
      if (!insert(keys[index], static_cast<uint32_t>(index)))
        throw std::logic_error("duplicate key in atomic directory bulk load");
    }
  }

  std::optional<uint32_t> find(int64_t key) const noexcept {
    size_t position = hasher_(key) & mask_;
    for (size_t probes = 0; probes < capacity_; ++probes) {
      const Bucket &bucket = buckets_[position];
      const uint8_t state = bucket.state.load(std::memory_order_acquire);
      if (state == kEmpty)
        return std::nullopt;
      if (state == kOccupied &&
          bucket.key.load(std::memory_order_relaxed) == key) {
        return bucket.locator.load(std::memory_order_relaxed);
      }
      position = (position + 1) & mask_;
    }
    return std::nullopt;
  }

  bool insert(int64_t key, uint32_t locator) {
    std::lock_guard lock(writer_mutex_);
    size_t position = hasher_(key) & mask_;
    size_t first_tombstone = capacity_;
    for (size_t probes = 0; probes < capacity_; ++probes) {
      Bucket &bucket = buckets_[position];
      const uint8_t state = bucket.state.load(std::memory_order_acquire);
      if (state == kOccupied &&
          bucket.key.load(std::memory_order_relaxed) == key) {
        return false;
      }
      if (state == kTombstone && first_tombstone == capacity_)
        first_tombstone = position;
      if (state == kEmpty) {
        publish(first_tombstone == capacity_ ? position : first_tombstone, key,
                locator);
        return true;
      }
      position = (position + 1) & mask_;
    }
    if (first_tombstone != capacity_) {
      publish(first_tombstone, key, locator);
      return true;
    }
    throw std::overflow_error("atomic linear directory is full");
  }

  bool update(int64_t key, uint32_t locator) noexcept {
    Bucket *bucket = find_bucket(key);
    if (bucket == nullptr)
      return false;
    bucket->locator.store(locator, std::memory_order_release);
    return true;
  }

  bool erase(int64_t key) {
    std::lock_guard lock(writer_mutex_);
    Bucket *bucket = find_bucket(key);
    if (bucket == nullptr)
      return false;
    bucket->state.store(kTombstone, std::memory_order_release);
    return true;
  }

private:
  static constexpr uint8_t kEmpty = 0;
  static constexpr uint8_t kOccupied = 1;
  static constexpr uint8_t kTombstone = 2;

  struct alignas(16) Bucket {
    std::atomic<int64_t> key{0};
    std::atomic<uint32_t> locator{0};
    std::atomic<uint8_t> state{kEmpty};
  };

  static size_t next_power_of_two(size_t value) {
    if (value < 2)
      throw std::invalid_argument("atomic directory capacity is too small");
    size_t result = 1;
    while (result < value) {
      if (result > std::numeric_limits<size_t>::max() / 2)
        throw std::overflow_error("atomic directory capacity overflow");
      result *= 2;
    }
    return result;
  }

  void publish(size_t position, int64_t key, uint32_t locator) noexcept {
    Bucket &bucket = buckets_[position];
    bucket.key.store(key, std::memory_order_relaxed);
    bucket.locator.store(locator, std::memory_order_relaxed);
    bucket.state.store(kOccupied, std::memory_order_release);
  }

  Bucket *find_bucket(int64_t key) const noexcept {
    size_t position = hasher_(key) & mask_;
    for (size_t probes = 0; probes < capacity_; ++probes) {
      Bucket &bucket = buckets_[position];
      const uint8_t state = bucket.state.load(std::memory_order_acquire);
      if (state == kEmpty)
        return nullptr;
      if (state == kOccupied &&
          bucket.key.load(std::memory_order_relaxed) == key) {
        return &bucket;
      }
      position = (position + 1) & mask_;
    }
    return nullptr;
  }

  size_t capacity_;
  size_t mask_;
  std::unique_ptr<Bucket[]> buckets_;
  mutable std::mutex writer_mutex_;
  SplitMixHash hasher_;
};

template <typename Directory> void prove_correctness(const char *name) {
  Directory directory(1024);
  const std::vector<int64_t> keys = {-17, -1, 0, 1, 91};
  directory.bulk_load(keys);
  for (size_t index = 0; index < keys.size(); ++index) {
    const auto found = directory.find(keys[index]);
    if (!found || *found != index)
      throw std::runtime_error(std::string(name) + " signed lookup failed");
  }
  if (directory.insert(-17, 99))
    throw std::runtime_error(std::string(name) + " duplicate insert succeeded");
  if (!directory.update(-1, 77) || directory.find(-1) != 77)
    throw std::runtime_error(std::string(name) + " locator update failed");
  if (!directory.erase(0) || directory.find(0).has_value())
    throw std::runtime_error(std::string(name) + " erase failed");
  if (!directory.insert(0, 88) || directory.find(0) != 88)
    throw std::runtime_error(std::string(name) + " tombstone reuse failed");

  std::atomic<size_t> inserted{0};
  std::vector<std::thread> workers;
  for (size_t worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&, worker] {
      if (directory.insert(-999, static_cast<uint32_t>(worker)))
        inserted.fetch_add(1, std::memory_order_relaxed);
    });
  }
  for (auto &worker : workers)
    worker.join();
  if (inserted.load(std::memory_order_relaxed) != 1 ||
      !directory.find(-999).has_value()) {
    throw std::runtime_error(std::string(name) +
                             " concurrent duplicate insert failed");
  }
}

template <typename Directory>
double benchmark_lookup(const char *variant, Directory &directory,
                        const std::vector<int64_t> &workload, size_t threads,
                        size_t operations, const std::vector<int> &cpus) {
  std::barrier start_barrier(static_cast<std::ptrdiff_t>(threads + 1));
  std::vector<std::thread> workers;
  workers.reserve(threads);
  std::atomic<uint64_t> checksum{0};
  for (size_t worker = 0; worker < threads; ++worker) {
    workers.emplace_back([&, worker] {
      se::ThreadAffinity::bind_current_thread_to_cpu(
          cpus[worker % cpus.size()]);
      start_barrier.arrive_and_wait();
      uint64_t local = 0;
      for (size_t operation = worker; operation < operations;
           operation += threads) {
        const auto found =
            directory.find(workload[operation % workload.size()]);
        if (!found)
          throw std::runtime_error("warm lookup missed a populated key");
        local += *found;
      }
      checksum.fetch_add(local, std::memory_order_relaxed);
    });
  }
  const auto started = std::chrono::steady_clock::now();
  start_barrier.arrive_and_wait();
  for (auto &worker : workers)
    worker.join();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  const double mops = static_cast<double>(operations) / seconds / 1.0e6;
  std::cout << variant << ",warm_lookup," << threads << ',' << std::fixed
            << std::setprecision(3) << mops << ',' << checksum.load() << '\n';
  return mops;
}

template <typename Directory>
double benchmark_update(const char *variant, Directory &directory,
                        const std::vector<int64_t> &keys, size_t threads,
                        size_t operations, const std::vector<int> &cpus) {
  std::barrier start_barrier(static_cast<std::ptrdiff_t>(threads + 1));
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (size_t worker = 0; worker < threads; ++worker) {
    workers.emplace_back([&, worker] {
      se::ThreadAffinity::bind_current_thread_to_cpu(
          cpus[worker % cpus.size()]);
      start_barrier.arrive_and_wait();
      for (size_t operation = worker; operation < operations;
           operation += threads) {
        const size_t index = operation % keys.size();
        if (!directory.update(keys[index], static_cast<uint32_t>(operation)))
          throw std::runtime_error("locator update missed a populated key");
      }
    });
  }
  const auto started = std::chrono::steady_clock::now();
  start_barrier.arrive_and_wait();
  for (auto &worker : workers)
    worker.join();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  const double mops = static_cast<double>(operations) / seconds / 1.0e6;
  std::cout << variant << ",locator_update," << threads << ',' << std::fixed
            << std::setprecision(3) << mops << ",0\n";
  return mops;
}

template <typename Directory>
double benchmark_cold_insert(const char *variant, size_t threads,
                             size_t operations, const std::vector<int> &cpus) {
  Directory directory(operations * 4);
  std::barrier start_barrier(static_cast<std::ptrdiff_t>(threads + 1));
  std::vector<std::thread> workers;
  workers.reserve(threads);
  std::atomic<size_t> inserted{0};
  for (size_t worker = 0; worker < threads; ++worker) {
    workers.emplace_back([&, worker] {
      se::ThreadAffinity::bind_current_thread_to_cpu(
          cpus[worker % cpus.size()]);
      start_barrier.arrive_and_wait();
      size_t local = 0;
      for (size_t operation = worker; operation < operations;
           operation += threads) {
        const int64_t key = -static_cast<int64_t>(operation) - 1;
        local += directory.insert(key, static_cast<uint32_t>(operation));
      }
      inserted.fetch_add(local, std::memory_order_relaxed);
    });
  }
  const auto started = std::chrono::steady_clock::now();
  start_barrier.arrive_and_wait();
  for (auto &worker : workers)
    worker.join();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  if (inserted.load(std::memory_order_relaxed) != operations)
    throw std::runtime_error("cold insert lost a unique key");
  const double mops = static_cast<double>(operations) / seconds / 1.0e6;
  std::cout << variant << ",cold_insert," << threads << ',' << std::fixed
            << std::setprecision(3) << mops << ',' << inserted.load() << '\n';
  return mops;
}

std::vector<int64_t> make_keys(size_t count) {
  std::vector<int64_t> keys(count);
  for (size_t index = 0; index < count; ++index)
    keys[index] = static_cast<int64_t>(index * 2 + 1);
  return keys;
}

std::vector<int64_t> make_workload(const std::vector<int64_t> &keys,
                                   size_t count) {
  std::vector<int64_t> workload(count);
  uint64_t state = 0x123456789abcdef0ULL;
  for (size_t index = 0; index < count; ++index) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    const size_t first = state % keys.size();
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    const size_t second = state % keys.size();
    workload[index] = keys[std::min(first, second)];
  }
  return workload;
}

} // namespace

int main(int argc, char **argv) {
  bool bind_affinity = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--bind-affinity")
      bind_affinity = true;
    else
      throw std::invalid_argument("unknown benchmark argument: " + argument);
  }
  if (!bind_affinity)
    throw std::invalid_argument("hash directory benchmark requires code "
                                "affinity; pass --bind-affinity");

  prove_correctness<CurrentParlayDirectory>("current");
  prove_correctness<SharedPtrCowDirectory>("shared_ptr_cow");
  prove_correctness<AtomicLinearDirectory>("atomic_linear");
  std::cout << "correctness=passed\n";
  std::cout << "variant,operation,threads,mops,checksum\n";

  constexpr size_t key_count = 500'000;
  constexpr size_t workload_size = 65'536;
  constexpr size_t lookup_operations = 4'194'304;
  constexpr size_t update_operations = 262'144;
  constexpr size_t insert_operations = 65'536;
  const auto cpus = se::ThreadAffinity::allowed_cpus();
  const auto keys = make_keys(key_count);
  const auto workload = make_workload(keys, workload_size);
  CurrentParlayDirectory current(key_count * 4);
  SharedPtrCowDirectory shared_ptr_cow(key_count * 4);
  AtomicLinearDirectory atomic_linear(key_count * 4);
  current.bulk_load(keys);
  shared_ptr_cow.bulk_load(keys);
  atomic_linear.bulk_load(keys);

  for (size_t threads : {size_t{1}, size_t{2}, size_t{4}, size_t{8}}) {
    benchmark_lookup("current", current, workload, threads, lookup_operations,
                     cpus);
    benchmark_lookup("shared_ptr_cow", shared_ptr_cow, workload, threads,
                     lookup_operations, cpus);
    benchmark_lookup("atomic_linear", atomic_linear, workload, threads,
                     lookup_operations, cpus);
    benchmark_update("current", current, keys, threads, update_operations,
                     cpus);
    benchmark_update("atomic_linear", atomic_linear, keys, threads,
                     update_operations, cpus);
    benchmark_cold_insert<CurrentParlayDirectory>("current", threads,
                                                  insert_operations, cpus);
    benchmark_cold_insert<AtomicLinearDirectory>("atomic_linear", threads,
                                                 insert_operations, cpus);
  }
}
