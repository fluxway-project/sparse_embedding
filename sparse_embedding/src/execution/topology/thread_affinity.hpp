#pragma once

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <numa.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace se {

class ThreadAffinity {
public:
  static std::vector<int> allowed_cpus() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
      throw std::runtime_error("thread affinity query failed");
    std::vector<int> cpus;
    const long configured = sysconf(_SC_NPROCESSORS_CONF);
    if (configured <= 0)
      throw std::runtime_error("thread affinity CPU count query failed");
    cpus.reserve(static_cast<size_t>(configured));
    for (int cpu = 0; cpu < configured; ++cpu) {
      if (CPU_ISSET(cpu, &allowed))
        cpus.push_back(cpu);
    }
    if (cpus.empty())
      throw std::runtime_error("thread affinity allowed CPU set is empty");
    return cpus;
  }

  static int cpu_for_worker(size_t worker) {
    const std::vector<int> &cpus = cached_allowed_cpus();
    const size_t physical_cores = cached_physical_core_count();
    if (physical_cores == 0)
      throw std::runtime_error("thread affinity physical core count is zero");
    if (physical_cores >= cpus.size())
      return cpus[worker % cpus.size()];
    if (worker < physical_cores)
      return cpus[worker];
    return cpus[physical_cores +
                (worker - physical_cores) % (cpus.size() - physical_cores)];
  }

  static void bind_current_thread_to_cpu(int cpu) {
    if (cpu < 0)
      throw std::invalid_argument("thread affinity CPU is negative");
    cpu_set_t selected;
    CPU_ZERO(&selected);
    CPU_SET(cpu, &selected);
    if (pthread_setaffinity_np(pthread_self(), sizeof(selected), &selected) !=
        0)
      throw std::runtime_error("thread affinity bind failed for CPU " +
                               std::to_string(cpu));
  }

  static int current_cpu() {
    const int cpu = sched_getcpu();
    if (cpu < 0)
      throw std::runtime_error("thread affinity current CPU query failed");
    return cpu;
  }

private:
  static const std::vector<int> &cached_allowed_cpus() {
    static const std::vector<int> cpus = allowed_cpus();
    return cpus;
  }

  static size_t cached_physical_core_count() {
    static const size_t count = physical_core_count(cached_allowed_cpus());
    return count;
  }

  static size_t physical_core_count(const std::vector<int> &cpus) {
    std::vector<std::pair<int, int>> cores;
    cores.reserve(cpus.size());
    for (int cpu : cpus) {
      const std::string base =
          "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
      const int package = read_topology_id(base + "physical_package_id");
      const int core = read_topology_id(base + "core_id");
      const std::pair<int, int> identity{package, core};
      if (std::find(cores.begin(), cores.end(), identity) == cores.end())
        cores.push_back(identity);
    }
    if (cores.empty() || cores.size() > cpus.size())
      throw std::runtime_error(
          "thread affinity physical core discovery failed");
    return cores.size();
  }

  static int read_topology_id(const std::string &path) {
    FILE *file = std::fopen(path.c_str(), "r");
    if (file == nullptr)
      throw std::runtime_error("thread affinity topology open failed");
    int value = -1;
    const int matched = std::fscanf(file, "%d", &value);
    const int close_status = std::fclose(file);
    if (matched != 1 || close_status != 0 || value < 0)
      throw std::runtime_error("thread affinity topology read failed");
    return value;
  }
};

class NumaTopology {
public:
  static bool available() noexcept { return numa_available() >= 0; }

  static size_t node_count() { return node_ids().size(); }

  static size_t physical_core_count() {
    size_t count = 0;
    for (const auto &cpus : cached_cpus_by_node())
      count += physical_core_prefix_count(cpus);
    if (count == 0)
      throw std::runtime_error("NUMA physical CPU topology is empty");
    return count;
  }

  static const std::vector<int> &node_ids() { return cached_node_ids(); }

  static int node_for_cpu(int cpu) {
    if (!available())
      return 0;
    const int node = numa_node_of_cpu(cpu);
    if (node < 0)
      throw std::runtime_error("NUMA CPU-to-node query failed");
    return node;
  }

  static int node_for_worker(size_t worker) {
    return node_for_cpu(cpu_for_worker(worker));
  }

  static int node_for_shard(size_t shard_id, size_t shard_count) {
    if (shard_count == 0)
      throw std::invalid_argument("NUMA shard count is zero");
    if (shard_id >= shard_count)
      throw std::out_of_range("NUMA shard id is out of range");
    const std::vector<int> &nodes = node_ids();
    return nodes[node_ordinal_for_shard(shard_id, shard_count)];
  }

  static size_t node_ordinal_for_shard(size_t shard_id, size_t shard_count) {
    if (shard_count == 0)
      throw std::invalid_argument("NUMA shard count is zero");
    if (shard_id >= shard_count)
      throw std::out_of_range("NUMA shard id is out of range");
    return (shard_id * node_count()) / shard_count;
  }

  static int cpu_for_worker(size_t worker) {
    const std::vector<int> &nodes = node_ids();
    const auto &cpus_by_node = cached_cpus_by_node();
    const size_t node_ordinal = worker % nodes.size();
    const size_t local_ordinal = worker / nodes.size();
    const std::vector<int> &node_cpus = cpus_by_node[node_ordinal];
    return node_cpus[local_ordinal % node_cpus.size()];
  }

  static int current_node() {
    return node_for_cpu(ThreadAffinity::current_cpu());
  }

private:
  static const std::vector<int> &cached_node_ids() {
    static const std::vector<int> nodes = [] {
      if (!available())
        return std::vector<int>{0};
      if (numa_all_nodes_ptr == nullptr)
        throw std::runtime_error("NUMA node mask is unavailable");
      const int maximum_node = numa_max_node();
      if (maximum_node < 0)
        throw std::runtime_error("NUMA maximum node query failed");
      std::vector<int> discovered;
      discovered.reserve(static_cast<size_t>(maximum_node) + 1);
      for (int node = 0; node <= maximum_node; ++node) {
        if (numa_bitmask_isbitset(numa_all_nodes_ptr,
                                  static_cast<unsigned>(node))) {
          discovered.push_back(node);
        }
      }
      if (discovered.empty())
        throw std::runtime_error("NUMA node mask is empty");
      return discovered;
    }();
    return nodes;
  }

  static const std::vector<std::vector<int>> &cached_cpus_by_node() {
    static const std::vector<std::vector<int>> cpus_by_node = [] {
      const std::vector<int> &nodes = node_ids();
      std::vector<std::vector<int>> discovered(nodes.size());
      for (int cpu : ThreadAffinity::allowed_cpus()) {
        const int node = node_for_cpu(cpu);
        const auto found = std::ranges::find(nodes, node);
        if (found == nodes.end()) {
          throw std::runtime_error(
              "allowed CPU belongs to an unknown NUMA node");
        }
        discovered[static_cast<size_t>(found - nodes.begin())].push_back(cpu);
      }
      for (auto &node_cpus : discovered) {
        if (node_cpus.empty())
          throw std::runtime_error("NUMA node has no allowed CPUs");
        node_cpus = physical_core_first(std::move(node_cpus));
      }
      return discovered;
    }();
    return cpus_by_node;
  }

  static size_t physical_core_prefix_count(const std::vector<int> &cpus) {
    std::vector<std::pair<int, int>> physical_cores;
    physical_cores.reserve(cpus.size());
    for (int cpu : cpus) {
      const std::string base =
          "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
      const std::pair<int, int> identity{
          read_topology_id(base + "physical_package_id"),
          read_topology_id(base + "core_id")};
      if (std::ranges::find(physical_cores, identity) != physical_cores.end())
        return physical_cores.size();
      physical_cores.push_back(identity);
    }
    return physical_cores.size();
  }

  static std::vector<int> physical_core_first(std::vector<int> cpus) {
    std::vector<std::pair<int, int>> physical_cores;
    std::vector<int> ordered;
    ordered.reserve(cpus.size());
    for (int cpu : cpus) {
      const std::string base =
          "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
      const std::pair<int, int> identity{
          read_topology_id(base + "physical_package_id"),
          read_topology_id(base + "core_id")};
      if (std::ranges::find(physical_cores, identity) == physical_cores.end()) {
        physical_cores.push_back(identity);
        ordered.push_back(cpu);
      }
    }
    for (int cpu : cpus) {
      const std::string base =
          "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
      const std::pair<int, int> identity{
          read_topology_id(base + "physical_package_id"),
          read_topology_id(base + "core_id")};
      const auto first =
          std::find_if(cpus.begin(), cpus.end(), [&](int candidate) {
            const std::string candidate_base = "/sys/devices/system/cpu/cpu" +
                                               std::to_string(candidate) +
                                               "/topology/";
            return std::pair<int, int>{
                       read_topology_id(candidate_base + "physical_package_id"),
                       read_topology_id(candidate_base + "core_id")} ==
                   identity;
          });
      if (first != cpus.end() && *first != cpu)
        ordered.push_back(cpu);
    }
    if (ordered.size() != cpus.size())
      throw std::runtime_error("NUMA CPU topology ordering failed");
    return ordered;
  }

  static int read_topology_id(const std::string &path) {
    FILE *file = std::fopen(path.c_str(), "r");
    if (file == nullptr)
      throw std::runtime_error("NUMA CPU topology open failed");
    int value = -1;
    const int matched = std::fscanf(file, "%d", &value);
    const int close_status = std::fclose(file);
    if (matched != 1 || close_status != 0 || value < 0)
      throw std::runtime_error("NUMA CPU topology read failed");
    return value;
  }
};

} // namespace se
