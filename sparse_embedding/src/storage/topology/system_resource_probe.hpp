#pragma once

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <set>
#include <utility>

#include "sparse_embedding/src/execution/topology/thread_affinity.hpp"

namespace se::storage {

class SystemResourceProbe {
public:
  static size_t physical_core_count() {
    std::set<std::pair<int, int>> socket_core_pairs;
    for (int cpu = 0;; ++cpu) {
      std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                         "/topology/core_id";
      std::ifstream f_core(path);
      if (!f_core.is_open())
        break;
      int core_id = -1;
      f_core >> core_id;

      std::string path_sock = "/sys/devices/system/cpu/cpu" +
                              std::to_string(cpu) +
                              "/topology/physical_package_id";
      std::ifstream f_sock(path_sock);
      int sock_id = 0;
      if (f_sock.is_open()) {
        f_sock >> sock_id;
      }
      socket_core_pairs.insert({sock_id, core_id});
    }
    if (socket_core_pairs.empty())
      throw std::runtime_error(
          "system resource probe found no physical CPU topology");
    return socket_core_pairs.size();
  }

  static size_t hardware_concurrency() noexcept {
    return std::max(1u, std::thread::hardware_concurrency());
  }

  static size_t available_ram_bytes() {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open())
      throw std::runtime_error(
          "system resource probe failed to open /proc/meminfo");
    std::string line;
    size_t mem_available_kb = 0;
    bool found = false;
    while (std::getline(meminfo, line)) {
      if (line.rfind("MemAvailable:", 0) == 0) {
        std::istringstream iss(line.substr(13));
        if (iss >> mem_available_kb) {
          found = true;
          break;
        }
      }
    }
    if (!found || mem_available_kb == 0)
      throw std::runtime_error(
          "system resource probe failed to parse MemAvailable from "
          "/proc/meminfo");
    return mem_available_kb * 1024ULL;
  }

  static size_t numa_node_count() {
    return NumaTopology::available() ? NumaTopology::node_count() : 1;
  }

  static bool numa_available() noexcept { return NumaTopology::available(); }
};

} // namespace se::storage
