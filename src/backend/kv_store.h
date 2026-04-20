#pragma once
#include "thread_pool.h"
#include <cstdint>
#include <future>
#include <iostream>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define FID_SLOT_SIZE 16
#define FID_VALUE_SIZE 48
#define FID_SLOT_MAX 0xFFFF

inline uint32_t fid_to_slot(uint64_t fid) {
  return uint32_t((fid >> FID_VALUE_SIZE) & FID_SLOT_MAX); // 16 bit
}

using KeyType = uint64_t;
using ExtValueType = std::vector<float>;

struct RaggedTensors {
  ExtValueType values;
  std::vector<int> row_lengths;
};

// Memory Allocation Strategy (Arena):
// Instead of storing std::vector directly in the map, store a reference
// (offset, size) to data in a continuous chunk (Arena). This minimizes memory
// fragmentation and improves cache locality. init_type: 0 = zeros; 1 = random
// (seeded by key, reproducible)
class KVStore {
public:
  KVStore(std::unordered_map<uint32_t, int> feature_metas, int init_type = 0,
          int num_shards = 16);

  // Batch insert or update (upsert).
  // Keys and values dimensions must match.
  void insert(const std::vector<KeyType> &keys,
              const std::vector<ExtValueType> &values);

  // Batch update. Unmatched keys are ignored.
  // Keys and values dimensions must match.
  void update(const std::vector<KeyType> &keys,
              const std::vector<ExtValueType> &values);

  // Batch read-only lookup.
  // Missing keys are returned as default values.
  RaggedTensors lookup(const std::vector<KeyType> &keys) const;

  // Batch lookup or insert.
  // Missing keys are inserted with default values.
  RaggedTensors lookup_or_insert(const std::vector<KeyType> &keys);

  std::vector<int> get_embedding_dim(const std::vector<KeyType> &fids) const {
    std::vector<int> dims;
    dims.reserve(fids.size());
    for (const auto &fid : fids) {
      uint32_t slot = fid_to_slot(fid);
      auto it = feature_metas_.find(slot);
      if (it != feature_metas_.end()) {
        dims.push_back(it->second);
      } else
        throw std::invalid_argument("Invalid fids.");
    }
    return dims;
  }

private:
  // Reference to data in the Arena
  struct ValueRef {
    int offset;
    int size;
  };

  static int get_shard_index(KeyType key, int num_shards) {
    return std::hash<KeyType>{}(key) % num_shards;
  }

  std::vector<std::unordered_map<KeyType, ValueRef>> stores_;
  std::vector<std::vector<float>> data_arenas_; // Arena for each shard
  mutable std::vector<std::shared_mutex> mutexes_;

  std::unordered_map<uint32_t, int> feature_metas_; // Maps slot -> dim
  int init_type_;                                   // 0=zeros, 1=random

  mutable ThreadPool pool_;
};