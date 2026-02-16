#include <stdexcept>
#include <future>
#include <numeric>
#include "kv_store.h"



KVStore::KVStore(std::unordered_map<uint32_t, int>& feature_metas, 
                 std::unordered_map<uint32_t, ExtValueType>& default_values, 
                 int num_shards) 
    : feature_metas_(feature_metas),
      default_values_(default_values),
      stores_(num_shards),
      data_arenas_(num_shards),
      mutexes_(num_shards),
      pool_(std::thread::hardware_concurrency()) { // Initialize the thread pool
    if (num_shards <= 0) {
        throw std::invalid_argument("Number of shards must be positive.");
    }
    // default_value_ = ExtValueType(default_embedding_dim, 0.0f);
    for (auto& arena : data_arenas_) {
        arena.reserve(1024 * 1024); // Pre-allocate 4MB per shard arena
     }
}


void KVStore::insert(const std::vector<KeyType>& keys, const std::vector<ExtValueType>& values) {
    if (keys.size() != values.size()) {
        throw std::invalid_argument("Keys and values must have the same size.");
    }

    auto num_shards = stores_.size();
    std::vector<std::vector<size_t>> per_shard_indices(num_shards);
    for (size_t i = 0; i < keys.size(); ++i) {
        per_shard_indices[get_shard_index(keys[i], num_shards)].push_back(i);
    }

    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_shards; ++i) {
        if (!per_shard_indices[i].empty()) {
            // futures.push_back(std::async(std::launch::async, [this, shard_idx = i, &per_shard_indices, &keys, &values]() {
            // Use the thread pool to execute tasks
            futures.push_back(pool_.enqueue([this, shard_idx = i, &per_shard_indices, &keys, &values]() {

                std::unique_lock<std::shared_mutex> lock(mutexes_[shard_idx]);
                auto& store = stores_[shard_idx];
                auto& arena = data_arenas_[shard_idx];

                for (const auto original_index : per_shard_indices[shard_idx]) {
                    const auto& key = keys[original_index];
                    const auto& value = values[original_index];
                    
                    int offset = arena.size();
                    arena.insert(arena.end(), value.begin(), value.end());
                    store[key] = ValueRef{offset, (int)value.size()};
                }
            }));
        }
    }
    for (auto& fut : futures) { fut.get(); }
}


void KVStore::update(const std::vector<KeyType>& keys, const std::vector<ExtValueType>& values) {
    if (keys.size() != values.size()) {
        throw std::invalid_argument("Keys and values must have the same size.");
    }
    
    auto num_shards = stores_.size();
    std::vector<std::vector<size_t>> per_shard_indices(num_shards);
    for (size_t i = 0; i < keys.size(); ++i) {
        per_shard_indices[get_shard_index(keys[i], num_shards)].push_back(i);
    }

    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_shards; ++i) {
        if (!per_shard_indices[i].empty()) {
            // futures.push_back(std::async(std::launch::async, [this, shard_idx = i, &per_shard_indices, &keys, &values]() {
            futures.push_back(pool_.enqueue([this, shard_idx = i, &per_shard_indices, &keys, &values]() {

                std::unique_lock<std::shared_mutex> lock(mutexes_[shard_idx]);
                auto& store = stores_[shard_idx];
                auto& arena = data_arenas_[shard_idx];

                for (const auto original_index : per_shard_indices[shard_idx]) {
                    const auto& key = keys[original_index];
                    auto it = store.find(key);
                    if (it != store.end()) {
                        const auto& value = values[original_index];
                        int offset = arena.size();
                        arena.insert(arena.end(), value.begin(), value.end());
                        it->second = ValueRef{offset, (int)value.size()};
                    }
                }
            }));
        }
    }
    for (auto& fut : futures) { fut.get(); }
}


std::vector<std::optional<ExtValueType>> KVStore::lookup(const std::vector<KeyType>& keys) const {
    std::vector<std::optional<ExtValueType>> results(keys.size());
    
    auto num_shards = stores_.size();
    std::vector<std::vector<size_t>> per_shard_indices(num_shards);
    for (size_t i = 0; i < keys.size(); ++i) {
        per_shard_indices[get_shard_index(keys[i], num_shards)].push_back(i);
    }

    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_shards; ++i) {
        if (!per_shard_indices[i].empty()) {
            // futures.push_back(std::async(std::launch::async, [this, shard_idx = i, &per_shard_indices, &keys, &results]() {
            futures.push_back(pool_.enqueue([this, shard_idx = i, &per_shard_indices, &keys, &results]() {

                std::shared_lock<std::shared_mutex> lock(mutexes_[shard_idx]);
                const auto& store = stores_[shard_idx];
                const auto& arena = data_arenas_[shard_idx];

                for (const auto original_index : per_shard_indices[shard_idx]) {
                    const auto& key = keys[original_index];
                    auto it = store.find(key);
                    if (it != store.end()) {
                        const auto& ref = it->second;
                        results[original_index].emplace(arena.begin() + ref.offset, arena.begin() + ref.offset + ref.size);
                    }
                }
            }));
        }
    }
    for (auto& fut : futures) { fut.get(); }
    
    return results;
}


std::vector<ExtValueType> KVStore::lookup_or_insert(const std::vector<KeyType>& keys) {
    std::vector<ExtValueType> results(keys.size());

    auto num_shards = stores_.size();
    std::vector<std::vector<size_t>> per_shard_indices(num_shards);
    for (size_t i = 0; i < keys.size(); ++i) {
        per_shard_indices[get_shard_index(keys[i], num_shards)].push_back(i);
    }

    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_shards; ++i) {
        if (!per_shard_indices[i].empty()) {
            // futures.push_back(std::async(std::launch::async, [this, shard_idx = i, &per_shard_indices, &keys, &results]() {
            futures.push_back(pool_.enqueue([this, shard_idx = i, &per_shard_indices, &keys, &results]() {

                std::unique_lock<std::shared_mutex> lock(mutexes_[shard_idx]);
                auto& store = stores_[shard_idx];
                auto& arena = data_arenas_[shard_idx];

                for (const auto original_index : per_shard_indices[shard_idx]) {
                    const auto& key = keys[original_index];
                    auto it = store.find(key);
                    if (it != store.end()) {
                        const auto& ref = it->second;
                        results[original_index].assign(arena.begin() + ref.offset, arena.begin() + ref.offset + ref.size);
                    } else {
                        int offset = arena.size();
                        auto slot = fid_to_slot(key);
                        auto it1 = default_values_.find(slot);
                        if (it1 == default_values_.end()) {
                            throw std::invalid_argument(std::to_string(key) + "-> "+std::to_string(fid_to_slot(key))+", Invalid slot.");
                        }
                        auto& default_value_ = it1->second;
                        arena.insert(arena.end(), default_value_.begin(), default_value_.end());
                        store[key] = ValueRef{offset, (int)default_value_.size()};
                        results[original_index] = default_value_;
                    }
                }
            }));
        }
    }
    for (auto& fut : futures) { fut.get(); }

    return results;
}
