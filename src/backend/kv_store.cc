#include <stdexcept>
#include <future>
#include <numeric>
#include <random>
#include "kv_store.h"


KVStore::KVStore(std::unordered_map<uint32_t, int> feature_metas,
                 int init_type,
                 int num_shards)
    : feature_metas_(feature_metas),
      init_type_(init_type),
      stores_(num_shards),
      data_arenas_(num_shards),
      mutexes_(num_shards),
      pool_(std::thread::hardware_concurrency()) {
    if (num_shards <= 0) {
        throw std::invalid_argument("Number of shards must be positive.");
    }
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


RaggedTensors KVStore::lookup(const std::vector<KeyType>& keys) const {
    RaggedTensors results;
    
    std::vector<int> dims = get_embedding_dim(keys);
    size_t total_size = std::accumulate(dims.begin(), dims.end(), size_t(0));
    results.values.resize(total_size);
    results.row_lengths.resize(keys.size(), 0);

    auto num_shards = stores_.size();
    std::vector<std::vector<size_t>> per_shard_indices(num_shards);
    for (size_t i = 0; i < keys.size(); ++i) {
        per_shard_indices[get_shard_index(keys[i], num_shards)].push_back(i);
    }

    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_shards; ++i) {
        if (!per_shard_indices[i].empty()) {
            futures.push_back(pool_.enqueue([this, shard_idx = i, &per_shard_indices, &keys, &results, &dims]() {

                std::shared_lock<std::shared_mutex> lock(mutexes_[shard_idx]);
                const auto& store = stores_[shard_idx];
                const auto& arena = data_arenas_[shard_idx];

                for (const auto original_index : per_shard_indices[shard_idx]) {
                    const auto& key = keys[original_index];
                    auto it = store.find(key);
                    if (it != store.end()) {
                        const auto& ref = it->second;
                        int offset = std::accumulate(dims.begin(), dims.begin() + original_index, 0);
                        std::copy(arena.begin() + ref.offset, arena.begin() + ref.offset + ref.size,
                                  results.values.begin() + offset);
                        results.row_lengths[original_index] = ref.size;
                    }
                }
            }));
        }
    }
    for (auto& fut : futures) { fut.get(); }
    
    return results;
}


RaggedTensors KVStore::lookup_or_insert(const std::vector<KeyType>& keys) {
    RaggedTensors results;

    std::vector<int> dims = get_embedding_dim(keys);
    size_t total_size = std::accumulate(dims.begin(), dims.end(), size_t(0));
    results.values.resize(total_size);
    results.row_lengths.resize(keys.size(), 0);

    auto num_shards = stores_.size();
    std::vector<std::vector<size_t>> per_shard_indices(num_shards);
    for (size_t i = 0; i < keys.size(); ++i) {
        per_shard_indices[get_shard_index(keys[i], num_shards)].push_back(i);
    }

    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_shards; ++i) {
        if (!per_shard_indices[i].empty()) {
            futures.push_back(pool_.enqueue([this, shard_idx = i, &per_shard_indices, &keys, &results, &dims]() {

                std::unique_lock<std::shared_mutex> lock(mutexes_[shard_idx]);
                auto& store = stores_[shard_idx];
                auto& arena = data_arenas_[shard_idx];

                for (const auto original_index : per_shard_indices[shard_idx]) {
                    const auto& key = keys[original_index];
                    auto it = store.find(key);
                    if (it != store.end()) {
                        const auto& ref = it->second;
                        int offset = std::accumulate(dims.begin(), dims.begin() + original_index, 0);
                        std::copy(arena.begin() + ref.offset, arena.begin() + ref.offset + ref.size,
                                  results.values.begin() + offset);
                        results.row_lengths[original_index] = ref.size;
                    } else {
                        auto slot = fid_to_slot(key);
                        auto it1 = feature_metas_.find(slot);
                        if (it1 == feature_metas_.end()) {
                            throw std::invalid_argument(std::to_string(key) + "-> " + std::to_string(slot) + ", Invalid slot.");
                        }
                        int dim = it1->second;
                        ExtValueType init_value(dim);
                        if (init_type_ == 0) {
                            std::fill(init_value.begin(), init_value.end(), 0.0f);
                        } else {
                            std::mt19937 rng(static_cast<uint32_t>(key ^ (key >> 32)));
                            std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
                            for (int i = 0; i < dim; ++i) init_value[i] = dist(rng);
                        }
                        int offset = arena.size();
                        arena.insert(arena.end(), init_value.begin(), init_value.end());
                        store[key] = ValueRef{offset, dim};

                        int offset_ = std::accumulate(dims.begin(), dims.begin() + original_index, 0);
                        std::copy(init_value.begin(), init_value.end(), results.values.begin() + offset_);
                        results.row_lengths[original_index] = dim;
                    }
                }
            }));
        }
    }
    for (auto& fut : futures) { fut.get(); }

    return results;
}
