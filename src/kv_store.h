# pragma once
#include <cstdint>
#include <iostream>
#include "thread_pool.h" // Include the new thread pool
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <optional>
#include <future>
#include <string>

#define FID_SLOT_SIZE 16
#define FID_VALUE_SIZE 48
#define FID_SLOT_MAX 0xFFFF

inline uint32_t fid_to_slot(uint64_t fid) {
    return uint32_t((fid >> FID_VALUE_SIZE) & FID_SLOT_MAX);// 16 bit
}


// 定义K和V的类型，方便修改
using KeyType = uint64_t;
// 对外接口的Value类型仍为std::vector<float>
using ExtValueType = std::vector<float>;


struct RaggedTensors {
    ExtValueType values;
    std::vector<int> row_lengths;
};

// 针对不定长embedding的优化：使用Arena内存分配模式。
// Map中不再直接存储std::vector, 而是存储一个指向连续内存块(Arena)中数据的引用(偏移量和大小)。
// 这样可以显著减少内存碎片、降低内存分配开销，并提高缓存局部性。
// init_type: 0 = 全 0 初始化；1 = 随机初始化（以 key 为 seed，可复现）
class KVStore {
public:
    KVStore(std::unordered_map<uint32_t, int> feature_metas, int init_type = 0, int num_shards = 16);

    // 批量插入/更新 (Upsert)。
    // keys 和 values 的大小必须一致。
    void insert(const std::vector<KeyType>& keys, const std::vector<ExtValueType>& values);

    // 批量更新，只更新存在的键。
    // keys 和 values 的大小必须一致。
    void update(const std::vector<KeyType>& keys, const std::vector<ExtValueType>& values);

    // 批量只读查找，不插入新值。
    // 对于不存在的key，返回的vector中对应位置为 std::nullopt。
    RaggedTensors lookup(const std::vector<KeyType>& keys) const;

    // 批量查找或插入，如果key不存在，会插入默认值。
    RaggedTensors lookup_or_insert(const std::vector<KeyType>& keys);

    std::vector<int> get_embedding_dim(const std::vector<KeyType>& fids) const {
        std::vector<int> dims;
        dims.reserve(fids.size());
        // std::cout << "feature_metas_.size() = " << feature_metas_.size() << std::endl;
        // for (const auto& [slot, dim] : feature_metas_) {
        //     std::cout << "slot = " << slot << ", dim = " << dim << std::endl;
        // }
        for (const auto& fid : fids) {
            uint32_t slot = fid_to_slot(fid);
            // std::cout << "slot = " << slot << std::endl;
            auto it = feature_metas_.find(slot);
            if (it != feature_metas_.end()) {
                // std::cout << "dim = " << it->second << std::endl;
                dims.push_back(it->second);
            } 
            else throw std::invalid_argument("Invalid fids.");
        }
        return dims;
    }

    // void save(const std::string& model_path);
    // void load(const std::string& model_path);

private:
    // 内部使用的值引用，指向它在Arena中的数据
    struct ValueRef {
        int offset;
        int size;
    };

    static int get_shard_index(KeyType key, int num_shards) {
        // std::hash is a simple choice. For performance-critical code, a faster hash
        // like MurmurHash3 or XXH3 might be better.
        return std::hash<KeyType>{}(key) % num_shards;
    }

    // int default_embedding_dim_;
    std::vector<std::unordered_map<KeyType, ValueRef>> stores_;
    std::vector<std::vector<float>> data_arenas_; // 每个分片拥有一个Arena
    mutable std::vector<std::shared_mutex> mutexes_; // 读写锁，允许多个读并发
    // ExtValueType default_value_; // 未找到键时返回的默认值

    std::unordered_map<uint32_t, int> feature_metas_;  // 存放 slot, dim pair
    int init_type_;  // 0=全0, 1=随机

    // The thread pool for parallel execution
    mutable ThreadPool pool_;
};