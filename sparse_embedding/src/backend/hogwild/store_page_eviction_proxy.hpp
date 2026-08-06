#pragma once

#ifndef SPARSE_EMBEDDING_STORE_PAGE_EVICTION_PROXY_HPP_
#define SPARSE_EMBEDDING_STORE_PAGE_EVICTION_PROXY_HPP_

#include "sparse_embedding/src/storage/page_storage.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace se {

class StorePageEvictionProxy {
public:
  StorePageEvictionProxy() = default;

  bool evaluate_page(size_t page_idx, uint64_t &score);
  void register_page(size_t page_idx, PageEvaluator evaluator);
  void unregister_page(size_t page_idx) noexcept;
  bool score_enabled() const noexcept;

private:
  std::mutex registry_mutex_;
  std::condition_variable callbacks_cv_;
  std::unordered_map<size_t, PageEvaluator> evaluators_;
  size_t active_callbacks_ = 0;
};

} // namespace se

#endif
