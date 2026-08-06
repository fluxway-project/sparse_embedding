#include "sparse_embedding/src/backend/hogwild/store_page_eviction_proxy.hpp"

#include <stdexcept>
#include <utility>

namespace se {
bool StorePageEvictionProxy::evaluate_page(size_t page_idx, uint64_t &score) {
  PageEvaluator evaluator;
  {
    std::lock_guard lock(registry_mutex_);
    auto it = evaluators_.find(page_idx);
    if (it == evaluators_.end())
      return false;
    evaluator = it->second;
    ++active_callbacks_;
  }
  struct CallbackExit {
    StorePageEvictionProxy &proxy;

    ~CallbackExit() {
      std::lock_guard lock(proxy.registry_mutex_);
      if (--proxy.active_callbacks_ == 0)
        proxy.callbacks_cv_.notify_all();
    }
  } callback_exit{*this};
  score = evaluator();
  return true;
}

void StorePageEvictionProxy::register_page(size_t page_idx,
                                           PageEvaluator evaluator) {
  if (!evaluator)
    throw std::invalid_argument("page evaluator is empty");
  std::lock_guard lock(registry_mutex_);
  evaluators_[page_idx] = std::move(evaluator);
}

void StorePageEvictionProxy::unregister_page(size_t page_idx) noexcept {
  std::unique_lock lock(registry_mutex_);
  evaluators_.erase(page_idx);
  callbacks_cv_.wait(lock, [this] { return active_callbacks_ == 0; });
}

bool StorePageEvictionProxy::score_enabled() const noexcept { return true; }

} // namespace se
