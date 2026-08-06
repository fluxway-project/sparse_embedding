#pragma once

#ifndef SPARSE_EMBEDDING_SHARD_OWNER_SCHEDULER_HPP_
#define SPARSE_EMBEDDING_SHARD_OWNER_SCHEDULER_HPP_

#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace se {

class ShardOwnerScheduler {
  struct Request {
    Request(void *request_context, void (*request_execute)(void *))
        : context(request_context), execute(request_execute) {}

    std::atomic<bool> done{false};
    void *context;
    void (*execute)(void *);
    std::exception_ptr error;
    Request *next = nullptr;
  };

public:
  ShardOwnerScheduler() = default;
  ShardOwnerScheduler(const ShardOwnerScheduler &) = delete;
  ShardOwnerScheduler &operator=(const ShardOwnerScheduler &) = delete;

  template <typename Function> void submit(Function &&function) {
    if (active_scheduler_ != nullptr)
      throw std::logic_error(
          "shard-owner callback cannot submit another shard request");
    using Callable = std::remove_reference_t<Function>;
    Request request(std::addressof(function), [](void *context) {
      std::invoke(*static_cast<Callable *>(context));
    });
    bool became_owner = false;
    {
      std::lock_guard lock(mutex_);
      if (tail_ == nullptr) {
        head_ = &request;
      } else {
        tail_->next = &request;
      }
      tail_ = &request;
      if (!owner_active_) {
        owner_active_ = true;
        became_owner = true;
      }
    }
    if (became_owner) {
      drain();
    } else {
      request.done.wait(false, std::memory_order_acquire);
    }
    if (request.error != nullptr)
      std::rethrow_exception(request.error);
  }

private:
  void drain() {
    active_scheduler_ = this;
    while (true) {
      Request *request;
      {
        std::lock_guard lock(mutex_);
        request = head_;
        if (request == nullptr) {
          owner_active_ = false;
          active_scheduler_ = nullptr;
          return;
        }
        head_ = request->next;
        if (head_ == nullptr)
          tail_ = nullptr;
      }
      try {
        request->execute(request->context);
      } catch (...) {
        request->error = std::current_exception();
      }
      request->done.store(true, std::memory_order_release);
      request->done.notify_all();
    }
  }

  inline static thread_local ShardOwnerScheduler *active_scheduler_ = nullptr;
  std::mutex mutex_;
  Request *head_ = nullptr;
  Request *tail_ = nullptr;
  bool owner_active_ = false;
};

} // namespace se

#endif
