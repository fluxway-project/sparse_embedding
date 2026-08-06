#include "sparse_embedding/src/storage/swap_scheduler.hpp"

#include <stdexcept>

namespace se::storage_detail {

SwapScheduler::~SwapScheduler() { stop_and_join(); }

void SwapScheduler::ensure_started(void *context, Task task) {
  if (context == nullptr || task == nullptr)
    throw std::invalid_argument("swap scheduler requires a task context");
  std::lock_guard scheduler_lock(scheduler_mutex_);
  if (active_.load(std::memory_order_acquire) &&
      !stop_requested_.load(std::memory_order_acquire))
    return;
  if (thread_.joinable())
    thread_.join();
  {
    std::lock_guard error_lock(error_mutex_);
    error_ = nullptr;
  }
  {
    std::lock_guard wait_lock(wait_mutex_);
    context_ = context;
    task_ = task;
    pending_ = false;
  }
  stop_requested_.store(false, std::memory_order_release);
  active_.store(true, std::memory_order_release);
  thread_ = std::thread([this] { run(); });
}

void SwapScheduler::wake() noexcept {
  {
    std::lock_guard lock(wait_mutex_);
    if (stop_requested_.load(std::memory_order_acquire))
      return;
    pending_ = true;
  }
  wake_cv_.notify_one();
}

void SwapScheduler::request_stop() noexcept {
  stop_requested_.store(true, std::memory_order_release);
  wake_cv_.notify_all();
}

void SwapScheduler::stop_and_join() noexcept {
  request_stop();
  std::lock_guard scheduler_lock(scheduler_mutex_);
  if (thread_.joinable())
    thread_.join();
}

void SwapScheduler::rethrow_if_failed() const {
  std::lock_guard lock(error_mutex_);
  if (error_)
    std::rethrow_exception(error_);
}

bool SwapScheduler::stop_requested() const noexcept {
  return stop_requested_.load(std::memory_order_acquire);
}

void SwapScheduler::run() noexcept {
  try {
    while (!stop_requested_.load(std::memory_order_acquire)) {
      Task task = nullptr;
      void *context = nullptr;
      {
        std::unique_lock lock(wait_mutex_);
        wake_cv_.wait(lock, [this] {
          return stop_requested_.load(std::memory_order_relaxed) || pending_;
        });
        if (stop_requested_.load(std::memory_order_relaxed))
          break;
        pending_ = false;
        task = task_;
        context = context_;
      }
      if (task == nullptr || context == nullptr)
        throw std::logic_error("swap scheduler task was not initialized");
      task(context);
    }
  } catch (...) {
    {
      std::lock_guard lock(error_mutex_);
      error_ = std::current_exception();
    }
    stop_requested_.store(true, std::memory_order_release);
    wake_cv_.notify_all();
  }
  active_.store(false, std::memory_order_release);
}

} // namespace se::storage_detail
