#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>

namespace se::storage_detail {

class SwapScheduler {
public:
  using Task = void (*)(void *);

  SwapScheduler() = default;
  ~SwapScheduler();

  SwapScheduler(const SwapScheduler &) = delete;
  SwapScheduler &operator=(const SwapScheduler &) = delete;

  void ensure_started(void *context, Task task);
  void wake() noexcept;
  void request_stop() noexcept;
  void stop_and_join() noexcept;
  void rethrow_if_failed() const;
  bool stop_requested() const noexcept;

private:
  void run() noexcept;

  mutable std::mutex scheduler_mutex_;
  mutable std::mutex error_mutex_;
  std::mutex wait_mutex_;
  std::condition_variable wake_cv_;
  std::thread thread_;
  std::exception_ptr error_;
  void *context_ = nullptr;
  Task task_ = nullptr;
  bool pending_ = false;
  std::atomic<bool> stop_requested_{true};
  std::atomic<bool> active_{false};
};

} // namespace se::storage_detail
