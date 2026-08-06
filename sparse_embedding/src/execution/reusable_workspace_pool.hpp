#pragma once

#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace se::execution {

template <typename Workspace> class ReusableWorkspacePool {
  struct Node {
    Workspace workspace;
    std::unique_ptr<Node> next;
  };

public:
  class Lease {
  public:
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
    Lease(Lease &&other) noexcept
        : pool_(std::exchange(other.pool_, nullptr)),
          node_(std::move(other.node_)) {}
    Lease &operator=(Lease &&other) noexcept {
      if (this == &other)
        return *this;
      release();
      pool_ = std::exchange(other.pool_, nullptr);
      node_ = std::move(other.node_);
      return *this;
    }
    ~Lease() { release(); }
    Workspace &operator*() const {
      if (node_ == nullptr)
        throw std::logic_error("workspace lease is empty");
      return node_->workspace;
    }
    Workspace *operator->() const { return &operator*(); }
    Workspace *get() const noexcept {
      return node_ == nullptr ? nullptr : &node_->workspace;
    }

  private:
    friend class ReusableWorkspacePool;
    Lease(ReusableWorkspacePool &pool, std::unique_ptr<Node> node)
        : pool_(&pool), node_(std::move(node)) {}
    void release() noexcept {
      if (node_ == nullptr)
        return;
      pool_->release(std::move(node_));
      pool_ = nullptr;
    }
    ReusableWorkspacePool *pool_;
    std::unique_ptr<Node> node_;
  };

  ReusableWorkspacePool() = default;
  ReusableWorkspacePool(const ReusableWorkspacePool &) = delete;
  ReusableWorkspacePool &operator=(const ReusableWorkspacePool &) = delete;
  Lease acquire() {
    std::unique_ptr<Node> node;
    {
      std::lock_guard lock(mutex_);
      if (available_ != nullptr) {
        node = std::move(available_);
        available_ = std::move(node->next);
      }
    }
    if (node == nullptr)
      node = std::make_unique<Node>();
    return Lease(*this, std::move(node));
  }

private:
  void release(std::unique_ptr<Node> node) noexcept {
    std::lock_guard lock(mutex_);
    node->next = std::move(available_);
    available_ = std::move(node);
  }
  std::mutex mutex_;
  std::unique_ptr<Node> available_;
};

} // namespace se::execution
