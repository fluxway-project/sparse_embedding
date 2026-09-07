#pragma once

#ifndef SPARSE_EMBEDDING_LANCE_STORAGE_OPTIONS_HPP_
#define SPARSE_EMBEDDING_LANCE_STORAGE_OPTIONS_HPP_

#include <span>
#include <string>
#include <utility>
#include <vector>

namespace se {

class LanceStorageOptions {
  std::vector<std::pair<std::string, std::string>> entries_;
  std::vector<const char *> encoded_options_;

public:
  LanceStorageOptions() = default;

  LanceStorageOptions(const LanceStorageOptions &other)
      : entries_(other.entries_) {
    init_encoded();
  }

  LanceStorageOptions &operator=(const LanceStorageOptions &other) {
    if (this != &other) {
      entries_ = other.entries_;
      encoded_options_.clear();
      init_encoded();
    }
    return *this;
  }

  LanceStorageOptions(LanceStorageOptions &&other) noexcept
      : entries_(std::move(other.entries_)) {
    init_encoded();
  }

  LanceStorageOptions &operator=(LanceStorageOptions &&other) noexcept {
    if (this != &other) {
      entries_ = std::move(other.entries_);
      encoded_options_.clear();
      init_encoded();
    }
    return *this;
  }

  explicit LanceStorageOptions(
      std::span<const std::pair<std::string, std::string>> options)
      : entries_(options.begin(), options.end()) {
    init_encoded();
  }

  explicit LanceStorageOptions(
      const std::vector<std::pair<std::string, std::string>> &options)
      : entries_(options) {
    init_encoded();
  }

  [[nodiscard]] const char *const *lance_options() const noexcept {
    return entries_.empty() ? nullptr : encoded_options_.data();
  }

  [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

  [[nodiscard]] const std::vector<std::pair<std::string, std::string>> &
  entries() const noexcept {
    return entries_;
  }

private:
  void init_encoded() {
    if (entries_.empty())
      return;
    encoded_options_.reserve(entries_.size() * 2 + 1);
    for (const auto &[key, val] : entries_) {
      encoded_options_.push_back(key.c_str());
      encoded_options_.push_back(val.c_str());
    }
    encoded_options_.push_back(nullptr);
  }
};

} // namespace se

#endif // SPARSE_EMBEDDING_LANCE_STORAGE_OPTIONS_HPP_
