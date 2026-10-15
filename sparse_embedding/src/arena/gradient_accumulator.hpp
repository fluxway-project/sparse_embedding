#pragma once

#ifndef SPARSE_EMBEDDING_GRADIENT_ACCUMULATOR_HPP_
#define SPARSE_EMBEDDING_GRADIENT_ACCUMULATOR_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "sparse_embedding/src/arena/transient_arena.hpp"
#include "sparse_embedding/src/common/numeric_types.hpp"

namespace se {

enum class GradientDType : uint8_t { kFloat32, kFloat16, kBFloat16 };

template <typename ParamT> class GradientBatchView {
public:
  size_t size() const noexcept { return keys_.size(); }
  uint64_t key(size_t position) const noexcept { return keys_[position]; }
  uint64_t hash(size_t position) const noexcept { return hashes_[position]; }
  const ParamT *gradient(size_t position) const noexcept {
    return gradients_.data() + position * dim_;
  }
  const std::vector<uint64_t> &keys() const noexcept { return keys_; }
  const std::vector<uint64_t> &hashes() const noexcept { return hashes_; }
  bool contiguous() const noexcept { return true; }

  template <typename RawBatch>
  void assign(const RawBatch &raw, size_t dim, float scale) {
    dim_ = dim;
    keys_.resize(raw.size());
    hashes_.resize(raw.size());
    gradients_.resize(raw.size() * dim);
    for (size_t row = 0; row < raw.size(); ++row) {
      keys_[row] = raw.key(row);
      hashes_[row] = raw.hash(row);
      const auto *source = raw.gradient(row);
      for (size_t col = 0; col < dim; ++col) {
        gradients_[row * dim + col] =
            static_cast<ParamT>(static_cast<float>(source[col]) * scale);
      }
    }
  }

private:
  size_t dim_ = 0;
  std::vector<uint64_t> keys_;
  std::vector<uint64_t> hashes_;
  std::vector<ParamT> gradients_;
};

template <typename ParamT> class GradientAccumulator {
  using FloatArena = std::unique_ptr<TransientArena<float>>;
  using Float16Arena = std::unique_ptr<TransientArena<float16_t>>;
  using BFloat16Arena = std::unique_ptr<TransientArena<bfloat16_t>>;
  using ArenaVariant = std::variant<FloatArena, Float16Arena, BFloat16Arena>;

public:
  GradientAccumulator(size_t dim, PageStorage storage,
                      bool page_eviction_enabled, GradientDType dtype,
                      float scale)
      : dim_(dim), scale_(scale),
        arena_(
            make_arena(dim, std::move(storage), page_eviction_enabled, dtype)) {
    validate_scale(scale_);
  }

  void reserve(size_t expected_unique_keys) {
    std::visit([&](auto &arena) { arena->reserve(expected_unique_keys); },
               arena_);
  }

  size_t active_slots() const {
    return std::visit([](const auto &arena) { return arena->active_slots(); },
                      arena_);
  }

  template <typename GradT>
  void validate_input(std::span<const GradT> gradients) const {
    std::visit(
        [&](const auto &arena) {
          using ArenaT = std::remove_reference_t<decltype(*arena)>;
          using StoredT = typename ArenaT::value_type;
          for (const GradT gradient : gradients) {
            const float scaled = static_cast<float>(gradient) / scale_;
            if (!std::isfinite(scaled) ||
                std::abs(scaled) > dtype_max<StoredT>() ||
                std::abs(scaled) > dtype_max<ParamT>())
              throw std::overflow_error(
                  "gradient scale storage exceeds configured dtype");
          }
        },
        arena_);
  }

  template <typename GradT>
  void accumulate_hashed(uint64_t feature_id, uint64_t feature_hash,
                         const GradT *gradient) {
    std::visit(
        [&](auto &arena) {
          using ArenaT = std::remove_reference_t<decltype(*arena)>;
          using StoredT = typename ArenaT::value_type;
          if constexpr (std::is_same_v<GradT, StoredT>) {
            if (scale_ == 1.0f) {
              arena->accumulate_hashed(feature_id, feature_hash, gradient);
              return;
            }
          }
          thread_local std::vector<StoredT> scaled;
          scaled.resize(dim_);
          for (size_t col = 0; col < dim_; ++col) {
            scaled[col] = static_cast<StoredT>(
                static_cast<float>(gradient[col]) / scale_);
          }
          arena->accumulate_hashed(feature_id, feature_hash, scaled.data());
        },
        arena_);
  }

  template <typename Fn>
  void with_accumulated_batch(Fn &&fn, size_t normalization_steps) {
    if (normalization_steps == 0)
      throw std::invalid_argument("normalization_steps must be positive");
    std::visit(
        [&](auto &arena) {
          arena->with_accumulated_batch([&](const auto &raw) {
            using ArenaT = std::remove_reference_t<decltype(*arena)>;
            using StoredT = typename ArenaT::value_type;
            if constexpr (std::is_same_v<StoredT, ParamT>) {
              if (normalization_steps == 1 && scale_ == 1.0f) {
                std::forward<Fn>(fn)(raw);
                return;
              }
            }
            thread_local GradientBatchView<ParamT> batch;
            batch.assign(raw, dim_,
                         scale_ / static_cast<float>(normalization_steps));
            std::forward<Fn>(fn)(batch);
          });
        },
        arena_);
  }

  void clear() {
    std::visit([](auto &arena) { arena->clear(); }, arena_);
  }

private:
  template <typename T> static constexpr float dtype_max() {
    if constexpr (std::is_same_v<T, float>)
      return std::numeric_limits<float>::max();
    if constexpr (std::is_same_v<T, float16_t>)
      return 65504.0f;
    if constexpr (std::is_same_v<T, bfloat16_t>)
      return 3.38953139e38f;
    return std::numeric_limits<T>::max();
  }

  static void validate_scale(float scale) {
    if (!std::isfinite(scale))
      throw std::invalid_argument("gradient_scale must be finite");
    if (scale <= 0.0f)
      throw std::invalid_argument("gradient_scale must be greater than zero");
  }

  static ArenaVariant make_arena(size_t dim, PageStorage storage,
                                 bool page_eviction_enabled,
                                 GradientDType dtype) {
    switch (dtype) {
    case GradientDType::kFloat32:
      return ArenaVariant(std::in_place_type<FloatArena>,
                          std::make_unique<TransientArena<float>>(
                              dim, storage, page_eviction_enabled));
    case GradientDType::kFloat16:
      return ArenaVariant(std::in_place_type<Float16Arena>,
                          std::make_unique<TransientArena<float16_t>>(
                              dim, storage, page_eviction_enabled));
    case GradientDType::kBFloat16:
      return ArenaVariant(std::in_place_type<BFloat16Arena>,
                          std::make_unique<TransientArena<bfloat16_t>>(
                              dim, storage, page_eviction_enabled));
    }
    throw std::invalid_argument("unsupported gradient dtype");
  }

  size_t dim_;
  float scale_;
  ArenaVariant arena_;
};

} // namespace se

#endif // SPARSE_EMBEDDING_GRADIENT_ACCUMULATOR_HPP_
