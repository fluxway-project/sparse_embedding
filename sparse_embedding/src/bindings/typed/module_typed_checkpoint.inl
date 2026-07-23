// Copyright 2026 The Fluxway Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0
// =============================================================================
// Implementation of CheckpointOpsImpl<ParamT, OptT> methods
// =============================================================================

#include "sparse_embedding/src/bindings/arrow_c_bridge.hpp"
#include "sparse_embedding/src/bindings/module_wrapper.hpp"
#ifdef SE_ENABLE_LANCE_CHECKPOINTER
#include "sparse_embedding/src/checkpoint/lance/lance_checkpoint.hpp"
#endif
#include "sparse_embedding/src/ops/partitioner.hpp"

namespace se::py::detail {

#ifdef SE_ENABLE_LANCE_CHECKPOINTER
template <typename ParamT, typename OptT>
void CheckpointOpsImpl<ParamT, OptT>::save_checkpoint(
    EmbeddingStoreWrapper &self, const std::string &uri,
    const std::string &mode, uint64_t min_score, bool include_optimizer_state,
    bool use_compression) {
  se::CheckpointExportOptions options;
  options.mode = (mode == "incremental")
                     ? se::CheckpointExportOptions::Mode::kIncremental
                     : se::CheckpointExportOptions::Mode::kFull;
  options.min_score = min_score;
  options.include_optimizer_state = include_optimizer_state;
  options.use_compression = use_compression;

  auto writer = self.checkpoint_writer_factory_;
  if (!writer.has_value())
    throw std::runtime_error("checkpoint backend not configured");
  bool has_score_policy = self.get_store<ParamT, OptT>()->has_score_policy();

  switch (self.key_type_enum_) {
    case KeyDType::kInt32:
      options.key_schema = "i";
      break;
    case KeyDType::kUInt32:
      options.key_schema = "I";
      break;
    case KeyDType::kInt64:
      options.key_schema = "l";
      break;
    case KeyDType::kUInt64:
      options.key_schema = "L";
      break;
  }

  auto dispatch_key = [&]<typename Fn>(Fn &&fn) {
    switch (self.key_type_enum_) {
      case KeyDType::kInt32:
        return fn.template operator()<int32_t>();
      case KeyDType::kUInt32:
        return fn.template operator()<uint32_t>();
      case KeyDType::kInt64:
        return fn.template operator()<int64_t>();
      case KeyDType::kUInt64:
        return fn.template operator()<uint64_t>();
    }
    throw std::invalid_argument("unsupported key_dtype");
  };

  dispatch_key([&]<typename KeyT>() {
    nb::gil_scoped_release release;
    auto dims = self.table_dims();
    for (size_t table_idx = 0; table_idx < dims.size(); ++table_idx) {
      size_t dim = dims[table_idx];

      if (include_optimizer_state) {
        // Use export_opt_state_batch to capture keys+vals+scores+opts+steps
        // in a single pass. Defer writer->begin until the first batch so we
        // know opt_state_dim in bytes.
        bool begun = false;
        self.get_store<ParamT, OptT>()->export_opt_state_batch(
            pro::make_proxy<se::FunctionFacade<bool(
                size_t, std::span<const KeyT>, std::span<const ParamT>,
                std::span<const uint64_t>, std::span<const OptT>,
                std::span<const uint32_t>)>>(
                [&](size_t /*dim_arg*/, std::span<const KeyT> keys,
                    std::span<const ParamT> vals,
                    std::span<const uint64_t> scores,
                    std::span<const OptT> opt_states,
                    std::span<const uint32_t> steps) -> bool {
                  if (!begun) {
                    size_t opt_dim_bytes =
                        keys.empty()
                            ? 0
                            : (opt_states.size() / keys.size()) * sizeof(OptT);
                    writer->begin(uri, static_cast<uint32_t>(dim),
                                  opt_dim_bytes, has_score_policy, options);
                    begun = true;
                  }
                  if (options.mode ==
                          se::CheckpointExportOptions::Mode::kIncremental &&
                      has_score_policy) {
                    std::vector<KeyT> fkeys;
                    std::vector<ParamT> fvals;
                    std::vector<uint64_t> fscores;
                    std::vector<OptT> fopts;
                    std::vector<uint32_t> fsteps;
                    size_t opt_per_key =
                        keys.empty() ? 0 : opt_states.size() / keys.size();
                    for (size_t i = 0; i < keys.size(); ++i) {
                      if (scores[i] >= options.min_score) {
                        fkeys.push_back(keys[i]);
                        fvals.insert(fvals.end(), vals.data() + i * dim,
                                     vals.data() + (i + 1) * dim);
                        if (!scores.empty()) fscores.push_back(scores[i]);
                        fopts.insert(fopts.end(),
                                     opt_states.data() + i * opt_per_key,
                                     opt_states.data() + (i + 1) * opt_per_key);
                        if (!steps.empty()) fsteps.push_back(steps[i]);
                      }
                    }
                    if (!fkeys.empty()) {
                      writer->write_batch(
                          fkeys.size(), fkeys.data(),
                          fkeys.size() * sizeof(KeyT), fvals.data(),
                          fvals.size() * sizeof(ParamT), fscores, fopts.data(),
                          fopts.size() * sizeof(OptT), fsteps);
                    }
                  } else {
                    writer->write_batch(
                        keys.size(), keys.data(), keys.size() * sizeof(KeyT),
                        vals.data(), vals.size() * sizeof(ParamT), scores,
                        opt_states.data(), opt_states.size() * sizeof(OptT),
                        steps);
                  }
                  return true;
                }),
            options.batch_size);
        if (!begun) {
          // Empty table — still need begin+commit.
          writer->begin(uri, static_cast<uint32_t>(dim), 0, has_score_policy,
                        options);
        }
      } else {
        writer->begin(uri, static_cast<uint32_t>(dim), 0, has_score_policy,
                      options);
        self.get_store<ParamT, OptT>()->export_batch(
            pro::make_proxy<se::FunctionFacade<bool(
                std::size_t, std::span<const KeyT>, std::span<const ParamT>,
                std::span<const uint64_t>)>>(
                [&](std::size_t /*dim_arg*/, std::span<const KeyT> keys,
                    std::span<const ParamT> vals,
                    std::span<const uint64_t> scores) -> bool {
                  if (options.mode ==
                          se::CheckpointExportOptions::Mode::kIncremental &&
                      has_score_policy) {
                    std::vector<KeyT> fkeys;
                    std::vector<ParamT> fvals;
                    std::vector<uint64_t> fscores;
                    for (size_t i = 0; i < keys.size(); ++i) {
                      if (scores[i] >= options.min_score) {
                        fkeys.push_back(keys[i]);
                        fvals.insert(fvals.end(), vals.data() + i * dim,
                                     vals.data() + (i + 1) * dim);
                        if (!scores.empty()) fscores.push_back(scores[i]);
                      }
                    }
                    if (!fkeys.empty()) {
                      writer->write_batch(fkeys.size(), fkeys.data(),
                                          fkeys.size() * sizeof(KeyT),
                                          fvals.data(),
                                          fvals.size() * sizeof(ParamT),
                                          fscores, nullptr, 0, {});
                    }
                  } else {
                    writer->write_batch(keys.size(), keys.data(),
                                        keys.size() * sizeof(KeyT), vals.data(),
                                        vals.size() * sizeof(ParamT), scores,
                                        nullptr, 0, {});
                  }
                  return true;
                }),
            options.batch_size);
      }

      writer->commit();
    }
  });
}

template <typename ParamT, typename OptT>
void CheckpointOpsImpl<ParamT, OptT>::load_checkpoint(
    EmbeddingStoreWrapper &self, const std::string &uri, uint64_t version,
    bool clear, bool load_optimizer_state, nb::object filter_fn,
    uint32_t target_shard_id, size_t batch_size) {
  se::CheckpointLoadOptions options;
  options.version = version;
  options.clear_before_load = clear;
  options.load_optimizer_state = load_optimizer_state;
  options.batch_size = batch_size;

  auto reader = self.checkpoint_reader_factory_;
  if (!reader.has_value())
    throw std::runtime_error("checkpoint backend not configured");

  std::shared_ptr<pro::proxy<se::ops::PartitionerFacade>> cpp_partitioner =
      nullptr;
  if (!filter_fn.is_none() && nb::isinstance<PyPartitioner>(filter_fn)) {
    cpp_partitioner = nb::cast<PyPartitioner>(filter_fn).facade;
  }

  auto dispatch_key = [&]<typename Fn>(Fn &&fn) {
    switch (self.key_type_enum_) {
      case KeyDType::kInt32:
        return fn.template operator()<int32_t>();
      case KeyDType::kUInt32:
        return fn.template operator()<uint32_t>();
      case KeyDType::kInt64:
        return fn.template operator()<int64_t>();
      case KeyDType::kUInt64:
        return fn.template operator()<uint64_t>();
    }
    throw std::invalid_argument("unsupported key_dtype");
  };

  dispatch_key([&]<typename KeyT>() {
    nb::gil_scoped_release release;
    if (clear) self.get_store<ParamT, OptT>()->clear();

    bool has_score = self.get_store<ParamT, OptT>()->has_score_policy();
    auto dims = self.table_dims();

    for (size_t table_idx = 0; table_idx < dims.size(); ++table_idx) {
      size_t dim = dims[table_idx];
      reader->open(uri, static_cast<uint32_t>(dim), version,
                   static_cast<int64_t>(options.batch_size));

      std::vector<std::byte> keys_buf;
      std::vector<std::byte> embs;
      std::vector<uint64_t> scores;
      std::vector<std::byte> opt_states;
      std::vector<uint32_t> steps;
      std::unique_ptr<bool[]> cpp_mask;
      size_t cpp_mask_capacity = 0;

      while (reader->read_batch(options.batch_size, keys_buf, embs, scores,
                                opt_states, steps)) {
        size_t n = keys_buf.size() / sizeof(KeyT);
        KeyT *keys = reinterpret_cast<KeyT *>(keys_buf.data());
        size_t opt_dim =
            (n > 0 && !opt_states.empty()) ? opt_states.size() / n : 0;
        if (n == 0) continue;

        auto compact_batch = [&](auto should_keep) {
          size_t keep_idx = 0;
          for (size_t i = 0; i < n; ++i) {
            if (should_keep(i)) {
              if (keep_idx != i) {
                keys[keep_idx] = keys[i];
                std::memcpy(embs.data() + keep_idx * dim * sizeof(ParamT),
                            embs.data() + i * dim * sizeof(ParamT),
                            dim * sizeof(ParamT));
                if (!scores.empty()) scores[keep_idx] = scores[i];
                if (!opt_states.empty())
                  std::memcpy(opt_states.data() + keep_idx * opt_dim,
                              opt_states.data() + i * opt_dim, opt_dim);
                if (!steps.empty()) steps[keep_idx] = steps[i];
              }
              keep_idx++;
            }
          }
          keys_buf.resize(keep_idx * sizeof(KeyT));
          embs.resize(keep_idx * dim * sizeof(ParamT));
          if (!scores.empty()) scores.resize(keep_idx);
          if (!opt_states.empty()) opt_states.resize(keep_idx * opt_dim);
          if (!steps.empty()) steps.resize(keep_idx);
          n = keep_idx;
        };

        if (cpp_partitioner) {
          if (n > cpp_mask_capacity) {
            cpp_mask.reset(new bool[n]);
            cpp_mask_capacity = n;
          }
          (*cpp_partitioner)
              ->filter_span(std::span<const KeyT>(keys, n), target_shard_id,
                            std::span<bool>(cpp_mask.get(), n));
          compact_batch([&](size_t i) { return cpp_mask[i]; });
        } else if (!filter_fn.is_none()) {
          nb::gil_scoped_acquire acquire;
          // Note: keys_buf is a vector of bytes, we need to create a shared_ptr
          // of KeyT
          auto keys_sp = std::shared_ptr<KeyT[]>(keys, [](KeyT *) {});
          DLManagedTensor *in_dlm =
              se::py::wrap_shared_ptr_as_tensor<KeyT>(keys_sp, n);
          nb::object keys_arr = se::py::wrap_output_tensor(in_dlm);
          nb::object shard_ids_obj = filter_fn(keys_arr);
          keys_arr = nb::none();
          auto [dlm, cap] = se::py::extract_dlpack_tensor(shard_ids_obj);
          if (dlm->dl_tensor.ndim != 1 ||
              dlm->dl_tensor.shape[0] != static_cast<int64_t>(n))
            throw std::runtime_error(
                "filter_fn returned dlpack tensor of incorrect size");
          auto span = se::py::dlpack_to_span<uint32_t>(dlm);
          const uint32_t *shard_ids_data = span.data();
          if (keys_sp.use_count() > 1)
            throw std::runtime_error(
                "filter_fn retained a reference to the keys tensor, which "
                "violates zero-copy memory safety.");
          compact_batch(
              [&](size_t i) { return shard_ids_data[i] == target_shard_id; });
        }

        if (n == 0) continue;

        auto vals = std::span<const ParamT>(
            reinterpret_cast<const ParamT *>(embs.data()), n * dim);
        thread_local std::vector<int64_t> tls_row_lengths;
        tls_row_lengths.resize(n);
        fill_uniform_row_lengths(tls_row_lengths, static_cast<int64_t>(dim));
        self.get_store<ParamT, OptT>()->assign(std::span<const KeyT>(keys, n),
                                               vals, tls_row_lengths);
        if (has_score && !scores.empty())
          self.get_store<ParamT, OptT>()->assign_scores(
              std::span<const KeyT>(keys, n),
              std::span<const uint64_t>(scores));
        if (load_optimizer_state && !opt_states.empty()) {
          self.get_store<ParamT, OptT>()->assign_opt_states(
              dim, std::span<const KeyT>(keys, n),
              std::span<const OptT>(
                  reinterpret_cast<const OptT *>(opt_states.data()),
                  opt_states.size() / sizeof(OptT)),
              std::span<const uint32_t>(steps));
        }
      }
    }
  });
}
#endif  // SE_ENABLE_LANCE_CHECKPOINTER

template <typename ParamT, typename OptT>
void CheckpointOpsImpl<ParamT, OptT>::export_batches_dlpack(
    EmbeddingStoreWrapper &self, nb::callable visitor, size_t batch_size,
    bool include_optimizer_state) {
  auto dispatch_key = [&]<typename Fn>(Fn &&fn) {
    switch (self.key_type_enum_) {
    case KeyDType::kInt32:
      return fn.template operator()<int32_t>();
    case KeyDType::kUInt32:
      return fn.template operator()<uint32_t>();
    case KeyDType::kInt64:
      return fn.template operator()<int64_t>();
    case KeyDType::kUInt64:
      return fn.template operator()<uint64_t>();
    }
    throw std::invalid_argument("unsupported key_dtype");
  };

  dispatch_key([&]<typename KeyT>() {
    nb::gil_scoped_release release;
    if (!include_optimizer_state) {
      self.get_store<ParamT, OptT>()->export_batch(
          pro::make_proxy<se::FunctionFacade<bool(
              std::size_t, std::span<const KeyT>, std::span<const ParamT>,
              std::span<const uint64_t>)>>(
              [&](std::size_t dim, std::span<const KeyT> keys,
                  std::span<const ParamT> vals,
                  std::span<const uint64_t>) -> bool {
                nb::gil_scoped_acquire acquire;
                auto keys_dlp = se::py::span_to_dlpack_no_delete(keys);
                auto vals_dlp = se::py::span_to_dlpack_no_delete(vals);
                visitor(dim, se::py::wrap_output_tensor(keys_dlp),
                        se::py::wrap_output_tensor(vals_dlp));
                return true;
              }),
          batch_size);
    } else {
      self.get_store<ParamT, OptT>()->export_opt_state_batch(
          pro::make_proxy<se::FunctionFacade<bool(
              size_t, std::span<const KeyT>, std::span<const ParamT>,
              std::span<const uint64_t>, std::span<const OptT>,
              std::span<const uint32_t>)>>(
              [&](size_t dim, std::span<const KeyT> keys,
                  std::span<const ParamT> vals,
                  std::span<const uint64_t> /*scores*/,
                  std::span<const OptT> opt_states,
                  std::span<const uint32_t>) -> bool {
                nb::gil_scoped_acquire acquire;
                auto keys_dlp = se::py::span_to_dlpack_no_delete(keys);
                auto vals_dlp = se::py::span_to_dlpack_no_delete(vals);
                auto opt_dlp = se::py::span_to_dlpack_no_delete(opt_states);
                visitor(dim, se::py::wrap_output_tensor(keys_dlp),
                        se::py::wrap_output_tensor(vals_dlp),
                        se::py::wrap_output_tensor(opt_dlp));
                return true;
              }),
          batch_size);
    }
  });
}

// Abstraction for casting an owned vector to a target type.
template <typename DstT, typename SrcT>
inline std::vector<DstT> cast_vector_owned(std::vector<SrcT> &&src) {
  if constexpr (std::is_same_v<DstT, SrcT>) {
    return std::move(src);
  } else {
    return std::vector<DstT>(src.begin(), src.end());
  }
}

template <typename ParamT, typename OptT>
void CheckpointOpsImpl<ParamT, OptT>::export_batches_arrow(
    EmbeddingStoreWrapper &self, nb::callable visitor, size_t batch_size,
    bool include_optimizer_state) {
  auto dispatch_key = [&]<typename Fn>(Fn &&fn) {
    switch (self.key_type_enum_) {
      case KeyDType::kInt32:
        return fn.template operator()<int32_t>();
      case KeyDType::kUInt32:
        return fn.template operator()<uint32_t>();
      case KeyDType::kInt64:
        return fn.template operator()<int64_t>();
      case KeyDType::kUInt64:
        return fn.template operator()<uint64_t>();
    }
    throw std::invalid_argument("unsupported key_dtype");
  };

  dispatch_key([&]<typename KeyT>() {
    nb::gil_scoped_release release;
    self.get_store<ParamT, OptT>()->export_opt_state_batch_owned(
        pro::make_proxy<se::FunctionFacade<bool(
            size_t, std::vector<KeyT> &&, std::vector<ParamT> &&,
            std::vector<uint64_t> &&, std::vector<OptT> &&,
            std::vector<uint32_t> &&)>>(
            [&, include_optimizer_state](
                size_t dim, std::vector<KeyT> &&keys,
                std::vector<ParamT> &&vals, std::vector<uint64_t> && /*scores*/,
                std::vector<OptT> &&opt_states,
                std::vector<uint32_t> &&steps) -> bool {
              // Escape Hatch for Arrow Schema compatibility:
              // PyArrow does not natively support bfloat16. If ParamT or OptT is bfloat16,
              // we MUST convert it to float32 to prevent PyArrow from throwing schema parsing errors.
              // When ParamT/OptT is float, cast_vector_owned acts as a zero-cost abstraction 
              // and safely steals the pointer.
              auto vals_f32 = cast_vector_owned<float>(std::move(vals));
              auto opt_f32 = cast_vector_owned<float>(std::move(opt_states));

              // For include_optimizer_state path, forward per-ID step counters
              // (empty for optimizers without step tracking, e.g. SGD). Empty
              // steps preserve the legacy 3-column layout.
              auto pair =
                  include_optimizer_state
                      ? se::arrow::build_opt_state_batch(
                            dim, std::move(keys), std::move(vals_f32),
                            std::move(opt_f32), std::move(steps))
                      : se::arrow::build_embedding_batch(
                            dim, std::move(keys), std::move(vals_f32));
              auto array_addr = reinterpret_cast<uintptr_t>(&pair.array);
              auto schema_addr = reinterpret_cast<uintptr_t>(&pair.schema);
              nb::gil_scoped_acquire acquire;
              visitor(array_addr, schema_addr);
              return true;
            }),
        batch_size);
  });
}

template <typename ParamT, typename OptT>
void CheckpointOpsImpl<ParamT, OptT>::assign_opt_state(
    EmbeddingStoreWrapper &self, nb::object keys_obj, nb::object opt_states_obj,
    size_t dimension, nb::object steps_obj) {
  auto dispatch_key = [&]<typename Fn>(Fn &&fn) {
    switch (self.key_type_enum_) {
    case KeyDType::kInt32:
      return fn.template operator()<int32_t>();
    case KeyDType::kUInt32:
      return fn.template operator()<uint32_t>();
    case KeyDType::kInt64:
      return fn.template operator()<int64_t>();
    case KeyDType::kUInt64:
      return fn.template operator()<uint64_t>();
    }
    throw std::invalid_argument("unsupported key_dtype");
  };

  dispatch_key([&]<typename KeyT>() {
    auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys_obj);
    auto [dlm_opt, cap_opt] = se::py::extract_dlpack_tensor(opt_states_obj);

    if (dlm_keys->dl_tensor.dtype.bits != sizeof(KeyT) * 8)
      throw std::invalid_argument(
          "assign_opt_state requires a matching key tensor bit-width; received " +
          std::to_string(dlm_keys->dl_tensor.dtype.bits) + "-bit input");
    
    const KeyT *keys = static_cast<const KeyT *>(dlm_keys->dl_tensor.data);
    size_t n = dlm_keys->dl_tensor.shape[0];
    const float *opt_flat = static_cast<const float *>(dlm_opt->dl_tensor.data);
    size_t total_opt = static_cast<size_t>(dlm_opt->dl_tensor.shape[0]);

    // Optional per-ID step tensor (uint32). Absent for SGD-like optimizers.
    const uint32_t *steps_ptr = nullptr;
    size_t steps_n = 0;
    std::optional<std::pair<DLManagedTensor *, nb::object>> steps_holder;
    if (!steps_obj.is_none()) {
      auto [dlm_steps, cap_steps] = se::py::extract_dlpack_tensor(steps_obj);
      if (dlm_steps->dl_tensor.dtype.bits != 32)
        throw std::invalid_argument(
            "assign_opt_state requires a 32-bit step tensor; received " +
            std::to_string(dlm_steps->dl_tensor.dtype.bits) + "-bit input");
      steps_holder.emplace(dlm_steps, cap_steps);
      steps_ptr = static_cast<const uint32_t *>(dlm_steps->dl_tensor.data);
      steps_n = static_cast<size_t>(dlm_steps->dl_tensor.shape[0]);
    }

    nb::gil_scoped_release release;
    if constexpr (std::is_same_v<OptT, float>) {
      self.get_store<ParamT, OptT>()->assign_opt_states(
          dimension, std::span<const KeyT>(keys, n),
          std::span<const OptT>(opt_flat, total_opt),
          std::span<const uint32_t>(steps_ptr, steps_n));
    } else {
      std::vector<OptT> opt_cast;
      opt_cast.reserve(total_opt);
      for (size_t i = 0; i < total_opt; ++i)
        opt_cast.push_back(static_cast<OptT>(opt_flat[i]));
      self.get_store<ParamT, OptT>()->assign_opt_states(
          dimension, std::span<const KeyT>(keys, n), std::span<const OptT>(opt_cast),
          std::span<const uint32_t>(steps_ptr, steps_n));
    }
  });
}

}  // namespace se::py::detail
