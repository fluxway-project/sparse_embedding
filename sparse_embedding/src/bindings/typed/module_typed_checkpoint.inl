// Copyright 2026 The Fluxway Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0
// =============================================================================
// Implementation of CheckpointOpsImpl<ParamT, OptT> methods
// =============================================================================

#include <unordered_map>

#include "sparse_embedding/src/bindings/arrow_c_bridge.hpp"
#include "sparse_embedding/src/bindings/module_wrapper.hpp"
#include "sparse_embedding/src/checkpoint/checkpoint_stream_policy.hpp"

#ifdef SE_ENABLE_LANCE_C
#include "sparse_embedding/src/checkpoint/lance/native_fragment_writer.hpp"
#include "sparse_embedding/src/checkpoint/lance/native_restore.hpp"
#include "sparse_embedding/src/ops/partitioner_registry.hpp"
#endif

namespace se::py::detail {

#ifdef SE_ENABLE_LANCE_C
template <typename ParamT, typename OptT>
nb::dict CheckpointOpsImpl<ParamT, OptT>::write_native_fragment(
    EmbeddingStoreWrapper &self, const std::string &uri,
    uint32_t participant_index, const std::vector<uint32_t> &dimensions,
    const std::string &embedding_dtype, const std::string &optimizer_dtype,
    const std::string &score_policy, const std::string &key_dtype,
    const std::string &schema_digest, bool include_page_images,
    bool incremental_mode, const std::string &expected_baseline,
    const std::string &page_dataset_uri, uint64_t page_dataset_version,
    uint64_t expected_page_layout_generation) {
  se::NativeFragmentWriter writer;
  writer.begin_fragment(
      uri, participant_index, dimensions, embedding_dtype, optimizer_dtype,
      score_policy, key_dtype, schema_digest,
      self.get_store<ParamT, OptT>()->has_score_policy(), include_page_images,
      static_cast<uint32_t>(self.get_store<ParamT, OptT>()->entries_per_page()),
      page_dataset_uri, page_dataset_version);
  se::CheckpointCaptureCut cut;
  {
    nb::gil_scoped_release release;
    cut = self.get_store<ParamT, OptT>()->capture_checkpoint(
        incremental_mode, expected_baseline, include_page_images,
        expected_page_layout_generation,
        pro::make_proxy<se::FunctionFacade<void(
            std::span<const se::CheckpointMutation>, uint64_t)>>(
            [&](std::span<const se::CheckpointMutation> mutations, uint64_t) {
              if (!incremental_mode)
                return;
              writer.set_incremental_mutations(mutations);
            }),
        pro::make_proxy<se::FunctionFacade<bool(
            size_t, se::PageImageExport<ParamT, OptT> &&)>>(
            [&](size_t dim, se::PageImageExport<ParamT, OptT> &&image) {
              if (image.deleted_page) {
                if (include_page_images) {
                  writer.write_page_tombstone(
                      dim, image.physical_page_id, image.page_slot_stride_bytes,
                      static_cast<uint32_t>(
                          self.get_store<ParamT, OptT>()->entries_per_page()));
                }
                return true;
              }
              writer.write_page_image(
                  dim, image.physical_page_id, image.page_slot_stride_bytes,
                  sizeof(ParamT) * dim, image.slot_ordinals, image.keys,
                  image.seq_counts,
                  std::as_bytes(std::span<const ParamT>(image.values)),
                  image.scores,
                  std::as_bytes(std::span<const OptT>(image.optimizer_states)),
                  image.optimizer_states.empty()
                      ? 0
                      : image.optimizer_states.size() / image.keys.size(),
                  sizeof(OptT), image.steps);
              return true;
            }));
  }
  se::NativeFragmentArtifact artifact;
  {
    nb::gil_scoped_release release;
    try {
      artifact = writer.finalize();
    } catch (...) {
      self.get_store<ParamT, OptT>()->cancel_checkpoint_capture(
          cut.capture_id, cut.journal_identity);
      throw;
    }
  }
  nb::dict result;
  artifact.page_layout_generation = cut.page_layout_generation;
  result["entries_per_page"] = artifact.entries_per_page;
  result["capture_id"] = cut.capture_id;
  result["journal_identity"] = cut.journal_identity;
  result["mutation_watermark"] = cut.mutation_watermark;
  result["locator"] = artifact.locator;
  result["entries_version"] = artifact.entries_version;
  result["page_images_version"] = artifact.page_images_version;
  result["page_layout_generation"] = artifact.page_layout_generation;
  result["entry_count"] = artifact.entry_count;
  result["page_count"] = artifact.page_count;
  result["physical_bytes"] = artifact.physical_bytes;
  result["schema_digest"] = artifact.schema_digest;
  result["content_digest"] = artifact.content_digest;
  return result;
}

template <typename ParamT, typename OptT>
void CheckpointOpsImpl<ParamT, OptT>::restore_native_partition_entries(
    EmbeddingStoreWrapper &self, const std::vector<std::string> &uris,
    uint64_t version, uint64_t partition_count, uint64_t partition_id,
    const std::string &key_dtype, const std::string &assignment_name,
    const std::unordered_map<std::string, std::string> &assignment_parameters,
    const se::LanceStorageOptions &storage_options, bool load_optimizer_state,
    std::optional<uint32_t> source_participant_index) {
  const uint32_t native_partition_count =
      se::ops::checked_partition_count(partition_count);
  if (partition_id >= partition_count)
    throw std::invalid_argument("partition_id is out of range");

  auto partitioner = se::ops::PartitionerRegistry::instance().create(
      assignment_name, native_partition_count, assignment_parameters);

  nb::gil_scoped_release release;
  se::NativeRestore<ParamT, OptT>::entries(
      self.get_store<ParamT, OptT>(), uris, version, std::move(partitioner),
      static_cast<uint32_t>(partition_id), key_dtype, load_optimizer_state,
      storage_options, source_participant_index);
}

template <typename ParamT, typename OptT>
void CheckpointOpsImpl<ParamT, OptT>::restore_native_page_images(
    EmbeddingStoreWrapper &self, const std::vector<std::string> &uris,
    uint64_t version, uint32_t participant_index, const std::string &key_dtype,
    const se::LanceStorageOptions &storage_options, bool load_optimizer_state,
    uint64_t page_layout_generation) {
  const auto optimizer_dtype = self.optimizer_dtype();
  nb::gil_scoped_release release;
  se::NativeRestore<ParamT, OptT>::pages(
      self.get_store<ParamT, OptT>(), uris, version, participant_index,
      key_dtype, optimizer_dtype, storage_options, load_optimizer_state,
      page_layout_generation);
}
#endif // SE_ENABLE_LANCE_C

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
    std::vector<DstT> result;
    result.reserve(src.size());
    for (SrcT value : src)
      result.push_back(static_cast<DstT>(static_cast<float>(value)));
    return result;
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
              // PyArrow does not natively support bfloat16. If ParamT or OptT
              // is bfloat16, we MUST convert it to float32 to prevent PyArrow
              // from throwing schema parsing errors. When ParamT/OptT is float,
              // cast_vector_owned acts as a zero-cost abstraction and safely
              // steals the pointer.
              auto vals_f32 = cast_vector_owned<float>(std::move(vals));
              auto opt_f32 = cast_vector_owned<float>(std::move(opt_states));

              // For include_optimizer_state path, forward per-ID step counters
              // (empty for optimizers without step tracking, e.g. SGD). Empty
              // steps preserve the legacy 3-column layout.
              auto pair = include_optimizer_state
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
          "assign_opt_state requires a matching key tensor bit-width; "
          "received " +
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
          dimension, std::span<const KeyT>(keys, n),
          std::span<const OptT>(opt_cast),
          std::span<const uint32_t>(steps_ptr, steps_n));
    }
  });
}

} // namespace se::py::detail
