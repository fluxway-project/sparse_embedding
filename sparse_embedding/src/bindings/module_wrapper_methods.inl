
namespace se::py::detail {

inline size_t EmbeddingStoreWrapper::total_size() {
  size_t size = 0;
  dispatch_types(
      [&]<typename P, typename O>() { size = get_store<P, O>()->size(); });
  return size;
}

inline size_t EmbeddingStoreWrapper::max_dim() const {
  return dims_.empty() ? 0 : std::ranges::max(dims_);
}

inline size_t EmbeddingStoreWrapper::num_slots() const { return dims_.size(); }

inline std::vector<size_t> EmbeddingStoreWrapper::table_dims() const {
  return dims_;
}

inline std::string EmbeddingStoreWrapper::value_dtype() const {
  switch (value_dtype_code_) {
  case ValueDType::kFloat32:
    return "float32";
  case ValueDType::kFloat16:
    return "float16";
  case ValueDType::kBFloat16:
    return "bfloat16";
  }
  throw std::invalid_argument("unsupported value dtype");
}

inline std::string EmbeddingStoreWrapper::optimizer_dtype() const {
  switch (optimizer_dtype_code_) {
  case ValueDType::kFloat32:
    return "float32";
  case ValueDType::kFloat16:
    return "float16";
  case ValueDType::kBFloat16:
    return "bfloat16";
  }
  throw std::invalid_argument("unsupported optimizer dtype");
}

inline bool EmbeddingStoreWrapper::optimizer_enabled() const {
  return optimizer_enabled_;
}

inline std::string EmbeddingStoreWrapper::score_policy() const {
  return score_policy_name_;
}

inline size_t EmbeddingStoreWrapper::internal_shard_count() {
  size_t count = 0;
  dispatch_types([&]<typename P, typename O>() {
    count = get_store<P, O>()->internal_shard_count();
  });
  return count;
}

inline nb::dict EmbeddingStoreWrapper::storage_stats() {
  if (storage_->consume_spill_warning() &&
      PyErr_WarnEx(PyExc_RuntimeWarning,
                   "SparseEmbeddingSpillWarning: embedding pages are using "
                   "file-backed storage",
                   1) < 0)
    throw nb::python_error();
  const auto stats = storage_->stats();
  nb::dict result;
  result["memory_bytes"] = stats.memory_bytes;
  result["evictable_memory_bytes"] = stats.evictable_memory_bytes;
  result["spill_bytes"] = stats.spill_bytes;
  result["memory_limit_bytes"] = stats.memory_limit_bytes;
  result["spill_limit_bytes"] = stats.spill_limit_bytes;
  result["spill_count"] = stats.spill_count;
  result["spill_directory"] = storage_->spill_directory();
  result["swap_out_count"] = stats.swap_out_count;
  result["eviction_selection_count"] = stats.eviction_selection_count;
  result["eviction_evaluated_pages"] = stats.eviction_evaluated_pages;
  result["eviction_selection_ns"] = stats.eviction_selection_ns;
  result["eviction_candidate_invalidated_count"] =
      stats.eviction_candidate_invalidated_count;
  result["retired_bytes"] = stats.retired_bytes;
  switch (stats.retirement_pressure_state) {
  case se::RetirementPressureState::kNormal:
    result["retirement_pressure_state"] = "normal";
    break;
  case se::RetirementPressureState::kBackpressure:
    result["retirement_pressure_state"] = "backpressure";
    break;
  case se::RetirementPressureState::kCircuitOpen:
    result["retirement_pressure_state"] = "circuit";
    break;
  }
  result["retirement_backpressure_count"] = stats.retirement_backpressure_count;
  result["oldest_unreclaimed_age_ns"] = stats.oldest_unreclaimed_age_ns;
  result["epoch_stall_count"] = stats.epoch_stall_count;
  return result;
}

inline void EmbeddingStoreWrapper::reserve(size_t keys_count, int64_t dim) {
  if (dim <= 0)
    throw std::invalid_argument("dim must be > 0");
  std::error_code error;
  dispatch_types([&]<typename P, typename O>() {
    error = get_store<P, O>()->reserve(keys_count, static_cast<size_t>(dim));
  });
  check_error(error);
}

inline void
EmbeddingStoreWrapper::reserve_per_shard(const std::vector<size_t> &counts,
                                         int64_t dim) {
  if (dim <= 0)
    throw std::invalid_argument("dim must be > 0");
  std::error_code error;
  dispatch_types([&]<typename P, typename O>() {
    error = get_store<P, O>()->reserve_per_shard(
        std::span<const size_t>(counts.data(), counts.size()),
        static_cast<size_t>(dim));
  });
  check_error(error);
}

inline nb::tuple EmbeddingStoreWrapper::lookup(nb::object keys, int64_t dim) {
  if (dim <= 0)
    throw std::invalid_argument("dim must be > 0");
  return dispatch_types([&]<typename P, typename O>() {
    return SyncOpsImpl<P, O>::lookup(*this, keys, dim);
  });
}

inline nb::tuple EmbeddingStoreWrapper::lookup_or_insert(nb::object keys,
                                                         int64_t dim) {
  if (dim <= 0)
    throw std::invalid_argument("dim must be > 0");
  return dispatch_types([&]<typename P, typename O>() {
    return SyncOpsImpl<P, O>::lookup_or_insert(*this, keys, dim);
  });
}

inline void EmbeddingStoreWrapper::assign(nb::object keys, nb::object values) {
  dispatch_types([&]<typename P, typename O>() {
    SyncOpsImpl<P, O>::assign(*this, keys, values);
  });
}

inline nb::object EmbeddingStoreWrapper::contains(nb::object keys) {
  return dispatch_types([&]<typename P, typename O>() {
    return SyncOpsImpl<P, O>::contains(*this, keys);
  });
}

inline void EmbeddingStoreWrapper::accumulate_gradients(nb::object keys,
                                                        nb::object grads) {
  if (!optimizer_enabled_)
    throw std::runtime_error("accumulate_gradients requires optimizer enabled");
  dispatch_types([&]<typename P, typename O>() {
    SyncOpsImpl<P, O>::accumulate_gradients(*this, keys, grads);
  });
}

inline void EmbeddingStoreWrapper::optimizer_step() {
  if (!optimizer_enabled_)
    throw std::runtime_error("optimizer_step requires optimizer enabled");
  dispatch_types([&]<typename P, typename O>() {
    SyncOpsImpl<P, O>::optimizer_step(*this);
  });
}

inline void EmbeddingStoreWrapper::erase(nb::object keys) {
  dispatch_types(
      [&]<typename P, typename O>() { SyncOpsImpl<P, O>::erase(*this, keys); });
}

inline void EmbeddingStoreWrapper::clear() {
  dispatch_types(
      [&]<typename P, typename O>() { SyncOpsImpl<P, O>::clear(*this); });
}

inline void EmbeddingStoreWrapper::assign_scores(nb::object keys,
                                                 nb::object scores) {
  dispatch_types([&]<typename P, typename O>() {
    SyncOpsImpl<P, O>::assign_scores(*this, keys, scores);
  });
}

inline size_t EmbeddingStoreWrapper::compact_page(size_t dim,
                                                  uint64_t page_id) {
  return dispatch_types([&]<typename P, typename O>() {
    return get_store<P, O>()->compact_page(dim, page_id);
  });
}

inline size_t EmbeddingStoreWrapper::promote_hot_keys(size_t dim,
                                                      uint64_t page_id,
                                                      size_t maximum_keys,
                                                      uint64_t minimum_score) {
  return dispatch_types([&]<typename P, typename O>() {
    return get_store<P, O>()->promote_hot_keys(dim, page_id, maximum_keys,
                                               minimum_score);
  });
}

inline size_t EmbeddingStoreWrapper::promoted_hot_page_size(size_t dim) {
  return dispatch_types([&]<typename P, typename O>() {
    return get_store<P, O>()->promoted_hot_page_size(dim);
  });
}

inline nb::list EmbeddingStoreWrapper::export_page_images() {
  nb::list result;
  dispatch_types([&]<typename P, typename O>() {
    get_store<P, O>()->export_page_images(
        pro::make_proxy<
            se::FunctionFacade<bool(size_t, se::PageImageExport<P, O> &&)>>(
            [&](size_t dim, se::PageImageExport<P, O> &&image) {
              if (image.embedding_dim != dim)
                throw std::logic_error(
                    "page image dimension does not match callback dimension");
              nb::dict page;
              page["embedding_dim"] = dim;
              page["physical_page_id"] = image.physical_page_id;
              page["page_slot_stride_bytes"] = image.page_slot_stride_bytes;
              page["slot_ordinals"] = std::move(image.slot_ordinals);
              page["keys"] = std::move(image.keys);
              page["seq_counts"] = std::move(image.seq_counts);
              page["scores"] = std::move(image.scores);
              page["steps"] = std::move(image.steps);
              std::vector<float> values;
              values.reserve(image.values.size());
              for (P value : image.values)
                values.push_back(static_cast<float>(static_cast<float>(value)));
              std::vector<float> optimizer_states;
              optimizer_states.reserve(image.optimizer_states.size());
              for (O value : image.optimizer_states)
                optimizer_states.push_back(static_cast<float>(static_cast<float>(value)));
              page["values"] = std::move(values);
              page["optimizer_states"] = std::move(optimizer_states);
              result.append(std::move(page));
              return true;
            }));
  });
  return result;
}

inline void EmbeddingStoreWrapper::import_page_images(nb::list pages) {
  dispatch_types([&]<typename P, typename O>() {
    std::vector<se::PageImageImport<P, O>> imports;
    imports.reserve(nb::len(pages));
    for (nb::handle handle : pages) {
      nb::dict page = nb::cast<nb::dict>(handle);
      se::PageImageExport<P, O> image{
          nb::cast<uint64_t>(page["physical_page_id"]),
          nb::cast<uint32_t>(page["embedding_dim"]),
          nb::cast<uint32_t>(page["page_slot_stride_bytes"]),
          nb::cast<std::vector<uint32_t>>(page["slot_ordinals"]),
          nb::cast<std::vector<uint64_t>>(page["keys"]),
          nb::cast<std::vector<uint64_t>>(page["seq_counts"]),
          {},
          nb::cast<std::vector<uint64_t>>(page["scores"]),
          {},
          nb::cast<std::vector<uint32_t>>(page["steps"])};
      const auto values = nb::cast<std::vector<float>>(page["values"]);
      image.values.reserve(values.size());
      for (float value : values)
        image.values.push_back(
            static_cast<P>(value));
      const auto optimizer_states =
          nb::cast<std::vector<float>>(page["optimizer_states"]);
      image.optimizer_states.reserve(optimizer_states.size());
      for (float value : optimizer_states)
        image.optimizer_states.push_back(
            static_cast<O>(value));
      imports.push_back(
          {nb::cast<size_t>(page["embedding_dim"]), std::move(image)});
    }
    get_store<P, O>()->import_page_images(imports);
  });
}

inline nb::object EmbeddingStoreWrapper::async_lookup(nb::object keys,
                                                      int64_t dim) {
  if (dim <= 0)
    throw std::invalid_argument("dim must be > 0");
  return dispatch_types([&]<typename P, typename O>() -> nb::object {
    return AsyncOpsImpl<P, O>::async_lookup(*this, keys, dim);
  });
}

inline nb::object EmbeddingStoreWrapper::async_assign(nb::object keys,
                                                      nb::object values) {
  return dispatch_types([&]<typename P, typename O>() -> nb::object {
    return AsyncOpsImpl<P, O>::async_assign(*this, keys, values);
  });
}

inline nb::object EmbeddingStoreWrapper::async_lookup_or_insert(nb::object keys,
                                                                int64_t dim) {
  if (dim <= 0)
    throw std::invalid_argument("dim must be > 0");
  return dispatch_types([&]<typename P, typename O>() -> nb::object {
    return AsyncOpsImpl<P, O>::async_lookup_or_insert(*this, keys, dim);
  });
}

inline nb::object
EmbeddingStoreWrapper::async_accumulate_gradients(nb::object keys,
                                                  nb::object grads) {
  if (!optimizer_enabled_)
    throw std::runtime_error(
        "async_accumulate_gradients requires optimizer enabled");
  return dispatch_types([&]<typename P, typename O>() -> nb::object {
    return AsyncOpsImpl<P, O>::async_accumulate_gradients(*this, keys, grads);
  });
}

inline nb::object EmbeddingStoreWrapper::async_optimizer_step() {
  if (!optimizer_enabled_)
    throw std::runtime_error("async_optimizer_step requires optimizer enabled");
  return dispatch_types([&]<typename P, typename O>() -> nb::object {
    return AsyncOpsImpl<P, O>::async_optimizer_step(*this);
  });
}

inline nb::object EmbeddingStoreWrapper::async_contains(nb::object keys) {
  return dispatch_types([&]<typename P, typename O>() -> nb::object {
    return AsyncOpsImpl<P, O>::async_contains(*this, keys);
  });
}

inline nb::object EmbeddingStoreWrapper::async_erase(nb::object keys) {
  return dispatch_types([&]<typename P, typename O>() -> nb::object {
    return AsyncOpsImpl<P, O>::async_erase(*this, keys);
  });
}

inline nb::object EmbeddingStoreWrapper::async_clear() {
  return dispatch_types([&]<typename P, typename O>() -> nb::object {
    return AsyncOpsImpl<P, O>::async_clear(*this);
  });
}

inline nb::object
EmbeddingStoreWrapper::async_assign_scores(nb::object keys, nb::object scores) {
  return dispatch_types([&]<typename P, typename O>() -> nb::object {
    return AsyncOpsImpl<P, O>::async_assign_scores(*this, keys, scores);
  });
}

inline void EmbeddingStoreWrapper::export_batches_dlpack(
    nb::callable visitor, size_t batch_size, bool include_optimizer_state) {
  dispatch_types([&]<typename P, typename O>() {
    CheckpointOpsImpl<P, O>::export_batches_dlpack(*this, visitor, batch_size,
                                                   include_optimizer_state);
  });
}

inline void EmbeddingStoreWrapper::export_batches_arrow(
    nb::callable visitor, size_t batch_size, bool include_optimizer_state) {
  dispatch_types([&]<typename P, typename O>() {
    CheckpointOpsImpl<P, O>::export_batches_arrow(*this, visitor, batch_size,
                                                  include_optimizer_state);
  });
}

inline void EmbeddingStoreWrapper::assign_opt_state(nb::object keys_obj,
                                                    nb::object opt_states_obj,
                                                    size_t dimension,
                                                    nb::object steps_obj) {
  dispatch_types([&]<typename P, typename O>() {
    CheckpointOpsImpl<P, O>::assign_opt_state(*this, keys_obj, opt_states_obj,
                                              dimension, steps_obj);
  });
}

} // namespace se::py::detail
