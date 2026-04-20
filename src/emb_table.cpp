#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

#include <folly/concurrency/ConcurrentHashMap.h>

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include "backend/dlpack_capsule_wrapper.hpp"

namespace nb = nanobind;

// TODO 多线程操作抽象成模板
// TODO 把模板展开分开在多个文件中——并行编译
// TODO hash value改成结构体{array + score}  score可以是访问次数或者时间戳

// TODO 要保证查找和插入不互相阻塞 (阻塞的原因是GIL)

namespace {

/// 调用 __dlpack__，取 dltensor capsule，不消费张量本体
std::pair<DLManagedTensor *, nb::object>
extract_dlpack_tensor(const nb::object &py_tensor) {
  if (!nb::hasattr(py_tensor, "__dlpack__")) {
    throw std::invalid_argument("object has no __dlpack__");
  }
  nb::object dlm_obj = py_tensor.attr("__dlpack__")();
  if (!PyCapsule_CheckExact(dlm_obj.ptr())) {
    throw std::invalid_argument("__dlpack__ must return a dltensor PyCapsule");
  }
  void *rawData = PyCapsule_GetPointer(dlm_obj.ptr(), "dltensor");
  if (!rawData) {
    PyErr_Clear();
    throw std::invalid_argument("invalid or consumed dltensor capsule");
  }
  return {static_cast<DLManagedTensor *>(rawData), std::move(dlm_obj)};
}

/// DLPack capsule 安全析构：未消费则调 deleter，已消费（renamed
/// "used_dltensor"）则跳过
inline void SafeDlpackCapsuleDestructor(PyObject *obj) {
  DLManagedTensor *dlm =
      static_cast<DLManagedTensor *>(PyCapsule_GetPointer(obj, "dltensor"));
  if (dlm == nullptr) {
    PyErr_Clear();
    return;
  }
  if (dlm->deleter) {
    dlm->deleter(dlm);
  }
}

} // namespace

/// 表内一行：仅保活 dltensor capsule + 缓存 DLManagedTensor*。
// todo 生命周期管理
struct EmbeddingRow {
  DLManagedTensor *dlm = nullptr;
  nb::object dlpack_owner;

  const float *data() const noexcept {
    if (!dlm) {
      return nullptr;
    }
    const DLTensor &t = dlm->dl_tensor;
    const auto *base = static_cast<const char *>(t.data) + t.byte_offset;
    return reinterpret_cast<const float *>(base);
  }

  float *mutable_data() noexcept {
    if (!dlm) {
      return nullptr;
    }
    DLTensor &t = dlm->dl_tensor;
    auto *base = static_cast<char *>(t.data) + t.byte_offset;
    return reinterpret_cast<float *>(base);
  }
};

/// 小批量不值得调度开销，直接走串行路径
static constexpr size_t kParallelThreshold = 2;

class EmbeddingTable {
  size_t embedding_dim_;
  // unique_ptr在堆上  改 std::array 展开
  folly::ConcurrentHashMap<int64_t, std::unique_ptr<EmbeddingRow>> table_;
  exec::static_thread_pool pool_;

public:
  explicit EmbeddingTable(
      size_t embedding_dim,
      size_t num_threads = std::max(1u, std::thread::hardware_concurrency()))
      : embedding_dim_(embedding_dim), pool_(num_threads) {}

  void insert_from_dlpack_object(int64_t key, const nb::object &py_tensor) {
    auto [dlm, owner] = extract_dlpack_tensor(py_tensor);
    auto row = std::make_unique<EmbeddingRow>(dlm, std::move(owner));
    table_.insert_or_assign(key, std::move(row));
  }

  void insert(int64_t key, const nb::object &vec) {
    if (nb::hasattr(vec, "__dlpack__")) {
      insert_from_dlpack_object(key, vec);
      return;
    }
    throw std::invalid_argument(
        "insert: vec must support __dlpack__ (e.g. JAX / NumPy array)");
  }

  /// 为单个 id 构建 DLManagedTensor*（纯 C++，无 GIL，可并发调用）
  DLManagedTensor *lookup_one(int64_t k) const {
    DLManagedTensor *dlm = new DLManagedTensor;
    auto it = table_.find(k);

    if (it != table_.cend() && it->second->dlm) {
      auto *ctx = new std::shared_ptr<EmbeddingRow>(it->second);
      dlm->dl_tensor = it->second->dlm->dl_tensor;
      dlm->manager_ctx = ctx;
      dlm->deleter = [](DLManagedTensor *self) {
        delete static_cast<std::shared_ptr<EmbeddingRow> *>(self->manager_ctx);
        delete self;
      };
    } else {
      struct ZeroBufCtx {
        std::unique_ptr<float[]> buf;
        int64_t shape_val;
      };
      auto *ctx = new ZeroBufCtx();
      ctx->buf = std::make_unique<float[]>(embedding_dim_);
      std::memset(ctx->buf.get(), 0, embedding_dim_ * sizeof(float));
      ctx->shape_val = static_cast<int64_t>(embedding_dim_);

      dlm->dl_tensor.data = ctx->buf.get();
      dlm->dl_tensor.device = {kDLCPU, 0};
      dlm->dl_tensor.ndim = 1;
      dlm->dl_tensor.dtype = {kDLFloat, 32, 1};
      dlm->dl_tensor.shape = &ctx->shape_val;
      dlm->dl_tensor.strides = nullptr;
      dlm->dl_tensor.byte_offset = 0;
      dlm->manager_ctx = ctx;
      dlm->deleter = [](DLManagedTensor *self) {
        delete static_cast<ZeroBufCtx *>(self->manager_ctx);
        delete self;
      };
    }
    return dlm;
  }

  nb::list
  batch_lookup(nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> ids) {
    size_t n = ids.shape(0);

    // 预分配结果槽（仅存指针，无 Python 对象）
    std::vector<DLManagedTensor *> dlm_vec(n, nullptr);
    const int32_t *ids_ptr = ids.data();

    // ── 阶段一：并行查表，释放 GIL ──────────────────────────────
    // folly::ConcurrentHashMap::find 并发读安全；
    // 每个 i 写独立槽，无数据竞争。
    {
      nb::gil_scoped_release release;

      if (n >= kParallelThreshold) {
        // bulk_chunked: 线程池直接调用 fn(begin, end)，无需包装层
        stdexec::sync_wait(stdexec::bulk_chunked(
            stdexec::schedule(pool_.get_scheduler()), stdexec::par, n,
            [&](size_t begin, size_t end) {
              for (size_t i = begin; i < end; ++i) {
                dlm_vec[i] = lookup_one(static_cast<int64_t>(ids_ptr[i]));
              }
            }));
      } else {
        for (size_t i = 0; i < n; ++i) {
          dlm_vec[i] = lookup_one(static_cast<int64_t>(ids_ptr[i]));
        }
      }
    }

    // ── 阶段二：串行构建 Python 对象，持有 GIL ──────────────────
    // TODO 返回聚合的tensor
    nb::list result;
    for (size_t i = 0; i < n; ++i) {
      DLManagedTensor *dlm = dlm_vec[i];
      DLDevice dev = dlm->dl_tensor.device;
      nb::object capsule = nb::steal<nb::object>(
          PyCapsule_New(dlm, "dltensor", SafeDlpackCapsuleDestructor));
      result.append(
          nb::cast(emb_dlpack::DLPackCapsuleWrapper(std::move(capsule), dev)));
    }
    return result;
  }

  // TODO 支持并行更新
  // TODO 用adam
  void batch_sgd_update(
      nb::ndarray<const int32_t, nb::shape<-1>, nb::device::cpu> ids,
      nb::ndarray<const float, nb::shape<-1, -1>, nb::device::cpu> grads,
      float lr) {
    size_t n = ids.shape(0);
    if (grads.shape(0) != n || grads.shape(1) != embedding_dim_) {
      throw std::invalid_argument(
          "grads shape must match (len(ids), embedding_dim)");
    }
    size_t grad_stride = grads.stride(0);

    for (size_t i = 0; i < n; ++i) {
      int64_t k = static_cast<int64_t>(ids(i));
      auto it = table_.find(k);
      if (it == table_.cend()) {
        continue;
      }

      float *w = it->second->mutable_data();
      if (!w) {
        continue;
      }
      const float *grad_row = grads.data() + i * grad_stride;
      for (size_t d = 0; d < embedding_dim_; ++d) {
        w[d] -= lr * grad_row[d];
      }
    }
  }

  nb::ndarray<nb::numpy, float, nb::shape<-1>> get(int64_t key) {
    float *out = new float[embedding_dim_];
    nb::capsule owner(
        out, [](void *p) noexcept { delete[] static_cast<float *>(p); });

    auto it = table_.find(key);
    if (it != table_.cend()) {
      const float *src = it->second->data();
      std::memcpy(out, src, embedding_dim_ * sizeof(float));
    } else {
      std::memset(out, 0, embedding_dim_ * sizeof(float));
    }

    return nb::ndarray<nb::numpy, float, nb::shape<-1>>(out, {embedding_dim_},
                                                        owner);
  }

  size_t size() const { return table_.size(); }
};

NB_MODULE(emb_table_ext, m) {
  nb::class_<emb_dlpack::DLPackCapsuleWrapper>(m, "DLPackCapsuleWrapper")
      .def("__dlpack__", &emb_dlpack::DLPackCapsuleWrapper::dlpack,
           nb::kw_only(), nb::arg("stream") = nb::none(),
           nb::arg("max_version") = nb::none(),
           nb::arg("dl_device") = nb::none(), nb::arg("copy") = nb::none())
      .def("__dlpack_device__",
           &emb_dlpack::DLPackCapsuleWrapper::dlpack_device);

  nb::class_<EmbeddingTable>(m, "EmbeddingTable")
      .def(nb::init<size_t, size_t>(), nb::arg("embedding_dim"),
           nb::arg("num_threads") =
               std::max(1u, std::thread::hardware_concurrency()))
      .def("insert", &EmbeddingTable::insert, nb::arg("key"), nb::arg("vec"))
      .def("batch_lookup", &EmbeddingTable::batch_lookup, nb::arg("ids"))
      .def("batch_sgd_update", &EmbeddingTable::batch_sgd_update,
           nb::arg("ids"), nb::arg("grads"), nb::arg("lr") = 0.01f)
      .def("get", &EmbeddingTable::get, nb::arg("key"))
      .def("size", &EmbeddingTable::size);
}
