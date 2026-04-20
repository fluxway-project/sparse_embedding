#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unique_ptr.h>
#include <memory>
#include <cstring>
#include "kv_store.h"

namespace nb = nanobind;

// Helper: RaggedTensors -> (values_ndarray, row_lengths_ndarray) for Python
static nb::tuple ragged_tensors_to_python(const RaggedTensors& rt) {
    float* copy_v = nullptr;
    if (!rt.values.empty()) {
        copy_v = new float[rt.values.size()];
        std::memcpy(copy_v, rt.values.data(), rt.values.size() * sizeof(float));
    }
    nb::capsule owner_v(copy_v, [](void *p) noexcept {
        delete[] (float*)p;
    });
    size_t shape_v[1] = { rt.values.size() };
    nb::ndarray<nb::numpy, float, nb::c_contig> values_arr(copy_v, 1, shape_v, owner_v);

    int* copy_l = nullptr;
    if (!rt.row_lengths.empty()) {
        copy_l = new int[rt.row_lengths.size()];
        std::memcpy(copy_l, rt.row_lengths.data(), rt.row_lengths.size() * sizeof(int));
    }
    nb::capsule owner_l(copy_l, [](void *p) noexcept {
        delete[] (int*)p;
    });
    size_t shape_l[1] = { rt.row_lengths.size() };
    nb::ndarray<nb::numpy, int, nb::c_contig> lengths_arr(copy_l, 1, shape_l, owner_l);

    return nb::make_tuple(values_arr, lengths_arr);
}

// Helper to convert a 2D NumPy array to std::vector<std::vector<float>>
std::vector<std::vector<float>> numpy_to_vec_of_vec(nb::ndarray<float, nb::c_contig, nb::device::cpu> arr) {
    if (arr.ndim() != 2) throw std::runtime_error("Input values should be a 2D array.");
    std::vector<std::vector<float>> result;
    result.reserve(arr.shape(0));
    const float* ptr = arr.data();
    size_t cols = arr.shape(1);
    for (size_t i = 0; i < arr.shape(0); ++i) {
        std::vector<float> row(cols);
        std::memcpy(row.data(), ptr + i * cols, cols * sizeof(float));
        result.push_back(std::move(row));
    }
    return result;
}

// KVStore constructor takes feature_metas (slot -> dim), init_type (0=zeros, 1=random)
// We use a custom __init__ for KVStore to construct it in-place

NB_MODULE(_core, m) {
    m.doc() = "Python bindings for the high-performance C++ KVStore";

    nb::class_<KVStore>(m, "KVStore")
        .def("__init__", [](KVStore *t, nb::dict feature_metas, int init_type, int num_shards) {
            std::unordered_map<uint32_t, int> fm;
            for (auto item : feature_metas) {
                fm[nb::cast<uint32_t>(item.first)] = nb::cast<int>(item.second);
            }
            new (t) KVStore(fm, init_type, num_shards);
        }, nb::arg("feature_metas"), nb::arg("init_type") = 0, nb::arg("num_shards") = 16,
             "Constructor: feature_metas maps slot_id->embedding_dim, init_type: 0=zeros, 1=random (seeded by key).")

        .def("insert", [](KVStore &self, nb::ndarray<uint64_t, nb::c_contig, nb::device::cpu> keys, nb::ndarray<float, nb::c_contig, nb::device::cpu> values) {
            if (keys.ndim() != 1) throw std::runtime_error("Keys should be a 1D array.");
            if (values.ndim() != 2) throw std::runtime_error("Values should be a 2D array.");
            if (keys.shape(0) != values.shape(0)) throw std::runtime_error("Number of keys and values must match.");
            
            std::vector<uint64_t> key_vec(keys.data(), keys.data() + keys.size());
            self.insert(key_vec, numpy_to_vec_of_vec(values));
        }, "Batch insert or update (upsert) key-value pairs from NumPy arrays.",
        nb::arg("keys"), nb::arg("values"))

        .def("update", [](KVStore &self, nb::ndarray<uint64_t, nb::c_contig, nb::device::cpu> keys, nb::ndarray<float, nb::c_contig, nb::device::cpu> values) {
            if (keys.ndim() != 1) throw std::runtime_error("Keys should be a 1D array.");
            if (values.ndim() != 2) throw std::runtime_error("Values should be a 2D array.");
            if (keys.shape(0) != values.shape(0)) throw std::runtime_error("Number of keys and values must match.");

            std::vector<uint64_t> key_vec(keys.data(), keys.data() + keys.size());
            self.update(key_vec, numpy_to_vec_of_vec(values));
        }, "Batch update values for existing keys from NumPy arrays. Keys that do not exist are ignored.",
        nb::arg("keys"), nb::arg("values"))

        .def("lookup", [](KVStore &self, nb::ndarray<uint64_t, nb::c_contig, nb::device::cpu> keys) {
            if (keys.ndim() != 1) throw std::runtime_error("Keys should be a 1D array.");
            std::vector<uint64_t> key_vec(keys.data(), keys.data() + keys.size());
            RaggedTensors rt = self.lookup(key_vec);
            return ragged_tensors_to_python(rt);
        }, "Batch read-only lookup from a NumPy array. Returns (values, row_lengths) as RaggedTensors.", nb::arg("keys"))

        .def("lookup_or_insert", [](KVStore &self, nb::ndarray<uint64_t, nb::c_contig, nb::device::cpu> keys) {
            if (keys.ndim() != 1) throw std::runtime_error("Keys should be a 1D array.");
            std::vector<uint64_t> key_vec(keys.data(), keys.data() + keys.size());
            RaggedTensors rt = self.lookup_or_insert(key_vec);
            return ragged_tensors_to_python(rt);
        }, "Batch lookup from a NumPy array. If a key is not found, it's inserted with a default value. Returns (values, row_lengths) as RaggedTensors.",
        nb::arg("keys"))

        .def("get_embedding_dim", [](KVStore &self, nb::ndarray<uint64_t, nb::c_contig, nb::device::cpu> fids) {
            if (fids.ndim() != 1) throw std::runtime_error("fids should be a 1D array.");
            std::vector<uint64_t> key_vec(fids.data(), fids.data() + fids.size());
            return self.get_embedding_dim(key_vec);
        }, nb::arg("fids"),
             "Return embedding dimension for the given feature ids (fids). Throws if slot is invalid.");
}

