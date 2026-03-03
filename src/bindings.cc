#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include "kv_store.h"

namespace py = pybind11;

// Helper: RaggedTensors -> (values_ndarray, row_lengths_ndarray) for Python
static py::tuple ragged_tensors_to_python(const RaggedTensors& rt) {
    py::array_t<float> values_arr(rt.values.size());
    if (!rt.values.empty()) {
        std::memcpy(values_arr.mutable_data(), rt.values.data(), rt.values.size() * sizeof(float));
    }
    py::array_t<int> lengths_arr(rt.row_lengths.size());
    if (!rt.row_lengths.empty()) {
        std::memcpy(lengths_arr.mutable_data(), rt.row_lengths.data(), rt.row_lengths.size() * sizeof(int));
    }
    return py::make_tuple(values_arr, lengths_arr);
}

// Helper to convert a 2D NumPy array to std::vector<std::vector<float>>
std::vector<std::vector<float>> numpy_to_vec_of_vec(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr) {
    if (arr.ndim() != 2) {
        throw std::runtime_error("Input values should be a 2D array.");
    }
    std::vector<std::vector<float>> result;
    result.reserve(arr.shape(0));
    auto unchecked = arr.unchecked<2>();
    for (ssize_t i = 0; i < unchecked.shape(0); ++i) {
        std::vector<float> row(unchecked.shape(1));
        // Using raw data pointer for faster copying
        std::memcpy(row.data(), unchecked.data(i, 0), unchecked.shape(1) * sizeof(float));
        result.push_back(std::move(row));
    }
    return result;
}


// Helper to build KVStore: feature_metas (slot -> dim), init_type (0=zeros, 1=random)
static std::unique_ptr<KVStore> make_kv_store(py::dict feature_metas, int init_type, int num_shards) {
    std::unordered_map<uint32_t, int> fm;
    for (auto item : feature_metas) {
        fm[item.first.cast<uint32_t>()] = item.second.cast<int>();
    }
    return std::make_unique<KVStore>(fm, init_type, num_shards);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Python bindings for the high-performance C++ KVStore";

    py::class_<KVStore, std::unique_ptr<KVStore>>(m, "KVStore")
        .def(py::init([](py::dict feature_metas, int init_type, int num_shards) {
            return make_kv_store(std::move(feature_metas), init_type, num_shards);
        }), py::arg("feature_metas"), py::arg("init_type") = 0, py::arg("num_shards") = 16,
             "Constructor: feature_metas maps slot_id->embedding_dim, init_type: 0=zeros, 1=random (seeded by key).")

        .def("insert", [](KVStore &self, const py::array_t<uint64_t, py::array::c_style | py::array::forcecast>& keys, const py::array_t<float, py::array::c_style | py::array::forcecast>& values) {
            if (keys.ndim() != 1) throw std::runtime_error("Keys should be a 1D array.");
            if (values.ndim() != 2) throw std::runtime_error("Values should be a 2D array.");
            if (keys.shape(0) != values.shape(0)) throw std::runtime_error("Number of keys and values must match.");
            
            std::vector<uint64_t> key_vec(keys.data(), keys.data() + keys.size());
            self.insert(key_vec, numpy_to_vec_of_vec(values));
        }, "Batch insert or update (upsert) key-value pairs from NumPy arrays.",
        py::arg("keys"), py::arg("values"))

        .def("update", [](KVStore &self, const py::array_t<uint64_t, py::array::c_style | py::array::forcecast>& keys, const py::array_t<float, py::array::c_style | py::array::forcecast>& values) {
            if (keys.ndim() != 1) throw std::runtime_error("Keys should be a 1D array.");
            if (values.ndim() != 2) throw std::runtime_error("Values should be a 2D array.");
            if (keys.shape(0) != values.shape(0)) throw std::runtime_error("Number of keys and values must match.");

            std::vector<uint64_t> key_vec(keys.data(), keys.data() + keys.size());
            self.update(key_vec, numpy_to_vec_of_vec(values));
        }, "Batch update values for existing keys from NumPy arrays. Keys that do not exist are ignored.",
        py::arg("keys"), py::arg("values"))

        .def("lookup", [](KVStore &self, const py::array_t<uint64_t, py::array::c_style | py::array::forcecast>& keys) {
            if (keys.ndim() != 1) throw std::runtime_error("Keys should be a 1D array.");
            std::vector<uint64_t> key_vec(keys.data(), keys.data() + keys.size());
            RaggedTensors rt = self.lookup(key_vec);
            return ragged_tensors_to_python(rt);
        }, "Batch read-only lookup from a NumPy array. Returns (values, row_lengths) as RaggedTensors.", py::arg("keys"))

        .def("lookup_or_insert", [](KVStore &self, const py::array_t<uint64_t, py::array::c_style | py::array::forcecast>& keys) {
            if (keys.ndim() != 1) throw std::runtime_error("Keys should be a 1D array.");
            std::vector<uint64_t> key_vec(keys.data(), keys.data() + keys.size());
            RaggedTensors rt = self.lookup_or_insert(key_vec);
            return ragged_tensors_to_python(rt);
        }, "Batch lookup from a NumPy array. If a key is not found, it's inserted with a default value. Returns (values, row_lengths) as RaggedTensors.",
        py::arg("keys"))

        .def("get_embedding_dim", [](KVStore &self, const py::array_t<uint64_t, py::array::c_style | py::array::forcecast>& fids) {
            if (fids.ndim() != 1) throw std::runtime_error("fids should be a 1D array.");
            std::vector<uint64_t> key_vec(fids.data(), fids.data() + fids.size());
            return self.get_embedding_dim(key_vec);
        }, py::arg("fids"),
             "Return embedding dimension for the given feature ids (fids). Throws if slot is invalid.");
}
