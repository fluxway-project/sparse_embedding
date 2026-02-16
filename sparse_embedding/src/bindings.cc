#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include "kv_store.h"

namespace py = pybind11;

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


// Helper to build KVStore from Python dicts: feature_metas (slot -> dim), default_values (slot -> list/array of float)
static std::unique_ptr<KVStore> make_kv_store(py::dict feature_metas, py::dict default_values, int num_shards) {
    std::unordered_map<uint32_t, int> fm;
    std::unordered_map<uint32_t, ExtValueType> dv;
    for (auto item : feature_metas) {
        fm[item.first.cast<uint32_t>()] = item.second.cast<int>();
    }
    for (auto item : default_values) {
        py::handle val = item.second;
        ExtValueType vec;
        if (py::isinstance<py::array>(val)) {
            auto arr = val.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            if (arr.ndim() != 1) throw std::runtime_error("Default value array must be 1D.");
            vec.assign(arr.data(), arr.data() + arr.size());
        } else {
            vec = val.cast<std::vector<float>>();
        }
        dv[item.first.cast<uint32_t>()] = std::move(vec);
    }
    return std::make_unique<KVStore>(fm, dv, num_shards);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Python bindings for the high-performance C++ KVStore";

    py::class_<KVStore, std::unique_ptr<KVStore>>(m, "KVStore")
        .def(py::init([](py::dict feature_metas, py::dict default_values, int num_shards) {
            return make_kv_store(std::move(feature_metas), std::move(default_values), num_shards);
        }), py::arg("feature_metas"), py::arg("default_values"), py::arg("num_shards") = 16,
             "Constructor: feature_metas maps slot_id->embedding_dim, default_values maps slot_id->default vector.")

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
            return self.lookup(key_vec);
        }, "Batch read-only lookup from a NumPy array.", py::arg("keys"))

        .def("lookup_or_insert", [](KVStore &self, const py::array_t<uint64_t, py::array::c_style | py::array::forcecast>& keys) {
            if (keys.ndim() != 1) throw std::runtime_error("Keys should be a 1D array.");
            std::vector<uint64_t> key_vec(keys.data(), keys.data() + keys.size());
            return self.lookup_or_insert(key_vec);
        }, "Batch lookup from a NumPy array. If a key is not found, it's inserted with a default value.",
        py::arg("keys"))

        .def("get_embedding_dim", &KVStore::get_embedding_dim, py::arg("fid"),
             "Return embedding dimension for the given feature id (fid). Throws if slot is invalid.");
}
