#!/bin/bash
set -euo pipefail

mkdir -p tmp

echo "=========================================="
echo "    Sparse Embedding Benchmark Script     "
echo "=========================================="

echo "[1/4] Running C++ Embedding Store Full Benchmark..."
bazel run -c opt //benchmarks:embedding_store_benchmark 2>&1 | tee tmp/benchmark_cpp_full.log

echo "[2/4] Running C++ Hotpath Microbenchmarks..."
bazel run -c opt //benchmarks:hotpath_microbench 2>&1 | tee tmp/benchmark_cpp_hotpath.log

echo "[3/4] Ensuring Python bindings are built..."
# Ensure the python extension is built, required for python benchmarks
pip install -e .

echo "[4/4] Running Python API Benchmarks..."
export PYTHONPATH=$(pwd)
echo "--- benchmark_ops.py ---"
python3 benchmarks/python/benchmark_ops.py 2>&1 | tee tmp/benchmark_py_ops.log

echo "--- benchmark_embedding_table.py ---"
python3 benchmarks/python/benchmark_embedding_table.py 2>&1 | tee tmp/benchmark_py_table.log

echo "--- benchmark_async_zipfian.py ---"
python3 benchmarks/python/benchmark_async_zipfian.py 2>&1 | tee tmp/benchmark_py_zipfian.log

echo "--- benchmark_storage_pressure.py ---"
python3 benchmarks/python/benchmark_storage_pressure.py 2>&1 | tee tmp/benchmark_py_storage_pressure.log

echo "--- benchmark_e2e.py ---"
python3 benchmarks/python/benchmark_e2e.py 2>&1 | tee tmp/benchmark_py_e2e.log

echo "=========================================="
echo "              All Done!                   "
echo "=========================================="
