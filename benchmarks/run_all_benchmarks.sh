#!/bin/bash
set -e

echo "=========================================="
echo "    Sparse Embedding Benchmark Script     "
echo "=========================================="

echo "[1/4] Running C++ Embedding Store Full Benchmark..."
bazel run -c opt //benchmarks:embedding_store_benchmark

echo "[2/4] Running C++ Hotpath Microbenchmarks..."
bazel run -c opt //benchmarks:hotpath_microbench

echo "[3/4] Ensuring Python bindings are built..."
# Ensure the python extension is built, required for python benchmarks
pip install -e .

echo "[4/4] Running Python API Benchmarks..."
export PYTHONPATH=$(pwd)
echo "--- benchmark_ops.py ---"
python3 benchmarks/python/benchmark_ops.py

echo "--- benchmark_embedding_table.py ---"
python3 benchmarks/python/benchmark_embedding_table.py

echo "--- benchmark_async_zipfian.py ---"
python3 benchmarks/python/benchmark_async_zipfian.py

echo "=========================================="
echo "              All Done!                   "
echo "=========================================="
