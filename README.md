# kv store

## 简介

```
kv存储, 目的存储大规模 <uint64, float*> 键值对，设计目标十亿条

通过分片切分，unordered_map存放embedding在内存中的引用

```

## 环境

```
本地测试环境准备
1.conda create -n py310 python=3.10

  conda activate py310      
  conda deactivate      
  python -m ensurepip
  pip install jax pybind11 flax
  python setup.py build_ext --inplace

  conda install mpi4py openmpi
2.
```
