#!/usr/bin/env bash
# 使用当前设置的 Python 虚拟环境（请先激活：source .venv/bin/activate 或 conda activate xxx）
# 若已激活，优先使用 $VIRTUAL_ENV/bin/python
set -e
dirpath=$(dirname "$0")
cd "$(dirname "$0")"

VIRTUAL_ENV=/Users/admin/miniforge3/envs/py310

if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi
echo "Using: $PYTHON ($($PYTHON --version 2>&1))"

# 单机 2 进程，CPU 模式；GLOO_IFACE 按本机网卡改（mac 常用 lo0）
# 若未安装 mpirun，可改为单进程：去掉 mpirun，直接 $PYTHON train_demo.py
if command -v mpirun >/dev/null 2>&1; then
  mpirun -np 2 \
    --tag-output \
    -x JAX_COORDINATOR_ADDRESS=localhost:34567 \
    -x JAX_NUM_PROCESSES=2 \
    -x JAX_PLATFORMS=cpu \
    -x GLOO_IFACE=lo0 \
    -x VIRTUAL_ENV \
    bash -c "cd $dirpath && \"$PYTHON\" sparse_train_demo_dist.py"
else
  echo "mpirun not found, running single process."
  export JAX_PLATFORMS=cpu
  "$PYTHON" train_demo.py
fi
