cmake_minimum_required(VERSION 3.18)
project(sparse_kv LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# 先找到 Python3，并启用 Development.Module / Development.Embed，确保定义 python3_add_library 和 Python3::Python
find_package(Python3 COMPONENTS Interpreter Development.Module Development.Embed REQUIRED)
message(STATUS "Using Python3: ${Python3_EXECUTABLE} (version ${Python3_VERSION})")

# 优先使用系统/环境已配置好的 pybind11
find_package(pybind11 QUIET)

# 如果没找到，则尝试通过当前 Python 自动查找 pybind11 的 CMake 路径
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
    RESULT_VARIABLE _pybind11_status
    OUTPUT_VARIABLE _pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(_pybind11_status EQUAL 0)
    set(pybind11_DIR "${_pybind11_cmake_dir}")
    message(STATUS "Using pybind11 from Python: ${pybind11_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
  else()
    message(FATAL_ERROR "pybind11 not found. Please install it in this Python (pip install pybind11) 或手动设置 pybind11_DIR。")
  endif()
endif()

# pybind11_add_module 会创建 Python 模块
# `_core` 是模块名，`src/bindings.cc` 等是源文件
pybind11_add_module(_core SHARED
    src/bindings.cc
    src/kv_store.cc
)