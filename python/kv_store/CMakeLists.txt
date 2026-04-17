cmake_minimum_required(VERSION 3.15...3.27)
project(emb_table_ext)

# ── CPM.cmake（CMake Package Manager）────────────────────────────────────────
# 构建时自动下载；可通过 CPM_SOURCE_CACHE 缓存避免重复拉取：
#   cmake -DCPM_SOURCE_CACHE=~/.cache/cpm ..
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.2/CPM.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake"
)
include("${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake")

# stdexec（NVIDIA P2300 reference impl，纯头文件）
# DOWNLOAD_ONLY YES：跳过其 CMakeLists（依赖 NVIDIA RAPIDS cmake 模块）
CPMAddPackage(
  NAME              stdexec
  GITHUB_REPOSITORY NVIDIA/stdexec
  GIT_TAG           main
  DOWNLOAD_ONLY     YES
)
find_package(Threads REQUIRED)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# 确保 CMake 能找到 nanobind（解决 "找不到头文件 nanobind"）
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import nanobind; import os; print(os.path.dirname(nanobind.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_PKG_DIR)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_CMAKE_DIR)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Folly（ConcurrentHashMap）- 手动指定路径，绕过 pkg-config 引用不存在的 SDK 路径
find_package(PkgConfig REQUIRED)
pkg_check_modules(FOLLY REQUIRED libfolly)
# 过滤掉 pkg-config 中不存在的 include 路径（macOS SDK 版本漂移问题）
set(FOLLY_VALID_INCLUDE_DIRS "")
foreach(dir ${FOLLY_INCLUDE_DIRS})
  if(EXISTS "${dir}")
    list(APPEND FOLLY_VALID_INCLUDE_DIRS "${dir}")
  endif()
endforeach()

# 导出 compile_commands.json 供 IDE（clangd/VSCode）解析 include 路径
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 生成 .clangd 配置，使 IDE 能解析 nanobind + folly 头文件
list(GET FOLLY_VALID_INCLUDE_DIRS 0 FOLLY_FIRST_INCLUDE)
file(WRITE "${CMAKE_SOURCE_DIR}/.clangd"
  "CompileFlags:\n  Add:\n    - -I${nanobind_PKG_DIR}/include\n    - -I${FOLLY_FIRST_INCLUDE}\n    - -I${CMAKE_SOURCE_DIR}/csrc/third_party\n    - -I${stdexec_SOURCE_DIR}/include\n")

# NOMINSIZE: optimize for speed (-O3) since C++ does heavy embedding lookup/SGD
nanobind_add_module(emb_table_ext NOMINSIZE csrc/emb_table.cpp)
set_target_properties(emb_table_ext PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
target_include_directories(emb_table_ext PRIVATE
  ${FOLLY_VALID_INCLUDE_DIRS}
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/third_party"
  "${stdexec_SOURCE_DIR}/include")
target_link_directories(emb_table_ext PRIVATE ${FOLLY_LIBRARY_DIRS} /opt/homebrew/lib)
target_link_libraries(emb_table_ext PRIVATE ${FOLLY_LIBRARIES} glog gflags Threads::Threads)
target_compile_options(emb_table_ext PRIVATE ${FOLLY_CFLAGS_OTHER})
