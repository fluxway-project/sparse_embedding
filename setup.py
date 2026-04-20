# coding=utf-8

# 一个简化的 setup.py，用于调用 CMake
# 在生产项目中，建议使用 pybind/cmake_example 中的模板
import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        build_args = ["--config", "Release", "--", "-j4"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


import sys

setup(
    name="sparse_embedding",
    version="0.1.0",
    author="Your Name",
    description="A high-performance C++/Python Sparse Embedding KV Store",
    long_description="",
    ext_modules=[CMakeExtension("sparse_embedding._core")],
    cmdclass={"build_ext": CMakeBuild},
    packages=["sparse_embedding", "sparse_embedding.optim"],
    package_dir={"": "."},
    zip_safe=False,
)
