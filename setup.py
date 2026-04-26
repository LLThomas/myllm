from setuptools import setup, find_packages
import torch
import os
import sys
from torch.utils import cpp_extension
import pybind11

is_windows = sys.platform.startswith("win")

sources = [
    "src/extensions/flash_attention.cpp",
    "src/extensions/flash_attention_cpu.cpp",
    "src/extensions/bindings.cpp",
]

if is_windows:
    extra_compile_args = {"cxx": ["/O2", "/std:c++17"]}
else:
    extra_compile_args = {"cxx": ["-O2", "-std=c++17", "-fPIC"]}

# Get pybind11 include directories
pybind11_include = pybind11.get_include()

torch_lib_dirs = []
torch_path = os.path.dirname(torch.__file__)
candidate_paths = [
    os.path.join(sys.prefix, "lib"),
    os.path.join(torch_path, "lib"),
]

lib_ext = ".lib" if is_windows else ".so"
for path in candidate_paths:
    if os.path.exists(path):
        torch_lib_dirs.append(path)
        break

ext_modules = [
    cpp_extension.CppExtension(
        name="my_llm_ext",
        sources=sources,
        extra_compile_args=extra_compile_args,
        include_dirs=["src/extensions", pybind11_include],
        library_dirs=torch_lib_dirs,
    )
]

setup(
    name="myllm",
    version="0.1.0",
    # src 布局: 指定包目录和发现规则
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    zip_safe=False,
)